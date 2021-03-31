from collections import defaultdict
from ast import literal_eval

import dask
import dask.array as da
from daskms.constants import DASKMS_PARTITION_KEY as PARTITION_KEY
from dask.highlevelgraph import HighLevelGraph
from distributed.diagnostics.plugin import SchedulerPlugin
from dask.core import get_deps
from dask.order import ndependencies
from dask.base import tokenize, unpack_collections, collections_to_dsk
import xarray


def annotate_dask_collection(collection, annotations):
    if not isinstance(collection, da.Array):
        raise NotImplementedError(f"annotation of {type(collection)} "
                                  f"not yet supported")

    hlg = collection.__dask_graph__()

    if type(hlg) is not HighLevelGraph:
        raise TypeError(f"hlg is not a HighLevelGraph: {type(hlg)}")

    top_layer = hlg.layers[collection.name]

    # Integrate array data with existing annotations
    annotations = {
        "__dask_array__": {
            "dtype": collection.dtype.name,
            "chunks": collection.chunks,
            **annotations
        }
    }

    # Try ensure the number of dims match the number of chunks (ndim)
    try:
        dims = annotations["dims"]
    except KeyError:
        pass
    else:
        assert len(annotations["chunks"]) == len(dims)

    if top_layer.annotations:
        # Annotations already exist, do some sanity checks
        try:
            a = top_layer.annotations["__dask_array__"]
        except KeyError:
            # Destructive update
            top_layer.annotations = annotations
        else:
            if "row" in a["dims"] and top_layer.annotations != annotations:
                raise ValueError(f"Trying to annotate a row array with "
                                 f"differing annotations\n."
                                 f"original: {top_layer.annotations}"
                                 f"new: {annotations}")
            else:
                # TODO(sjperkins) Improve this
                # We hit this case when trying to annotate a single array
                # that has been assigned to multiple datasets.
                # i.e. POSITION: ((ant,), position) from the ANTENNA subtable
                # This causes issues because partitioning information differs
                # between datasets, but we don't really care about annotations
                # for this type of array
                pass
    else:
        top_layer.annotations = annotations


def dataset_partition(ds):
    # Add any dataset partitioning information to annotations
    try:
        partition = getattr(ds, PARTITION_KEY)
    except AttributeError:
        raise ValueError(f"{ds} has no {PARTITION_KEY} attribute")
    else:
        return tuple((p, getattr(ds, p)) for p, _ in partition)


def annotate(obj, **annotations):
    if isinstance(obj, (tuple, list)):
        for o in obj:
            annotate(o, **annotations)
    elif isinstance(obj, xarray.Dataset):
        partition = dataset_partition(obj)

        if partition:
            annotations = {**annotations, "partition": partition}

        for var in obj.data_vars.values():
            annotate(var, **annotations)

    elif isinstance(obj, xarray.DataArray):
        # Annotate any wrapped dask collections
        # with dimension information
        if not dask.is_dask_collection(obj.data):
            return  # Ignore numpy arrays etc.

        annotate_dask_collection(obj.data, {**annotations, "dims": obj.dims})
    elif dask.is_dask_collection(obj):
        annotate_dask_collection(obj, annotations)
    else:
        raise TypeError(f"obj must be a dask collection, "
                        f"xarray.Dataset, xarray.DataArray, "
                        f"or list of the previous types. "
                        f"Got a: {type(obj)}")


class QuarticalScheduler(SchedulerPlugin):
    def update_graph(self, scheduler, dsk=None, keys=None, restrictions=None,
                     **kw):
        try:
            annotations = kw["annotations"]
        except KeyError:
            return

        tasks = scheduler.tasks
        workers = list(scheduler.workers.keys())
        partitions = defaultdict(list)

        for k, a in annotations.get("__dask_array__", {}).items():
            try:
                p = a["partition"]
                dims = a["dims"]
                chunks = a["chunks"]
            except KeyError:
                continue

            try:
                ri = dims.index("row")
            except ValueError:
                # print(f"{k} not understood")
                # print(f"{tasks.get(k)}")
                continue

            # Map block id's and chunks to dimensions
            block = tuple(map(int, literal_eval(k)[1:]))
            pkey = p + (("__row_block__", block[ri]),)
            partitions[pkey].append(k)

        npartitions = len(partitions)

        for p, (partition, keys) in enumerate(sorted(partitions.items())):
            roots = {k for k in keys if len(tasks.get(k)._dependencies) == 0}

        # Stripe partitions across workers
        for p, (partition, keys) in enumerate(sorted(partitions.items())):
            worker = set([workers[int(len(workers) * p / npartitions)]])

            for k in keys:
                ts = tasks.get(k)
                ts._worker_restrictions = worker
                ts._loose_restrictions = False


def install_plugin(dask_scheduler=None, **kwargs):
    dask_scheduler.add_plugin(QuarticalScheduler(**kwargs), idempotent=True)


def interrogate_annotations(collection):

    hlg = collection.__dask_graph__()
    layers = hlg.layers
    deps = hlg.dependencies

    for k, v in layers.items():
        if v.annotations is None:
            print(k, deps[k])

    for k, v in deps.items():
        if layers[k].annotations is None:
            print(k, v)

    # import pdb; pdb.set_trace()

    return


def grouped_annotate(*args):

    group_offset = 0

    for collections in zip(*args):
        group_offset = annotate_traversal(*collections,
                                          group_offset=group_offset)


def annotate_traversal(*args, group_offset=0):

    collections, _ = unpack_collections(*args, traverse=True)

    hlg = collections_to_dsk(collections, optimize_graph=False)
    layers = hlg.layers

    dependencies, dependents = get_deps(hlg)
    _, total_dependencies = ndependencies(dependencies, dependents)

    terminal_nodes = {k for (k, v) in dependents.items() if v == set()}
    root_nodes = {k for (k, v) in dependencies.items() if v == set()}

    terminal_roots = {}
    unravelled_deps = {}

    for group_id, task_name in enumerate(terminal_nodes):
        unravelled_deps[task_name] = unravel_deps(dependencies, task_name)
        terminal_roots[task_name] = root_nodes & unravelled_deps[task_name]

    root_tokens = {tokenize(*sorted(v)): v for v in terminal_roots.values()}

    hash_map = defaultdict(set)

    for k, v in root_tokens.items():
        if any([v < vv for vv in root_tokens.values()]):  # Strict subset.
            continue
        else:
            hash_map[k] |= set([group_offset])
            group_offset += 1

    for k, v in root_tokens.items():
        shared_roots = {kk: None for kk, vv in root_tokens.items() if v < vv}
        if shared_roots:
            hash_map[k] = \
                set().union(*[hash_map[kk] for kk in shared_roots.keys()])

    for k, v in unravelled_deps.items():

        group = hash_map[tokenize(*sorted(terminal_roots[k]))]

        annotate_layers(layers, v, k, group)

    return group_offset


def annotate_layers(layers, unravelled_deps, task_name, group):

    for name in [task_name, *unravelled_deps]:

        layer_name = name[0]

        annotation = layers[layer_name].annotations

        if annotation is None:
            annotation = layers[layer_name].annotations = {}

        if isinstance(annotation, dict):
            if not ("__group__" in annotation):
                annotation["__group__"] = defaultdict(set)
        else:
            raise ValueError(f"Annotations are expected to be a dictionary - "
                             f"got {type(annotation)}.")

        annotation["__group__"][name] |= group

    return


def unravel_deps(hlg_deps, name, unravelled_deps=None):

    if unravelled_deps is None:
        unravelled_deps = set()

    for dep in hlg_deps[name]:
        unravelled_deps |= {dep}
        unravel_deps(hlg_deps, dep, unravelled_deps)

    return unravelled_deps
