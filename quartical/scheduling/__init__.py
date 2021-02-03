from collections import defaultdict
from ast import literal_eval

import dask
import dask.array as da
from daskms.reads import PARTITION_KEY
from dask.highlevelgraph import HighLevelGraph
from distributed.diagnostics import SchedulerPlugin
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
        return None
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
    def update_graph(self, scheduler, dsk=None, keys=None, restrictions=None, **kw):
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

            ri = dims.index("row")
            if ri == -1:
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


def install_plugin(dask_scheduler=None, **kwargs):
    dask_scheduler.add_plugin(QuarticalScheduler(**kwargs))