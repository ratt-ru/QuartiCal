from collections import defaultdict
import warnings

from distributed.diagnostics.plugin import SchedulerPlugin
from dask.core import get_deps
from dask.order import ndependencies
from dask.base import tokenize, unpack_collections, collections_to_dsk


def install_plugin(dask_scheduler=None, **kwargs):
    dask_scheduler.add_plugin(AutoScheduler(**kwargs), idempotent=True)


def interrogate_annotations(collection):
    """Utitlity function to identify unannotated layers in a collection."""

    hlg = collection.__dask_graph__()
    layers = hlg.layers
    deps = hlg.dependencies

    for k, v in layers.items():
        if v.annotations is None:
            print(k, deps[k])

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

        layer_name = name[0] if isinstance(name, tuple) else name

        annotation = layers[layer_name].annotations

        if annotation is None:
            annotation = layers[layer_name].annotations = {}

        if isinstance(annotation, dict):
            if not ("__group__" in annotation):
                annotation["__group__"] = defaultdict(set)
        else:
            raise ValueError(f"Annotations are expected to be a dictionary - "
                             f"got {type(annotation)}.")

        annotation["__group__"][str(name)] |= group

    return


def unravel_deps(hlg_deps, name, unravelled_deps=None):
    """Recursively construct a set of all dependencies for a specific task."""

    if unravelled_deps is None:
        unravelled_deps = set()

    for dep in hlg_deps[name]:
        unravelled_deps |= {dep}
        unravel_deps(hlg_deps, dep, unravelled_deps)

    return unravelled_deps


class AutoScheduler(SchedulerPlugin):
    def update_graph(self, scheduler, dsk=None, keys=None, restrictions=None,
                     **kw):
        """Uses __group__ annotations to assign tasks to specific workers."""
        try:
            annotations = kw["annotations"]["__group__"]
        except KeyError:
            return

        tasks = scheduler.tasks
        workers = list(scheduler.workers.keys())
        n_worker = len(workers)

        for k, tsk in tasks.items():
            try:
                # From all annotations, get this task's annotation. Double get
                # arises as we have a dict per layer.
                grp = annotations.get(k).get(k)
            except AttributeError:  # Annotations on delayed objects disappear?
                warnings.warn(
                    f"__group__ annotations appear to be absent for {tsk}. "
                    f"This usually happens for annotated Delayed objects "
                    f"when optimization is enabled.")
                continue
            except KeyError:
                warnings.warn(f"No valid __group__ annotation for {k}.")
                continue

            # TODO: This is a little simplisitic - a round-robin stategy will
            # be suboptimal when the graph is inhomogeonous.
            tsk._worker_restrictions = {workers[g % n_worker] for g in grp}
            tsk._loose_restrictions = False
