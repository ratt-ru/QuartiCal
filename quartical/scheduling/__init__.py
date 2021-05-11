from ast import literal_eval
from collections import defaultdict
import itertools
import logging

from distributed.diagnostics.plugin import SchedulerPlugin
import dask
from dask.core import reverse_dict
from dask.base import tokenize
from dask.order import order


log = logging.getLogger(__file__)


def unravel_deps(hlg_deps, name, unravelled_deps=None):
    """Recursively construct a set of all dependencies for a specific task."""

    if unravelled_deps is None:
        unravelled_deps = set()

    for dep in hlg_deps[name]:
        unravelled_deps |= {dep}
        unravel_deps(hlg_deps, dep, unravelled_deps)

    return unravelled_deps


class AutoRestrictor(SchedulerPlugin):

    def update_graph(self, scheduler, dsk=None, keys=None, restrictions=None,
                     **kw):
        """Processes dependencies to assign tasks to specific workers."""
        tasks = scheduler.tasks
        dependencies = kw["dependencies"]
        dependents = reverse_dict(dependencies)

        # Terminal nodes have no dependents, root nodes have no dependencies.
        terminal_nodes = {k for (k, v) in dependents.items() if not v}
        root_nodes = {k for (k, v) in dependencies.items() if not v}

        roots_per_terminal = {}
        terminal_deps = {}

        for tn in terminal_nodes:
            # Get dependencies per terminal node.
            terminal_deps[tn] = unravel_deps(dependencies, tn)
            # Associate terminal nodes with root nodes.
            roots_per_terminal[tn] = root_nodes & terminal_deps[tn]

        # Create a unique token for each set of terminal roots. TODO: This is
        # very strict. What about nodes with very similar roots? Tokenization
        # may be overkill too.
        root_tokens = \
            {tokenize(*sorted(v)): v for v in roots_per_terminal.values()}

        hash_map = defaultdict(set)
        group_offset = 0

        # Associate terminal roots with a specific group if they are not a
        # subset of another larger root set. TODO: This can likely be improved.
        for k, v in root_tokens.items():
            if any([v < vv for vv in root_tokens.values()]):  # Strict subset.
                continue
            else:
                hash_map[k] |= set([group_offset])
                group_offset += 1

        # If roots were a subset, they should share the annotation of their
        # superset/s.
        for k, v in root_tokens.items():
            shared_roots = \
                {kk: None for kk, vv in root_tokens.items() if v < vv}
            if shared_roots:
                hash_map[k] = \
                    set().union(*[hash_map[kk] for kk in shared_roots.keys()])

        workers = list(scheduler.workers.keys())
        n_worker = len(workers)

        for k, deps in terminal_deps.items():

            # TODO: This can likely be improved.
            group = hash_map[tokenize(*sorted(roots_per_terminal[k]))]

            # Set restrictions on a terminal node and its dependencies.
            for tn in [k, *deps]:
                try:
                    task = tasks[tn]
                except KeyError:  # Keys may not have an associated task.
                    continue
                if task._worker_restrictions is None:
                    task._worker_restrictions = set()
                task._worker_restrictions |= \
                    {workers[g % n_worker] for g in group}
                task._loose_restrictions = False
                # (user priority, graph generation, dask.order priority)
                task._priority = (min(group),) + task._priority[1:]


COLOUR = "__quartical_colour__"


class Dummy:
    def __init__(self, name):
        self.__name__ = name

    def __call__(self, *args, **kwargs):
        pass

    def __reduce__(self):
        return (Dummy, (self.__name__,))

    def __repr__(self):
        return self.__name__

    __str__ = __repr__


class BreadthFirstSearch:
    def __init__(self, roots, colour):
        if not isinstance(roots, (tuple, list)):
            roots = [roots]

        print(f"{colour} has {len(roots)} roots")

        self.roots = roots
        self.colour = colour

    def initialise(self):
        for r in self.roots:
            r._annotations[COLOUR] = self.colour

        self.frontier = self.roots.copy()
        self.dsk = {r.key: () for r in self.frontier}
        self.visited = set()
        self.n = 0

    def iterate(self):
        colour = self.colour
        from operator import add

        try:
            # Remove first node in the frontier
            node = self.frontier.pop(0)
        except IndexError:
            raise StopIteration
        else:
            assert colour == node._annotations[COLOUR]
            self.visited.add(node)

        for child in itertools.chain(node.dependents, node.dependencies):
            if child in self.frontier or child in self.visited:
                continue

            if child._annotations.get(COLOUR, None) is not None:
                continue

            child._annotations[COLOUR] = colour
            self.frontier.append(child)

        from dask.utils import key_split
        self.dsk[node.key] = (Dummy(key_split(node.key)), *(cd.key for cd in node.dependencies))

        self.n += 1
        return node

    @property
    def order(self):
        return order(self.dsk)

    def __next__(self):
        return self.iterate()

    def __iter__(self):
        self.initialise()
        return self



class ColouringPlugin(SchedulerPlugin):
    def update_graph(self, scheduler, tasks=None,
                     annotations=None, dependencies=None,
                     **kw):
        dsk = tasks

        if not annotations:
            return

        tasks = scheduler.tasks
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
                continue

            row_block = int(literal_eval(k)[1 + ri])
            pkey = p + (("__row_block__", row_block))
            partitions[pkey].append(k)

        searches = []

        for colour, frontier in enumerate(partitions.values()):
            roots = [tasks.get(k) for k in frontier]
            assert all(t is not None for t in roots)
            bfs = BreadthFirstSearch(roots, colour)
            searches.append(bfs)

        for bfs in searches:
            bfs.initialise()

        success = True

        while success:
            success = False

            # Do one iteration of each bfs
            for bfs in searches:
                try:
                    bfs.iterate()
                except StopIteration:
                    pass
                else:
                    # We just try iterate over the searches
                    # once more
                    success = True

        workers = list(scheduler.workers.keys())
        n_workers = len(workers)

        colours = {}

        for s, bfs in enumerate(searches):
            w = {workers[int(n_workers * s / len(searches))]}

            # order = {}

            for k, p in bfs.order.items():
                priority = float(1 + s)*10.0 + p / 1000.0
                # priority = float(1 + s)*10.0
                t = tasks.get(k)
                t._priority = (priority,) + t.priority[1:]
                t._worker_restrictions = w
                t._loose_restrictions = False

                colours[k] = priority
                # order[k] = priority

            # if s == 1:
            #     from pprint import pprint
            #     dask.visualize(bfs.dsk, filename="order.pdf", color="order")
            #     pprint(bfs.dsk)

        if False:
            # Graph visualization and debugging
            import pickle
            import logging
            from distributed.protocol import deserialize, Serialized

            dsk2 = {}

            for k, v in dsk.items():
                if isinstance(v, dict):
                    dsk2[k] = (pickle.loads(v["function"]), *dependencies[k])
                elif isinstance(v, Serialized):
                    dsk2[k] = deserialize(v.header, v.frames)
                else:
                    dsk2[k] = v

            dask.visualize(dsk2, color=colours, filename="graph.pdf")

        log.info("Plugin done")

def install_plugin(dask_scheduler=None, **kwargs):
    # dask_scheduler.add_plugin(AutoRestrictor(**kwargs), idempotent=True)
    dask_scheduler.add_plugin(ColouringPlugin(**kwargs), idempotent=True)


