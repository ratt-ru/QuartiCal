from ast import literal_eval
from collections import defaultdict

from distributed.diagnostics.plugin import SchedulerPlugin
from dask.core import reverse_dict
from dask.base import tokenize


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

class BreadthFirstSearch:
    def __init__(self, dsk, roots, colour):
        if not isinstance(roots, (tuple, list)):
            roots = [roots]

        print(f"{colour} has {len(roots)} roots")

        self.roots = roots
        self.dsk = dsk
        self.colour = colour

    def initialise(self):
        for r in self.roots:
            r._annotations[COLOUR] = self.colour

        self.frontier = self.roots.copy()
        self.n = 0

    def iterate(self):
        try:
            # Remove first node in the frontier
            node = self.frontier.pop(0)
        except IndexError:
            print(f"Stopping {self.colour} after {self.n} iterations")
            raise StopIteration
        # else:
        #     assert self.colour == node._annotations[COLOUR]

        for child in node.dependents:
            if child in self.frontier:
                continue

            if COLOUR in child.annotations:
                continue

            child._annotations[COLOUR] = self.colour
            self.frontier.append(child)

        self.n += 1
        return node

    def __next__(self):
        return self.iterate()


    def __iter__(self):
        self.initialise()
        return self



class ColouringPlugin(SchedulerPlugin):
    def update_graph(self, scheduler, dsk=None,
                     annotations=None, dependencies=None,
                     **kw):

        if not annotations:
            return

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

        # print(f"partitions {len(partitions)}")
        # from pprint import pprint
        # pprint(partitions)

        searches = []
        tasks = scheduler.tasks

        for colour, frontier in enumerate(partitions.values()):
            roots = [tasks.get(k) for k in frontier]
            assert all(t is not None for t in roots)
            bfs = BreadthFirstSearch(dsk, roots, colour)
            searches.append(bfs)

        for bfs in searches:
            bfs.initialise()

        iterated = True

        while iterated:
            iterated = False

            for bfs in searches:
                try:
                    bfs.iterate()
                except StopIteration:
                    pass
                else:
                    iterated = True

        colour_counts = defaultdict(set)

        for k, t in tasks.items():
            try:
                colour = t.annotations["colour"]
            except KeyError:
                pass
            else:
                print(f"{k} = {colour}")
                colour_counts[colour].add(k)

        from pprint import pprint
        pprint({k: len(v) for k, v in colour_counts.items()})



def install_plugin(dask_scheduler=None, **kwargs):
    dask_scheduler.add_plugin(ColouringPlugin(**kwargs), idempotent=True)


