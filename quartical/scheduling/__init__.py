from collections import defaultdict
from ast import literal_eval
from distributed.diagnostics import SchedulerPlugin


class QuarticalScheduler(SchedulerPlugin):
    def update_graph(self, scheduler, dsk=None, keys=None, restrictions=None, **kw):
        if "annotations" not in kw:
            return

        tasks = scheduler.tasks
        workers = list(scheduler.workers.keys())

        partitions = defaultdict(list)

        for k, a in kw["annotations"].get("__dask_array__", {}).items():
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

        # Stripe partitions across workers
        for p, (partition, keys) in enumerate(sorted(partitions.items())):
            wid = int(len(workers) * p / npartitions)

            for k in keys:
                ts = tasks.get(k)
                ts._worker_restrictions = set([workers[wid]])


def install_plugin(dask_scheduler=None, **kwargs):
    dask_scheduler.add_plugin(QuarticalScheduler(**kwargs))