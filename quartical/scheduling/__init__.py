from ast import literal_eval
from distributed.diagnostics import SchedulerPlugin


class QuarticalScheduler(SchedulerPlugin):
    def update_graph(self, scheduler, dsk=None, keys=None, restrictions=None, **kw):
        if "annotations" not in kw:
            return

        tasks = scheduler.tasks
        workers = list(scheduler.workers.keys())

        for k, a in kw["annotations"].get("__dask_array__", {}).items():
            # Map block id's and chunks to dimensions
            dims = {d: int(b) for d, b in zip(a["dims"], literal_eval(k)[1:])}
            chunks = {d: c for d, c in zip(a["dims"], a["chunks"])}

            # Extract row block and number of row blocks
            try:
                row_block = dims["row"]
                nrow_blocks = len(chunks["row"])
            except KeyError:
                continue

            # Stripe across workers
            wid = int((row_block / nrow_blocks) * len(workers))
            ts = tasks.get(k)
            ts._worker_restrictions = set([workers[wid]])


def install_plugin(dask_scheduler=None, **kwargs):
    dask_scheduler.add_plugin(QuarticalScheduler(**kwargs))