from argparse import Namespace
from contextlib import ExitStack

import dask
from distributed import Client, LocalCluster
import pytest

from quartical.data_handling.ms_handler import read_xds_list

from quartical.scheduling import install_plugin

def test_distributed(base_opts):
    opts = Namespace(**vars(base_opts))

    with ExitStack() as stack:
        cluster = stack.enter_context(LocalCluster(processes=False, n_workers=4))
        client = stack.enter_context(Client(cluster))
        scheduler = cluster.scheduler

        opts.input_ms_time_chunk = 4
        opts.parallel = Namespace(scheduler='distributed',
                                  address=scheduler.address)

        opts._model_columns = ["MODEL_DATA"]

        datasets, _, _ = read_xds_list(opts)
        assert len(datasets) == 2
        assert len(datasets[0].chunks["row"]) == 29
        assert len(datasets[1].chunks["row"]) == 27

        client.run_on_scheduler(install_plugin)
        assert len(scheduler.plugins) > 1

        columns = ("DATA", "UVW", "TIME")
        computes = [getattr(ds, c).data.sum() for c in columns for ds in datasets]

        with dask.config.set(optimization__fuse__active=False):
            dask.compute(computes)
