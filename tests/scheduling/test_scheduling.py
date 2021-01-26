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
        cluster = stack.enter_context(LocalCluster(processes=False))
        scheduler = cluster.scheduler
        client = stack.enter_context(Client(cluster))

        opts.parallel = Namespace(scheduler='distributed',
                                  address=scheduler.address)

        opts._model_columns = ["MODEL_DATA"]

        datasets, _, _ = read_xds_list(opts)

        client.run_on_scheduler(install_plugin)
        assert len(scheduler.plugins) > 1

        with dask.config.set(optimization__fuse__active=False):
            datasets[0].DATA.data.compute()