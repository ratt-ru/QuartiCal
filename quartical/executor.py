# -*- coding: utf-8 -*-
# Sets up logger - hereafter import logger from Loguru.
from contextlib import ExitStack
import quartical.logging.init_logger  # noqa
from loguru import logger
import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
import time
from quartical.parser import parser, preprocess, helper
from quartical.data_handling.ms_handler import (read_xds_list,
                                                write_xds_list,
                                                preprocess_xds_list)
from quartical.data_handling.model_handler import add_model_graph
from quartical.calibration.calibrate import add_calibration_graph
from quartical.flagging.flagging import finalise_flags, add_mad_graph
from quartical.scheduling import install_plugin
from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
from quartical.gains.datasets import write_gain_datasets


@logger.catch
def execute():
    with ExitStack() as stack:
        _execute(stack)


def _execute(exitstack):
    """Runs the application."""

    helper.help()  # Check to see if the user asked for help.
    opts = parser.parse_inputs()

    # TODO: This check needs to be fleshed out substantially.

    preprocess.check_opts(opts)
    preprocess.interpret_model(opts)

    if opts.parallel.scheduler == "distributed":
        if opts.parallel.address:
            logger.info("Initializing distributed client.")
            client = exitstack.enter_context(Client(opts.parallel.address))
        else:
            logger.info("Initializing distributed client using LocalCluster.")
            cluster = LocalCluster(processes=opts.parallel.n_worker > 1,
                                   n_workers=opts.parallel.n_worker,
                                   threads_per_worker=opts.parallel.n_thread,
                                   memory_limit=0)
            cluster = exitstack.enter_context(cluster)
            client = exitstack.enter_context(Client(cluster))

        # Install Quartical Scheduler Plugin
        # Controversial from a security POV,
        # run_on_scheduler is a debugging function
        # `dask-scheduler --preload install_plugin.py`
        # is the standard but less convenient pattern
        client.run_on_scheduler(install_plugin)

        logger.info("Distributed client sucessfully initialized.")

    t0 = time.time()

    # Reads the measurement set using the relavant configuration from opts.
    data_xds_list, ref_xds_list = read_xds_list(opts)

    # logger.info("Reading data from zms.")
    # data_xds_list = xds_from_zarr("/home/jonathan/3C147_tests/3C147_daskms.zms")

    # writes = xds_to_zarr(data_xds_list, "/home/jonathan/3C147_tests/3C147_daskms.zms")
    # dask.compute(writes)
    # return

    # Preprocess the xds_list - initialise some values and fix bad data.
    data_xds_list = preprocess_xds_list(data_xds_list, opts)

    # Model xds is a list of xdss onto which appropriate model data has been
    # assigned.
    data_xds_list = add_model_graph(data_xds_list, opts)

    # Adds the dask graph describing the calibration of the data.
    gain_xds_lol, data_xds_list = \
        add_calibration_graph(data_xds_list, opts)

    if opts.mad_flags.enable:
        data_xds_list = add_mad_graph(data_xds_list, opts)

    writable_xds = finalise_flags(data_xds_list, opts)

    writes = write_xds_list(writable_xds, ref_xds_list, opts)

    gain_writes = write_gain_datasets(gain_xds_lol, opts)

    logger.success("{:.2f} seconds taken to build graph.", time.time() - t0)

    t0 = time.time()

    with ProgressBar():

        dask.compute(writes, gain_writes,
                     num_workers=opts.parallel.n_thread,
                     optimize_graph=True,
                     scheduler=opts.parallel.scheduler)

    logger.success("{:.2f} seconds taken to execute graph.", time.time() - t0)

    # dask.visualize(*writes[:1], *gain_writes[:1],
    #                color='order', cmap='autumn',
    #                filename='order.pdf', node_attr={'penwidth': '10'},
    #                optimize_graph=True)

    # dask.visualize(*writes[:1], *gain_writes[:1],
    #                filename='graph.pdf',
    #                optimize_graph=True)

    # dask.visualize(*gains_per_xds["G"],
    #                filename='gain_graph',
    #                format="pdf",
    #                optimize_graph=True,
    #                collapse_outputs=True,
    #                node_attr={'penwidth': '4',
    #                           'fontsize': '18',
    #                           'fontname': 'helvetica'},
    #                edge_attr={'penwidth': '2', })
