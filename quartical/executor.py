# -*- coding: utf-8 -*-
# Sets up logger - hereafter import logger from Loguru.
from contextlib import ExitStack
import quartical.logging.init_logger  # noqa
from loguru import logger
import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
import time
from quartical.parser import parser, preprocess
from quartical.data_handling.ms_handler import (read_xds_list,
                                                write_xds_list,
                                                preprocess_xds_list)
from quartical.data_handling.model_handler import add_model_graph
from quartical.calibration.calibrate import add_calibration_graph
from quartical.flagging.flagging import finalise_flags, add_mad_graph
from quartical.scheduling import install_plugin, grouped_annotate
from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr
from quartical.calibration.gain_datasets import write_gain_datasets


@logger.catch
def execute():
    with ExitStack() as stack:
        _execute(stack)


def _execute(exitstack):
    """Runs the application."""

    opts = parser.parse_inputs()

    # TODO: This check needs to be fleshed out substantially.

    preprocess.check_opts(opts)
    preprocess.interpret_model(opts)

    if opts.parallel_scheduler == "distributed":
        optimize_graph = False
        if opts.parallel_address:
            logger.info("Initializing distributed client.")
            client = exitstack.enter_context(Client(opts.parallel_address))
        else:
            logger.info("Initializing distributed client using LocalCluster.")
            cluster = LocalCluster(processes=opts.parallel_nworker > 1,
                                   n_workers=opts.parallel_nworker,
                                   threads_per_worker=opts.parallel_nthread,
                                   memory_limit=0)
            cluster = exitstack.enter_context(cluster)
            client = exitstack.enter_context(Client(cluster))

        # Install Quartical Scheduler Plugin
        # Controversial from a security POV,
        # run_on_scheduler is a debugging function
        # `dask-scheduler --preload install_plugin.py`
        # is the standard but less convenient pattern
        client.run_on_scheduler(install_plugin)

        # Disable fuse optimisation: https://github.com/dask/dask/issues/7036
        opt_ctx = dask.config.set(optimization__fuse__active=False)
        exitstack.enter_context(opt_ctx)
        logger.info("Distributed client sucessfully initialized.")
    else:
        optimize_graph = True

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

    if opts.flags_mad_enable:
        data_xds_list = add_mad_graph(data_xds_list, opts)

    writable_xds = finalise_flags(data_xds_list, opts)

    writes = write_xds_list(writable_xds, ref_xds_list, opts)

    gain_writes = write_gain_datasets(gain_xds_lol, opts)

    logger.success("{:.2f} seconds taken to build graph.", time.time() - t0)

    if opts.parallel_scheduler == "distributed":
        t0 = time.time()
        # TODO: Dirty hack to coerce coordinate writes into graph. This should
        # should probably be handled by daskms. Requires disabled optimization.
        gain_writes = [dask.delayed(bool)(gw) for gw in gain_writes]
        grouped_annotate(gain_writes, writes)
        logger.success(f"{time.time() - t0:.2f} seconds taken to annotate "
                       f"graph.")

    t0 = time.time()

    with ProgressBar():

        dask.compute(writes, gain_writes,
                     num_workers=opts.parallel_nthread,
                     optimize_graph=optimize_graph,
                     scheduler=opts.parallel_scheduler)

    logger.success("{:.2f} seconds taken to execute graph.", time.time() - t0)

    # dask.visualize(writes[:2],# gain_writes[:2],
    #                color='order', cmap='autumn',
    #                filename='order.pdf', node_attr={'penwidth': '10'},
    #                optimize_graph=False)

    # dask.visualize(writes[:2],# gain_writes[:2],
    #                filename='graph.pdf',
    #                optimize_graph=False)

    # dask.visualize(*gains_per_xds["G"],
    #                filename='gain_graph',
    #                format="pdf",
    #                optimize_graph=True,
    #                collapse_outputs=True,
    #                node_attr={'penwidth': '4',
    #                           'fontsize': '18',
    #                           'fontname': 'helvetica'},
    #                edge_attr={'penwidth': '2', })
