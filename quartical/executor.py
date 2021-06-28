# -*- coding: utf-8 -*-
# Sets up logger - hereafter import logger from Loguru.
from contextlib import ExitStack
from loguru import logger
import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
import time
from quartical.config import parser, preprocess, helper, internal
from quartical.logging import configure_loguru
from quartical.data_handling.ms_handler import (read_xds_list,
                                                write_xds_list,
                                                preprocess_xds_list)
from quartical.data_handling.model_handler import add_model_graph
from quartical.calibration.calibrate import add_calibration_graph
from quartical.flagging.flagging import finalise_flags, add_mad_graph
from quartical.scheduling import install_plugin
from quartical.gains.datasets import write_gain_datasets
# from daskms.experimental.zarr import xds_from_zarr, xds_to_zarr


@logger.catch
def execute():
    with ExitStack() as stack:
        _execute(stack)


def _execute(exitstack):
    """Runs the application."""

    helper.help()  # Check to see if the user asked for help.
    configure_loguru()

    # Get all the config. This should never be used directly.
    opts = parser.parse_inputs()

    # Split out all the configuration objects. Mitigates god-object problems.
    ms_opts = opts.input_ms
    model_opts = opts.input_model
    solver_opts = opts.solver
    output_opts = opts.output
    mad_flag_opts = opts.mad_flags
    parallel_opts = opts.parallel
    gain_opts = internal.convert_gain_config(opts)  # Special handling.

    model_vis_recipe = preprocess.transcribe_recipe(model_opts.recipe)

    if parallel_opts.scheduler == "distributed":
        if parallel_opts.address:
            logger.info("Initializing distributed client.")
            client = exitstack.enter_context(Client(parallel_opts.address))
        else:
            logger.info("Initializing distributed client using LocalCluster.")
            cluster = LocalCluster(processes=parallel_opts.n_worker > 1,
                                   n_workers=parallel_opts.n_worker,
                                   threads_per_worker=parallel_opts.n_thread,
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
    model_columns = model_vis_recipe.ingredients.model_columns
    data_xds_list, ref_xds_list = read_xds_list(model_columns, ms_opts)

    # logger.info("Reading data from zms.")
    # data_xds_list = xds_from_zarr(
    #     "/home/jonathan/3C147_tests/3C147_daskms.zms"
    # )

    # writes = xds_to_zarr(
    #     data_xds_list,
    #     "/home/jonathan/3C147_tests/3C147_daskms.zms"
    # )
    # dask.compute(writes)
    # return

    # Preprocess the xds_list - initialise some values and fix bad data.
    data_xds_list = preprocess_xds_list(data_xds_list,
                                        ms_opts.weight_column)

    # A list of xdss onto which appropriate model data has been assigned.
    data_xds_list = add_model_graph(data_xds_list,
                                    model_vis_recipe,
                                    ms_opts.path,
                                    model_opts)

    # Adds the dask graph describing the calibration of the data.
    gain_xds_lol, data_xds_list = add_calibration_graph(data_xds_list,
                                                        solver_opts,
                                                        gain_opts)

    if mad_flag_opts.enable:
        data_xds_list = add_mad_graph(data_xds_list, mad_flag_opts)

    writable_xds = finalise_flags(data_xds_list)

    writes = write_xds_list(writable_xds,
                            ref_xds_list,
                            ms_opts.path,
                            output_opts)

    gain_writes = write_gain_datasets(gain_xds_lol,
                                      solver_opts.terms,
                                      output_opts)

    logger.success("{:.2f} seconds taken to build graph.", time.time() - t0)

    t0 = time.time()

    with ProgressBar():

        dask.compute(writes, gain_writes,
                     num_workers=parallel_opts.n_thread,
                     optimize_graph=True,
                     scheduler=parallel_opts.scheduler)

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
