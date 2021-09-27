# -*- coding: utf-8 -*-
# Sets up logger - hereafter import logger from Loguru.
from contextlib import ExitStack
from loguru import logger
import sys
import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster, performance_report
import time
from pathlib import Path
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


@logger.catch(onerror=lambda _: sys.exit(1))
def execute():
    with ExitStack() as stack:
        _execute(stack)


def _execute(exitstack):
    """Runs the application."""

    helper.help()  # Check to see if the user asked for help.

    # Get all the config. This should never be used directly.
    opts = parser.parse_inputs()

    # Split out all the configuration objects. Mitigates god-object problems.
    ms_opts = opts.input_ms
    model_opts = opts.input_model
    solver_opts = opts.solver
    output_opts = opts.output
    mad_flag_opts = opts.mad_flags
    dask_opts = opts.dask
    chain_opts = internal.gains_to_chain(opts)  # Special handling.

    # Make sure that the output directory is correctly cleaned up.
    preprocess.prepare_output_directory(output_opts.directory)

    # Init the logger once we know where to put the output.
    configure_loguru(output_opts.directory)

    # Now that we know where to put the log, log the final config state.
    parser.log_final_config(opts)

    model_vis_recipe = preprocess.transcribe_recipe(model_opts.recipe)

    if dask_opts.scheduler == "distributed":
        if dask_opts.address:
            logger.info("Initializing distributed client.")
            client = exitstack.enter_context(Client(dask_opts.address))
        else:
            logger.info("Initializing distributed client using LocalCluster.")
            cluster = LocalCluster(
                processes=dask_opts.workers > 1,
                n_workers=dask_opts.workers,
                threads_per_worker=dask_opts.threads,
                memory_limit=0
            )
            cluster = exitstack.enter_context(cluster)
            client = exitstack.enter_context(Client(cluster))

        # Install Quartical Scheduler Plugin. Controversial from a security
        # POV, run_on_scheduler is a debugging function.
        # `dask-scheduler --preload install_plugin.py` is the standard but
        # less convenient pattern.
        client.run_on_scheduler(install_plugin)

        logger.info("Distributed client sucessfully initialized.")

    t0 = time.time()

    # Reads the measurement set using the relavant configuration from opts.
    model_columns = model_vis_recipe.ingredients.model_columns
    data_xds_list, ref_xds_list = read_xds_list(model_columns, ms_opts)

    # Preprocess the xds_list - initialise some values and fix bad data.
    data_xds_list = preprocess_xds_list(data_xds_list, ms_opts)

    # A list of xdss onto which appropriate model data has been assigned.
    data_xds_list = add_model_graph(data_xds_list,
                                    model_vis_recipe,
                                    ms_opts.path,
                                    model_opts)

    # Adds the dask graph describing the calibration of the data.
    gain_xds_lod, net_xds_list, data_xds_list = add_calibration_graph(
        data_xds_list,
        solver_opts,
        chain_opts
    )

    if mad_flag_opts.enable:
        data_xds_list = add_mad_graph(data_xds_list, mad_flag_opts)

    data_xds_list = finalise_flags(data_xds_list)

    ms_writes = write_xds_list(data_xds_list,
                               ref_xds_list,
                               ms_opts.path,
                               output_opts)

    gain_writes = write_gain_datasets(gain_xds_lod,
                                      net_xds_list,
                                      output_opts)

    logger.success("{:.2f} seconds taken to build graph.", time.time() - t0)

    def compute_context(dask_opts, output_opts):
        if dask_opts.scheduler == "distributed":
            root_path = Path(output_opts.directory).absolute()
            report_path = root_path / Path("dask_report.html.qc")
            return performance_report(filename=str(report_path))
        else:
            return ProgressBar()

    t0 = time.time()

    with compute_context(dask_opts, output_opts):

        dask.compute(ms_writes, gain_writes,
                     num_workers=dask_opts.threads,
                     optimize_graph=True,
                     scheduler=dask_opts.scheduler)

    logger.success("{:.2f} seconds taken to execute graph.", time.time() - t0)

    if dask_opts.scheduler == "distributed":
        client.close()  # Close this client, hopefully gracefully.
