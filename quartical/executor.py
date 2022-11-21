# -*- coding: utf-8 -*-
# Sets up logger - hereafter import logger from Loguru.
from contextlib import ExitStack
import sys
from loguru import logger
import dask
from dask.distributed import Client, LocalCluster
import time
from quartical.config import parser, preprocess, helper, internal
from quartical.logging import (ProxyLogger, LoggerPlugin)
from quartical.data_handling.ms_handler import (read_xds_list,
                                                write_xds_list,
                                                preprocess_xds_list,
                                                postprocess_xds_list)
from quartical.data_handling.model_handler import add_model_graph
from quartical.data_handling.angles import make_parangle_xds_list
from quartical.calibration.calibrate import add_calibration_graph
from quartical.statistics.statistics import (make_stats_xds_list,
                                             assign_presolve_chisq,
                                             assign_postsolve_chisq)
from quartical.statistics.logging import (embed_stats_logging,
                                          log_summary_stats)
from quartical.flagging.flagging import finalise_flags, add_mad_graph
from quartical.scheduling import install_plugin
from quartical.gains.datasets import write_gain_datasets
from quartical.utils.dask import compute_context


@logger.catch(onerror=lambda _: sys.exit(1))
def execute():
    with ExitStack() as stack:
        _execute(stack)


def _execute(exitstack):
    """Runs the application."""

    helper.help()  # Check to see if the user asked for help.

    # Get all the config. This should never be used directly.
    opts, config_files = parser.parse_inputs()

    # Split out all the configuration objects. Mitigates god-object problems.
    ms_opts = opts.input_ms
    model_opts = opts.input_model
    solver_opts = opts.solver
    output_opts = opts.output
    mad_flag_opts = opts.mad_flags
    dask_opts = opts.dask
    chain_opts = internal.gains_to_chain(opts)  # Special handling.

    # Init the logging proxy - an object which helps us ensure that logging
    # works for both threads an processes. It is easily picklable.

    time_str = time.strftime("%Y%m%d_%H%M%S")
    proxy_logger = ProxyLogger(
        output_opts.log_directory,
        time_str,
        output_opts.log_to_terminal
    )
    proxy_logger.configure()

    # Now that we know where to put the log, log the final config state.
    parser.log_final_config(opts, config_files)

    model_vis_recipe = preprocess.transcribe_recipe(model_opts.recipe)

    if dask_opts.scheduler == "distributed":

        # NOTE: This is needed to allow threads to spawn processes in a
        # distributed enviroment. This *may* be dangerous. Monitor.
        dask.config.set({"distributed.worker.daemon": False})

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

        client.register_worker_plugin(LoggerPlugin(proxy_logger=proxy_logger))

        # Install Quartical Scheduler Plugin. Controversial from a security
        # POV, run_on_scheduler is a debugging function.
        # `dask-scheduler --preload install_plugin.py` is the standard but
        # less convenient pattern.
        client.run_on_scheduler(install_plugin)

        client.wait_for_workers(dask_opts.workers)

        logger.info("Distributed client sucessfully initialized.")

    t0 = time.time()

    # Reads the measurement set using the relavant configuration from opts.
    model_columns = model_vis_recipe.ingredients.model_columns
    data_xds_list, ref_xds_list = read_xds_list(model_columns, ms_opts)

    # Preprocess the xds_list - initialise some values and fix bad data.
    data_xds_list = preprocess_xds_list(data_xds_list, ms_opts)

    # Make a list of datasets containing the parallactic angles as these
    # can be expensive to compute and may be used several times. NOTE: At
    # present, these also include the effect of the receptor angles.
    parangle_xds_list = make_parangle_xds_list(ms_opts.path, data_xds_list)

    # A list of xdss onto which appropriate model data has been assigned.
    data_xds_list = add_model_graph(data_xds_list,
                                    parangle_xds_list,
                                    model_vis_recipe,
                                    ms_opts.path,
                                    model_opts)

    stats_xds_list = make_stats_xds_list(data_xds_list)
    stats_xds_list = assign_presolve_chisq(data_xds_list, stats_xds_list)

    # Adds the dask graph describing the calibration of the data. TODO:
    # This call has excess functionality now. Split out mapping and outputs.
    gain_xds_lod, net_xds_lod, data_xds_list = add_calibration_graph(
        data_xds_list,
        solver_opts,
        chain_opts,
        output_opts
    )

    stats_xds_list = assign_postsolve_chisq(data_xds_list, stats_xds_list)
    stats_xds_list = embed_stats_logging(stats_xds_list)

    if mad_flag_opts.enable:
        data_xds_list = add_mad_graph(data_xds_list, mad_flag_opts)

    data_xds_list = finalise_flags(data_xds_list)

    # This will apply the inverse of P-Jones but can also be extended.
    data_xds_list = postprocess_xds_list(data_xds_list,
                                         parangle_xds_list,
                                         output_opts)

    ms_writes = write_xds_list(data_xds_list,
                               ref_xds_list,
                               ms_opts.path,
                               output_opts)

    gain_writes = write_gain_datasets(gain_xds_lod,
                                      net_xds_lod,
                                      output_opts)

    logger.success("{:.2f} seconds taken to build graph.", time.time() - t0)

    logger.info("Computation starting. Please be patient - log messages will "
                "only appear per completed chunk.")

    t0 = time.time()

    with compute_context(dask_opts, output_opts, time_str):

        _, _, stats_xds_list = dask.compute(
            ms_writes,
            gain_writes,
            stats_xds_list,
            num_workers=dask_opts.threads,
            optimize_graph=True,
            scheduler=dask_opts.scheduler
        )

    logger.success("{:.2f} seconds taken to execute graph.", time.time() - t0)

    log_summary_stats(stats_xds_list)

    if dask_opts.scheduler == "distributed":
        client.close()  # Close this client, hopefully gracefully.
