# -*- coding: utf-8 -*-
# Sets up logger - hereafter import logger from Loguru.
import cubicalv2.logging.init_logger  # noqa
from loguru import logger
from cubicalv2.parser import parser, preprocess
from cubicalv2.data_handling.ms_handler import read_ms, write_column
from cubicalv2.data_handling.model_handler import add_model_graph
from cubicalv2.calibration.calibrate import add_calibration_graph
import dask.array as da
import time
from dask.diagnostics import ProgressBar
import dask
from dask.distributed import Client


@logger.catch
def execute():
    """Runs the application."""

    opts = parser.parse_inputs()

    # Add this functionality - should check opts for problems in addition
    # to interpreting weird options. Can also raise flags for different modes
    # of operation. The idea is that all our configuration state lives in this
    # options dictionary. Down with OOP!

    # There needs to be a validation step which checks that the config is
    # possible.

    preprocess.preprocess_opts(opts)

    if opts.parallel_scheduler == "distributed":
        logger.info("Initializing distributed client.")
        client = Client(processes=False,                            # noqa
                        n_workers=opts.parallel_nworker,
                        threads_per_worker=opts.parallel_nthread)
        logger.info("Distributed client sucessfully initialized.")

    t0 = time.time()

    # Reads the measurement set using the relavant configuration from opts.
    ms_xds, col_kwrds = read_ms(opts)

    # Model xds is a list of xdss onto which appropriate model data has been
    # assigned.
    model_xds = add_model_graph(ms_xds, opts)

    gains_per_xds, col_kwrds = add_calibration_graph(model_xds, col_kwrds, opts)

    writes = write_column(model_xds, col_kwrds, opts)

    # write_columns = ms_handler.write_ms(updated_data_xds, opts)
    logger.success("{:.2f} seconds taken to build graph.", time.time() - t0)

    t0 = time.time()
    with ProgressBar():
        gains, _ = da.compute(gains_per_xds, writes,
                           # write_columns,
                           # scheduler="sync")
                           num_workers=opts.parallel_nthread)
    logger.success("{:.2f} seconds taken to execute graph.", time.time() - t0)

    # for gain in gains[0]["G"]:
    #     print(np.max(np.abs(gain)))
    # for gain in gains[0]["dE"]:
    #     print(np.max(np.abs(gain)))

    # dask.visualize(gains_per_xds[0],
    #                filename='graph.pdf',
    #                optimize_graph=True)
