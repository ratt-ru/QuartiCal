# -*- coding: utf-8 -*-
# Sets up logger - hereafter import logger from Loguru.
import cubicalv2.logging.init_logger  # noqa
from loguru import logger
from cubicalv2.parser import parser
from cubicalv2.data_handling import data_handler
from cubicalv2.calibration.calibrate import calibrate
import dask.array as da
import time

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

    # preprocess.preprocess_opts(opts)

    # Give opts to the data handler, which returns a list of xarray data sets.

    t0 = time.time()
    data_xds = data_handler.read_ms(opts)

    gains_per_xds, updated_data_xds = calibrate(data_xds, opts)

    write_columns = data_handler.write_ms(updated_data_xds, opts)
    logger.success("{:.2f} seconds taken to build graph.", time.time() - t0)

    t0 = time.time()
    gains, _ = da.compute(gains_per_xds, write_columns)
    logger.success("{:.2f} seconds taken to execute graph.", time.time() - t0)
