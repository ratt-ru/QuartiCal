# -*- coding: utf-8 -*-
# Sets up logger - hereafter import logger from Loguru.
import cubicalv2.logging.init_logger  # noqa
from loguru import logger
from cubicalv2.parser import parser
from cubicalv2.data_handling import data_handler
from cubicalv2.calibration.calibrate import calibrate


def execute():
    """Runs the application."""

    opts = parser.parse_inputs()

    # Add this functionality - should check opts for problems in addition
    # to interpreting weird options. Can also raise flags for different modes
    # of operation. The idea is that all our configuration state lives in this
    # options dictionary. Down with OOP!

    # preprocess.preprocess_opts(opts)

    # Give opts to the data handler, which returns a list of xarray data sets.

    data_xds = data_handler.read_ms(opts)

    import time
    t0 = time.time()
    calibrate(data_xds, opts)
    logger.success("{:.2f} seconds taken to execute graph.", time.time() - t0)
