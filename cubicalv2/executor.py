# -*- coding: utf-8 -*-
# Sets up logger - hereafter import logger from Loguru.
import cubicalv2.logging.init_logger  # noqa.
from cubicalv2.parser import parser
from cubicalv2.data_handling import data_handler
import numpy as np


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

    for xds in data_xds:

        # Submit the xds

        data_col = xds.DATA.data
        model_col = xds.MODEL_DATA.data
        ant1_col = xds.ANTENNA1.data
        ant2_col = xds.ANTENNA2.data
        time_col = xds.TIME.data
        utime_ind = \
            time_col.map_blocks(lambda d: np.unique(d, return_inverse=True)[1])
        utime_per_chunk = \
            utime_ind.map_blocks(lambda f: np.max(f, keepdims=True) + 1,
                                 chunks=(1,),
                                 dtype=utime_ind.dtype)
