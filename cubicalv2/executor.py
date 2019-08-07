# -*- coding: utf-8 -*-
from cubicalv2.parser import parser, preprocess
from cubicalv2.data_handling import data_handler
from loguru import logger
import sys

fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "\
      "<level>{level}</level> | "\
      "<cyan>{module}</cyan>:<cyan>{function}</cyan> "\
      "<level>{message}</level>"

config = {
    "handlers": [
        {"sink": sys.stderr,
         "level": "INFO",
         "format": fmt},
        {"sink": "cubicalv2.log",
         "level": "DEBUG",
         "rotation": "100 MB",
         "format": fmt}
    ],
}
logger.configure(**config)


def execute():
    """Runs the application."""

    opts = parser.parse_inputs()

    # Add this functionality - should check opts for problems in addition
    # to interpreting weird options. Can also raise flags for different modes
    # of operation. The idea is that all our configuration state lives in this
    # options dictionary. Down with OOP!

    preprocess.preprocess_opts(opts)

    # Give opts to the data handler, which returns a list of xarray data sets.

    data_xds = data_handler.read_ms(opts)

    for xds in data_xds:

        # Submit the xds

        pass
