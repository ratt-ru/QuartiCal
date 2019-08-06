# -*- coding: utf-8 -*-
from cubicalv2.parser import parser, preprocess
from cubicalv2.data_handling import data_handler


def execute():
    """Runs the application."""

    opts = parser.parse_inputs()

    # Add this functionality - should check opts for problems in addition
    # to interpreting weird options. Can also raise flags for differrent modes
    # of operation. The idea is that all our configuration state lives in this
    # options dictionary. Down with OOP!

    preprocess.preprocess_opts(opts)

    # Give opts to the data handler, which in turn should return some dask
    # arrays. This needs to be implemented and will require further thought.

    data_handler.handle_ms(opts)

# if __name__ == "__main__":
#     pass
