# -*- coding: utf-8 -*-
from argparse import ArgumentTypeError
from loguru import logger

custom_types = {}


def custom_type(fn):
    """Adds decorated function to the custom type dictionary."""

    logger.trace("Registering custom type {}", fn.__name__)

    custom_types[fn.__name__] = fn

    return fn


@custom_type
def TIME(arg):
    """Defines the custom argument type TIME.

    Converts its input into an integer if it lacks a unit suffix, otherwise a
    float.

    Args:
        arg (str): A command line argument.

    Returns:
        Value of arg converted to a float (duration) or integer (integrations).

    Raises:
        ArgumentTypeError: If unit characters are not understood.

    """

    if sum(not char.isnumeric() for char in arg) > 1:
        raise ArgumentTypeError("Too many non-numeric characters for custom "
                                "type TIME.")

    if arg.isnumeric():
        arg = int(arg)
    elif arg.endswith('s'):
        arg = float(arg.rstrip('s'))
    else:
        raise ArgumentTypeError("Unit not understood. TIME values must be "
                                "either an integer number of intergrations "
                                "or a duration in seconds.")

    if arg == 0:
        arg = int(1e99)
        logger.info("Argument of custom type TIME is zero - treating as "
                    "infinite.")

    return arg


@custom_type
def FREQ(arg):
    logger.critical("Custom type FREQ not implemented.")
    return str(arg)

@custom_type
def DDID(arg):
    logger.critical("Custom type DDID not implemented.")
    return str(arg)

@custom_type
def CHAN(arg):
    logger.critical("Custom type CHAN not implemented.")
    return str(arg)

@custom_type
def DIRECTIONS(arg):
    logger.critical("Custom type DIRECTIONS not implemented.")
    return str(arg)