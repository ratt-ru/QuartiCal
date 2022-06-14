# -*- coding: utf-8 -*-
import re


def as_time(arg):
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
        raise ValueError("Too many non-numeric characters in time value.")

    if arg.isnumeric():
        arg = int(arg)
    elif arg.endswith('s'):
        arg = float(arg.rstrip('s'))
    else:
        raise ValueError("Units not understood. Time values must be "
                         "either an integer number of intergrations "
                         "or a duration in seconds.")

    if arg == 0:
        arg = int(arg)

    return arg


def as_freq(arg):
    """Defines the custom argument type FREQ.

    Converts its input into an integer if it lacks a unit suffix, otherwise a
    float.

    Args:
        arg (str): A command line argument.

    Returns:
        Value of arg converted to a float (bandwidth) or integer (number of
        channels).

    Raises:
        ArgumentTypeError: If unit characters are not understood.
    """

    if sum(not char.isnumeric() for char in arg) > 3:
        raise ValueError("Too many non-numeric characters in freq value.")

    unit_magnitudes = {"HZ":  1e0,
                       "KHZ": 1e3,
                       "MHZ": 1e6,
                       "GHZ": 1e9}

    pattern = ",".join(unit_magnitudes.keys())

    if arg.isnumeric():
        arg = int(arg)
    else:
        match = re.match(r"([0-9]+)([{}]+)".format(pattern), arg, re.I)
        if match:
            bw = float(match.group(1))
            mag = unit_magnitudes[match.group(2).upper()]
            arg = bw*mag
        else:
            raise ValueError("Unit not understood. Freq values must be "
                             "either an integer number of channels "
                             "or a bandwidth in Hz/kHz/MHz/GHz.")

    if arg == 0:
        arg = int(arg)

    return arg
