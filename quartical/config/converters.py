# -*- coding: utf-8 -*-
import re
from loguru import logger


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



def as_antenna_index(arg, ant_names):
    """Defines the custom argument type ANTENNA.

    Resolves a user-specified antenna into an antenna index. An integer is an
    index and a string is a name. The schema types the option as
    Union[int, str] so that the distinction survives config files, the command
    line and direct assignment in Python.

    Args:
        arg (int or str): An antenna index or an antenna name.
        ant_names (list of str): Antenna names, ordered by antenna index.

    Returns:
        The integer index of the selected antenna.

    Raises:
        ValueError: If arg is not a valid antenna index or antenna name.
    """

    ant_names = [str(name) for name in ant_names]
    n_ant = len(ant_names)

    if isinstance(arg, str):
        if arg not in ant_names:
            raise ValueError(
                f"Antenna name '{arg}' not found in the antenna table. "
                f"Valid names are {ant_names}."
            )
        return ant_names.index(arg)

    # An antenna whose name is an integer can only be selected by quoting the
    # name, so an unquoted integer which is also a name is worth flagging.
    is_name = str(arg) in ant_names

    if not 0 <= arg < n_ant:
        hint = f" Quote the value to select the antenna named '{arg}'." \
            if is_name else ""
        raise ValueError(
            f"Antenna index {arg} is out of range for the {n_ant} antennas "
            f"in the antenna table.{hint}"
        )

    if is_name:
        logger.warning(
            f"Antenna {arg} is both a valid antenna index and the name of "
            f"antenna {ant_names.index(str(arg))}. Interpreting it as an "
            f"index. Quote the value to select the antenna by name."
        )

    return arg
