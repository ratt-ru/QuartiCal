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

    Resolves a user-specified antenna into an antenna index. The value may be
    an antenna name, an integer index, or either of those disambiguated by an
    explicit "name:" or "index:" prefix. A bare value which is simultaneously
    a valid index and a valid name is interpreted as an index, with a
    warning.

    Args:
        arg (str): A command line argument.
        ant_names (list of str): Antenna names, ordered by antenna index.

    Returns:
        The integer index of the selected antenna.

    Raises:
        ValueError: If arg cannot be resolved to a valid antenna index.
    """

    ant_names = [str(name) for name in ant_names]
    n_ant = len(ant_names)

    def by_name(value):
        if value not in ant_names:
            raise ValueError(
                f"Antenna name '{value}' not found in the antenna table. "
                f"Valid names are {ant_names}."
            )
        return ant_names.index(value)

    def by_index(value):
        if not value.isdigit():
            raise ValueError(
                f"Antenna index '{value}' is not a non-negative integer."
            )
        index = int(value)
        if index >= n_ant:
            raise ValueError(
                f"Antenna index {index} is out of range for the {n_ant} "
                f"antennas in the antenna table."
            )
        return index

    if arg.startswith("name:"):
        return by_name(arg[len("name:"):])
    elif arg.startswith("index:"):
        return by_index(arg[len("index:"):])

    is_name = arg in ant_names
    is_index = arg.isdigit() and int(arg) < n_ant

    if is_name and is_index:
        logger.warning(
            f"Antenna '{arg}' is ambiguous - it is both a valid antenna "
            f"index and the name of antenna {ant_names.index(arg)}. "
            f"Interpreting it as an index. Use 'index:{arg}' or "
            f"'name:{arg}' to select explicitly and silence this warning."
        )

    if is_index:
        return int(arg)
    elif is_name:
        return by_name(arg)

    raise ValueError(
        f"Antenna '{arg}' could not be resolved to an antenna index. It is "
        f"neither a valid index for the {n_ant} antennas in the antenna "
        f"table nor one of their names ({ant_names})."
    )
