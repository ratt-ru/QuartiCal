# -*- coding: utf-8 -*-
from numba import jit, types, generated_jit
import numpy as np
from collections import namedtuple


extent_tuple = namedtuple(
    "extent_tuple",
    (
        "row_starts",
        "row_stops",
        "chan_starts",
        "chan_stops"
    )
)

# Handy alias for functions that need to be jitted in this way.
qcjit = jit(nogil=True,
            nopython=True,
            fastmath=True,
            cache=True,
            inline="always")

qcgjit = generated_jit(nogil=True,
                       nopython=True,
                       fastmath=True,
                       cache=True)

# TODO: Consider whether these should have true optionals. This will likely
# require them all to be implemented as overloads.


@qcgjit
def get_dims(col, row_map):
    """Returns effective column dimensions. This may be larger than col.

    Args:
        col: A numpy.ndrray containing column-like data.
        row_map: A nominal to effective row mapping. If not None, the returned
            shape will have the effective number of rows.

    Returns:
        A shape tuple.
    """

    if isinstance(row_map, types.NoneType):
        def impl(col, row_map):
            return col.shape
    else:
        if col.ndim == 4:
            def impl(col, row_map):
                _, n_chan, n_dir, n_corr = col.shape
                return (row_map.size, n_chan, n_dir, n_corr)
        else:
            def impl(col, row_map):
                _, n_chan, n_corr = col.shape
                return (row_map.size, n_chan, n_corr)

    return impl


@qcgjit
def get_row(row_ind, row_map):
    """Gets the current row index. Row map is needed for the BDA case.

    Args:
        row_ind: Integer index of current row.
        row_map: A nominal to effective row mapping.

    Returns:
        Integer index of effective row.
    """
    if isinstance(row_map, types.NoneType):
        def impl(row_ind, row_map):
            return row_ind
    else:
        def impl(row_ind, row_map):
            return row_map[row_ind]

    return impl


@qcjit
def get_extents(t_map, f_map):
    """Given the time/freq mappings, determine run start and stop indices."""

    row_starts, row_stops = get_row_extents(t_map)
    chan_starts, chan_stops = get_chan_extents(f_map)

    return extent_tuple(row_starts, row_stops, chan_starts, chan_stops)


@qcjit
def get_chan_extents(f_map_arr):
    """Given the frequency mappings, determines the start/stop indices."""

    n_fint = f_map_arr.max() + 1
    n_chan = f_map_arr.size

    chan_starts = np.empty(n_fint, dtype=np.int32)
    chan_starts[0] = 0

    chan_stops = np.empty(n_fint, dtype=np.int32)
    chan_stops[-1] = n_chan

    # NOTE: This might not be correct for decreasing channel freqs.
    if n_fint > 1:
        chan_starts[1:] = 1 + np.where(f_map_arr[1:] - f_map_arr[:-1])[0]
        chan_stops[:-1] = chan_starts[1:]

    return chan_starts, chan_stops


@qcjit
def get_row_extents(t_map_arr):
    """Given the time mappings, determines the row start/stop indices."""

    n_tint = t_map_arr.max() + 1

    row_starts = np.empty(n_tint, dtype=np.int32)
    row_starts[0] = 0

    row_stops = np.empty(n_tint, dtype=np.int32)
    row_stops[-1] = t_map_arr.size

    # NOTE: This assumes time ordered data (row runs).
    if n_tint > 1:
        row_starts[1:] = 1 + np.where(t_map_arr[1:] - t_map_arr[:-1])[0]
        row_stops[:-1] = row_starts[1:]

    return row_starts, row_stops
