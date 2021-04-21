# -*- coding: utf-8 -*-
from numba.extending import overload
from numba import jit, types
import numpy as np


# Handy alias for functions that need to be jitted in this way.
qcjit = jit(nogil=True,
            nopython=True,
            fastmath=True,
            cache=True,
            inline="always")

# TODO: Consider whether these should have true optionals. This will likely
# require them all to be implemented as overloads.


def get_dims(col, row_map):
    """Returns effective column dimensions. This may be larger than col.

    Args:
        col: A numpy.ndrray containing column-like data.
        row_map: A nominal to effective row mapping. If not None, the returned
            shape will have the effective number of rows.

    Returns:
        A shape tuple.
    """
    return


@overload(get_dims, inline="always")
def _get_dims(col, row_map):

    if isinstance(row_map, types.NoneType):
        def impl(col, row_map):
            return col.shape
        return impl
    else:
        if col.ndim == 4:
            def impl(col, row_map):
                _, n_chan, n_dir, n_corr = col.shape
                return (row_map.size, n_chan, n_dir, n_corr)
            return impl
        else:
            def impl(col, row_map):
                _, n_chan, n_corr = col.shape
                return (row_map.size, n_chan, n_corr)
            return impl


def get_row(row_ind, row_map):
    """Gets the current row index. Row map is needed for the BDA case.

    Args:
        row_ind: Integer index of current row.
        row_map: A nominal to effective row mapping.

    Returns:
        Integer index of effective row.
    """
    return


@overload(get_row, inline="always")
def _get_row(row_ind, row_map):

    if isinstance(row_map, types.NoneType):
        def impl(row_ind, row_map):
            return row_ind
        return impl
    else:
        def impl(row_ind, row_map):
            return row_map[row_ind]
        return impl


def old_mul_rweight(vis, weight, ind):
    """Multiplies the row weight into a visiblity if weight is not None.

    Args:
        vis: An complex valued visibility.
        weight: A float row weight.
        ind: Integer row index for selecting weight.

    Returns:
        Product of visilbity and weight, if weight is not None.
    """
    return


@overload(old_mul_rweight, inline="always")
def _old_mul_rweight(vis, weight, ind):

    if isinstance(weight, types.NoneType):
        def impl(vis, weight, ind):
            return vis
        return impl
    else:
        def impl(vis, weight, ind):
            return vis*weight[ind]
        return impl


@qcjit
def get_chan_extents(f_map_arr, active_term, n_fint, n_chan):
    """Given the frequency mappings, determines the start/stop indices."""

    chan_starts = np.empty(n_fint, dtype=np.int32)
    chan_starts[0] = 0

    chan_stops = np.empty(n_fint, dtype=np.int32)
    chan_stops[-1] = n_chan

    # NOTE: This might not be correct for decreasing channel freqs.
    if n_fint > 1:
        chan_starts[1:] = 1 + np.where(
            f_map_arr[1:, active_term] - f_map_arr[:-1, active_term])[0]
        chan_stops[:-1] = chan_starts[1:]

    return chan_starts, chan_stops


@qcjit
def get_row_extents(t_map_arr, active_term, n_tint):
    """Given the time mappings, determines the row start/stop indices."""

    row_starts = np.empty(n_tint, dtype=np.int32)
    row_starts[0] = 0

    row_stops = np.empty(n_tint, dtype=np.int32)
    row_stops[-1] = t_map_arr[1:, active_term].size

    # NOTE: This assumes time ordered data (row runs).
    if n_tint > 1:
        row_starts[1:] = 1 + np.where(
            t_map_arr[1:, active_term] - t_map_arr[:-1, active_term])[0]
        row_stops[:-1] = row_starts[1:]

    return row_starts, row_stops
