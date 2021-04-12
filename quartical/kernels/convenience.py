# -*- coding: utf-8 -*-
from numba.extending import overload, register_jitable
from numba import jit, types, generated_jit
import numpy as np


# Handy alias for functions that need to be jitted in this way.
injit = jit(nogil=True,
            nopython=True,
            fastmath=True,
            cache=True,
            inline="always")

gjit = generated_jit(nopython=True,
                     fastmath=True,
                     parallel=False,
                     cache=True,
                     nogil=True)

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


def mul_rweight(vis, weight, ind):
    """Multiplies the row weight into a visiblity if weight is not None.

    Args:
        vis: An complex valued visibility.
        weight: A float row weight.
        ind: Integer row index for selecting weight.

    Returns:
        Product of visilbity and weight, if weight is not None.
    """
    return


@overload(mul_rweight, inline="always")
def _mul_rweight(vis, weight, ind):

    if isinstance(weight, types.NoneType):
        def impl(vis, weight, ind):
            return vis
        return impl
    else:
        def impl(vis, weight, ind):
            return vis*weight[ind]
        return impl


@injit
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


@injit
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


@register_jitable(inline="always")
def _v1_mul_v2(v1, v2, md):

    v100, v101, v110, v111 = _unpack(v1, md)
    v200, v201, v210, v211 = _unpack(v2, md)

    v300 = (v100*v200 + v101*v210)
    v301 = (v100*v201 + v101*v211)
    v310 = (v110*v200 + v111*v210)
    v311 = (v110*v201 + v111*v211)

    return v300, v301, v310, v311


@register_jitable(inline="always")
def _v1_mul_v2ct(v1, v2, md):

    v100, v101, v110, v111 = _unpack(v1, md)
    v200, v201, v210, v211 = _unpack_ct(v2, md)

    v300 = (v100*v200 + v101*v210)
    v301 = (v100*v201 + v101*v211)
    v310 = (v110*v200 + v111*v210)
    v311 = (v110*v201 + v111*v211)

    return v300, v301, v310, v311


@register_jitable(inline="always")
def _v1_wmul_v2ct(v1, v2, w1, md):

    v100, v101, v110, v111 = _unpack(v1, md)
    v200, v201, v210, v211 = _unpack_ct(v2, md)
    w100, w101, w110, w111 = _unpack(w1, md)

    v300 = (v100*w100*v200 + v101*w111*v210)
    v301 = (v100*w100*v201 + v101*w111*v211)
    v310 = (v110*w100*v200 + v111*w111*v210)
    v311 = (v110*w100*v201 + v111*w111*v211)

    return v300, v301, v310, v311


@register_jitable(inline="always")
def _v1ct_wmul_v2(v1, v2, w1, md):

    v100, v101, v110, v111 = _unpack_ct(v1, md)
    v200, v201, v210, v211 = _unpack(v2, md)
    w100, w101, w110, w111 = _unpack(w1, md)

    v300 = (v100*w100*v200 + v101*w111*v210)
    v301 = (v100*w100*v201 + v101*w111*v211)
    v310 = (v110*w100*v200 + v111*w111*v210)
    v311 = (v110*w100*v201 + v111*w111*v211)

    return v300, v301, v310, v311


@gjit
def _unpack(vec, md):

    if md.literal_value == "full":
        def impl(vec, md):
            return vec[0], vec[1], vec[2], vec[3]
    else:
        def impl(vec, md):
            return vec[0], 0, 0, vec[1]

    return impl


@gjit
def _unpack_ct(vec, md):
    if md.literal_value == "full":
        def impl(vec, md):
            return vec[0].conjugate(), \
                   vec[2].conjugate(), \
                   vec[1].conjugate(), \
                   vec[3].conjugate()
    else:
        def impl(vec, md):
            return vec[0].conjugate(), 0, 0, vec[1].conjugate()

    return impl


@gjit
def _iunpack(out, vec, md):
    if md.literal_value == "full":
        def impl(out, vec, md):
            out[0], out[1], out[2], out[3] = vec[0], vec[1], vec[2], vec[3]
    else:
        def impl(out, vec, md):
            out[0], out[1], out[2], out[3] = vec[0], 0, 0, vec[1]

    return impl


@gjit
def _iunpack_ct(out, vec, md):
    if md.literal_value == "full":
        def impl(out, vec, md):
            out[0] = vec[0].conjugate()
            out[1] = vec[2].conjugate()
            out[2] = vec[1].conjugate()
            out[3] = vec[3].conjugate()
    else:
        def impl(out, vec, md):
            out[0] = vec[0].conjugate()
            out[1] = 0
            out[2] = 0
            out[3] = vec[1].conjugate()

    return impl


@register_jitable(inline="always")
def _iadd4(out, vec):
    out[0] += vec[0]
    out[1] += vec[1]
    out[2] += vec[2]
    out[3] += vec[3]
