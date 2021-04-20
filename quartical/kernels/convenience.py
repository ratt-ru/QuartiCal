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


def imul_rweight_factory(mode, weight):

    if isinstance(weight, types.NoneType):
        if mode.literal_value == "full" or mode.literal_value == "mixed":
            def impl(invec, outvec, weight, ind):
                outvec[0] = invec[0]
                outvec[1] = invec[1]
                outvec[2] = invec[2]
                outvec[3] = invec[3]
        else:
            def impl(invec, outvec, weight, ind):
                outvec[0] = invec[0]
                outvec[1] = invec[1]
    else:

        unpack = unpack_factory(mode)

        if mode.literal_value == "full" or mode.literal_value == "mixed":
            def impl(invec, outvec, weight, ind):
                v00, v01, v10, v11 = unpack(invec)
                w = weight[ind]
                outvec[0] = w*v00
                outvec[1] = w*v01
                outvec[2] = w*v10
                outvec[3] = w*v11
        else:
            def impl(invec, outvec, weight, ind):
                v00, v11 = unpack(invec)
                w = weight[ind]
                outvec[0] = w*v00
                outvec[1] = w*v11
    return qcjit(impl)


def v1_mul_v2_factory(mode):

    unpack = unpack_factory(mode)

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(v1, v2):
            v1_00, v1_01, v1_10, v1_11 = unpack(v1)
            v2_00, v2_01, v2_10, v2_11 = unpack(v2)

            v3_00 = (v1_00*v2_00 + v1_01*v2_10)
            v3_01 = (v1_00*v2_01 + v1_01*v2_11)
            v3_10 = (v1_10*v2_00 + v1_11*v2_10)
            v3_11 = (v1_10*v2_01 + v1_11*v2_11)

            return v3_00, v3_01, v3_10, v3_11
    else:
        def impl(v1, v2):
            v1_00, v1_11 = unpack(v1)
            v2_00, v2_11 = unpack(v2)

            v3_00 = v1_00*v2_00
            v3_11 = v1_11*v2_11

            return v3_00, v3_11
    return qcjit(impl)


def v1_imul_v2_factory(mode):

    unpack = unpack_factory(mode)

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(v1, v2, o1):
            v1_00, v1_01, v1_10, v1_11 = unpack(v1)
            v2_00, v2_01, v2_10, v2_11 = unpack(v2)

            o1[0] = (v1_00*v2_00 + v1_01*v2_10)
            o1[1] = (v1_00*v2_01 + v1_01*v2_11)
            o1[2] = (v1_10*v2_00 + v1_11*v2_10)
            o1[3] = (v1_10*v2_01 + v1_11*v2_11)
    else:
        def impl(v1, v2, o1):
            v1_00, v1_11 = unpack(v1)
            v2_00, v2_11 = unpack(v2)

            o1[0] = v1_00*v2_00
            o1[1] = v1_11*v2_11
    return qcjit(impl)


def v1_mul_v2ct_factory(mode):

    unpack = unpack_factory(mode)
    unpackct = unpackct_factory(mode)

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(v1, v2):
            v1_00, v1_01, v1_10, v1_11 = unpack(v1)
            v2_00, v2_01, v2_10, v2_11 = unpackct(v2)

            v3_00 = (v1_00*v2_00 + v1_01*v2_10)
            v3_01 = (v1_00*v2_01 + v1_01*v2_11)
            v3_10 = (v1_10*v2_00 + v1_11*v2_10)
            v3_11 = (v1_10*v2_01 + v1_11*v2_11)

            return v3_00, v3_01, v3_10, v3_11
    else:
        def impl(v1, v2):
            v1_00, v1_11 = unpack(v1)
            v2_00, v2_11 = unpackct(v2)

            v3_00 = v1_00*v2_00
            v3_11 = v1_11*v2_11

            return v3_00, v3_11
    return qcjit(impl)


def v1_imul_v2ct_factory(mode):

    unpack = unpack_factory(mode)
    unpackct = unpackct_factory(mode)

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(v1, v2, o1):
            v1_00, v1_01, v1_10, v1_11 = unpack(v1)
            v2_00, v2_01, v2_10, v2_11 = unpackct(v2)

            o1[0] = (v1_00*v2_00 + v1_01*v2_10)
            o1[1] = (v1_00*v2_01 + v1_01*v2_11)
            o1[2] = (v1_10*v2_00 + v1_11*v2_10)
            o1[3] = (v1_10*v2_01 + v1_11*v2_11)
    else:
        def impl(v1, v2, o1):
            v1_00, v1_11 = unpack(v1)
            v2_00, v2_11 = unpackct(v2)

            o1[0] = v1_00*v2_00
            o1[1] = v1_11*v2_11
    return qcjit(impl)


def iwmul_factory(mode):

    unpack = unpack_factory(mode)

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(v1, w1):
            w1_00, w1_01, w1_10, w1_11 = unpack(w1)

            v1[0] *= w1_00
            v1[1] *= w1_00
            v1[2] *= w1_11
            v1[3] *= w1_11
    else:
        def impl(v1, w1):
            w1_00, w1_11 = unpack(w1)

            v1[0] *= w1_00
            v1[1] *= w1_11
    return qcjit(impl)


def v1_wmul_v2ct_factory(mode):

    unpack = unpack_factory(mode)
    unpackct = unpackct_factory(mode)

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(v1, v2, w1):
            v1_00, v1_01, v1_10, v1_11 = unpack(v1)
            v2_00, v2_01, v2_10, v2_11 = unpackct(v2)
            w1_00, w1_01, w1_10, w1_11 = unpack(w1)

            v3_00 = (v1_00*w1_00*v2_00 + v1_01*w1_11*v2_10)
            v3_01 = (v1_00*w1_00*v2_01 + v1_01*w1_11*v2_11)
            v3_10 = (v1_10*w1_00*v2_00 + v1_11*w1_11*v2_10)
            v3_11 = (v1_10*w1_00*v2_01 + v1_11*w1_11*v2_11)

            return v3_00, v3_01, v3_10, v3_11
    else:
        def impl(v1, v2, w1):
            v1_00, v1_11 = unpack(v1)
            v2_00, v2_11 = unpackct(v2)
            w1_00, w1_11 = unpack(w1)

            v3_00 = v1_00*w1_00*v2_00
            v3_11 = v1_11*w1_11*v2_11

            return v3_00, v3_11
    return qcjit(impl)


def v1ct_wmul_v2_factory(mode):

    unpack = unpack_factory(mode)
    unpackct = unpackct_factory(mode)

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(v1, v2, w1):
            v1_00, v1_01, v1_10, v1_11 = unpackct(v1)
            v2_00, v2_01, v2_10, v2_11 = unpack(v2)
            w1_00, w1_01, w1_10, w1_11 = unpack(w1)

            v3_00 = (v1_00*w1_00*v2_00 + v1_01*w1_11*v2_10)
            v3_01 = (v1_00*w1_00*v2_01 + v1_01*w1_11*v2_11)
            v3_10 = (v1_10*w1_00*v2_00 + v1_11*w1_11*v2_10)
            v3_11 = (v1_10*w1_00*v2_01 + v1_11*w1_11*v2_11)

            return v3_00, v3_01, v3_10, v3_11
    else:
        def impl(v1, v2, w1):
            v1_00, v1_11 = unpackct(v1)
            v2_00, v2_11 = unpack(v2)
            w1_00, w1_11 = unpack(w1)

            v3_00 = v1_00*w1_00*v2_00
            v3_11 = v1_11*w1_11*v2_11

            return v3_00, v3_11
    return qcjit(impl)


def unpack_factory(mode):

    if mode.literal_value == "full":
        def impl(invec):
            return invec[0], invec[1], invec[2], invec[3]
    elif mode.literal_value == "diag":
        def impl(invec):
            return invec[0], invec[1]
    else:
        def impl(invec):
            if len(invec) == 4:
                return invec[0], invec[1], invec[2], invec[3]
            else:
                return invec[0], 0, 0, invec[1]
    return qcjit(impl)


def unpackct_factory(mode):

    if mode.literal_value == "full":
        def impl(invec):
            return np.conjugate(invec[0]), \
                   np.conjugate(invec[2]), \
                   np.conjugate(invec[1]), \
                   np.conjugate(invec[3])
    elif mode.literal_value == "diag":
        def impl(invec):
            return np.conjugate(invec[0]), \
                   np.conjugate(invec[1])
    else:
        def impl(invec):
            if len(invec) == 4:
                return np.conjugate(invec[0]), \
                       np.conjugate(invec[2]), \
                       np.conjugate(invec[1]), \
                       np.conjugate(invec[3])
            else:
                return np.conjugate(invec[0]), \
                       0, \
                       0, \
                       np.conjugate(invec[1])
    return qcjit(impl)


def iunpack_factory(mode):

    if mode.literal_value == "full":
        def impl(outvec, invec):
            outvec[0] = invec[0]
            outvec[1] = invec[1]
            outvec[2] = invec[2]
            outvec[3] = invec[3]
    elif mode.literal_value == "diag":
        def impl(outvec, invec):
            outvec[0] = invec[0]
            outvec[1] = invec[1]
    else:
        def impl(outvec, invec):
            if len(invec) == 4:
                outvec[0] = invec[0]
                outvec[1] = invec[1]
                outvec[2] = invec[2]
                outvec[3] = invec[3]
            else:
                outvec[0] = invec[0]
                outvec[1] = 0
                outvec[2] = 0
                outvec[3] = invec[1]
    return qcjit(impl)


def iunpackct_factory(mode):

    if mode.literal_value == "full":
        def impl(outvec, invec):
            outvec[0] = np.conjugate(invec[0])
            outvec[1] = np.conjugate(invec[2])
            outvec[2] = np.conjugate(invec[1])
            outvec[3] = np.conjugate(invec[3])
    elif mode.literal_value == "diag":
        def impl(outvec, invec):
            outvec[0] = np.conjugate(invec[0])
            outvec[1] = np.conjugate(invec[1])
    else:
        def impl(outvec, invec):
            if len(invec) == 4:
                outvec[0] = np.conjugate(invec[0])
                outvec[1] = np.conjugate(invec[2])
                outvec[2] = np.conjugate(invec[1])
                outvec[3] = np.conjugate(invec[3])
            else:
                outvec[0] = np.conjugate(invec[0])
                outvec[1] = 0
                outvec[2] = 0
                outvec[3] = np.conjugate(invec[1])
    return qcjit(impl)


def iadd_factory(mode):

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(outvec, invec):
            outvec[0] += invec[0]
            outvec[1] += invec[1]
            outvec[2] += invec[2]
            outvec[3] += invec[3]
    else:
        def impl(outvec, invec):
            outvec[0] += invec[0]
            outvec[1] += invec[1]
    return qcjit(impl)


def valloc_factory(mode):
    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(dtype, leading_dims=()):
            return np.empty((*leading_dims, 4), dtype=dtype)
    else:
        def impl(dtype, leading_dims=()):
            return np.empty((*leading_dims, 2), dtype=dtype)
    return qcjit(impl)


def loop_var_factory(mode):
    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(n_gains, active_term):
            all_terms = np.arange(n_gains - 1, -1, -1)
            gt_active = np.arange(n_gains - 1, active_term, -1)
            lt_active = np.arange(active_term)
            return all_terms, gt_active, lt_active
    else:
        def impl(n_gains, active_term):
            all_terms = np.arange(n_gains - 1, -1, -1)
            gt_active = np.where(np.arange(n_gains) != active_term)[0]
            lt_active = np.arange(0)
            return all_terms, gt_active, lt_active
    return qcjit(impl)