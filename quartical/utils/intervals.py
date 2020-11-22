import numpy as np
from numba import jit, types
from numba.extending import overload


model_schema = ("rowlike", "chan", "ant", "dir", "corr")
data_schema = ("rowlike", "chan", "ant", "corr")
gain_schema = ("rowlike", "chan", "ant", "dir", "corr")


# @jit(nopython=True, fastmath=False, parallel=False, cache=True, nogil=True)
# def column_to_tifiac(in_col, t_map, f_map, ant1_col, ant2_col, n_ti, n_fi,
#                      n_a):
#     """Go from a column-like input to a (ti, fi, a, c) output."""

#     n_row = in_col.shape[0]
#     n_chan = in_col.shape[1]
#     n_corr = in_col.shape[-1]

#     out_dtype = get_output_dtype(in_col)

#     out_arr = np.zeros((n_ti.item(), n_fi.item(), n_a, n_corr),
#                        dtype=out_dtype)

#     for row in range(n_row):

#         t_m = t_map[row]
#         a1_m = ant1_col[row]
#         a2_m = ant2_col[row]

#         for chan in range(n_chan):

#             f_m = f_map[chan]

#             tifiac_inner_loop(out_arr, in_col, t_m, f_m, row, chan, a1_m,
#                               a2_m)

#     return out_arr


# def tifiac_inner_loop(out_arr, in_col, t_m, f_m, row, chan, a1_m, a2_m):
#     pass


# @overload(tifiac_inner_loop, inline='always')
# def tifiac_inner_loop_impl(out_arr, in_col, t_m, f_m, row, chan, a1_m, a2_m):

#     if isinstance(in_col.dtype, types.Complex):
#         return tifiac_inner_loop_cmplx
#     else:
#         return tifiac_inner_loop_noncmplx


# def tifiac_inner_loop_noncmplx(out_arr, in_col, t_m, f_m, row, chan, a1_m,
#                                a2_m):

#     out_arr[t_m, f_m, a1_m, :] += in_col[row, chan, :]
#     out_arr[t_m, f_m, a2_m, :] += in_col[row, chan, :]


# def tifiac_inner_loop_cmplx(out_arr, in_col, t_m, f_m, row, chan, a1_m,
#                             a2_m):

#     out_arr[t_m, f_m, a1_m, :] += in_col[row, chan, :]
#     out_arr[t_m, f_m, a2_m, :] += in_col[row, chan, :].conjugate()


@jit(nopython=True, fastmath=False, parallel=False, cache=True, nogil=True)
def rfc_to_tfac(in_col, ant1_col, ant2_col, utime_ind, n_ut, n_a):
    """Accumulate a (r, f, c) column into (t, f, a, c) array."""

    n_row, n_chan, n_corr = in_col.shape

    out_dtype = get_output_dtype(in_col)

    out_arr = np.zeros((n_ut.item(), n_chan, n_a, n_corr), dtype=out_dtype)

    for row in range(n_row):

        t_m = utime_ind[row]
        a1_m = ant1_col[row]
        a2_m = ant2_col[row]

        for chan in range(n_chan):

            tfac_inner_loop(out_arr, in_col, t_m, row, chan, a1_m, a2_m,
                            n_corr)

    return out_arr


def tfac_inner_loop(out_arr, in_col, t_m, row, chan, a1_m, a2_m, n_corr):
    pass


@overload(tfac_inner_loop, inline='always')
def tfac_inner_loop_impl(out_arr, in_col, t_m, row, chan, a1_m, a2_m, n_corr):

    if isinstance(in_col.dtype, types.Complex):
        return tfac_inner_loop_cmplx
    else:
        return tfac_inner_loop_noncmplx


def tfac_inner_loop_noncmplx(out_arr, in_col, t_m, row, chan, a1_m, a2_m,
                             n_corr):

    for c in range(n_corr):
        out_arr[t_m, chan, a1_m, c] += in_col[row, chan, c]
        out_arr[t_m, chan, a2_m, c] += in_col[row, chan, c]


def tfac_inner_loop_cmplx(out_arr, in_col, t_m, row, chan, a1_m, a2_m, n_corr):

    for c in range(n_corr):
        out_arr[t_m, chan, a1_m, c] += in_col[row, chan, c]
        out_arr[t_m, chan, a2_m, c] += in_col[row, chan, c].conjugate()


def get_output_dtype(in_col):
    pass


@overload(get_output_dtype, inline='always')
def get_output_dtype_impl(in_col):

    if isinstance(in_col.dtype, types.Boolean):
        return lambda in_col: np.int32
    else:
        return lambda in_col: in_col.dtype


# @jit(nopython=True, fastmath=False, parallel=False, cache=True, nogil=True)
# def rfdc_to_tfadc(in_col, ant1_col, ant2_col, utime_ind, n_ut, n_a):
#     """Accumulate a (r, f, d, c) column into (t, f, a, d, c) array."""

#     n_row, n_chan, n_dir, n_corr = in_col.shape

#     out_arr = np.zeros((n_ut.item(), n_chan, n_a, n_dir, n_corr),
#                        dtype=in_col.dtype)

#     for row in range(n_row):

#         t_m = utime_ind[row]
#         a1_m = ant1_col[row]
#         a2_m = ant2_col[row]

#         for chan in range(n_chan):

#             for d in range(n_dir):

#                 for c in range(n_corr):

#                     out_arr[t_m, chan, a1_m, d, c] += \
#                         in_col[row, chan, d, c]
#                     out_arr[t_m, chan, a2_m, d, c] += \
#                         in_col[row, chan, d, c].conjugate()

#     return out_arr


@jit(nopython=True, fastmath=False, parallel=False, cache=True, nogil=True)
def rfdc_to_abs_tfadc(in_col, ant1_col, ant2_col, utime_ind, n_ut, n_a):
    """Accumulate (r, f, d, c) column into abs**2 (t, f, a, d, c) array."""

    n_row, n_chan, n_dir, n_corr = in_col.shape

    out_arr = np.zeros((n_ut.item(), n_chan, n_a, n_dir, n_corr),
                       dtype=in_col.real.dtype)

    for row in range(n_row):

        t_m = utime_ind[row]
        a1_m = ant1_col[row]
        a2_m = ant2_col[row]

        for chan in range(n_chan):

            for d in range(n_dir):

                for c in range(n_corr):

                    elem = in_col[row, chan, d, c]

                    abs_val = elem.real**2 + elem.imag**2

                    out_arr[t_m, chan, a1_m, d, c] += abs_val
                    out_arr[t_m, chan, a2_m, d, c] += abs_val

    return out_arr


@jit(nopython=True, fastmath=False, parallel=False, cache=True, nogil=True)
def tfx_to_tifix(in_arr, t_map, f_map):
    """Sum a (t, f, ...) array into a (ti, fi, ...) array."""

    in_arr_shape = in_arr.shape

    n_time = in_arr_shape[0]
    n_chan = in_arr_shape[1]

    n_tint = np.max(t_map) + 1
    n_fint = np.max(f_map) + 1

    out_dtype = get_output_dtype(in_arr)

    trailing_dims = in_arr_shape[2:]

    out_arr = np.zeros((n_tint, n_fint, *trailing_dims), dtype=out_dtype)

    for t in range(n_time):

        t_m = t_map[t]

        for f in range(n_chan):

            f_m = f_map[f]

            tifix_inner_loop(out_arr, in_arr, t, f, t_m, f_m, trailing_dims)

    return out_arr


def tifix_inner_loop(out_arr, in_arr, t, f, t_m, f_m, trailing_dims):
    pass


@overload(tifix_inner_loop, inline='always')
def tifix_inner_loop_impl(out_arr, in_arr, t, f, t_m, f_m, trailing_dims):

    if len(trailing_dims) == 3:
        return tifix_inner_loop_3D
    elif len(trailing_dims) == 2:
        return tifix_inner_loop_2D
    elif len(trailing_dims) == 1:
        return tifix_inner_loop_1D
    else:
        return tifix_inner_loop_0D


def tifix_inner_loop_3D(out_arr, in_arr, t, f, t_m, f_m, trailing_dims):

    for i in range(trailing_dims[0]):
        for j in range(trailing_dims[1]):
            for k in range(trailing_dims[2]):

                out_arr[t_m, f_m, i, j, k] += in_arr[t, f, i, j, k]


def tifix_inner_loop_2D(out_arr, in_arr, t, f, t_m, f_m, trailing_dims):

    for i in range(trailing_dims[0]):
        for j in range(trailing_dims[1]):

            out_arr[t_m, f_m, i, j] += in_arr[t, f, i, j]


def tifix_inner_loop_1D(out_arr, in_arr, t, f, t_m, f_m, trailing_dims):

    for i in range(trailing_dims[0]):

        out_arr[t_m, f_m, i] += in_arr[t, f, i]


def tifix_inner_loop_0D(out_arr, in_arr, t, f, t_m, f_m, trailing_dims):

    out_arr[t_m, f_m] += in_arr[t, f]
