import numpy as np
from numba import jit, types
from numba.extending import overload


model_schema = ("rowlike", "chan", "ant", "dir", "corr")
data_schema = ("rowlike", "chan", "ant", "corr")
gain_schema = ("rowlike", "chan", "ant", "dir", "corr")


@jit(nopython=True, fastmath=False, parallel=False, cache=True, nogil=True)
def column_to_tifiac(in_col, t_map, f_map, ant1_col, ant2_col, n_ti, n_fi,
                     n_a):

    n_row = in_col.shape[0]
    n_chan = in_col.shape[1]
    n_corr = in_col.shape[-1]

    out_dtype = get_output_dtype(in_col)

    out_arr = np.zeros((n_ti.item(), n_fi.item(), n_a, n_corr),
                       dtype=out_dtype)

    for row in range(n_row):

        t_m = t_map[row]
        a1_m = ant1_col[row]
        a2_m = ant2_col[row]

        for chan in range(n_chan):

            f_m = f_map[chan]

            tifiac_inner_loop(out_arr, in_col, t_m, f_m, row, chan, a1_m, a2_m)

    return out_arr


def tifiac_inner_loop(out_arr, in_col, t_m, f_m, row, chan, a1_m, a2_m):
    pass


@overload(tifiac_inner_loop, inline='always')
def tifiac_inner_loop_impl(out_arr, in_col, t_m, f_m, row, chan, a1_m, a2_m):

    if isinstance(in_col.dtype, types.Complex):
        return tifiac_inner_loop_cmplx
    else:
        return tifiac_inner_loop_noncmplx


def tifiac_inner_loop_noncmplx(out_arr, in_col, t_m, f_m, row, chan, a1_m,
                               a2_m):

    out_arr[t_m, f_m, a1_m, :] += in_col[row, chan, :]
    out_arr[t_m, f_m, a2_m, :] += in_col[row, chan, :]


def tifiac_inner_loop_cmplx(out_arr, in_col, t_m, f_m, row, chan, a1_m, a2_m):

    out_arr[t_m, f_m, a1_m, :] += in_col[row, chan, :]
    out_arr[t_m, f_m, a2_m, :] += in_col[row, chan, :].conjugate()


@jit(nopython=True, fastmath=False, parallel=False, cache=True, nogil=True)
def column_to_tfac(in_col, ant1_col, ant2_col, utime_ind, n_ut, n_a):
    """Accumulate a column into (time, freq, antenna, correlation) array."""

    n_row, n_chan, n_corr = in_col.shape

    out_dtype = get_output_dtype(in_col)

    out_arr = np.zeros((n_ut.item(), n_chan, n_a, n_corr), dtype=out_dtype)

    for row in range(n_row):

        t_m = utime_ind[row]
        a1_m = ant1_col[row]
        a2_m = ant2_col[row]

        for chan in range(n_chan):

            tfac_inner_loop(out_arr, in_col, t_m, row, chan, a1_m, a2_m)

    return out_arr


def tfac_inner_loop(out_arr, in_col, t_m, row, chan, a1_m, a2_m):
    pass


@overload(tfac_inner_loop, inline='always')
def tfac_inner_loop_impl(out_arr, in_col, t_m, row, chan, a1_m, a2_m):

    if isinstance(in_col.dtype, types.Complex):
        return tfac_inner_loop_cmplx
    else:
        return tfac_inner_loop_noncmplx


def tfac_inner_loop_noncmplx(out_arr, in_col, t_m, row, chan, a1_m, a2_m):

    out_arr[t_m, chan, a1_m, :] += in_col[row, chan, :]
    out_arr[t_m, chan, a2_m, :] += in_col[row, chan, :]


def tfac_inner_loop_cmplx(out_arr, in_col, t_m, row, chan, a1_m, a2_m):

    out_arr[t_m, chan, a1_m, :] += in_col[row, chan, :]
    out_arr[t_m, chan, a2_m, :] += in_col[row, chan, :].conjugate()


def get_output_dtype(in_col):
    pass


@overload(get_output_dtype, inline='always')
def get_output_dtype_impl(in_col):

    if isinstance(in_col.dtype, types.Boolean):
        return lambda in_col: np.int32
    else:
        return lambda in_col: in_col.dtype


@jit(nopython=True, fastmath=False, parallel=False, cache=True, nogil=True)
def column_to_tfadc(in_col, ant1_col, ant2_col, utime_ind, n_ut, n_a):

    n_row, n_chan, n_dir, n_corr = in_col.shape

    out_arr = np.zeros((n_ut.item(), n_chan, n_a, n_dir, n_corr),
                       dtype=in_col.dtype)

    for row in range(n_row):

        t_m = utime_ind[row]
        a1_m = ant1_col[row]
        a2_m = ant2_col[row]

        for chan in range(n_chan):

            out_arr[t_m, chan, a1_m, :, :] += \
                in_col[row, chan, :, :]
            out_arr[t_m, chan, a2_m, :, :] += \
                in_col[row, chan, :, :].conjugate()

    return out_arr


@jit(nopython=True, fastmath=False, parallel=False, cache=True, nogil=True)
def column_to_abs_tfadc(in_col, ant1_col, ant2_col, utime_ind, n_ut, n_a):

    n_row, n_chan, n_dir, n_corr = in_col.shape

    out_arr = np.zeros((n_ut.item(), n_chan, n_a, n_dir, n_corr),
                       dtype=in_col.real.dtype)

    for row in range(n_row):

        t_m = utime_ind[row]
        a1_m = ant1_col[row]
        a2_m = ant2_col[row]

        for chan in range(n_chan):

            selection = in_col[row, chan, :, :]

            abs_val = selection.real**2 + selection.imag**2

            out_arr[t_m, chan, a1_m, :, :] += abs_val
            out_arr[t_m, chan, a2_m, :, :] += abs_val

    return out_arr


@jit(nopython=True, fastmath=False, parallel=False, cache=True, nogil=True)
def sum_intervals(in_arr, t_int, f_int):

    in_arr_shape = in_arr.shape

    n_time = in_arr_shape[0]
    n_chan = in_arr_shape[1]

    n_tint = np.int64(np.ceil(n_time/t_int))
    n_fint = np.int64(np.ceil(n_chan/f_int))

    out_dtype = get_output_dtype(in_arr)

    out_arr = np.zeros((n_tint, n_fint, *in_arr_shape[2:]), dtype=out_dtype)

    for t in range(n_time):

        t_m = t//t_int

        for f in range(n_chan):

            f_m = f//f_int

            out_arr[t_m, f_m, ...] += in_arr[t, f, ...]

    return out_arr
