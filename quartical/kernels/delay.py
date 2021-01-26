# -*- coding: utf-8 -*-
import numpy as np
from numba import jit, prange, literally, types
from numba.extending import overload, register_jitable
from quartical.kernels.generics import (invert_gains,
                                        compute_residual,
                                        compute_convergence)
from quartical.kernels.helpers import get_row, mul_rweight
from collections import namedtuple


# This can be done without a named tuple now. TODO: Add unpacking to
# constructor.
stat_fields = {"conv_iters": np.int64,
               "conv_perc": np.float64}

term_conv_info = namedtuple("term_conv_info", " ".join(stat_fields.keys()))


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def delay_solver(model, data, a1, a2, weights, t_map_arr, f_map_arr,
                 d_map_arr, corr_mode, active_term, inverse_gain_list,
                 gains, flags, params, chan_freqs, row_map, row_weights,
                 t_bin_arr):

    n_tint, n_fint, n_ant, n_dir, n_param, n_corr = params.shape

    invert_gains(gains, inverse_gain_list, literally(corr_mode))

    dd_term = n_dir > 1

    last_gain = gains[active_term].copy()

    cnv_perc = 0.

    real_dtype = gains[active_term].real.dtype

    intemediary_shape = (n_tint, n_fint, n_ant, n_dir, n_param, n_corr)

    jhj = np.empty(intemediary_shape, dtype=real_dtype)
    jhr = np.empty(intemediary_shape, dtype=real_dtype)
    update = np.empty(intemediary_shape, dtype=real_dtype)

    for i in range(20):

        if dd_term:
            residual = compute_residual(data, model, gains, a1, a2,
                                        t_map_arr, f_map_arr, d_map_arr,
                                        row_map, row_weights,
                                        literally(corr_mode))
        else:
            residual = data

        compute_jhj_jhr(jhj,
                        jhr,
                        model,
                        gains,
                        inverse_gain_list,
                        chan_freqs,
                        residual,
                        a1,
                        a2,
                        weights,
                        t_map_arr,
                        f_map_arr,
                        d_map_arr,
                        row_map,
                        row_weights,
                        active_term,
                        corr_mode)

        compute_update(update,
                       jhj,
                       jhr,
                       literally(corr_mode))

        finalize_update(update,
                        params,
                        gains[active_term],
                        chan_freqs,
                        t_bin_arr,
                        f_map_arr,
                        d_map_arr,
                        dd_term,
                        literally(corr_mode),
                        active_term)

        # Check for gain convergence. TODO: This can be affected by the
        # weights. Currently unsure how or why, but using unity weights
        # leads to monotonic convergence in all solution intervals.

        cnv_perc = compute_convergence(gains[active_term][:], last_gain)

        last_gain[:] = gains[active_term][:]

        if cnv_perc > 0.99:
            break

    return term_conv_info(i, cnv_perc)


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def compute_jhj_jhr(jhj, jhr, model, gains, inverse_gain_list, chan_freqs,
                    residual, a1, a2, weights, t_map_arr, f_map_arr,
                    d_map_arr, row_map, row_weights, active_term, corr_mode):

    return _compute_jhj_jhr(jhj, jhr, model, gains, inverse_gain_list,
                            chan_freqs, residual, a1, a2, weights, t_map_arr,
                            f_map_arr, d_map_arr, row_map, row_weights,
                            active_term, corr_mode)


def _compute_jhj_jhr(jhj, jhr, model, gains, inverse_gain_list, chan_freqs,
                     residual, a1, a2, weights, t_map_arr, f_map_arr,
                     d_map_arr, row_map, row_weights, active_term, corr_mode):
    pass


@overload(_compute_jhj_jhr, inline="always")
def _compute_jhj_jhr_impl(jhj, jhr, model, gains, inverse_gain_list,
                          chan_freqs, residual, a1, a2, weights, t_map_arr,
                          f_map_arr, d_map_arr, row_map, row_weights,
                          active_term, corr_mode):

    if not isinstance(corr_mode, types.Literal):
        return lambda mode: literally(mode)

    if corr_mode.literal_value == "diag":
        return jhj_jhr_diag
    else:
        raise NotImplementedError("Delay not yet supported in "
                                  "non-diagonal modes.")


@register_jitable
def jhj_jhr_diag(jhj, jhr, model, gains, inverse_gain_list, chan_freqs,
                 residual, a1, a2, weights, t_map_arr, f_map_arr, d_map_arr,
                 row_map, row_weights, active_term, corr_mode):

    _, n_chan, n_dir, n_corr = model.shape
    n_out_dir = gains[active_term].shape[3]

    jhj[:] = 0
    jhr[:] = 0

    n_tint, n_fint, n_ant, n_gdir, _ = gains[active_term].shape

    cmplx_dtype = gains[active_term].dtype

    n_gains = len(gains)

    inactive_terms = list(range(n_gains))
    inactive_terms.pop(active_term)

    for ti in prange(n_tint):
        row_sel = np.where(t_map_arr[:, active_term] == ti)[0]

        # Two here is the number of parameters. TODO: Flexible?
        tmp_jh_p = np.zeros((n_out_dir, n_corr), dtype=cmplx_dtype)
        tmp_jh_q = np.zeros((n_out_dir, n_corr), dtype=cmplx_dtype)

        for row_ind in row_sel:

            row = get_row(row_ind, row_map)
            a1_m, a2_m = a1[row], a2[row]

            for f in range(n_chan):

                r = residual[row, f]
                w = weights[row, f]  # Consider a map?

                w00 = w[0]
                w11 = w[1]

                tmp_jh_p[:, :] = 0
                tmp_jh_q[:, :] = 0

                nu = chan_freqs[f]

                for d in range(n_dir):

                    out_d = d_map_arr[active_term, d]

                    r00 = w00*mul_rweight(r[0], row_weights, row_ind)
                    r11 = w11*mul_rweight(r[1], row_weights, row_ind)

                    rh00 = r00.conjugate()
                    rh11 = r11.conjugate()

                    m = model[row, f, d]

                    m00 = mul_rweight(m[0], row_weights, row_ind)
                    m11 = mul_rweight(m[1], row_weights, row_ind)

                    mh00 = m00.conjugate()
                    mh11 = m11.conjugate()

                    for g in range(n_gains - 1, -1, -1):

                        d_m = d_map_arr[g, d]  # Broadcast dir.
                        t_m = t_map_arr[row_ind, g]
                        f_m = f_map_arr[f, g]
                        gb = gains[g][t_m, f_m, a2_m, d_m]

                        g00 = gb[0]
                        g11 = gb[1]

                        jh00 = (g00*mh00)
                        jh11 = (g11*mh11)

                        mh00 = jh00
                        mh11 = jh11

                    for g in inactive_terms:

                        d_m = d_map_arr[g, d]  # Broadcast dir.
                        t_m = t_map_arr[row_ind, g]
                        f_m = f_map_arr[f, g]
                        ga = gains[g][t_m, f_m, a1_m, d_m]

                        gh00 = ga[0].conjugate()
                        gh11 = ga[1].conjugate()

                        jh00 = (mh00*gh00)
                        jh11 = (mh11*gh11)

                        mh00 = jh00
                        mh11 = jh11

                    t_m = t_map_arr[row_ind, active_term]
                    f_m = f_map_arr[f, active_term]

                    ga = gains[active_term][t_m, f_m, a1_m, d_m]

                    ga00 = -1j*ga[0].conjugate()
                    ga11 = -1j*ga[1].conjugate()

                    upd00 = (ga00*r00*jh00).real
                    upd11 = (ga11*r11*jh11).real

                    jhr[t_m, f_m, a1_m, out_d, 0, 0] += upd00
                    jhr[t_m, f_m, a1_m, out_d, 0, 1] += upd11

                    jhr[t_m, f_m, a1_m, out_d, 1, 0] += nu*upd00
                    jhr[t_m, f_m, a1_m, out_d, 1, 1] += nu*upd11

                    tmp_jh_p[out_d, 0] += jh00
                    tmp_jh_p[out_d, 1] += jh11

                    for g in range(n_gains-1, -1, -1):

                        d_m = d_map_arr[g, d]  # Broadcast dir.
                        t_m = t_map_arr[row_ind, g]
                        f_m = f_map_arr[f, g]
                        ga = gains[g][t_m, f_m, a1_m, d_m]

                        g00 = ga[0]
                        g11 = ga[1]

                        jh00 = (g00*m00)
                        jh11 = (g11*m11)

                        m00 = jh00
                        m11 = jh11

                    for g in inactive_terms:

                        d_m = d_map_arr[g, d]  # Broadcast dir.
                        t_m = t_map_arr[row_ind, g]
                        f_m = f_map_arr[f, g]
                        gb = gains[g][t_m, f_m, a2_m, d_m]

                        gh00 = gb[0].conjugate()
                        gh11 = gb[1].conjugate()

                        jh00 = (m00*gh00)
                        jh11 = (m11*gh11)

                        m00 = jh00
                        m11 = jh11

                    t_m = t_map_arr[row_ind, active_term]
                    f_m = f_map_arr[f, active_term]

                    gb = gains[active_term][t_m, f_m, a2_m, d_m]

                    gb00 = -1j*gb[0].conjugate()
                    gb11 = -1j*gb[1].conjugate()

                    upd00 = (gb00*rh00*jh00).real
                    upd11 = (gb11*rh11*jh11).real

                    jhr[t_m, f_m, a2_m, out_d, 0, 0] += upd00
                    jhr[t_m, f_m, a2_m, out_d, 0, 1] += upd11

                    jhr[t_m, f_m, a2_m, out_d, 1, 0] += nu*upd00
                    jhr[t_m, f_m, a2_m, out_d, 1, 1] += nu*upd11

                    tmp_jh_q[out_d, 0] += jh00
                    tmp_jh_q[out_d, 1] += jh11

                for d in range(n_out_dir):

                    jh00 = tmp_jh_p[d, 0]
                    jh11 = tmp_jh_p[d, 1]

                    j00 = jh00.conjugate()
                    j11 = jh11.conjugate()

                    jhj[t_m, f_m, a1_m, d, 0, 0] += (j00*w00*jh00).real
                    jhj[t_m, f_m, a1_m, d, 0, 1] += (j11*w11*jh11).real

                    jh00 = tmp_jh_q[d, 0]
                    jh11 = tmp_jh_q[d, 1]

                    j00 = jh00.conjugate()
                    j11 = jh11.conjugate()

                    jhj[t_m, f_m, a2_m, d, 0, 0] += (j00*w00*jh00).real
                    jhj[t_m, f_m, a2_m, d, 0, 1] += (j11*w11*jh11).real

    return


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def compute_update(update, jhj, jhr, corr_mode):

    return _compute_update(update, jhj, jhr, literally(corr_mode))


def _compute_update(update, jhj, jhr, corr_mode):
    pass


@overload(_compute_update, inline="always")
def _compute_update_impl(update, jhj, jhr, corr_mode):

    if corr_mode.literal_value == "diag":
        return update_diag
    else:
        raise NotImplementedError("Phase-only gain not supported in "
                                  "non-diagonal modes.")


def update_diag(update, jhj, jhr, corr_mode):

    n_tint, n_fint, n_ant, n_dir, n_param, n_corr = jhj.shape

    for t in range(n_tint):
        for f in range(n_fint):
            for a in range(n_ant):
                for d in range(n_dir):

                    jhj00 = jhj[t, f, a, d, 0, 0]
                    jhj11 = jhj[t, f, a, d, 0, 1]

                    det = (jhj00*jhj11)

                    if det.real < 1e-6:
                        jhjinv00 = 0
                        jhjinv11 = 0
                    else:
                        jhjinv00 = 1/jhj00
                        jhjinv11 = 1/jhj11

                    jhr00 = jhr[t, f, a, d, 0, 0]
                    jhr11 = jhr[t, f, a, d, 0, 1]

                    update[t, f, a, d, 0, 0] = (jhr00*jhjinv00)
                    update[t, f, a, d, 0, 1] = (jhr11*jhjinv11)

    return


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def finalize_update(update, params, gain, chan_freqs, t_bin_arr, f_map_arr,
                    d_map_arr, dd_term, corr_mode, active_term):

    return _finalize_update(update, params, gain, chan_freqs, t_bin_arr,
                            f_map_arr, d_map_arr, dd_term, corr_mode,
                            active_term)


def _finalize_update(update, params, gain, chan_freqs, t_bin_arr, f_map_arr,
                     d_map_arr, dd_term, corr_mode, active_term):
    pass


@overload(_finalize_update, inline="always")
def _finalize_update_impl(update, params, gain, chan_freqs, t_bin_arr,
                          f_map_arr, d_map_arr, dd_term, corr_mode,
                          active_term):

    if corr_mode.literal_value == "diag":
        return finalize_diag
    else:
        raise NotImplementedError("Delay not yet supported in non-diagonal "
                                  "modes.")


def finalize_diag(update, params, gain, chan_freqs, t_bin_arr, f_map_arr,
                  d_map_arr, dd_term, corr_mode, active_term):

    params[:] = params[:] + update/2

    n_tint, n_fint, n_ant, n_dir, n_param, n_corr = params.shape

    n_time, n_freq, _, _, _ = gain.shape

    for t in range(n_time):
        for f in range(n_freq):
            for a in range(n_ant):
                for d in range(n_dir):

                    t_m = t_bin_arr[t, active_term]
                    f_m = f_map_arr[f, active_term]
                    d_m = d_map_arr[d, active_term]

                    inter0 = params[t_m, f_m, a, d_m, 0, 0]
                    delay0 = params[t_m, f_m, a, d_m, 1, 0]
                    inter1 = params[t_m, f_m, a, d_m, 0, 1]
                    delay1 = params[t_m, f_m, a, d_m, 1, 1]

                    cf = chan_freqs[f]

                    gain[t, f, a, d, 0] = np.exp(1j*(cf*delay0 + inter0))
                    gain[t, f, a, d, 1] = np.exp(1j*(cf*delay1 + inter1))