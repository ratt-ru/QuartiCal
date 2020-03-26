# -*- coding: utf-8 -*-
import numpy as np
from numba import jit, prange, literally
from numba.extending import overload
from cubicalv2.kernels.generics import (invert_gains,
                                        compute_residual,
                                        compute_convergence)
from collections import namedtuple


# This can be done without a named tuple now. TODO: Add unpacking to
# constructor.
stat_fields = {"conv_iters": np.int64,
               "conv_perc": np.float64}

term_conv_info = namedtuple("term_conv_info", " ".join(stat_fields.keys()))


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def phase_solver(model, data, a1, a2, weights, t_map_arr, f_map_arr,
                 d_map_arr, corr_mode, gain_list, gain_flag_list,
                 inverse_gain_list, gain_ind, params):

    n_tint, t_fint, n_ant, n_dir, n_corr = gain_list[gain_ind].shape

    invert_gains(gain_list, inverse_gain_list, literally(corr_mode))

    dd_term = n_dir > 1

    last_gain = gain_list[gain_ind].copy()

    cnv_perc = 0.

    for i in range(20):

        if dd_term:
            residual = compute_residual(data, model, gain_list, a1, a2,
                                        t_map_arr, f_map_arr, d_map_arr,
                                        literally(corr_mode))
        else:
            residual = data

        jhj, jhr = compute_jhj_jhr(model,
                                   gain_list,
                                   inverse_gain_list,
                                   residual,
                                   a1,
                                   a2,
                                   weights,
                                   t_map_arr,
                                   f_map_arr,
                                   d_map_arr,
                                   gain_ind,
                                   literally(corr_mode))

        update = compute_update(jhj,
                                jhr,
                                literally(corr_mode))

        finalize_update(update,
                        params,
                        gain_list[gain_ind],
                        i,
                        dd_term,
                        literally(corr_mode))

        # Check for gain convergence. TODO: This can be affected by the
        # weights. Currently unsure how or why, but using unity weights
        # leads to monotonic convergence in all solution intervals.

        cnv_perc = compute_convergence(gain_list[gain_ind][:], last_gain)

        last_gain[:] = gain_list[gain_ind][:]

        if cnv_perc > 0.99:
            break

    return term_conv_info(i, cnv_perc)


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def compute_jhj_jhr(model, gain_list, inverse_gain_list, residual, a1, a2,
                    weights, t_map_arr, f_map_arr, d_map_arr, active_term,
                    corr_mode):

    return _compute_jhj_jhr(model, gain_list, inverse_gain_list, residual, a1,
                            a2, weights, t_map_arr, f_map_arr, d_map_arr,
                            active_term, literally(corr_mode))


def _compute_jhj_jhr(model, gain_list, inverse_gain_list, residual, a1, a2,
                     weights, t_map_arr, f_map_arr, d_map_arr, active_term,
                     corr_mode):
    pass


@overload(_compute_jhj_jhr, inline="always")
def _compute_jhj_jhr_impl(model, gain_list, inverse_gain_list, residual, a1,
                          a2, weights, t_map_arr, f_map_arr, d_map_arr,
                          active_term, corr_mode):

    if corr_mode.literal_value == "diag":
        return jhj_jhr_diag
    else:
        raise NotImplementedError("Phase-only gain not yet supported in "
                                  "non-diagonal modes.")


def jhj_jhr_diag(model, gain_list, inverse_gain_list, residual, a1, a2,
                 weights, t_map_arr, f_map_arr, d_map_arr, active_term,
                 corr_mode):

    n_rows, n_chan, n_dir, n_corr = model.shape
    n_out_dir = gain_list[active_term].shape[3]

    cmplx_dtype = gain_list[active_term].dtype
    real_dtype = gain_list[active_term].real.dtype

    n_tint, n_fint, n_ant, n_gdir, _ = gain_list[active_term].shape
    n_param = 1  # One phase per antenna.

    jhj = np.zeros((n_tint, n_fint, n_ant, n_gdir, n_param, n_corr),
                   dtype=real_dtype)
    jhr = np.zeros((n_tint, n_fint, n_ant, n_gdir, n_param, n_corr),
                   dtype=real_dtype)

    n_gains = len(gain_list)

    inactive_terms = list(range(n_gains))
    inactive_terms.pop(active_term)

    for ti in prange(n_tint):
        row_sel = np.where(t_map_arr[:, active_term] == ti)[0]

        tmp_jh_p = np.zeros((n_out_dir, n_corr), dtype=cmplx_dtype)
        tmp_jh_q = np.zeros((n_out_dir, n_corr), dtype=cmplx_dtype)

        for row in row_sel:

            a1_m, a2_m = a1[row], a2[row]

            for f in range(n_chan):

                r = residual[row, f]
                w = weights[row, f]  # Consider a map?

                w00 = w[0]
                w11 = w[1]

                tmp_jh_p[:, :] = 0
                tmp_jh_q[:, :] = 0

                for d in range(n_dir):

                    out_d = d_map_arr[active_term, d]

                    r00 = w00*r[0]
                    r11 = w11*r[1]

                    rh00 = r00.conjugate()
                    rh11 = r11.conjugate()

                    m = model[row, f, d]

                    m00 = m[0]
                    m11 = m[1]

                    mh00 = m[0].conjugate()
                    mh11 = m[1].conjugate()

                    for g in range(n_gains - 1, -1, -1):

                        d_m = d_map_arr[g, d]  # Broadcast dir.
                        t_m = t_map_arr[row, g]
                        f_m = f_map_arr[f, g]
                        gb = gain_list[g][t_m, f_m, a2_m, d_m]

                        g00 = gb[0]
                        g11 = gb[1]

                        jh00 = (g00*mh00)
                        jh11 = (g11*mh11)

                        mh00 = jh00
                        mh11 = jh11

                    for g in inactive_terms:

                        d_m = d_map_arr[g, d]  # Broadcast dir.
                        t_m = t_map_arr[row, g]
                        f_m = f_map_arr[f, g]
                        ga = gain_list[g][t_m, f_m, a1_m, d_m]

                        gh00 = ga[0].conjugate()
                        gh11 = ga[1].conjugate()

                        jh00 = (mh00*gh00)
                        jh11 = (mh11*gh11)

                        mh00 = jh00
                        mh11 = jh11

                    t_m = t_map_arr[row, active_term]
                    f_m = f_map_arr[f, active_term]

                    ga = gain_list[active_term][t_m, f_m, a1_m, d_m]

                    ga00 = -1j*ga[0].conjugate()
                    ga11 = -1j*ga[1].conjugate()

                    jhr[t_m, f_m, a1_m, out_d, 0, 0] += (ga00*r00*jh00).real
                    jhr[t_m, f_m, a1_m, out_d, 0, 1] += (ga11*r11*jh11).real

                    tmp_jh_p[out_d, 0] += jh00
                    tmp_jh_p[out_d, 1] += jh11

                    for g in range(n_gains-1, -1, -1):

                        d_m = d_map_arr[g, d]  # Broadcast dir.
                        t_m = t_map_arr[row, g]
                        f_m = f_map_arr[f, g]
                        ga = gain_list[g][t_m, f_m, a1_m, d_m]

                        g00 = ga[0]
                        g11 = ga[1]

                        jh00 = (g00*m00)
                        jh11 = (g11*m11)

                        m00 = jh00
                        m11 = jh11

                    for g in inactive_terms:

                        d_m = d_map_arr[g, d]  # Broadcast dir.
                        t_m = t_map_arr[row, g]
                        f_m = f_map_arr[f, g]
                        gb = gain_list[g][t_m, f_m, a2_m, d_m]

                        gh00 = gb[0].conjugate()
                        gh11 = gb[1].conjugate()

                        jh00 = (m00*gh00)
                        jh11 = (m11*gh11)

                        m00 = jh00
                        m11 = jh11

                    t_m = t_map_arr[row, active_term]
                    f_m = f_map_arr[f, active_term]

                    gb = gain_list[active_term][t_m, f_m, a2_m, d_m]

                    gb00 = -1j*gb[0].conjugate()
                    gb11 = -1j*gb[1].conjugate()

                    jhr[t_m, f_m, a2_m, out_d, 0, 0] += (gb00*rh00*jh00).real
                    jhr[t_m, f_m, a2_m, out_d, 0, 1] += (gb11*rh11*jh11).real

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

    return jhj, jhr


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def compute_update(jhj, jhr, corr_mode):

    return _compute_update(jhj, jhr, literally(corr_mode))


def _compute_update(jhj, jhr, corr_mode):
    pass


@overload(_compute_update, inline="always")
def _compute_update_impl(jhj, jhr, corr_mode):

    if corr_mode.literal_value == "diag":
        return update_diag
    else:
        raise NotImplementedError("Phase-only gain not supported in "
                                  "non-diagonal modes.")


def update_diag(jhj, jhr, corr_mode):

    n_tint, n_fint, n_ant, n_dir, n_param, n_corr = jhj.shape

    update = np.empty_like(jhr)

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

    return update


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def finalize_update(update, params, gain, i_num, dd_term, corr_mode):

    return _finalize_update(update, params, gain, i_num, dd_term, corr_mode)


def _finalize_update(update, params, gain, i_num, dd_term, corr_mode):
    pass


@overload(_finalize_update, inline="always")
def _finalize_update_impl(update, params, gain, i_num, dd_term, corr_mode):

    if corr_mode.literal_value == "diag":
        return finalize_diag
    else:
        raise NotImplementedError("Phase-only gain not yet supported in "
                                  "non-diagonal modes.")


def finalize_diag(update, params, gain, i_num, dd_term, corr_mode):

    params[:] = params[:] + update/2

    gain[:] = np.exp(1j*params[:, :, :, :, 0, :])  # This may be a bit slow.
