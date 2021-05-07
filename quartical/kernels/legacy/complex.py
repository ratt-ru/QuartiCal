# -*- coding: utf-8 -*-
import numpy as np
from numba import prange, literally, generated_jit, types
from numba.extending import register_jitable
from quartical.kernels.generics import (invert_gains,
                                        compute_residual,
                                        compute_convergence)
from quartical.kernels.convenience import (get_row,
                                           old_mul_rweight as mul_rweight,
                                           get_chan_extents,
                                           get_row_extents)
from collections import namedtuple


# This can be done without a named tuple now. TODO: Add unpacking to
# constructor.
stat_fields = {"conv_iters": np.int64,
               "conv_perc": np.float64}

term_conv_info = namedtuple("term_conv_info", " ".join(stat_fields.keys()))


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def complex_solver(model, data, a1, a2, weights, t_map_arr, f_map_arr,
                   d_map_arr, corr_mode, active_term, inverse_gains,
                   gains, flags, row_map, row_weights):

    if not isinstance(corr_mode, types.Literal):
        return lambda model, data, a1, a2, weights, t_map_arr, f_map_arr, \
                   d_map_arr, corr_mode, active_term, inverse_gains, \
                   gains, flags, row_map, row_weights: literally(corr_mode)

    if corr_mode.literal_value == "diag":
        compute_jhj_jhr = jhj_jhr_diag
        compute_update = update_diag
        finalize_update = finalize_full
    else:
        compute_jhj_jhr = jhj_jhr_full
        compute_update = update_full
        finalize_update = finalize_full

    def impl(model, data, a1, a2, weights, t_map_arr, f_map_arr,
             d_map_arr, corr_mode, active_term, inverse_gains,
             gains, flags, row_map, row_weights):

        n_tint, t_fint, n_ant, n_dir, n_corr = gains[active_term].shape

        invert_gains(gains, inverse_gains, corr_mode)

        dd_term = n_dir > 1

        last_gain = gains[active_term].copy()

        cnv_perc = 0.

        jhj = np.empty_like(gains[active_term])
        jhr = np.empty_like(gains[active_term])
        update = np.empty_like(gains[active_term])

        for i in range(20):

            if dd_term:
                residual = compute_residual(data, model, gains, a1, a2,
                                            t_map_arr, f_map_arr, d_map_arr,
                                            row_map, row_weights,
                                            corr_mode)
            else:
                residual = data

            compute_jhj_jhr(jhj,
                            jhr,
                            model,
                            gains,
                            inverse_gains,
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
                           corr_mode)

            finalize_update(update,
                            gains[active_term],
                            i,
                            dd_term,
                            corr_mode)

            # Check for gain convergence. TODO: This can be affected by the
            # weights. Currently unsure how or why, but using unity weights
            # leads to monotonic convergence in all solution intervals.

            cnv_perc = compute_convergence(gains[active_term][:], last_gain)

            last_gain[:] = gains[active_term][:]

            if cnv_perc > 0.99:
                break

        return term_conv_info(i, cnv_perc)

    return impl


@register_jitable
def jhj_jhr_diag(jhj, jhr, model, gains, inverse_gains, residual, a1,
                 a2, weights, t_map_arr, f_map_arr, d_map_arr, row_map,
                 row_weights, active_term, corr_mode):

    _, n_chan, n_dir, n_corr = model.shape

    jhj[:] = 0
    jhr[:] = 0

    n_tint, n_fint, n_ant, n_gdir, _ = gains[active_term].shape
    n_int = n_tint*n_fint

    n_gains = len(gains)

    # This works but is suboptimal, particularly when it results in an empty
    # array. Can be circumvented with overloads, but that is future work.
    inactive_terms = list(range(n_gains))
    inactive_terms.pop(active_term)
    inactive_terms = np.array(inactive_terms)

    # Determine the starts and stops of the rows and channels associated with
    # each solution interval. This could even be moved out for speed.
    row_starts, row_stops = get_row_extents(t_map_arr,
                                            active_term,
                                            n_tint)

    chan_starts, chan_stops = get_chan_extents(f_map_arr,
                                               active_term,
                                               n_fint,
                                               n_chan)

    for i in prange(n_int):

        ti = i//n_fint
        fi = i - ti*n_fint

        rs = row_starts[ti]
        re = row_stops[ti]
        fs = chan_starts[fi]
        fe = chan_stops[fi]

        tmp_jh_p = np.zeros((n_gdir, n_corr), dtype=jhj.dtype)
        tmp_jh_q = np.zeros((n_gdir, n_corr), dtype=jhj.dtype)

        for row_ind in range(rs, re):

            row = get_row(row_ind, row_map)
            a1_m, a2_m = a1[row], a2[row]

            for f in range(fs, fe):

                r = residual[row, f]
                w = weights[row, f]  # Consider a map?

                w00 = w[0]
                w11 = w[1]

                tmp_jh_p[:, :] = 0
                tmp_jh_q[:, :] = 0

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

                    jhr[t_m, f_m, a1_m, out_d, 0] += (r00*jh00)
                    jhr[t_m, f_m, a1_m, out_d, 1] += (r11*jh11)

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

                    jhr[t_m, f_m, a2_m, out_d, 0] += (rh00*jh00)
                    jhr[t_m, f_m, a2_m, out_d, 1] += (rh11*jh11)

                    tmp_jh_q[out_d, 0] += jh00
                    tmp_jh_q[out_d, 1] += jh11

                for d in range(n_gdir):

                    jh00 = tmp_jh_p[d, 0]
                    jh11 = tmp_jh_p[d, 1]

                    j00 = jh00.conjugate()
                    j11 = jh11.conjugate()

                    jhj[t_m, f_m, a1_m, d, 0] += (j00*w00*jh00)
                    jhj[t_m, f_m, a1_m, d, 1] += (j11*w11*jh11)

                    jh00 = tmp_jh_q[d, 0]
                    jh11 = tmp_jh_q[d, 1]

                    j00 = jh00.conjugate()
                    j11 = jh11.conjugate()

                    jhj[t_m, f_m, a2_m, d, 0] += (j00*w00*jh00)
                    jhj[t_m, f_m, a2_m, d, 1] += (j11*w11*jh11)

    return


@register_jitable
def jhj_jhr_full(jhj, jhr, model, gains, inverse_gains, residual, a1,
                 a2, weights, t_map_arr, f_map_arr, d_map_arr, row_map,
                 row_weights, active_term, corr_mode):

    _, n_chan, n_dir, n_corr = model.shape

    jhj[:] = 0
    jhr[:] = 0

    n_tint, n_fint, n_ant, n_gdir, _ = gains[active_term].shape
    n_int = n_tint*n_fint

    n_gains = len(gains)

    # Determine the starts and stops of the rows and channels associated with
    # each solution interval. This could even be moved out for speed.
    row_starts, row_stops = get_row_extents(t_map_arr,
                                            active_term,
                                            n_tint)

    chan_starts, chan_stops = get_chan_extents(f_map_arr,
                                               active_term,
                                               n_fint,
                                               n_chan)

    # Parallel over all solution intervals.
    for i in prange(n_int):

        ti = i//n_fint
        fi = i - ti*n_fint

        rs = row_starts[ti]
        re = row_stops[ti]
        fs = chan_starts[fi]
        fe = chan_stops[fi]

        tmp_jh_p = np.zeros((n_gdir, n_corr), dtype=jhj.dtype)
        tmp_jh_q = np.zeros((n_gdir, n_corr), dtype=jhj.dtype)

        for row_ind in range(rs, re):

            row = get_row(row_ind, row_map)
            a1_m, a2_m = a1[row], a2[row]

            for f in range(fs, fe):

                r = residual[row, f]
                w = weights[row, f]  # Consider a map?

                w0 = w[0]
                # w1 = w[1]  # We do not use the off diagonal weights.
                # w2 = w[2]  # We do not use the off diagonal weights.
                w3 = w[3]

                tmp_jh_p[:, :] = 0
                tmp_jh_q[:, :] = 0

                for d in range(n_dir):

                    out_d = d_map_arr[active_term, d]

                    r00 = w0*mul_rweight(r[0], row_weights, row_ind)
                    r01 = w0*mul_rweight(r[1], row_weights, row_ind)
                    r10 = w3*mul_rweight(r[2], row_weights, row_ind)
                    r11 = w3*mul_rweight(r[3], row_weights, row_ind)

                    rh00 = r00.conjugate()
                    rh01 = r10.conjugate()
                    rh10 = r01.conjugate()
                    rh11 = r11.conjugate()

                    m = model[row, f, d]

                    m00 = mul_rweight(m[0], row_weights, row_ind)
                    m01 = mul_rweight(m[1], row_weights, row_ind)
                    m10 = mul_rweight(m[2], row_weights, row_ind)
                    m11 = mul_rweight(m[3], row_weights, row_ind)

                    mh00 = m00.conjugate()
                    mh01 = m10.conjugate()
                    mh10 = m01.conjugate()
                    mh11 = m11.conjugate()

                    for g in range(n_gains - 1, -1, -1):

                        d_m = d_map_arr[g, d]  # Broadcast dir.
                        t_m = t_map_arr[row_ind, g]
                        f_m = f_map_arr[f, g]
                        gb = gains[g][t_m, f_m, a2_m, d_m]

                        g00 = gb[0]
                        g01 = gb[1]
                        g10 = gb[2]
                        g11 = gb[3]

                        jh00 = (g00*mh00 + g01*mh10)
                        jh01 = (g00*mh01 + g01*mh11)
                        jh10 = (g10*mh00 + g11*mh10)
                        jh11 = (g10*mh01 + g11*mh11)

                        mh00 = jh00
                        mh01 = jh01
                        mh10 = jh10
                        mh11 = jh11

                    for g in range(n_gains - 1, active_term, -1):

                        d_m = d_map_arr[g, d]  # Broadcast dir.
                        t_m = t_map_arr[row_ind, g]
                        f_m = f_map_arr[f, g]
                        ga = gains[g][t_m, f_m, a1_m, d_m]

                        gh00 = ga[0].conjugate()
                        gh01 = ga[2].conjugate()
                        gh10 = ga[1].conjugate()
                        gh11 = ga[3].conjugate()

                        jh00 = (mh00*gh00 + mh01*gh10)
                        jh01 = (mh00*gh01 + mh01*gh11)
                        jh10 = (mh10*gh00 + mh11*gh10)
                        jh11 = (mh10*gh01 + mh11*gh11)

                        mh00 = jh00
                        mh01 = jh01
                        mh10 = jh10
                        mh11 = jh11

                    for g in range(active_term):

                        d_m = d_map_arr[g, d]  # Broadcast dir.
                        t_m = t_map_arr[row_ind, g]
                        f_m = f_map_arr[f, g]
                        gai = inverse_gains[g][t_m, f_m, a1_m, d_m]

                        ginv00 = gai[0]
                        ginv01 = gai[1]
                        ginv10 = gai[2]
                        ginv11 = gai[3]

                        jhr00 = (ginv00*r00 + ginv01*r10)
                        jhr01 = (ginv00*r01 + ginv01*r11)
                        jhr10 = (ginv10*r00 + ginv11*r10)
                        jhr11 = (ginv10*r01 + ginv11*r11)

                        r00 = jhr00
                        r01 = jhr01
                        r10 = jhr10
                        r11 = jhr11

                    t_m = t_map_arr[row_ind, active_term]
                    f_m = f_map_arr[f, active_term]

                    jhr[t_m, f_m, a1_m, out_d, 0] += (r00*jh00 + r01*jh10)
                    jhr[t_m, f_m, a1_m, out_d, 1] += (r00*jh01 + r01*jh11)
                    jhr[t_m, f_m, a1_m, out_d, 2] += (r10*jh00 + r11*jh10)
                    jhr[t_m, f_m, a1_m, out_d, 3] += (r10*jh01 + r11*jh11)

                    tmp_jh_p[out_d, 0] += jh00
                    tmp_jh_p[out_d, 1] += jh01
                    tmp_jh_p[out_d, 2] += jh10
                    tmp_jh_p[out_d, 3] += jh11

                    for g in range(n_gains-1, -1, -1):

                        d_m = d_map_arr[g, d]  # Broadcast dir.
                        t_m = t_map_arr[row_ind, g]
                        f_m = f_map_arr[f, g]
                        ga = gains[g][t_m, f_m, a1_m, d_m]

                        g00 = ga[0]
                        g01 = ga[1]
                        g10 = ga[2]
                        g11 = ga[3]

                        jh00 = (g00*m00 + g01*m10)
                        jh01 = (g00*m01 + g01*m11)
                        jh10 = (g10*m00 + g11*m10)
                        jh11 = (g10*m01 + g11*m11)

                        m00 = jh00
                        m01 = jh01
                        m10 = jh10
                        m11 = jh11

                    for g in range(n_gains - 1, active_term, -1):

                        d_m = d_map_arr[g, d]  # Broadcast dir.
                        t_m = t_map_arr[row_ind, g]
                        f_m = f_map_arr[f, g]
                        gb = gains[g][t_m, f_m, a2_m, d_m]

                        gh00 = gb[0].conjugate()
                        gh01 = gb[2].conjugate()
                        gh10 = gb[1].conjugate()
                        gh11 = gb[3].conjugate()

                        jh00 = (m00*gh00 + m01*gh10)
                        jh01 = (m00*gh01 + m01*gh11)
                        jh10 = (m10*gh00 + m11*gh10)
                        jh11 = (m10*gh01 + m11*gh11)

                        m00 = jh00
                        m01 = jh01
                        m10 = jh10
                        m11 = jh11

                    for g in range(active_term):

                        d_m = d_map_arr[g, d]  # Broadcast dir.
                        t_m = t_map_arr[row_ind, g]
                        f_m = f_map_arr[f, g]
                        gbi = inverse_gains[g][t_m, f_m, a2_m, d_m]

                        ginv00 = gbi[0]
                        ginv01 = gbi[1]
                        ginv10 = gbi[2]
                        ginv11 = gbi[3]

                        jhr00 = (ginv00*rh00 + ginv01*rh10)
                        jhr01 = (ginv00*rh01 + ginv01*rh11)
                        jhr10 = (ginv10*rh00 + ginv11*rh10)
                        jhr11 = (ginv10*rh01 + ginv11*rh11)

                        rh00 = jhr00
                        rh01 = jhr01
                        rh10 = jhr10
                        rh11 = jhr11

                    t_m = t_map_arr[row_ind, active_term]
                    f_m = f_map_arr[f, active_term]

                    jhr[t_m, f_m, a2_m, out_d, 0] += (rh00*jh00 + rh01*jh10)
                    jhr[t_m, f_m, a2_m, out_d, 1] += (rh00*jh01 + rh01*jh11)
                    jhr[t_m, f_m, a2_m, out_d, 2] += (rh10*jh00 + rh11*jh10)
                    jhr[t_m, f_m, a2_m, out_d, 3] += (rh10*jh01 + rh11*jh11)

                    tmp_jh_q[out_d, 0] += jh00
                    tmp_jh_q[out_d, 1] += jh01
                    tmp_jh_q[out_d, 2] += jh10
                    tmp_jh_q[out_d, 3] += jh11

                for d in range(n_gdir):

                    jh00 = tmp_jh_p[d, 0]
                    jh01 = tmp_jh_p[d, 1]
                    jh10 = tmp_jh_p[d, 2]
                    jh11 = tmp_jh_p[d, 3]

                    j00 = jh00.conjugate()
                    j01 = jh10.conjugate()
                    j10 = jh01.conjugate()
                    j11 = jh11.conjugate()

                    jhj[t_m, f_m, a1_m, d, 0] += (j00*w0*jh00 + j01*w3*jh10)
                    jhj[t_m, f_m, a1_m, d, 1] += (j00*w0*jh01 + j01*w3*jh11)
                    jhj[t_m, f_m, a1_m, d, 2] += (j10*w0*jh00 + j11*w3*jh10)
                    jhj[t_m, f_m, a1_m, d, 3] += (j10*w0*jh01 + j11*w3*jh11)

                    jh00 = tmp_jh_q[d, 0]
                    jh01 = tmp_jh_q[d, 1]
                    jh10 = tmp_jh_q[d, 2]
                    jh11 = tmp_jh_q[d, 3]

                    j00 = jh00.conjugate()
                    j01 = jh10.conjugate()
                    j10 = jh01.conjugate()
                    j11 = jh11.conjugate()

                    jhj[t_m, f_m, a2_m, d, 0] += (j00*w0*jh00 + j01*w3*jh10)
                    jhj[t_m, f_m, a2_m, d, 1] += (j00*w0*jh01 + j01*w3*jh11)
                    jhj[t_m, f_m, a2_m, d, 2] += (j10*w0*jh00 + j11*w3*jh10)
                    jhj[t_m, f_m, a2_m, d, 3] += (j10*w0*jh01 + j11*w3*jh11)

    return


@register_jitable
def update_diag(update, jhj, jhr, corr_mode):

    n_tint, n_fint, n_ant, n_dir, n_corr = jhj.shape

    for t in range(n_tint):
        for f in range(n_fint):
            for a in range(n_ant):
                for d in range(n_dir):

                    jhj00 = jhj[t, f, a, d, 0]
                    jhj11 = jhj[t, f, a, d, 1]

                    det = (jhj00*jhj11)

                    if det.real < 1e-6:
                        jhjinv00 = 0
                        jhjinv11 = 0
                    else:
                        jhjinv00 = 1/jhj00
                        jhjinv11 = 1/jhj11

                    jhr00 = jhr[t, f, a, d, 0]
                    jhr11 = jhr[t, f, a, d, 1]

                    update[t, f, a, d, 0] = (jhr00*jhjinv00)
                    update[t, f, a, d, 1] = (jhr11*jhjinv11)

    return


@register_jitable
def update_full(update, jhj, jhr, corr_mode):

    n_tint, n_fint, n_ant, n_dir, n_corr = jhj.shape

    for t in range(n_tint):
        for f in range(n_fint):
            for a in range(n_ant):
                for d in range(n_dir):

                    jhj00 = jhj[t, f, a, d, 0]
                    jhj01 = jhj[t, f, a, d, 1]
                    jhj10 = jhj[t, f, a, d, 2]
                    jhj11 = jhj[t, f, a, d, 3]

                    det = (jhj00*jhj11 - jhj01*jhj10)

                    if det.real < 1e-6:
                        jhjinv00 = 0
                        jhjinv01 = 0
                        jhjinv10 = 0
                        jhjinv11 = 0
                    else:
                        jhjinv00 = jhj11/det
                        jhjinv01 = -jhj01/det
                        jhjinv10 = -jhj10/det
                        jhjinv11 = jhj00/det

                    jhr00 = jhr[t, f, a, d, 0]
                    jhr01 = jhr[t, f, a, d, 1]
                    jhr10 = jhr[t, f, a, d, 2]
                    jhr11 = jhr[t, f, a, d, 3]

                    update[t, f, a, d, 0] = (jhr00*jhjinv00 + jhr01*jhjinv10)
                    update[t, f, a, d, 1] = (jhr00*jhjinv01 + jhr01*jhjinv11)
                    update[t, f, a, d, 2] = (jhr10*jhjinv00 + jhr11*jhjinv10)
                    update[t, f, a, d, 3] = (jhr10*jhjinv01 + jhr11*jhjinv11)

    return


@register_jitable
def finalize_full(update, gain, i_num, dd_term, corr_mode):

    if dd_term:
        gain[:] = gain[:] + update/2
    elif i_num % 2 == 0:
        gain[:] = update
    else:
        gain[:] = (gain[:] + update)/2
