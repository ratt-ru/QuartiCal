# -*- coding: utf-8 -*-
import numpy as np
from numba import prange, literally, generated_jit, types
from quartical.kernels.generics import (invert_gains,
                                        compute_residual,
                                        compute_convergence)
from quartical.kernels.convenience import (get_row,
                                           get_chan_extents,
                                           get_row_extents,
                                           imul_rweight_factory,
                                           v1_imul_v2_factory,
                                           v1_imul_v2ct_factory,
                                           v1ct_wmul_v2_factory,
                                           iunpack_factory,
                                           iunpackct_factory,
                                           iadd_factory,
                                           iwmul_factory,
                                           valloc_factory,
                                           loop_var_factory,
                                           compute_det_factory,
                                           iinverse_factory)
from collections import namedtuple


# This can be done without a named tuple now. TODO: Add unpacking to
# constructor.
stat_fields = {"conv_iters": np.int64,
               "conv_perc": np.float64}

term_conv_info = namedtuple("term_conv_info", " ".join(stat_fields.keys()))


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def generated_solver(model, data, a1, a2, weights, t_map_arr, f_map_arr,
                     d_map_arr, corr_mode, active_term, inverse_gains,
                     gains, flags, row_map, row_weights):

    if not isinstance(corr_mode, types.Literal):
        return lambda model, data, a1, a2, weights, t_map_arr, f_map_arr, \
                   d_map_arr, corr_mode, active_term, inverse_gains, \
                   gains, flags, row_map, row_weights: literally(corr_mode)

    compute_jhj_jhr = jhj_jhr
    compute_update = update
    finalize_update = finalize

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


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def jhj_jhr(jhj, jhr, model, gains, inverse_gains, residual, a1,
            a2, weights, t_map_arr, f_map_arr, d_map_arr, row_map,
            row_weights, active_term, corr_mode):

    imul_rweight = imul_rweight_factory(corr_mode, row_weights)
    v1_imul_v2 = v1_imul_v2_factory(corr_mode)
    v1_imul_v2ct = v1_imul_v2ct_factory(corr_mode)
    v1ct_wmul_v2 = v1ct_wmul_v2_factory(corr_mode)
    iunpack = iunpack_factory(corr_mode)
    iunpackct = iunpackct_factory(corr_mode)
    iadd = iadd_factory(corr_mode)
    iwmul = iwmul_factory(corr_mode)
    valloc = valloc_factory(corr_mode)
    loop_var = loop_var_factory(corr_mode)

    def impl(jhj, jhr, model, gains, inverse_gains, residual, a1,
             a2, weights, t_map_arr, f_map_arr, d_map_arr, row_map,
             row_weights, active_term, corr_mode):
        _, n_chan, n_dir, n_corr = model.shape

        jhj[:] = 0
        jhr[:] = 0

        n_tint, n_fint, n_ant, n_gdir, _ = gains[active_term].shape
        n_int = n_tint*n_fint

        n_gains = len(gains)

        # Determine the starts and stops of the rows and channels associated
        # with each solution interval. This could even be moved out for speed.
        row_starts, row_stops = get_row_extents(t_map_arr,
                                                active_term,
                                                n_tint)

        chan_starts, chan_stops = get_chan_extents(f_map_arr,
                                                   active_term,
                                                   n_fint,
                                                   n_chan)

        all_terms, gt_active, lt_active = loop_var(n_gains, active_term)

        # Parallel over all solution intervals.
        for i in prange(n_int):

            ti = i//n_fint
            fi = i - ti*n_fint

            rs = row_starts[ti]
            re = row_stops[ti]
            fs = chan_starts[fi]
            fe = chan_stops[fi]

            m_vec = valloc(jhj.dtype)
            mh_vec = valloc(jhj.dtype)
            r_vec = valloc(jhj.dtype)
            rh_vec = valloc(jhj.dtype)

            gains_a = valloc(jhj.dtype, leading_dims=(n_gains,))
            gains_b = valloc(jhj.dtype, leading_dims=(n_gains,))
            igains_a = valloc(jhj.dtype, leading_dims=(n_gains,))
            igains_b = valloc(jhj.dtype, leading_dims=(n_gains,))

            tmp_jh_p = valloc(jhj.dtype, leading_dims=(n_gdir,))
            tmp_jh_q = valloc(jhj.dtype, leading_dims=(n_gdir,))

            for row_ind in range(rs, re):

                row = get_row(row_ind, row_map)
                a1_m, a2_m = a1[row], a2[row]

                for f in range(fs, fe):

                    r = residual[row, f]
                    w = weights[row, f]  # Consider a map?

                    tmp_jh_p[:, :] = 0
                    tmp_jh_q[:, :] = 0

                    for d in range(n_dir):

                        # Construct a small contiguous gain array.
                        for gi in range(n_gains):
                            d_m = d_map_arr[gi, d]  # Broadcast dir.
                            t_m = t_map_arr[row_ind, gi]
                            f_m = f_map_arr[f, gi]

                            gain = gains[gi][t_m, f_m]
                            inverse_gain = inverse_gains[gi][t_m, f_m]

                            iunpack(gains_a[gi], gain[a1_m, d_m])
                            iunpack(gains_b[gi], gain[a2_m, d_m])
                            iunpack(igains_a[gi], inverse_gain[a1_m, d_m])
                            iunpack(igains_b[gi], inverse_gain[a2_m, d_m])

                        imul_rweight(r, r_vec, row_weights, row_ind)
                        iwmul(r_vec, w)
                        iunpackct(rh_vec, r_vec)

                        m = model[row, f, d]
                        imul_rweight(m, m_vec, row_weights, row_ind)
                        iunpackct(mh_vec, m_vec)

                        for g in all_terms:

                            gb = gains_b[g]
                            v1_imul_v2(gb, mh_vec, mh_vec)

                            ga = gains_a[g]
                            v1_imul_v2(ga, m_vec, m_vec)

                        for g in gt_active:

                            ga = gains_a[g]
                            v1_imul_v2ct(mh_vec, ga, mh_vec)

                            gb = gains_b[g]
                            v1_imul_v2ct(m_vec, gb, m_vec)

                        for g in lt_active:

                            gai = igains_a[g]
                            v1_imul_v2(gai, r_vec, r_vec)

                            gbi = igains_b[g]
                            v1_imul_v2(gbi, rh_vec, rh_vec)

                        t_m = t_map_arr[row_ind, active_term]
                        f_m = f_map_arr[f, active_term]
                        out_d = d_map_arr[active_term, d]

                        v1_imul_v2(r_vec, mh_vec, r_vec)

                        iadd(jhr[t_m, f_m, a1_m, out_d], r_vec)
                        iadd(tmp_jh_p[out_d], mh_vec)

                        v1_imul_v2(rh_vec, m_vec, rh_vec)

                        iadd(jhr[t_m, f_m, a2_m, out_d], rh_vec)
                        iadd(tmp_jh_q[out_d], m_vec)

                    for d in range(n_gdir):

                        jhp = tmp_jh_p[d]
                        jhj_vec = v1ct_wmul_v2(jhp, jhp, w)
                        jhj_sel = jhj[t_m, f_m, a1_m, d]
                        iadd(jhj_sel, jhj_vec)

                        jhq = tmp_jh_q[d]
                        jhj_vec = v1ct_wmul_v2(jhq, jhq, w)
                        jhj_sel = jhj[t_m, f_m, a2_m, d]
                        iadd(jhj_sel, jhj_vec)

        return
    return impl


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def update(update, jhj, jhr, corr_mode):

    v1_imul_v2 = v1_imul_v2_factory(corr_mode)
    compute_det = compute_det_factory(corr_mode)
    iinverse = iinverse_factory(corr_mode)

    def impl(update, jhj, jhr, corr_mode):
        n_tint, n_fint, n_ant, n_dir, n_corr = jhj.shape

        for t in range(n_tint):
            for f in range(n_fint):
                for a in range(n_ant):
                    for d in range(n_dir):

                        jhj_sel = jhj[t, f, a, d]

                        det = compute_det(jhj_sel)

                        if det.real < 1e-6:
                            jhj_sel[:] = 0
                        else:
                            iinverse(jhj_sel, det, jhj_sel)

                        v1_imul_v2(jhr[t, f, a, d],
                                   jhj_sel,
                                   update[t, f, a, d])
    return impl


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def finalize(update, gain, i_num, dd_term, corr_mode):

    def impl(update, gain, i_num, dd_term, corr_mode):
        if dd_term:
            gain[:] = gain[:] + update/2
        elif i_num % 2 == 0:
            gain[:] = update
        else:
            gain[:] = (gain[:] + update)/2
    return impl
