# -*- coding: utf-8 -*-
import numpy as np
from numba import prange, generated_jit
from quartical.utils.numba import coerce_literal
from quartical.gains.general.generics import (invert_gains,
                                              compute_residual,
                                              compute_convergence,
                                              per_array_jhj_jhr)
from quartical.gains.general.convenience import (get_row,
                                                 get_chan_extents,
                                                 get_row_extents)
import quartical.gains.general.factories as factories
from collections import namedtuple


# This can be done without a named tuple now. TODO: Add unpacking to
# constructor.
stat_fields = {"conv_iters": np.int64,
               "conv_perc": np.float64}

term_conv_info = namedtuple("term_conv_info", " ".join(stat_fields.keys()))

complex_args = namedtuple("complex_args", ())


@generated_jit(nopython=True,
               fastmath=True,
               parallel=False,
               cache=True,
               nogil=True)
def complex_solver(base_args, term_args, meta_args, corr_mode):

    coerce_literal(complex_solver, ["corr_mode"])

    def impl(base_args, term_args, meta_args, corr_mode):

        model = base_args.model
        data = base_args.data
        a1 = base_args.a1
        a2 = base_args.a2
        weights = base_args.weights
        flags = base_args.flags
        t_map_arr = base_args.t_map_arr
        f_map_arr = base_args.f_map_arr
        d_map_arr = base_args.d_map_arr
        inverse_gains = base_args.inverse_gains
        gains = base_args.gains
        gain_flags = base_args.flags
        row_map = base_args.row_map
        row_weights = base_args.row_weights

        stop_frac = meta_args.stop_frac
        stop_crit = meta_args.stop_crit
        active_term = meta_args.active_term
        solve_per = meta_args.solve_per

        n_tint, t_fint, n_ant, n_dir, n_corr = gains[active_term].shape

        t_map_arr = t_map_arr[0]  # We don't need the parameter mappings.
        f_map_arr = f_map_arr[0]  # We don't need the parameter mappings.

        invert_gains(gains, inverse_gains, corr_mode)

        dd_term = n_dir > 1

        last_gain = gains[active_term].copy()

        cnv_perc = 0.

        jhj = np.empty_like(gains[active_term])
        jhr = np.empty_like(gains[active_term])
        update = np.empty_like(gains[active_term])

        for i in range(meta_args.iters):

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
                            flags,
                            t_map_arr,
                            f_map_arr,
                            d_map_arr,
                            row_map,
                            row_weights,
                            active_term,
                            corr_mode)

            if solve_per == "array":
                per_array_jhj_jhr(jhj, jhr)

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

            cnv_perc = compute_convergence(gains[active_term][:],
                                           last_gain,
                                           stop_crit)

            last_gain[:] = gains[active_term][:]

            if cnv_perc >= stop_frac:
                break

        return jhj, term_conv_info(i + 1, cnv_perc)

    return impl


@generated_jit(nopython=True,
               fastmath=True,
               parallel=True,
               cache=True,
               nogil=True)
def compute_jhj_jhr(jhj, jhr, model, gains, inverse_gains, residual, a1,
                    a2, weights, flags, t_map_arr, f_map_arr, d_map_arr,
                    row_map, row_weights, active_term, corr_mode):

    imul_rweight = factories.imul_rweight_factory(corr_mode, row_weights)
    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    v1_imul_v2ct = factories.v1_imul_v2ct_factory(corr_mode)
    v1ct_wmul_v2 = factories.v1ct_wmul_v2_factory(corr_mode)
    iunpack = factories.iunpack_factory(corr_mode)
    iunpackct = factories.iunpackct_factory(corr_mode)
    iadd = factories.iadd_factory(corr_mode)
    iwmul = factories.iwmul_factory(corr_mode)
    valloc = factories.valloc_factory(corr_mode)
    loop_var = factories.loop_var_factory(corr_mode)

    def impl(jhj, jhr, model, gains, inverse_gains, residual, a1,
             a2, weights, flags, t_map_arr, f_map_arr, d_map_arr,
             row_map, row_weights, active_term, corr_mode):
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

        # Determine loop variables based on where we are in the chain.
        # gt means greater than (n>j) and lt means less than (n<j).
        all_terms, gt_active, lt_active = loop_var(n_gains, active_term)

        # Parallel over all solution intervals.
        for i in prange(n_int):

            ti = i//n_fint
            fi = i - ti*n_fint

            rs = row_starts[ti]
            re = row_stops[ti]
            fs = chan_starts[fi]
            fe = chan_stops[fi]

            rop_pq = valloc(jhj.dtype)  # Right-multiply operator for pq.
            rop_qp = valloc(jhj.dtype)  # Right-multiply operator for qp.
            r_pq = valloc(jhj.dtype)
            r_qp = valloc(jhj.dtype)

            gains_p = valloc(jhj.dtype, leading_dims=(n_gains,))
            gains_q = valloc(jhj.dtype, leading_dims=(n_gains,))
            igains_p = valloc(jhj.dtype, leading_dims=(n_gains,))
            igains_q = valloc(jhj.dtype, leading_dims=(n_gains,))

            tmp_jh_p = valloc(jhj.dtype, leading_dims=(n_gdir,))
            tmp_jh_q = valloc(jhj.dtype, leading_dims=(n_gdir,))

            for row_ind in range(rs, re):

                row = get_row(row_ind, row_map)
                a1_m, a2_m = a1[row], a2[row]

                for f in range(fs, fe):

                    if flags[row, f]:  # Skip flagged data points.
                        continue

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

                            iunpack(gains_p[gi], gain[a1_m, d_m])
                            iunpack(gains_q[gi], gain[a2_m, d_m])
                            iunpack(igains_p[gi], inverse_gain[a1_m, d_m])
                            iunpack(igains_q[gi], inverse_gain[a2_m, d_m])

                        imul_rweight(r, r_pq, row_weights, row_ind)
                        iwmul(r_pq, w)
                        iunpackct(r_qp, r_pq)

                        m = model[row, f, d]
                        imul_rweight(m, rop_qp, row_weights, row_ind)
                        iunpackct(rop_pq, rop_qp)

                        for g in all_terms:

                            g_q = gains_q[g]
                            v1_imul_v2(g_q, rop_pq, rop_pq)

                            g_p = gains_p[g]
                            v1_imul_v2(g_p, rop_qp, rop_qp)

                        for g in gt_active:

                            g_p = gains_p[g]
                            v1_imul_v2ct(rop_pq, g_p, rop_pq)

                            g_q = gains_q[g]
                            v1_imul_v2ct(rop_qp, g_q, rop_qp)

                        for g in lt_active:

                            ig_p = igains_p[g]
                            v1_imul_v2(ig_p, r_pq, r_pq)

                            ig_q = igains_q[g]
                            v1_imul_v2(ig_q, r_qp, r_qp)

                        t_m = t_map_arr[row_ind, active_term]
                        f_m = f_map_arr[f, active_term]
                        out_d = d_map_arr[active_term, d]

                        v1_imul_v2(r_pq, rop_pq, r_pq)

                        iadd(jhr[t_m, f_m, a1_m, out_d], r_pq)
                        iadd(tmp_jh_p[out_d], rop_pq)

                        v1_imul_v2(r_qp, rop_qp, r_qp)

                        iadd(jhr[t_m, f_m, a2_m, out_d], r_qp)
                        iadd(tmp_jh_q[out_d], rop_qp)

                    for d in range(n_gdir):

                        jh_p = tmp_jh_p[d]
                        jhj_vec = v1ct_wmul_v2(jh_p, jh_p, w)
                        jhj_p = jhj[t_m, f_m, a1_m, d]
                        iadd(jhj_p, jhj_vec)

                        jh_q = tmp_jh_q[d]
                        jhj_vec = v1ct_wmul_v2(jh_q, jh_q, w)
                        jhj_q = jhj[t_m, f_m, a2_m, d]
                        iadd(jhj_q, jhj_vec)

        return
    return impl


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def compute_update(update, jhj, jhr, corr_mode):

    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    compute_det = factories.compute_det_factory(corr_mode)
    iinverse = factories.iinverse_factory(corr_mode)

    def impl(update, jhj, jhr, corr_mode):
        n_tint, n_fint, n_ant, n_dir, n_corr = jhj.shape

        for t in range(n_tint):
            for f in range(n_fint):
                for a in range(n_ant):
                    for d in range(n_dir):

                        jhj_sel = jhj[t, f, a, d]
                        upd_sel = update[t, f, a, d]

                        det = compute_det(jhj_sel)

                        if det.real < 1e-6:
                            upd_sel[:] = 0
                        else:
                            iinverse(jhj_sel, det, upd_sel)

                        v1_imul_v2(jhr[t, f, a, d],
                                   upd_sel,
                                   upd_sel)
    return impl


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def finalize_update(update, gain, i_num, dd_term, corr_mode):

    def impl(update, gain, i_num, dd_term, corr_mode):
        if dd_term:
            gain[:] = gain[:] + update/2
        elif i_num % 2 == 0:
            gain[:] = update
        else:
            gain[:] = (gain[:] + update)/2
    return impl
