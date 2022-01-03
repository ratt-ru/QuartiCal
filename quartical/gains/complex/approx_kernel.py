# -*- coding: utf-8 -*-
import numpy as np
from numba import prange, generated_jit
from quartical.utils.numba import coerce_literal
from quartical.gains.general.generics import (solver_intermediaries,
                                              invert_gains,
                                              compute_residual_solver,
                                              per_array_jhj_jhr)
from quartical.gains.general.flagging import (flag_intermediaries,
                                              update_gain_flags,
                                              finalize_gain_flags,
                                              apply_gain_flags)
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
def approx_complex_solver(base_args, term_args, meta_args, corr_mode):

    coerce_literal(approx_complex_solver, ["corr_mode"])

    def impl(base_args, term_args, meta_args, corr_mode):

        data = base_args.data
        gains = base_args.gains
        gain_flags = base_args.gain_flags
        inverse_gains = base_args.inverse_gains

        active_term = meta_args.active_term
        max_iter = meta_args.iters
        solve_per = meta_args.solve_per
        dd_term = meta_args.dd_term

        active_gain = gains[active_term]
        active_gain_flags = gain_flags[active_term]

        # Set up some intemediaries used for flagging. TODO: Move?
        km1_gain = active_gain.copy()
        km1_abs2_diffs = np.zeros_like(active_gain_flags, dtype=np.float64)
        abs2_diffs_trend = np.zeros_like(active_gain_flags, dtype=np.float64)
        flag_imdry = \
            flag_intermediaries(km1_gain, km1_abs2_diffs, abs2_diffs_trend)

        # Set up some intemediaries used for solving. TODO: Move?
        jhj = np.empty_like(active_gain)
        jhr = np.empty_like(active_gain)
        residual = data.astype(np.complex128)  # Make a high precision copy.
        update = np.zeros_like(active_gain)
        solver_imdry = solver_intermediaries(jhj, jhr, residual, update)

        invert_gains(gains, inverse_gains, corr_mode)

        for loop_idx in range(max_iter):

            if dd_term:
                compute_residual_solver(base_args,
                                        solver_imdry,
                                        corr_mode)

            compute_jhj_jhr(base_args,
                            term_args,
                            meta_args,
                            solver_imdry,
                            corr_mode)

            if solve_per == "array":
                per_array_jhj_jhr(solver_imdry)

            compute_update(solver_imdry,
                           corr_mode)

            finalize_update(base_args,
                            term_args,
                            meta_args,
                            solver_imdry,
                            loop_idx,
                            corr_mode)

            # Check for gain convergence. Produced as a side effect of
            # flagging. The converged percentage is based on unflagged
            # intervals.
            conv_perc = update_gain_flags(base_args,
                                          term_args,
                                          meta_args,
                                          flag_imdry,
                                          loop_idx,
                                          corr_mode)

            if conv_perc >= meta_args.stop_frac:
                break

        # NOTE: Removes soft flags and flags points which have bad trends.
        finalize_gain_flags(base_args,
                            meta_args,
                            flag_imdry,
                            corr_mode)

        # Call this one last time to ensure points flagged by finialize are
        # propagated (in the DI case).
        if not dd_term:
            apply_gain_flags(base_args,
                             meta_args)

        return jhj, term_conv_info(loop_idx + 1, conv_perc)

    return impl


@generated_jit(nopython=True,
               fastmath=True,
               parallel=True,
               cache=True,
               nogil=True)
def compute_jhj_jhr(base_args, term_args, meta_args, solver_imdry, corr_mode):

    # We want to dispatch based on this field so we need its type.
    row_weights = base_args[base_args.fields.index('row_weights')]

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

    def impl(base_args, term_args, meta_args, solver_imdry, corr_mode):

        active_term = meta_args.active_term

        model = base_args.model
        weights = base_args.weights
        flags = base_args.flags
        a1 = base_args.a1
        a2 = base_args.a2
        row_map = base_args.row_map
        row_weights = base_args.row_weights

        gains = base_args.gains
        inverse_gains = base_args.inverse_gains
        t_map_arr = base_args.t_map_arr[0]  # We only need the gain mappings.
        f_map_arr = base_args.f_map_arr[0]  # We only need the gain mappings.
        d_map_arr = base_args.d_map_arr

        jhj = solver_imdry.jhj
        jhr = solver_imdry.jhr
        residual = solver_imdry.residual

        _, n_chan, n_dir, n_corr = model.shape

        jhj[:] = 0
        jhr[:] = 0

        n_tint, n_fint, n_ant, n_gdir, n_corr = jhr.shape
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
def compute_update(solver_imdry, corr_mode):

    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    compute_det = factories.compute_det_factory(corr_mode)
    iinverse = factories.iinverse_factory(corr_mode)

    def impl(solver_imdry, corr_mode):

        jhj = solver_imdry.jhj
        jhr = solver_imdry.jhr
        update = solver_imdry.update

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
def finalize_update(base_args, term_args, meta_args, solver_imdry, loop_idx,
                    corr_mode):

    set_identity = factories.set_identity_factory(corr_mode)

    def impl(base_args, term_args, meta_args, solver_imdry, loop_idx,
             corr_mode):

        dd_term = meta_args.dd_term
        active_term = meta_args.active_term

        gain = base_args.gains[active_term]
        gain_flags = base_args.gain_flags[active_term]

        update = solver_imdry.update

        n_tint, n_fint, n_ant, n_dir, n_corr = gain.shape

        for ti in range(n_tint):
            for fi in range(n_fint):
                for a in range(n_ant):
                    for d in range(n_dir):

                        g = gain[ti, fi, a, d]
                        fl = gain_flags[ti, fi, a, d]
                        upd = update[ti, fi, a, d]

                        if fl == 1:
                            set_identity(g)
                        elif dd_term:
                            upd /= 2
                            g += upd
                        elif loop_idx % 2 == 0:
                            g[:] = upd
                        else:
                            g += upd
                            g /= 2

    return impl
