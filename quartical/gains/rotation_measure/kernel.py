# -*- coding: utf-8 -*-
import numpy as np
from numba import prange, generated_jit
from quartical.utils.numba import coerce_literal
from quartical.gains.general.generics import (compute_residual,
                                              per_array_jhj_jhr)
from quartical.gains.general.flagging import (update_gain_flags,
                                              finalize_gain_flags,
                                              apply_gain_flags,
                                              gain_flags_to_param_flags)
from quartical.gains.general.convenience import (get_row,
                                                 get_chan_extents,
                                                 get_row_extents)
import quartical.gains.general.factories as factories
from quartical.gains.general.inversion import (invert_factory,
                                               inversion_buffer_factory)
from collections import namedtuple


# This can be done without a named tuple now. TODO: Add unpacking to
# constructor.
stat_fields = {"conv_iters": np.int64,
               "conv_perc": np.float64}

term_conv_info = namedtuple("term_conv_info", " ".join(stat_fields.keys()))

rm_args = namedtuple(
    "rm_args",
    (
        "params",
        "chan_freqs",
        "param_flags",
        "t_bin_arr"
    )
)


@generated_jit(nopython=True,
               fastmath=True,
               parallel=False,
               cache=True,
               nogil=True)
def rm_solver(base_args, term_args, meta_args, corr_mode):

    coerce_literal(rm_solver, ["corr_mode"])

    def impl(base_args, term_args, meta_args, corr_mode):

        model = base_args.model
        data = base_args.data
        a1 = base_args.a1
        a2 = base_args.a2
        weights = base_args.weights
        flags = base_args.flags
        t_map_arr_g = base_args.t_map_arr[0]  # Don't need time param mappings.
        f_map_arr = base_args.f_map_arr
        f_map_arr_g = base_args.f_map_arr[0]  # Gain mappings.
        f_map_arr_p = base_args.f_map_arr[1]  # Parameter mappings.
        d_map_arr = base_args.d_map_arr
        gains = base_args.gains
        gain_flags = base_args.gain_flags
        row_map = base_args.row_map
        row_weights = base_args.row_weights

        stop_frac = meta_args.stop_frac
        stop_crit = meta_args.stop_crit
        active_term = meta_args.active_term
        iters = meta_args.iters
        solve_per = meta_args.solve_per

        active_params = term_args.params[active_term]  # Params for this term.
        active_param_flags = term_args.param_flags[active_term]
        t_bin_arr = term_args.t_bin_arr
        chan_freqs = term_args.chan_freqs
        lambda_sq = (299792458/chan_freqs)**2

        n_term = len(gains)

        active_gain = gains[active_term]
        active_gain_flags = gain_flags[active_term]

        dd_term = np.any(d_map_arr[active_term])

        # Set up some intemediaries used for flagging.
        last_gain = active_gain.copy()
        km1_abs2_diffs = np.zeros_like(active_gain_flags, dtype=np.float64)
        abs2_diffs_trend = np.zeros_like(active_gain_flags, dtype=np.float64)
        cnv_perc = 0.
        real_dtype = active_gain.real.dtype

        pshape = active_params.shape
        jhj = np.empty(pshape + (pshape[-1],), dtype=real_dtype)
        jhr = np.empty(pshape, dtype=real_dtype)
        update = np.zeros_like(jhr)

        for i in range(iters):

            if dd_term or n_term > 1:
                residual = compute_residual(data,
                                            model,
                                            gains,
                                            a1,
                                            a2,
                                            t_map_arr_g,
                                            f_map_arr_g,
                                            d_map_arr,
                                            row_map,
                                            row_weights,
                                            corr_mode)
            else:
                residual = data

            compute_jhj_jhr(jhj,
                            jhr,
                            model,
                            gains,
                            active_params,
                            residual,
                            a1,
                            a2,
                            weights,
                            flags,
                            t_map_arr_g,
                            f_map_arr_g,
                            f_map_arr_p,
                            d_map_arr,
                            lambda_sq,
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
                            active_params,
                            active_gain,
                            active_gain_flags,
                            lambda_sq,
                            t_bin_arr[:, :, active_term],
                            f_map_arr[:, :, active_term],
                            d_map_arr[active_term, :],
                            corr_mode)

            # Check for gain convergence. Produced as a side effect of
            # flagging. The converged percentage is based on unflagged
            # intervals.
            cnv_perc = update_gain_flags(active_gain,
                                         last_gain,
                                         active_gain_flags,
                                         km1_abs2_diffs,
                                         abs2_diffs_trend,
                                         stop_crit,
                                         corr_mode,
                                         i)

            if not dd_term:
                apply_gain_flags(active_gain_flags,
                                 flags,
                                 active_term,
                                 a1,
                                 a2,
                                 t_map_arr_g,
                                 f_map_arr_g)

            # Don't update the last gain if converged/on final iteration.
            if (cnv_perc >= stop_frac) or (i == iters - 1):
                break
            else:
                last_gain[:] = active_gain

        # NOTE: Removes soft flags and flags points which have bad trends.
        finalize_gain_flags(active_gain,
                            active_gain_flags,
                            abs2_diffs_trend,
                            corr_mode)

        # Propagate gain flags to parameter flags. TODO: Verify that this
        # is adequate. Do we need to consider setting the identity params.
        gain_flags_to_param_flags(active_gain_flags,
                                  active_param_flags,
                                  t_bin_arr[:, :, active_term],
                                  f_map_arr[:, :, active_term],
                                  d_map_arr)

        # Call this one last time to ensure points flagged by finialize are
        # propagated (in the DI case).
        if not dd_term:
            apply_gain_flags(active_gain_flags,
                             flags,
                             active_term,
                             a1,
                             a2,
                             t_map_arr_g,
                             f_map_arr_g)

        return jhj, term_conv_info(i + 1, cnv_perc)

    return impl


@generated_jit(nopython=True,
               fastmath=True,
               parallel=True,
               cache=True,
               nogil=True)
def compute_jhj_jhr(jhj, jhr, model, gains, params, residual, a1, a2, weights,
                    flags, t_map_arr, f_map_arr_g, f_map_arr_p, d_map_arr,
                    lambda_sq, row_map, row_weights, active_term, corr_mode):

    imul_rweight = factories.imul_rweight_factory(corr_mode, row_weights)
    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    v1_imul_v2ct = factories.v1_imul_v2ct_factory(corr_mode)
    v1ct_imul_v2 = factories.v1ct_imul_v2_factory(corr_mode)
    iunpack = factories.iunpack_factory(corr_mode)
    iunpackct = factories.iunpackct_factory(corr_mode)
    imul = factories.imul_factory(corr_mode)
    iadd = factories.iadd_factory(corr_mode)
    valloc = factories.valloc_factory(corr_mode)
    make_loop_vars = factories.loop_var_factory(corr_mode)
    set_identity = factories.set_identity_factory(corr_mode)
    compute_jhwj_jhwr_elem = compute_jhwj_jhwr_elem_factory(corr_mode)

    def impl(jhj, jhr, model, gains, params, residual, a1, a2, weights,
             flags, t_map_arr, f_map_arr_g, f_map_arr_p, d_map_arr,
             lambda_sq, row_map, row_weights, active_term, corr_mode):
        _, n_chan, n_dir, n_corr = model.shape

        jhj[:] = 0
        jhr[:] = 0

        n_tint, n_fint, n_ant, n_gdir, n_param = jhr.shape
        n_int = n_tint*n_fint

        complex_dtype = gains[active_term].dtype

        n_gains = len(gains)

        # Determine the starts and stops of the rows and channels associated
        # with each solution interval. This could even be moved out for speed.
        row_starts, row_stops = get_row_extents(t_map_arr,
                                                active_term,
                                                n_tint)

        chan_starts, chan_stops = get_chan_extents(f_map_arr_p,
                                                   active_term,
                                                   n_fint,
                                                   n_chan)

        # Determine loop variables based on where we are in the chain.
        # gt means greater than (n>j) and lt means less than (n<j).
        all_terms, gt_active, lt_active = make_loop_vars(n_gains, active_term)

        # Parallel over all solution intervals.
        for i in prange(n_int):

            ti = i//n_fint
            fi = i - ti*n_fint

            rs = row_starts[ti]
            re = row_stops[ti]
            fs = chan_starts[fi]
            fe = chan_stops[fi]

            rop_pq = valloc(complex_dtype)  # Right-multiply operator for pq.
            rop_qp = valloc(complex_dtype)  # Right-multiply operator for qp.
            lop_pq = valloc(complex_dtype)  # Left-multiply operator for pq.
            lop_qp = valloc(complex_dtype)  # Left-multiply operator for qp.
            r_pq = valloc(complex_dtype)
            r_qp = valloc(complex_dtype)

            gains_p = valloc(complex_dtype, leading_dims=(n_gains,))
            gains_q = valloc(complex_dtype, leading_dims=(n_gains,))

            lop_pq_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
            rop_pq_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
            lop_qp_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
            rop_qp_arr = valloc(complex_dtype, leading_dims=(n_gdir,))

            tmp_kprod = np.zeros((4, 4), dtype=complex_dtype)
            tmp_jhr = jhr[ti, fi]
            tmp_jhj = jhj[ti, fi]

            for row_ind in range(rs, re):

                row = get_row(row_ind, row_map)
                a1_m, a2_m = a1[row], a2[row]

                rm_t = t_map_arr[row_ind, active_term]

                for f in range(fs, fe):

                    if flags[row, f]:  # Skip flagged data points.
                        continue

                    r = residual[row, f]
                    w = weights[row, f]  # Consider a map?

                    lop_pq_arr[:] = 0
                    rop_pq_arr[:] = 0
                    lop_qp_arr[:] = 0
                    rop_qp_arr[:] = 0

                    rm_f = f_map_arr_p[f, active_term]

                    for d in range(n_dir):

                        set_identity(lop_pq)
                        set_identity(lop_qp)

                        # Construct a small contiguous gain array. This makes
                        # the single term case fractionally slower.
                        for gi in range(n_gains):
                            d_m = d_map_arr[gi, d]  # Broadcast dir.
                            t_m = t_map_arr[row_ind, gi]
                            f_m = f_map_arr_g[f, gi]

                            gain = gains[gi][t_m, f_m]

                            iunpack(gains_p[gi], gain[a1_m, d_m])
                            iunpack(gains_q[gi], gain[a2_m, d_m])

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

                            g_p = gains_p[g]
                            v1ct_imul_v2(g_p, lop_pq, lop_pq)

                            g_q = gains_q[g]
                            v1ct_imul_v2(g_q, lop_qp, lop_qp)

                        out_d = d_map_arr[active_term, d]

                        iunpack(lop_pq_arr[out_d], lop_pq)
                        iadd(rop_pq_arr[out_d], rop_pq)

                        iunpack(lop_qp_arr[out_d], lop_qp)
                        iadd(rop_qp_arr[out_d], rop_qp)

                    lsq = lambda_sq[f]

                    for d in range(n_gdir):

                        rm_p = params[rm_t, rm_f, a1_m, d, 0]
                        rm_q = params[rm_t, rm_f, a2_m, d, 0]

                        imul_rweight(r, r_pq, row_weights, row_ind)
                        imul(r_pq, w)  # Check: Not needed as we compute jhwr.
                        iunpackct(r_qp, r_pq)

                        lop_pq_d = lop_pq_arr[d]
                        rop_pq_d = rop_pq_arr[d]

                        compute_jhwj_jhwr_elem(lop_pq_d,
                                               rop_pq_d,
                                               w,
                                               rm_p,
                                               lsq,
                                               gains_p[active_term],
                                               tmp_kprod,
                                               r_pq,
                                               tmp_jhr[a1_m, d],
                                               tmp_jhj[a1_m, d])

                        lop_qp_d = lop_qp_arr[d]
                        rop_qp_d = rop_qp_arr[d]

                        compute_jhwj_jhwr_elem(lop_qp_d,
                                               rop_qp_d,
                                               w,
                                               rm_q,
                                               lsq,
                                               gains_q[active_term],
                                               tmp_kprod,
                                               r_qp,
                                               tmp_jhr[a2_m, d],
                                               tmp_jhj[a2_m, d])
        return
    return impl


@generated_jit(nopython=True,
               fastmath=True,
               parallel=True,
               cache=True,
               nogil=True)
def compute_update(update, jhj, jhr, corr_mode):

    generalised = jhj.ndim == 6
    inversion_buffer = inversion_buffer_factory(generalised=generalised)
    invert = invert_factory(corr_mode, generalised=generalised)

    def impl(update, jhj, jhr, corr_mode):
        n_tint, n_fint, n_ant, n_dir, n_param = jhr.shape

        n_int = n_tint * n_fint

        result_dtype = jhr.dtype

        for i in prange(n_int):

            t = i // n_fint
            f = i - t * n_fint

            buffers = inversion_buffer(n_param, result_dtype)

            for a in range(n_ant):
                for d in range(n_dir):

                    invert(jhj[t, f, a, d],
                           jhr[t, f, a, d],
                           update[t, f, a, d],
                           buffers)

    return impl


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def finalize_update(update, params, gain, gain_flags, lambda_sq, t_bin_arr,
                    f_map_arr, d_map_arr, corr_mode):

    set_identity = factories.set_identity_factory(corr_mode)

    if corr_mode.literal_value == 4:
        def impl(update, params, gain, gain_flags, lambda_sq, t_bin_arr,
                 f_map_arr, d_map_arr, corr_mode):

            update /= 2
            params += update

            n_time, n_freq, n_ant, n_dir, _ = gain.shape

            for t in range(n_time):
                for f in range(n_freq):
                    lsq = lambda_sq[f]
                    for a in range(n_ant):
                        for d in range(n_dir):

                            f_m = f_map_arr[1, f]
                            d_m = d_map_arr[d]
                            fl = gain_flags[t, f, a, d]

                            if fl == 1:
                                set_identity(gain[t, f, a, d])
                            else:
                                rm = params[t, f_m, a, d_m, 0]

                                beta = lsq*rm

                                cos_beta = np.cos(beta)
                                sin_beta = np.sin(beta)

                                gain[t, f, a, d, 0] = cos_beta
                                gain[t, f, a, d, 1] = -sin_beta
                                gain[t, f, a, d, 2] = sin_beta
                                gain[t, f, a, d, 3] = cos_beta
    else:
        raise ValueError("Rotation measure can only be solved for with four "
                         "correlation data.")

    return impl


def compute_jhwj_jhwr_elem_factory(corr_mode):

    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    a_kron_bt = factories.a_kron_bt_factory(corr_mode)
    unpack = factories.unpack_factory(corr_mode)

    if corr_mode.literal_value == 4:
        def impl(lop, rop, w, rm, lsq, gain, tmp_kprod, res, jhr, jhj):

            # Accumulate an element of jhwr.
            v1_imul_v2(res, rop, res)
            v1_imul_v2(lop, res, res)

            # Accumulate an element of jhwj.

            # WARNING: In this instance we are using the row-major
            # version of the kronecker product identity. This is because the
            # MS stores the correlations in row-major order (XX, XY, YX, YY),
            # whereas the standard maths assumes column-major ordering
            # (XX, YX, XY, YY). This subtle change means we can use the MS
            # data directly without worrying about swapping elements around.
            a_kron_bt(lop, rop, tmp_kprod)

            w_0, w_1, w_2, w_3 = unpack(w)  # NOTE: XX, XY, YX, YY
            r_0, r_1, r_2, r_3 = unpack(res)

            beta = lsq*rm

            sin_beta = np.sin(beta)
            cos_beta = np.cos(beta)

            dh_0 = -lsq*sin_beta
            dh_1 = -lsq*cos_beta
            dh_2 = lsq*cos_beta
            dh_3 = -lsq*sin_beta

            jh_0, jh_1, jh_2, jh_3 = unpack(tmp_kprod[:, 0])

            dhjh = dh_0*jh_0 + dh_1*jh_1 + dh_2*jh_2 + dh_3*jh_3

            jhj[0, 0] += (dhjh * w_0 * dhjh.conjugate()).real
            jhr[0] += (dh_0 * r_0).real

            jh_0, jh_1, jh_2, jh_3 = unpack(tmp_kprod[:, 1])

            dhjh = dh_0*jh_0 + dh_1*jh_1 + dh_2*jh_2 + dh_3*jh_3

            jhj[0, 0] += (dhjh * w_1 * dhjh.conjugate()).real
            jhr[0] += (dh_1 * r_1).real

            jh_0, jh_1, jh_2, jh_3 = unpack(tmp_kprod[:, 2])

            dhjh = dh_0*jh_0 + dh_1*jh_1 + dh_2*jh_2 + dh_3*jh_3

            jhj[0, 0] += (dhjh * w_2 * dhjh.conjugate()).real
            jhr[0] += (dh_2 * r_2).real

            jh_0, jh_1, jh_2, jh_3 = unpack(tmp_kprod[:, 3])

            dhjh = dh_0*jh_0 + dh_1*jh_1 + dh_2*jh_2 + dh_3*jh_3

            jhj[0, 0] += (dhjh * w_3 * dhjh.conjugate()).real
            jhr[0] += (dh_3 * r_3).real

    else:
        raise ValueError("Rotation measure can only be solved for with four "
                         "correlation data.")

    return factories.qcjit(impl)
