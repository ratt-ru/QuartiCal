# -*- coding: utf-8 -*-
import numpy as np
from numba import prange, generated_jit
from quartical.utils.numba import coerce_literal
from quartical.gains.general.generics import (native_intermediaries,
                                              upsampled_itermediaries,
                                              per_array_jhj_jhr,
                                              resample_solints,
                                              downsample_jhj_jhr)
from quartical.gains.general.flagging import (flag_intermediaries,
                                              update_gain_flags,
                                              finalize_gain_flags,
                                              apply_gain_flags,
                                              update_param_flags)
from quartical.gains.general.convenience import (get_row,
                                                 get_extents)
import quartical.gains.general.factories as factories
from quartical.gains.general.inversion import (invert_factory,
                                               inversion_buffer_factory)
from collections import namedtuple


# This can be done without a named tuple now. TODO: Add unpacking to
# constructor.
stat_fields = {"conv_iters": np.int64,
               "conv_perc": np.float64}

term_conv_info = namedtuple("term_conv_info", " ".join(stat_fields.keys()))

rotation_args = namedtuple(
    "rotation_args",
    (
        "params",
        "param_flags",
        "t_bin_arr"
    )
)


def get_identity_params(corr_mode):

    if corr_mode.literal_value == 4:
        return np.zeros((1,), dtype=np.float64)
    else:
        raise ValueError("Unsupported number of correlations.")


@generated_jit(nopython=True,
               fastmath=True,
               parallel=False,
               cache=True,
               nogil=True)
def rotation_solver(base_args, term_args, meta_args, corr_mode):

    coerce_literal(rotation_solver, ["corr_mode"])

    identity_params = get_identity_params(corr_mode)

    def impl(base_args, term_args, meta_args, corr_mode):

        gains = base_args.gains
        gain_flags = base_args.gain_flags

        active_term = meta_args.active_term
        max_iter = meta_args.iters
        solve_per = meta_args.solve_per
        dd_term = meta_args.dd_term
        n_thread = meta_args.threads

        active_gain = gains[active_term]
        active_gain_flags = gain_flags[active_term]
        active_params = term_args.params[active_term]

        # Set up some intemediaries used for flagging. TODO: Move?
        km1_gain = active_gain.copy()
        km1_abs2_diffs = np.zeros_like(active_gain_flags, dtype=np.float64)
        abs2_diffs_trend = np.zeros_like(active_gain_flags, dtype=np.float64)
        flag_imdry = \
            flag_intermediaries(km1_gain, km1_abs2_diffs, abs2_diffs_trend)

        # Set up some intemediaries used for solving.
        real_dtype = active_gain.real.dtype
        param_shape = active_params.shape

        active_t_map_g = base_args.t_map_arr[0, :, active_term]
        active_f_map_p = base_args.f_map_arr[1, :, active_term]

        # Create more work to do in paralllel when needed, else no-op.
        resampler = resample_solints(active_t_map_g, param_shape, n_thread)

        # Determine the starts and stops of the rows and channels associated
        # with each solution interval.
        extents = get_extents(resampler.upsample_t_map, active_f_map_p)

        upsample_shape = resampler.upsample_shape
        upsampled_jhj = np.empty(upsample_shape + (upsample_shape[-1],),
                                 dtype=real_dtype)
        upsampled_jhr = np.empty(upsample_shape, dtype=real_dtype)
        jhj = upsampled_jhj[:param_shape[0]]
        jhr = upsampled_jhr[:param_shape[0]]
        update = np.zeros(param_shape, dtype=real_dtype)

        upsampled_imdry = upsampled_itermediaries(upsampled_jhj, upsampled_jhr)
        native_imdry = native_intermediaries(jhj, jhr, update)

        for loop_idx in range(max_iter):

            compute_jhj_jhr(base_args,
                            term_args,
                            meta_args,
                            upsampled_imdry,
                            extents,
                            corr_mode)

            if resampler.active:
                downsample_jhj_jhr(upsampled_imdry, resampler.downsample_t_map)

            if solve_per == "array":
                per_array_jhj_jhr(native_imdry)

            compute_update(native_imdry,
                           corr_mode)

            finalize_update(base_args,
                            term_args,
                            meta_args,
                            native_imdry,
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
                                          corr_mode,
                                          numbness=1e9)

            # Propagate gain flags to parameter flags.
            update_param_flags(base_args,
                               term_args,
                               meta_args,
                               identity_params)

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
def compute_jhj_jhr(
    base_args,
    term_args,
    meta_args,
    upsampled_imdry,
    extents,
    corr_mode
):

    # We want to dispatch based on this field so we need its type.
    row_weights = base_args[base_args.fields.index('row_weights')]

    imul_rweight = factories.imul_rweight_factory(corr_mode, row_weights)
    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    v1_imul_v2ct = factories.v1_imul_v2ct_factory(corr_mode)
    v1ct_imul_v2 = factories.v1ct_imul_v2_factory(corr_mode)
    iunpack = factories.iunpack_factory(corr_mode)
    iunpackct = factories.iunpackct_factory(corr_mode)
    imul = factories.imul_factory(corr_mode)
    iadd = factories.iadd_factory(corr_mode)
    isub = factories.isub_factory(corr_mode)
    valloc = factories.valloc_factory(corr_mode)
    make_loop_vars = factories.loop_var_factory(corr_mode)
    set_identity = factories.set_identity_factory(corr_mode)
    compute_jhwj_jhwr_elem = compute_jhwj_jhwr_elem_factory(corr_mode)

    def impl(
        base_args,
        term_args,
        meta_args,
        upsampled_imdry,
        extents,
        corr_mode
    ):

        active_term = meta_args.active_term

        data = base_args.data
        model = base_args.model
        weights = base_args.weights
        flags = base_args.flags
        antenna1 = base_args.a1
        antenna2 = base_args.a2
        row_map = base_args.row_map
        row_weights = base_args.row_weights

        gains = base_args.gains
        params = term_args.params[active_term]
        t_map_arr = base_args.t_map_arr[0]  # We only need the gain mappings.
        f_map_arr_g = base_args.f_map_arr[0]
        f_map_arr_p = base_args.f_map_arr[1]
        d_map_arr = base_args.d_map_arr

        jhj = upsampled_imdry.jhj
        jhr = upsampled_imdry.jhr

        _, n_chan, n_dir, n_corr = model.shape

        jhj[:] = 0
        jhr[:] = 0

        n_tint, n_fint, n_ant, n_gdir, n_param = jhr.shape
        n_int = n_tint*n_fint

        complex_dtype = gains[active_term].dtype
        weight_dtype = weights.dtype

        n_gains = len(gains)

        row_starts = extents.row_starts
        row_stops = extents.row_stops
        chan_starts = extents.chan_starts
        chan_stops = extents.chan_stops

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

            w = valloc(weight_dtype)
            r_pq = valloc(complex_dtype)
            wr_pq = valloc(complex_dtype)
            wr_qp = valloc(complex_dtype)
            v_pqd = valloc(complex_dtype)
            v_pq = valloc(complex_dtype)

            gains_p = valloc(complex_dtype, leading_dims=(n_gains,))
            gains_q = valloc(complex_dtype, leading_dims=(n_gains,))

            lop_pq_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
            rop_pq_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
            lop_qp_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
            rop_qp_arr = valloc(complex_dtype, leading_dims=(n_gdir,))

            tmp_kprod = np.zeros((4, 4), dtype=complex_dtype)
            jhr_tifi = jhr[ti, fi]
            jhj_tifi = jhj[ti, fi]

            for row_ind in range(rs, re):

                row = get_row(row_ind, row_map)
                a1_m, a2_m = antenna1[row], antenna2[row]

                rm_t = t_map_arr[row_ind, active_term]

                for f in range(fs, fe):

                    if flags[row, f]:  # Skip flagged data points.
                        continue

                    # Apply row weights in the BDA case, otherwise a no-op.
                    imul_rweight(weights[row, f], w, row_weights, row_ind)
                    iunpack(r_pq, data[row, f])

                    lop_pq_arr[:] = 0
                    rop_pq_arr[:] = 0
                    lop_qp_arr[:] = 0
                    rop_qp_arr[:] = 0
                    v_pq[:] = 0

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
                        iunpack(rop_qp, m)
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

                        v1ct_imul_v2(lop_pq, gains_p[active_term], v_pqd)
                        v1_imul_v2ct(v_pqd, rop_pq, v_pqd)
                        iadd(v_pq, v_pqd)

                    isub(r_pq, v_pq)

                    for d in range(n_gdir):

                        theta_p = params[rm_t, rm_f, a1_m, d, 0]
                        theta_q = params[rm_t, rm_f, a2_m, d, 0]

                        iunpack(wr_pq, r_pq)
                        imul(wr_pq, w)
                        iunpackct(wr_qp, wr_pq)

                        lop_pq_d = lop_pq_arr[d]
                        rop_pq_d = rop_pq_arr[d]

                        compute_jhwj_jhwr_elem(lop_pq_d,
                                               rop_pq_d,
                                               w,
                                               theta_p,
                                               gains_p[active_term],
                                               tmp_kprod,
                                               wr_pq,
                                               jhr_tifi[a1_m, d],
                                               jhj_tifi[a1_m, d])

                        lop_qp_d = lop_qp_arr[d]
                        rop_qp_d = rop_qp_arr[d]

                        compute_jhwj_jhwr_elem(lop_qp_d,
                                               rop_qp_d,
                                               w,
                                               theta_q,
                                               gains_q[active_term],
                                               tmp_kprod,
                                               wr_qp,
                                               jhr_tifi[a2_m, d],
                                               jhj_tifi[a2_m, d])
        return
    return impl


@generated_jit(nopython=True,
               fastmath=True,
               parallel=True,
               cache=True,
               nogil=True)
def compute_update(native_imdry, corr_mode):

    # We want to dispatch based on this field so we need its type.
    jhj = native_imdry[native_imdry.fields.index('jhj')]

    generalised = jhj.ndim == 6
    inversion_buffer = inversion_buffer_factory(generalised=generalised)
    invert = invert_factory(corr_mode, generalised=generalised)

    def impl(native_imdry, corr_mode):

        jhj = native_imdry.jhj
        jhr = native_imdry.jhr
        update = native_imdry.update

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
def finalize_update(base_args, term_args, meta_args, native_imdry, loop_idx,
                    corr_mode):

    set_identity = factories.set_identity_factory(corr_mode)

    if corr_mode.literal_value == 4:
        def impl(base_args, term_args, meta_args, native_imdry, loop_idx,
                 corr_mode):

            active_term = meta_args.active_term

            gain = base_args.gains[active_term]
            gain_flags = base_args.gain_flags[active_term]
            d_map_arr = base_args.d_map_arr[active_term, :]

            params = term_args.params[active_term]

            update = native_imdry.update

            update /= 2
            params += update

            n_time, n_freq, n_ant, n_dir, _ = gain.shape

            for t in range(n_time):
                for f in range(n_freq):
                    for a in range(n_ant):
                        for d in range(n_dir):

                            d_m = d_map_arr[d]
                            fl = gain_flags[t, f, a, d]

                            if fl == 1:
                                set_identity(gain[t, f, a, d])
                            else:
                                theta = params[t, f, a, d_m, 0]

                                cos_theta = np.cos(theta)
                                sin_theta = np.sin(theta)

                                gain[t, f, a, d, 0] = cos_theta
                                gain[t, f, a, d, 1] = -sin_theta
                                gain[t, f, a, d, 2] = sin_theta
                                gain[t, f, a, d, 3] = cos_theta
    else:
        raise ValueError("Rotation can only be solved for with four "
                         "correlation data.")

    return impl


def compute_jhwj_jhwr_elem_factory(corr_mode):

    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    a_kron_bt = factories.a_kron_bt_factory(corr_mode)
    unpack = factories.unpack_factory(corr_mode)

    if corr_mode.literal_value == 4:
        def impl(lop, rop, w, theta, gain, tmp_kprod, res, jhr, jhj):

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

            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            dh_0 = -sin_theta
            dh_1 = -cos_theta
            dh_2 = cos_theta
            dh_3 = -sin_theta

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
        raise ValueError("Rotation can only be solved for with four "
                         "correlation data.")

    return factories.qcjit(impl)