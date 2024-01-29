# -*- coding: utf-8 -*-
import numpy as np
from numba import prange, njit
from numba.extending import overload
from quartical.utils.numba import (coerce_literal,
                                   JIT_OPTIONS,
                                   PARALLEL_JIT_OPTIONS)
from quartical.gains.general.generics import (native_intermediaries,
                                              upsampled_itermediaries,
                                              per_array_jhj_jhr,
                                              resample_solints,
                                              downsample_jhj_jhr)
from quartical.gains.general.flagging import (flag_intermediaries,
                                              update_gain_flags,
                                              finalize_gain_flags,
                                              apply_gain_flags_to_flag_col,
                                              apply_gain_flags_to_gains)
from quartical.gains.general.convenience import (get_row,
                                                 get_extents)
import quartical.gains.general.factories as factories
from quartical.gains.general.inversion import (invert_factory,
                                               inversion_buffer_factory)


@njit(**JIT_OPTIONS)
def diag_complex_solver(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    return diag_complex_solver_impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        corr_mode
    )


def diag_complex_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    raise NotImplementedError


@overload(diag_complex_solver_impl, jit_options=JIT_OPTIONS)
def nb_diag_complex_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):

    coerce_literal(nb_diag_complex_solver_impl, ["corr_mode"])

    def impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        corr_mode
    ):

        gains = chain_inputs.gains
        gain_flags = chain_inputs.gain_flags

        active_term = meta_inputs.active_term
        max_iter = meta_inputs.iters
        solve_per = meta_inputs.solve_per
        dd_term = meta_inputs.dd_term
        n_thread = meta_inputs.threads

        active_gain = gains[active_term]
        active_gain_flags = gain_flags[active_term]

        # Set up some intemediaries used for flagging. TODO: Move?
        km1_gain = active_gain.copy()
        km1_abs2_diffs = np.zeros_like(active_gain_flags, dtype=np.float64)
        abs2_diffs_trend = np.zeros_like(active_gain_flags, dtype=np.float64)
        flag_imdry = \
            flag_intermediaries(km1_gain, km1_abs2_diffs, abs2_diffs_trend)

        # Set up some intemediaries used for solving.
        complex_dtype = active_gain.dtype
        gain_shape = active_gain.shape

        active_t_map_g = mapping_inputs.time_maps[active_term]
        active_f_map_g = mapping_inputs.freq_maps[active_term]

        # Create more work to do in paralllel when needed, else no-op.
        resampler = resample_solints(active_t_map_g, gain_shape, n_thread)

        # Determine the starts and stops of the rows and channels associated
        # with each solution interval.
        extents = get_extents(resampler.upsample_t_map, active_f_map_g)

        upsample_shape = resampler.upsample_shape
        upsampled_jhj = np.empty(upsample_shape, dtype=complex_dtype)
        upsampled_jhr = np.empty(upsample_shape, dtype=complex_dtype)
        jhj = upsampled_jhj[:gain_shape[0]]
        jhr = upsampled_jhr[:gain_shape[0]]
        update = np.zeros(gain_shape, dtype=complex_dtype)

        upsampled_imdry = upsampled_itermediaries(upsampled_jhj, upsampled_jhr)
        native_imdry = native_intermediaries(jhj, jhr, update)

        for loop_idx in range(max_iter or 1):

            compute_jhj_jhr(
                ms_inputs,
                mapping_inputs,
                chain_inputs,
                meta_inputs,
                upsampled_imdry,
                extents,
                corr_mode
            )

            if resampler.active:
                downsample_jhj_jhr(upsampled_imdry, resampler.downsample_t_map)

            if solve_per == "array":
                per_array_jhj_jhr(native_imdry)

            if not max_iter:  # Non-solvable term, we just want jhj.
                conv_perc = 0  # Didn't converge.
                loop_idx = -1  # Did zero iterations.
                break

            compute_update(native_imdry, corr_mode)

            finalize_update(
                chain_inputs,
                meta_inputs,
                native_imdry,
                loop_idx,
                corr_mode
            )

            # Check for gain convergence. Produced as a side effect of
            # flagging. The converged percentage is based on unflagged
            # intervals.
            conv_perc = update_gain_flags(
                chain_inputs,
                meta_inputs,
                flag_imdry,
                loop_idx,
                corr_mode
            )

            if conv_perc >= meta_inputs.stop_frac:
                break

        # NOTE: Removes soft flags and flags points which have bad trends.
        finalize_gain_flags(
            chain_inputs,
            meta_inputs,
            flag_imdry,
            corr_mode
        )

        reference_gains(chain_inputs, meta_inputs, corr_mode)

        # Call this one last time to ensure points flagged by finialize are
        # propagated (in the DI case).
        if not dd_term:
            apply_gain_flags_to_flag_col(
                ms_inputs,
                mapping_inputs,
                chain_inputs,
                meta_inputs
            )

        return native_imdry.jhj, loop_idx + 1, conv_perc

    return impl


def compute_jhj_jhr(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    upsampled_imdry,
    extents,
    corr_mode
):
    return NotImplementedError


@overload(compute_jhj_jhr, jit_options=PARALLEL_JIT_OPTIONS)
def nb_compute_jhj_jhr(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    upsampled_imdry,
    extents,
    corr_mode
):

    coerce_literal(nb_compute_jhj_jhr, ["corr_mode"])

    # We want to dispatch based on this field so we need its type.
    row_weights_idx = ms_inputs.fields.index('ROW_WEIGHTS')
    row_weights_type = ms_inputs[row_weights_idx]

    imul_rweight = factories.imul_rweight_factory(corr_mode, row_weights_type)
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
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        upsampled_imdry,
        extents,
        corr_mode
    ):

        active_term = meta_inputs.active_term

        data = ms_inputs.DATA
        model = ms_inputs.MODEL_DATA
        weights = ms_inputs.WEIGHT
        flags = ms_inputs.FLAG
        antenna1 = ms_inputs.ANTENNA1
        antenna2 = ms_inputs.ANTENNA2
        row_map = ms_inputs.ROW_MAP
        row_weights = ms_inputs.ROW_WEIGHTS

        time_maps = mapping_inputs.time_maps
        freq_maps = mapping_inputs.freq_maps
        dir_maps = mapping_inputs.dir_maps

        gains = chain_inputs.gains

        jhj = upsampled_imdry.jhj
        jhr = upsampled_imdry.jhr

        n_row, n_chan, n_dir, n_corr = model.shape

        jhj[:] = 0
        jhr[:] = 0

        n_tint, n_fint, n_ant, n_gdir, n_corr = jhr.shape
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

                    for d in range(n_dir):

                        set_identity(lop_pq)
                        set_identity(lop_qp)

                        # Construct a small contiguous gain array. This makes
                        # the single term case fractionally slower.
                        for gi in range(n_gains):
                            d_m = dir_maps[gi][d]  # Broadcast dir.
                            t_m = time_maps[gi][row_ind]
                            f_m = freq_maps[gi][f]

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

                        out_d = dir_maps[active_term][d]

                        iunpack(lop_pq_arr[out_d], lop_pq)
                        iadd(rop_pq_arr[out_d], rop_pq)

                        iunpack(lop_qp_arr[out_d], lop_qp)
                        iadd(rop_qp_arr[out_d], rop_qp)

                        v1ct_imul_v2(lop_pq, gains_p[active_term], v_pqd)
                        v1_imul_v2ct(v_pqd, rop_pq, v_pqd)
                        iadd(v_pq, v_pqd)

                    isub(r_pq, v_pq)

                    for d in range(n_gdir):

                        iunpack(wr_pq, r_pq)
                        imul(wr_pq, w)
                        iunpackct(wr_qp, wr_pq)

                        lop_pq_d = lop_pq_arr[d]
                        rop_pq_d = rop_pq_arr[d]

                        compute_jhwj_jhwr_elem(lop_pq_d,
                                               rop_pq_d,
                                               w,
                                               tmp_kprod,
                                               wr_pq,
                                               jhr_tifi[a1_m, d],
                                               jhj_tifi[a1_m, d])

                        lop_qp_d = lop_qp_arr[d]
                        rop_qp_d = rop_qp_arr[d]

                        compute_jhwj_jhwr_elem(lop_qp_d,
                                               rop_qp_d,
                                               w,
                                               tmp_kprod,
                                               wr_qp,
                                               jhr_tifi[a2_m, d],
                                               jhj_tifi[a2_m, d])
        return
    return impl


def compute_update(native_imdry, corr_mode):
    raise NotImplementedError


@overload(compute_update, jit_options=PARALLEL_JIT_OPTIONS)
def nb_compute_update(native_imdry, corr_mode):

    coerce_literal(nb_compute_update, ["corr_mode"])

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


def finalize_update(
    chain_inputs,
    meta_inputs,
    native_imdry,
    loop_idx,
    corr_mode
):
    raise NotImplementedError


@overload(finalize_update, jit_options=JIT_OPTIONS)
def nb_finalize_update(
    chain_inputs,
    meta_inputs,
    native_imdry,
    loop_idx,
    corr_mode
):

    coerce_literal(nb_finalize_update, ["corr_mode"])

    set_identity = factories.set_identity_factory(corr_mode)

    def impl(
        chain_inputs,
        meta_inputs,
        native_imdry,
        loop_idx,
        corr_mode
    ):

        dd_term = meta_inputs.dd_term
        active_term = meta_inputs.active_term
        pinned_directions = meta_inputs.pinned_directions

        gain = chain_inputs.gains[active_term]
        gain_flags = chain_inputs.gain_flags[active_term]

        update = native_imdry.update

        n_tint, n_fint, n_ant, n_dir, n_corr = gain.shape

        if dd_term:
            dir_loop = [d for d in range(n_dir) if d not in pinned_directions]
        else:
            dir_loop = [d for d in range(n_dir)]

        for ti in range(n_tint):
            for fi in range(n_fint):
                for a in range(n_ant):
                    for d in dir_loop:

                        g = gain[ti, fi, a, d]
                        fl = gain_flags[ti, fi, a, d]
                        upd = update[ti, fi, a, d]

                        if fl == 1:
                            set_identity(g)
                        elif dd_term or (loop_idx % 2 == 0):
                            upd /= 2
                            g += upd
                        else:
                            g += upd

    return impl


def compute_jhwj_jhwr_elem_factory(corr_mode):

    iadd = factories.iadd_factory(corr_mode)
    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    a_kron_bt = factories.a_kron_bt_factory(corr_mode)
    unpack = factories.unpack_factory(corr_mode)
    unpackc = factories.unpackc_factory(corr_mode)

    if corr_mode.literal_value == 4:
        def impl(lop, rop, w, tmp_kprod, res, jhr, jhj):

            # Effectively apply zero weight to off-diagonal terms.
            # TODO: Can be tidied but requires moving other weighting code.
            res[1] = 0
            res[2] = 0

            # Accumulate an element of jhwr.
            v1_imul_v2(lop, res, res)
            v1_imul_v2(res, rop, res)

            # Accumulate an element of jhwj.

            # WARNING: In this instance we are using the row-major
            # version of the kronecker product identity. This is because the
            # MS stores the correlations in row-major order (XX, XY, YX, YY),
            # whereas the standard maths assumes column-major ordering
            # (XX, YX, XY, YY). This subtle change means we can use the MS
            # data directly without worrying about swapping elements around.
            a_kron_bt(lop, rop, tmp_kprod)

            w_0, w_1, w_2, w_3 = unpack(w)  # NOTE: XX, XY, YX, YY
            w_1 = 0  # Effectively ignore the off-diagonal contributions.
            w_2 = 0  # Effectively ignore the off-diagonal contributions.
            r_0, _, _, r_3 = unpack(res)  # NOTE: XX, XY, YX, YY

            jhr[0] += r_0
            jhr[3] += r_3

            jh_0, jh_1, jh_2, jh_3 = unpack(tmp_kprod[0])
            j_0, j_1, j_2, j_3 = unpackc(tmp_kprod[0])

            jhwj_00 = jh_0*w_0*j_0 + jh_1*w_1*j_1 + jh_2*w_2*j_2 + jh_3*w_3*j_3

            j_0, j_1, j_2, j_3 = unpackc(tmp_kprod[3])

            jhwj_03 = jh_0*w_0*j_0 + jh_1*w_1*j_1 + jh_2*w_2*j_2 + jh_3*w_3*j_3

            jh_0, jh_1, jh_2, jh_3 = unpack(tmp_kprod[3])
            jhwj_33 = jh_0*w_0*j_0 + jh_1*w_1*j_1 + jh_2*w_2*j_2 + jh_3*w_3*j_3

            jhj[0] += jhwj_00
            jhj[1] += jhwj_03
            jhj[2] += jhwj_03.conjugate()
            jhj[3] += jhwj_33
    elif corr_mode.literal_value == 2:
        def impl(lop, rop, w, tmp_kprod, res, jhr, jhj):

            # Accumulate an element of jhwr.
            v1_imul_v2(res, rop, res)
            iadd(jhr, res)

            # Accumulate an element of jhwj.
            jh_00, jh_11 = unpack(rop)
            j_00, j_11 = unpackc(rop)
            w_00, w_11 = unpack(w)

            jhj[0] += jh_00*w_00*j_00
            jhj[1] += jh_11*w_11*j_11
    elif corr_mode.literal_value == 1:
        def impl(lop, rop, w, tmp_kprod, res, jhr, jhj):

            # Accumulate an element of jhwr.
            v1_imul_v2(res, rop, res)
            iadd(jhr, res)

            # Accumulate an element of jhwj.
            jh_00 = unpack(rop)
            j_00 = unpackc(rop)
            w_00 = unpack(w)

            jhj[0] += jh_00*w_00*j_00
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


@njit(**JIT_OPTIONS)
def reference_gains(chain_inputs, meta_inputs, mode):
    return reference_gains_impl(chain_inputs, meta_inputs, mode)


def reference_gains_impl(chain_inputs, meta_inputs, mode):
    raise NotImplementedError


@overload(reference_gains_impl, jit_options=JIT_OPTIONS)
def nb_reference_gains_impl(chain_inputs, meta_inputs, mode):

    coerce_literal(nb_reference_gains_impl, ["mode"])
    v1_imul_v2 = factories.v1_imul_v2_factory(mode)

    def impl(chain_inputs, meta_inputs, mode):

        active_term = meta_inputs.active_term
        ref_ant = meta_inputs.reference_antenna

        gains = chain_inputs.gains[active_term]
        gain_flags = chain_inputs.gain_flags[active_term]

        n_ti, n_fi, n_ant, n_dir, n_corr = gains.shape

        ref_gains = gains[:, :, ref_ant: ref_ant + 1, :, :].copy()

        for t in range(n_ti):
            for f in range(n_fi):
                for d in range(n_dir):

                    if gain_flags[t, f, ref_ant, d]:  # TODO: Flagged refant?
                        continue
                    elif n_corr in (1, 2):
                        rg = ref_gains[t, f, 0, d]
                        rg[...] = rg.conjugate()/np.abs(rg)
                    else:
                        rg = ref_gains[t, f, 0, d]
                        rg[1:3] = 0
                        rg[::3] = rg[::3].conjugate()/np.abs(rg[::3])

        for t in range(n_ti):
            for f in range(n_fi):
                for a in range(n_ant):
                    for d in range(n_dir):

                        g = gains[t, f, a, d]
                        rg = ref_gains[t, f, 0, d]

                        v1_imul_v2(g, rg, g)

        apply_gain_flags_to_gains(gain_flags, gains)

    return impl
