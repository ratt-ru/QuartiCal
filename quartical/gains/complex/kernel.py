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
                                              apply_gain_flags_to_flag_col)
from quartical.gains.general.convenience import (get_row,
                                                 get_extents)
import quartical.gains.general.factories as factories
from quartical.gains.general.inversion import (invert_factory,
                                               inversion_buffer_factory)


@njit(**JIT_OPTIONS)
def complex_solver(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    return complex_solver_impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        corr_mode
    )


def complex_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    raise NotImplementedError


@overload(complex_solver_impl, jit_options=JIT_OPTIONS)
def nb_complex_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):

    coerce_literal(nb_complex_solver_impl, ["corr_mode"])

    get_jhj_dims = get_jhj_dims_factory(corr_mode)

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
        scalar = meta_inputs.scalar
        dd_term = meta_inputs.dd_term
        n_thread = meta_inputs.threads

        active_gain = gains[active_term]
        active_gain_flags = gain_flags[active_term]

        # Set up some intemediaries used for flagging. TODO: Move?
        km1_gain = active_gain.copy()
        km1_abs2_diffs = np.zeros_like(active_gain_flags, dtype=np.float64)
        abs2_diffs_trend = np.zeros_like(active_gain_flags, dtype=np.float64)
        flag_imdry = flag_intermediaries(
            km1_gain, km1_abs2_diffs, abs2_diffs_trend
        )

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
        upsampled_jhj = np.empty(get_jhj_dims(upsample_shape),
                                 dtype=complex_dtype)
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

            if scalar:
                raise ValueError("Scalar mode not supported for complex terms.")

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

    # NOTE: The per-visibility work below is deliberately expressed using the
    # tuple_* factories - intermediary values live in tuples (registers)
    # rather than small array buffers (memory). See factories.py.
    tuple_unpack = factories.tuple_unpack_factory(corr_mode)
    tuple_unpackct = factories.tuple_unpackct_factory(corr_mode)
    tuple_unpack_rweight = factories.tuple_unpack_rweight_factory(
        corr_mode, row_weights_type
    )
    tuple_zeros = factories.tuple_zeros_factory(corr_mode)
    tuple_identity = factories.tuple_identity_factory(corr_mode)
    tuple_add = factories.tuple_add_factory(corr_mode)
    tuple_sub = factories.tuple_sub_factory(corr_mode)
    tuple_wmul = factories.tuple_wmul_factory(corr_mode)
    tuple_v1_mul_v2 = factories.tuple_v1_mul_v2_factory(corr_mode)
    tuple_v1_mul_v2ct = factories.tuple_v1_mul_v2ct_factory(corr_mode)
    tuple_v1ct_mul_v2 = factories.tuple_v1ct_mul_v2_factory(corr_mode)
    iunpack = factories.iunpack_factory(corr_mode)
    iadd = factories.iadd_factory(corr_mode)
    valloc = factories.valloc_factory(corr_mode)
    make_loop_vars = factories.loop_var_factory(corr_mode)
    compute_jhwj_jhwr_elem = compute_jhwj_jhwr_elem_factory(corr_mode)
    flush_jhwj_jhwr = flush_jhwj_jhwr_factory(corr_mode)
    jhwj_jhwr_zeros = jhwj_jhwr_zeros_factory(corr_mode)
    mirror_jhj = mirror_jhj_factory(corr_mode)

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

        # In the (very common) single direction case the per-direction
        # accumulators are unnecessary - the loop-invariant flag below lets
        # the compiler produce a fast path which skips them entirely.
        single_dir = (n_dir == 1) and (n_gdir == 1)

        complex_dtype = gains[active_term].dtype

        n_gains = len(gains)

        row_starts = extents.row_starts
        row_stops = extents.row_stops
        chan_starts = extents.chan_starts
        chan_stops = extents.chan_stops

        # Determine loop variables based on where we are in the chain.
        # gt means greater than (n>j) and lt means less than (n<j).
        all_terms, gt_active, lt_active = make_loop_vars(n_gains, active_term)

        active_t_map = time_maps[active_term]
        active_f_map = freq_maps[active_term]
        active_d_map = dir_maps[active_term]

        # Parallel over all solution intervals.
        for i in prange(n_int):

            ti = i//n_fint
            fi = i - ti*n_fint

            rs = row_starts[ti]
            re = row_stops[ti]
            fs = chan_starts[fi]
            fe = chan_stops[fi]

            # Per-direction accumulators - these are the only per-visibility
            # intermediaries which need to live in memory (a direction loop
            # may accumulate several model directions into one gain
            # direction). Everything else is a tuple i.e. register-resident.
            lop_pq_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
            rop_pq_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
            lop_qp_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
            rop_qp_arr = valloc(complex_dtype, leading_dims=(n_gdir,))

            jhr_tifi = jhr[ti, fi]
            jhj_tifi = jhj[ti, fi]

            # Zero/identity tuples in the compute (gain) dtype. Adding the
            # zero tuple is also used to promote lower precision inputs.
            zero_vec = tuple_zeros(jhr_tifi[0, 0])
            identity_vec = tuple_identity(jhr_tifi[0, 0])
            acc_zero = jhwj_jhwr_zeros(jhr_tifi[0, 0])

            for row_ind in range(rs, re):

                row = get_row(row_ind, row_map)
                a1_m, a2_m = antenna1[row], antenna2[row]

                if single_dir:

                    # Fast path: a single direction accumulates into a single
                    # jhr/jhj element per antenna. As the antennas are fixed
                    # for the duration of a row, the accumulation can be done
                    # in registers and flushed to memory once per row.
                    acc_p = acc_zero
                    acc_q = acc_zero

                    for f in range(fs, fe):

                        if flags[row, f]:  # Skip flagged data points.
                            continue

                        # Apply row weights in the BDA case, else a no-op.
                        w = tuple_unpack_rweight(
                            weights[row, f], row_weights, row_ind
                        )
                        r_pq = tuple_unpack(data[row, f])

                        # NOTE: This block is duplicated in the general path
                        # below - a shared helper returning the four operator
                        # tuples trips a numba parfor array analysis bug
                        # (nested tuple returns are misread as array shapes).
                        lop_pq = identity_vec
                        lop_qp = identity_vec

                        # Promote the model to the compute dtype - the zero
                        # add is free but ensures type stability below.
                        rop_qp = tuple_add(model[row, f, 0], zero_vec)
                        rop_pq = tuple_unpackct(rop_qp)

                        for gi in all_terms:

                            d_m = dir_maps[gi][0]  # Broadcast dir.
                            t_m = time_maps[gi][row_ind]
                            f_m = freq_maps[gi][f]

                            gain = gains[gi][t_m, f_m]

                            g_q = tuple_unpack(gain[a2_m, d_m])
                            rop_pq = tuple_v1_mul_v2(g_q, rop_pq)

                            g_p = tuple_unpack(gain[a1_m, d_m])
                            rop_qp = tuple_v1_mul_v2(g_p, rop_qp)

                        for gi in gt_active:

                            d_m = dir_maps[gi][0]
                            t_m = time_maps[gi][row_ind]
                            f_m = freq_maps[gi][f]

                            gain = gains[gi][t_m, f_m]

                            g_p = tuple_unpack(gain[a1_m, d_m])
                            rop_pq = tuple_v1_mul_v2ct(rop_pq, g_p)

                            g_q = tuple_unpack(gain[a2_m, d_m])
                            rop_qp = tuple_v1_mul_v2ct(rop_qp, g_q)

                        for gi in lt_active:

                            d_m = dir_maps[gi][0]
                            t_m = time_maps[gi][row_ind]
                            f_m = freq_maps[gi][f]

                            gain = gains[gi][t_m, f_m]

                            g_p = tuple_unpack(gain[a1_m, d_m])
                            lop_pq = tuple_v1ct_mul_v2(g_p, lop_pq)

                            g_q = tuple_unpack(gain[a2_m, d_m])
                            lop_qp = tuple_v1ct_mul_v2(g_q, lop_qp)

                        g_active = tuple_unpack(
                            gains[active_term][
                                active_t_map[row_ind], active_f_map[f]
                            ][a1_m, 0]
                        )
                        v_pq = tuple_v1ct_mul_v2(lop_pq, g_active)
                        v_pq = tuple_v1_mul_v2ct(v_pq, rop_pq)

                        r_pq = tuple_sub(r_pq, v_pq)

                        wr_pq = tuple_wmul(r_pq, w)
                        wr_qp = tuple_unpackct(wr_pq)

                        acc_p = compute_jhwj_jhwr_elem(
                            lop_pq, rop_pq, w, wr_pq, acc_p
                        )
                        acc_q = compute_jhwj_jhwr_elem(
                            lop_qp, rop_qp, w, wr_qp, acc_q
                        )

                    flush_jhwj_jhwr(
                        jhr_tifi[a1_m, 0], jhj_tifi[a1_m, 0], acc_p
                    )
                    flush_jhwj_jhwr(
                        jhr_tifi[a2_m, 0], jhj_tifi[a2_m, 0], acc_q
                    )

                    continue

                # General path: multiple directions require per-direction
                # accumulation through memory.
                for f in range(fs, fe):

                    if flags[row, f]:  # Skip flagged data points.
                        continue

                    # Apply row weights in the BDA case, otherwise a no-op.
                    w = tuple_unpack_rweight(
                        weights[row, f], row_weights, row_ind
                    )
                    r_pq = tuple_unpack(data[row, f])

                    lop_pq_arr[:] = 0
                    rop_pq_arr[:] = 0
                    lop_qp_arr[:] = 0
                    rop_qp_arr[:] = 0
                    v_pq = zero_vec

                    for d in range(n_dir):

                        lop_pq = identity_vec
                        lop_qp = identity_vec

                        # Promote the model to the compute dtype - the zero
                        # add is free but ensures type stability below.
                        rop_qp = tuple_add(model[row, f, d], zero_vec)
                        rop_pq = tuple_unpackct(rop_qp)

                        for gi in all_terms:

                            d_m = dir_maps[gi][d]  # Broadcast dir.
                            t_m = time_maps[gi][row_ind]
                            f_m = freq_maps[gi][f]

                            gain = gains[gi][t_m, f_m]

                            g_q = tuple_unpack(gain[a2_m, d_m])
                            rop_pq = tuple_v1_mul_v2(g_q, rop_pq)

                            g_p = tuple_unpack(gain[a1_m, d_m])
                            rop_qp = tuple_v1_mul_v2(g_p, rop_qp)

                        for gi in gt_active:

                            d_m = dir_maps[gi][d]
                            t_m = time_maps[gi][row_ind]
                            f_m = freq_maps[gi][f]

                            gain = gains[gi][t_m, f_m]

                            g_p = tuple_unpack(gain[a1_m, d_m])
                            rop_pq = tuple_v1_mul_v2ct(rop_pq, g_p)

                            g_q = tuple_unpack(gain[a2_m, d_m])
                            rop_qp = tuple_v1_mul_v2ct(rop_qp, g_q)

                        for gi in lt_active:

                            d_m = dir_maps[gi][d]
                            t_m = time_maps[gi][row_ind]
                            f_m = freq_maps[gi][f]

                            gain = gains[gi][t_m, f_m]

                            g_p = tuple_unpack(gain[a1_m, d_m])
                            lop_pq = tuple_v1ct_mul_v2(g_p, lop_pq)

                            g_q = tuple_unpack(gain[a2_m, d_m])
                            lop_qp = tuple_v1ct_mul_v2(g_q, lop_qp)

                        out_d = active_d_map[d]

                        iunpack(lop_pq_arr[out_d], lop_pq)
                        iadd(rop_pq_arr[out_d], rop_pq)

                        iunpack(lop_qp_arr[out_d], lop_qp)
                        iadd(rop_qp_arr[out_d], rop_qp)

                        g_active = tuple_unpack(
                            gains[active_term][
                                active_t_map[row_ind], active_f_map[f]
                            ][a1_m, out_d]
                        )
                        v_pqd = tuple_v1ct_mul_v2(lop_pq, g_active)
                        v_pqd = tuple_v1_mul_v2ct(v_pqd, rop_pq)
                        v_pq = tuple_add(v_pq, v_pqd)

                    r_pq = tuple_sub(r_pq, v_pq)

                    # Weighted residual and its conjugate transpose. These
                    # are direction independent and can be computed once.
                    wr_pq = tuple_wmul(r_pq, w)
                    wr_qp = tuple_unpackct(wr_pq)

                    for d in range(n_gdir):

                        lop_pq_d = tuple_unpack(lop_pq_arr[d])
                        rop_pq_d = tuple_unpack(rop_pq_arr[d])

                        acc = compute_jhwj_jhwr_elem(
                            lop_pq_d, rop_pq_d, w, wr_pq, acc_zero
                        )
                        flush_jhwj_jhwr(
                            jhr_tifi[a1_m, d], jhj_tifi[a1_m, d], acc
                        )

                        lop_qp_d = tuple_unpack(lop_qp_arr[d])
                        rop_qp_d = tuple_unpack(rop_qp_arr[d])

                        acc = compute_jhwj_jhwr_elem(
                            lop_qp_d, rop_qp_d, w, wr_qp, acc_zero
                        )
                        flush_jhwj_jhwr(
                            jhr_tifi[a2_m, d], jhj_tifi[a2_m, d], acc
                        )

            # Accumulation only touches the upper triangle of each jhj
            # element (4 correlation case) - fill in the lower triangle.
            mirror_jhj(jhj_tifi)
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


def jhwj_jhwr_zeros_factory(corr_mode):
    """Produce the zero jhwr/jhwj accumulator tuple for a given corr mode.

    The accumulator is a single flat tuple holding the jhwr element followed
    by the (upper triangle, row-major, in the 4 correlation case) jhwj
    element. A flat tuple is used deliberately - returning nested tuples
    from inlined functions inside a prange trips a numba parfor array
    analysis bug. In the diagonal cases the jhwj entries are real, as
    w|rop|^2 has no imaginary part.
    """

    if corr_mode.literal_value == 4:
        def impl(invec):
            z = invec[0]*0
            return z, z, z, z, z, z, z, z, z, z, z, z, z, z
    elif corr_mode.literal_value == 2:
        def impl(invec):
            z = invec[0]*0
            zr = invec[0].real*0
            return z, z, zr, zr
    elif corr_mode.literal_value == 1:
        def impl(invec):
            return invec[0]*0, invec[0].real*0
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def flush_jhwj_jhwr_factory(corr_mode):
    """Add a register-accumulated jhwr/jhwj accumulator into the arrays.

    In the 4 correlation case the jhj part of the accumulator holds the
    upper triangle of the (4, 4) jhj element in row-major order - the lower
    triangle is filled in by mirror_jhj once per solution interval.
    """

    if corr_mode.literal_value == 4:
        def impl(jhr, jhj, acc):

            jhr[0] += acc[0]
            jhr[1] += acc[1]
            jhr[2] += acc[2]
            jhr[3] += acc[3]

            jhj[0, 0] += acc[4]
            jhj[0, 1] += acc[5]
            jhj[0, 2] += acc[6]
            jhj[0, 3] += acc[7]
            jhj[1, 1] += acc[8]
            jhj[1, 2] += acc[9]
            jhj[1, 3] += acc[10]
            jhj[2, 2] += acc[11]
            jhj[2, 3] += acc[12]
            jhj[3, 3] += acc[13]
    elif corr_mode.literal_value == 2:
        def impl(jhr, jhj, acc):

            jhr[0] += acc[0]
            jhr[1] += acc[1]

            jhj[0] += acc[2]
            jhj[1] += acc[3]
    elif corr_mode.literal_value == 1:
        def impl(jhr, jhj, acc):

            jhr[0] += acc[0]

            jhj[0] += acc[1]
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def compute_jhwj_jhwr_elem_factory(corr_mode):
    """Accumulate a jhwr/jhwj element into a register-resident accumulator.

    All inputs and the returned accumulator are tuples (register-resident
    values) - the accumulator is only flushed to memory by flush_jhwj_jhwr.
    The accumulator is a single flat tuple (jhwr followed by jhwj - see
    jhwj_jhwr_zeros_factory). In the 4 correlation case only the upper
    triangle of the (4, 4) jhj element is accumulated (in row-major order) -
    the lower triangle is filled in once per solution interval by
    mirror_jhj.
    """

    tuple_v1_mul_v2 = factories.tuple_v1_mul_v2_factory(corr_mode)

    if corr_mode.literal_value == 4:
        def impl(lop, rop, w, res, acc):

            # Accumulate an element of jhwr.
            upd = tuple_v1_mul_v2(lop, res)
            upd = tuple_v1_mul_v2(upd, rop)

            # Accumulate an element of jhwj.

            # WARNING: In this instance we are using the row-major
            # version of the kronecker product identity. This is because the
            # MS stores the correlations in row-major order (XX, XY, YX, YY),
            # whereas the standard maths assumes column-major ordering
            # (XX, YX, XY, YY). This subtle change means we can use the MS
            # data directly without worrying about swapping elements around.
            #
            # Row (2i + j), column (2p + q) of J^H is a_ip*b_jq, so an
            # element of J^HWJ factorises as
            #   jhj[2i+j, 2k+l] = sum_pq w_pq (a_ip b_jq) conj(a_kp b_lq)
            #                   = sum_p a_ip conj(a_kp) C_jl^p,
            #   C_jl^p = w_p0 B_jl^0 + w_p1 B_jl^1,  B_jl^q = b_jq conj(b_lq).
            # Exploiting this (and hermitian symmetry, i.e. only computing
            # the upper triangle - see mirror_jhj) roughly halves the number
            # of multiplications relative to forming the kronecker product
            # rows explicitly.
            a00, a01, a10, a11 = lop[0], lop[1], lop[2], lop[3]
            b00, b10, b01, b11 = rop[0], rop[1], rop[2], rop[3]  # Transpose.

            w_0, w_1, w_2, w_3 = w[0], w[1], w[2], w[3]  # NOTE: XX XY YX YY

            # A_ik^p = a_ip conj(a_kp); A_10^p = conj(A_01^p) is not needed
            # as only the upper triangle is accumulated. The diagonal (in ik)
            # entries are |a_ip|^2 i.e. real - keeping them as real scalars
            # (likewise for B and C below) roughly halves the flops relative
            # to treating every factor as complex.
            a00_0 = a00.real*a00.real + a00.imag*a00.imag
            a00_1 = a01.real*a01.real + a01.imag*a01.imag
            a01_0, a01_1 = a00*a10.conjugate(), a01*a11.conjugate()
            a11_0 = a10.real*a10.real + a10.imag*a10.imag
            a11_1 = a11.real*a11.real + a11.imag*a11.imag

            # B_jl^q = b_jq conj(b_lq).
            b00_0 = b00.real*b00.real + b00.imag*b00.imag
            b00_1 = b01.real*b01.real + b01.imag*b01.imag
            b01_0, b01_1 = b00*b10.conjugate(), b01*b11.conjugate()
            b11_0 = b10.real*b10.real + b10.imag*b10.imag
            b11_1 = b11.real*b11.real + b11.imag*b11.imag

            # C_jl^p; the weights are real so C_10^p = conj(C_01^p) and the
            # diagonal (in jl) entries remain real.
            c00_0, c00_1 = w_0*b00_0 + w_1*b00_1, w_2*b00_0 + w_3*b00_1
            c01_0, c01_1 = w_0*b01_0 + w_1*b01_1, w_2*b01_0 + w_3*b01_1
            c10_0, c10_1 = c01_0.conjugate(), c01_1.conjugate()
            c11_0, c11_1 = w_0*b11_0 + w_1*b11_1, w_2*b11_0 + w_3*b11_1

            return (
                acc[0] + upd[0],
                acc[1] + upd[1],
                acc[2] + upd[2],
                acc[3] + upd[3],
                acc[4] + (a00_0*c00_0 + a00_1*c00_1),
                acc[5] + (a00_0*c01_0 + a00_1*c01_1),
                acc[6] + (a01_0*c00_0 + a01_1*c00_1),
                acc[7] + (a01_0*c01_0 + a01_1*c01_1),
                acc[8] + (a00_0*c11_0 + a00_1*c11_1),
                acc[9] + (a01_0*c10_0 + a01_1*c10_1),
                acc[10] + (a01_0*c11_0 + a01_1*c11_1),
                acc[11] + (a11_0*c00_0 + a11_1*c00_1),
                acc[12] + (a11_0*c01_0 + a11_1*c01_1),
                acc[13] + (a11_0*c11_0 + a11_1*c11_1),
            )

    elif corr_mode.literal_value == 2:
        def impl(lop, rop, w, res, acc):

            # Accumulate an element of jhwr.
            upd = tuple_v1_mul_v2(res, rop)

            # Accumulate an element of jhwj: w|rop|^2, which is real.
            jh_00, jh_11 = rop[0], rop[1]

            return (
                acc[0] + upd[0],
                acc[1] + upd[1],
                acc[2] + w[0]*(jh_00.real*jh_00.real +
                               jh_00.imag*jh_00.imag),
                acc[3] + w[1]*(jh_11.real*jh_11.real +
                               jh_11.imag*jh_11.imag),
            )
    elif corr_mode.literal_value == 1:
        def impl(lop, rop, w, res, acc):

            # Accumulate an element of jhwr.
            upd = tuple_v1_mul_v2(res, rop)

            # Accumulate an element of jhwj: w|rop|^2, which is real.
            jh_00 = rop[0]

            return (
                acc[0] + upd[0],
                acc[1] + w[0]*(jh_00.real*jh_00.real +
                               jh_00.imag*jh_00.imag),
            )
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def mirror_jhj_factory(corr_mode):
    """Fill in the lower triangle of the per-interval jhj elements.

    In the 4 correlation case, accumulation in compute_jhwj_jhwr_elem only
    writes the upper triangle of each (4, 4) jhj element. As jhj is Hermitian,
    the lower triangle is the conjugate of the upper triangle and can be
    filled in once per solution interval rather than once per visibility.
    This is a no-op in the 1/2 correlation (diagonal) cases.
    """

    if corr_mode.literal_value == 4:
        def impl(jhj_tifi):
            n_ant, n_gdir = jhj_tifi.shape[:2]
            for a in range(n_ant):
                for d in range(n_gdir):
                    jhj_ad = jhj_tifi[a, d]
                    for i in range(1, 4):
                        for j in range(i):
                            jhj_ad[i, j] = jhj_ad[j, i].conjugate()
    else:
        def impl(jhj_tifi):
            pass

    return factories.qcjit(impl)


def get_jhj_dims_factory(corr_mode):

    if corr_mode.literal_value == 4:
        def impl(shape):
            return shape[:4] + (4, 4)
    elif corr_mode.literal_value in (1, 2):
        def impl(shape):
            return shape
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)
