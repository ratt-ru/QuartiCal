# -*- coding: utf-8 -*-
import numpy as np
from numba import njit
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
                                              update_param_flags)
from quartical.gains.general.convenience import get_extents
import quartical.gains.general.factories as factories
from quartical.gains.general.accumulation import build_jhj_jhr_impl
from quartical.gains.general.solver_ops import compute_update  # noqa
# Rotation's residual is the plain complex residual (r - v), so it reuses the
# complex term's residual hook rather than duplicating it.
from quartical.gains.complex.kernel import resid_factory


def get_identity_params(corr_mode):

    if corr_mode.literal_value == 4:
        return np.zeros((1,), dtype=np.float64)
    else:
        raise ValueError("Unsupported number of correlations.")


@njit(**JIT_OPTIONS)
def rotation_solver(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    return rotation_solver_impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        corr_mode
    )


def rotation_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    raise NotImplementedError


@overload(rotation_solver_impl, jit_options=JIT_OPTIONS)
def nb_rotation_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):

    coerce_literal(nb_rotation_solver_impl, ["corr_mode"])

    identity_params = get_identity_params(corr_mode)

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
        active_params = chain_inputs.params[active_term]

        # Set up some intemediaries used for flagging. TODO: Move?
        km1_gain = active_gain.copy()
        km1_abs2_diffs = np.zeros_like(active_gain_flags, dtype=np.float64)
        abs2_diffs_trend = np.zeros_like(active_gain_flags, dtype=np.float64)
        flag_imdry = flag_intermediaries(
            km1_gain, km1_abs2_diffs, abs2_diffs_trend
        )

        # Set up some intemediaries used for solving.
        real_dtype = active_gain.real.dtype
        param_shape = active_params.shape

        active_t_map_g = mapping_inputs.time_maps[active_term]
        active_f_map_p = mapping_inputs.param_freq_maps[active_term]

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
                raise ValueError(
                    "Scalar mode not supported for rotation terms."
                )

            if not max_iter:  # Non-solvable term, we just want jhj.
                conv_perc = 0  # Didn't converge.
                loop_idx = -1  # Did zero iterations.
                break

            compute_update(native_imdry,
                           corr_mode)

            finalize_update(
                ms_inputs,
                mapping_inputs,
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
                corr_mode,
                numbness=1e9
            )

            # Propagate gain flags to parameter flags.
            update_param_flags(
                mapping_inputs,
                chain_inputs,
                meta_inputs,
                identity_params
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

        return jhj, loop_idx + 1, conv_perc

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

    # The accumulation loop itself is shared between kernels - only the hooks
    # below (the per-term maths) are specific to rotation terms. Rotation's
    # residual is the plain complex residual (r - v), so it reuses complex's
    # residual hook and appends no auxiliary values (n_resid_aux is zero).
    # Rotation solves a single parameter, so its jhj is (1, 1) and the mirror
    # hook is a no-op (mirror_factory is None). There are no per-channel
    # coefficients, so there is no stage hook.
    # The shared loop is inlined into the module-local trampoline below
    # rather than returned directly. This gives rotation a private on-disk
    # cache namespace - see the cache correctness constraint in
    # accumulation.py.
    shared_impl = build_jhj_jhr_impl(
        corr_mode,
        row_weights_type,
        elem_factory=compute_jhwj_jhwr_elem_factory,
        acc_zeros_factory=jhwj_jhwr_zeros_factory,
        flush_factory=flush_jhwj_jhwr_factory,
        resid_factory=resid_factory,
        n_resid_aux=0,
        stage_factory=None,
        mirror_factory=None,
    )

    def impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        upsampled_imdry,
        extents,
        corr_mode
    ):
        return shared_impl(
            ms_inputs,
            mapping_inputs,
            chain_inputs,
            meta_inputs,
            upsampled_imdry,
            extents,
            corr_mode
        )

    return impl


def finalize_update(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    native_imdry,
    loop_idx,
    corr_mode
):
    raise NotImplementedError


@overload(finalize_update, jit_options=JIT_OPTIONS)
def nb_finalize_update(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    native_imdry,
    loop_idx,
    corr_mode
):

    coerce_literal(nb_finalize_update, ["corr_mode"])

    set_identity = factories.set_identity_factory(corr_mode)

    if corr_mode.literal_value == 4:
        def impl(
            ms_inputs,
            mapping_inputs,
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
            params = chain_inputs.params[active_term]

            d_map_arr = mapping_inputs.dir_maps[active_term]

            update = native_imdry.update

            update /= 2
            params += update

            n_time, n_freq, n_ant, n_dir, _ = gain.shape

            if dd_term:
                for pd in pinned_directions:
                    params[..., pd, :] = 0

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


def jhwj_jhwr_zeros_factory(corr_mode):
    """Produce the zero jhr/jhj accumulator tuple for a given corr mode.

    Rotation solves a single parameter, so the accumulator is a flat tuple
    holding the one real jhr entry followed by the single (1, 1) jhj element:
    (jhr0, jhj00). The reference element is a jhr slice, whose dtype is real,
    so both accumulator values are real zeros.
    """

    if corr_mode.literal_value == 4:
        def impl(invec):
            z = invec[0]*0
            return z, z
    else:
        raise ValueError("Rotation can only be solved for with four "
                         "correlation data.")

    return factories.qcjit(impl)


def flush_jhwj_jhwr_factory(corr_mode):
    """Add a register-accumulated jhr/jhj accumulator into the arrays.

    Rotation's jhj is (1, 1), so there is no upper triangle to mirror - the
    mirror hook is a no-op (see nb_compute_jhj_jhr).
    """

    if corr_mode.literal_value == 4:
        def impl(jhr, jhj, acc):

            jhr[0] += acc[0]

            jhj[0, 0] += acc[1]
    else:
        raise ValueError("Rotation can only be solved for with four "
                         "correlation data.")

    return factories.qcjit(impl)


def compute_jhwj_jhwr_elem_factory(corr_mode):
    """Accumulate a jhr/jhj element into a register-resident accumulator.

    All inputs and the returned accumulator are tuples (register-resident
    values) - the accumulator is only flushed to memory by flush_jhwj_jhwr.
    The accumulator is a flat tuple (jhr0, jhj00) - see jhwj_jhwr_zeros_factory.

    The signature follows the unified elem contract of the shared accumulation
    loop (see accumulation.py). The active-term gain IS the rotation matrix
    [cos, -sin; sin, cos] (row-major XX, XY, YX, YY), so the derivative row
    dh = [-sin, -cos, cos, -sin] is read directly from the gain tuple
    (cos = gain[0].real, sin = gain[2].real) rather than recomputing sin/cos
    from theta - this is bit-identical to the original (which used
    np.sin(theta)/np.cos(theta) with theta = params) because the gain entries
    were themselves set to np.cos(theta)/np.sin(theta), and it avoids any
    arctan2 wrapping near theta = +/-pi. The aux argument is unused (rotation
    appends no auxiliary residual values).

    The original array kernel built the full (4, 4) row-major kronecker product
    a_kron_bt(lop, rop) and contracted every column with dh. Here that temp
    array is eliminated: the four column contractions dhjh_j are inlined
    symbolically from the kronecker entries the dh contraction actually touches
    (all 16, but expressed directly from lop/rop). jhr is dh . (lop @ res @ rop)
    and jhj sums w_j * |dhjh_j|^2 over the four correlations.
    """

    tuple_v1_mul_v2 = factories.tuple_v1_mul_v2_factory(corr_mode)

    if corr_mode.literal_value == 4:
        def impl(lop, rop, w, gain, aux, res, acc):

            lop_0, lop_1, lop_2, lop_3 = lop[0], lop[1], lop[2], lop[3]
            rop_0, rop_1, rop_2, rop_3 = rop[0], rop[1], rop[2], rop[3]
            w_0, w_1, w_2, w_3 = w[0], w[1], w[2], w[3]

            # jhwr element: r = lop @ (res @ rop), where res is the weighted
            # residual. Matches the array kernel's in-place matmuls exactly.
            r_0, r_1, r_2, r_3 = tuple_v1_mul_v2(
                lop, tuple_v1_mul_v2(res, rop)
            )

            # Derivative of the rotation matrix wrt theta, read straight from
            # the gain: d/dtheta [cos, -sin; sin, cos] = [-sin, -cos; cos, -sin]
            # (row-major XX, XY, YX, YY).
            cos_theta = gain[0].real
            sin_theta = gain[2].real

            dh_0 = -sin_theta
            dh_1 = -cos_theta
            dh_2 = cos_theta
            dh_3 = -sin_theta

            upd = (dh_0*r_0).real + (dh_1*r_1).real + \
                (dh_2*r_2).real + (dh_3*r_3).real

            # jhwj element: dh contracted with each column of the row-major
            # kronecker product of (lop, rop). The kronecker entries are
            # inlined so the (4, 4) temp array disappears. rop uses the
            # effective-transpose ordering (rop_1 <-> rop_2) of a_kron_bt.
            dhjh_0 = dh_0*lop_0*rop_0 + dh_1*lop_0*rop_1 + \
                dh_2*lop_2*rop_0 + dh_3*lop_2*rop_1
            dhjh_1 = dh_0*lop_0*rop_2 + dh_1*lop_0*rop_3 + \
                dh_2*lop_2*rop_2 + dh_3*lop_2*rop_3
            dhjh_2 = dh_0*lop_1*rop_0 + dh_1*lop_1*rop_1 + \
                dh_2*lop_3*rop_0 + dh_3*lop_3*rop_1
            dhjh_3 = dh_0*lop_1*rop_2 + dh_1*lop_1*rop_3 + \
                dh_2*lop_3*rop_2 + dh_3*lop_3*rop_3

            jhj_00 = (dhjh_0 * w_0 * dhjh_0.conjugate()).real + \
                (dhjh_1 * w_1 * dhjh_1.conjugate()).real + \
                (dhjh_2 * w_2 * dhjh_2.conjugate()).real + \
                (dhjh_3 * w_3 * dhjh_3.conjugate()).real

            return (
                acc[0] + upd,
                acc[1] + jhj_00,
            )

    else:
        raise ValueError("Rotation can only be solved for with four "
                         "correlation data.")

    return factories.qcjit(impl)


@njit(**JIT_OPTIONS)
def rotation_params_to_gains(
    params,
    gains
):

    n_time, n_freq, n_ant, n_dir, n_corr = gains.shape

    for t in range(n_time):
        for f in range(n_freq):
            for a in range(n_ant):
                for d in range(n_dir):

                    g = gains[t, f, a, d]
                    theta = params[t, f, a, d, 0]

                    cos_theta = np.cos(theta)
                    sin_theta = np.sin(theta)

                    g[0] = cos_theta
                    g[1] = -sin_theta
                    g[2] = sin_theta
                    g[3] = cos_theta
