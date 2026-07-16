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
# Crosshand phase's residual is amplitude-normalised in exactly the same way
# as the phase term (r_i*|v_i|/|r_i| - v_i), so it reuses phase's residual hook
# rather than duplicating it.
from quartical.gains.phase.kernel import resid_factory


def get_identity_params(corr_mode):

    if corr_mode.literal_value == 4:
        return np.zeros((1,), dtype=np.float64)
    else:
        raise ValueError("Unsupported number of correlations.")


@njit(**JIT_OPTIONS)
def crosshand_phase_solver(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    return crosshand_phase_solver_impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        corr_mode
    )


def crosshand_phase_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    raise NotImplementedError


@overload(crosshand_phase_solver_impl, jit_options=JIT_OPTIONS)
def nb_crosshand_phase_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):

    coerce_literal(nb_crosshand_phase_solver_impl, ["corr_mode"])

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

        # Set up some intemediaries used for flagging.
        km1_gain = active_gain.copy()
        km1_abs2_diffs = np.zeros_like(active_gain_flags, dtype=np.float64)
        abs2_diffs_trend = np.zeros_like(active_gain_flags, dtype=np.float64)
        flag_imdry = \
            flag_intermediaries(km1_gain, km1_abs2_diffs, abs2_diffs_trend)

        # Set up some intemediaries used for solving.
        real_dtype = active_gain.real.dtype
        param_shape = active_params.shape

        active_t_map_g = mapping_inputs.time_maps[active_term]
        active_f_map_g = mapping_inputs.freq_maps[active_term]

        # Create more work to do in paralllel when needed, else no-op.
        resampler = resample_solints(active_t_map_g, param_shape, n_thread)

        # Determine the starts and stops of the rows and channels associated
        # with each solution interval.
        extents = get_extents(resampler.upsample_t_map, active_f_map_g)

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
                    "Scalar mode not supported for crosshand phase terms."
                )

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

    # The accumulation loop itself is shared between kernels - only the hooks
    # below (the per-term maths) are specific to crosshand phase terms. The
    # loop body is phase's verbatim, so crosshand reuses phase's amplitude-
    # normalised residual hook (n_resid_aux is n_corr). Crosshand solves a
    # single parameter, so its jhj is (1, 1) and the mirror hook is a no-op
    # (mirror_factory is None). There are no per-channel coefficients, so there
    # is no stage hook.
    # The shared loop is inlined into the module-local trampoline below
    # rather than returned directly. This gives crosshand_phase a private on-disk
    # cache namespace - see the cache correctness constraint in
    # accumulation.py.
    shared_impl = build_jhj_jhr_impl(
        corr_mode,
        row_weights_type,
        elem_factory=compute_jhwj_jhwr_elem_factory,
        acc_zeros_factory=jhwj_jhwr_zeros_factory,
        flush_factory=flush_jhwj_jhwr_factory,
        resid_factory=resid_factory,
        n_resid_aux=corr_mode.literal_value,
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
    param_to_gain = param_to_gain_factory(corr_mode)

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
        params = chain_inputs.params[active_term]

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

                        p = params[ti, fi, a, d]
                        g = gain[ti, fi, a, d]
                        fl = gain_flags[ti, fi, a, d]
                        upd = update[ti, fi, a, d]

                        if fl == 1:
                            p[:] = 0
                            set_identity(g)
                        else:
                            p += upd
                            param_to_gain(p, g)

    return impl


def param_to_gain_factory(corr_mode):

    if corr_mode.literal_value == 4:
        def impl(params, gain):
            gain[0] = np.exp(1j*params[0])
    else:
        raise ValueError("Crosshand phase can only be solved for with four "
                         "correlation data.")

    return factories.qcjit(impl)


def jhwj_jhwr_zeros_factory(corr_mode):
    """Produce the zero jhr/jhj accumulator tuple for a given corr mode.

    Crosshand phase solves a single parameter, so the accumulator is a flat
    tuple holding the one real jhr entry followed by the single (1, 1) jhj
    element: (jhr0, jhj00). The reference element is a jhr slice, whose dtype
    is real, so both accumulator values are real zeros.
    """

    if corr_mode.literal_value == 4:
        def impl(invec):
            z = invec[0]*0
            return z, z
    else:
        raise ValueError("Crosshand phase can only be solved for with four "
                         "correlation data.")

    return factories.qcjit(impl)


def flush_jhwj_jhwr_factory(corr_mode):
    """Add a register-accumulated jhr/jhj accumulator into the arrays.

    Crosshand phase's jhj is (1, 1), so there is no upper triangle to mirror -
    the mirror hook is a no-op (see nb_compute_jhj_jhr).
    """

    if corr_mode.literal_value == 4:
        def impl(jhr, jhj, acc):

            jhr[0] += acc[0]

            jhj[0, 0] += acc[1]
    else:
        raise ValueError("Crosshand phase can only be solved for with four "
                         "correlation data.")

    return factories.qcjit(impl)


def compute_jhwj_jhwr_elem_factory(corr_mode):
    """Accumulate a jhr/jhj element into a register-resident accumulator.

    All inputs and the returned accumulator are tuples (register-resident
    values) - the accumulator is only flushed to memory by flush_jhwj_jhwr.
    The accumulator is a flat tuple (jhr0, jhj00) - see jhwj_jhwr_zeros_factory.

    The signature follows the unified elem contract of the shared accumulation
    loop (see accumulation.py). The crosshand chain rule uses the active-term
    gain (drv = -1j*conj(g)), so the gain argument is consumed. The aux
    argument (the per-correlation normalisation factor from the residual hook)
    is not used - this elem recomputes its own operator-based normalisation,
    exactly as the original array-buffer kernel did.

    Unlike phase, crosshand keeps the full (2, 2) operator product rather than
    only its diagonal: the derivative is with respect to the single crosshand
    phase and only the [0] (XX) component of lop @ (normalised residual) @ rop
    is retained for jhr, while jhj sums all four elements of the first row of
    the row-major kronecker product.
    """

    if corr_mode.literal_value == 4:
        def impl(lop, rop, w, gain, aux, res, acc):

            lop_0, lop_1, lop_2, lop_3 = lop[0], lop[1], lop[2], lop[3]
            rop_0, rop_1, rop_2, rop_3 = rop[0], rop[1], rop[2], rop[3]

            # Normalisation factor: 1/|lop @ rop|^2 elementwise, with the same
            # zero guard as the array kernel (iabsdivsq of the full 2x2
            # product).
            nf0 = lop_0*rop_0 + lop_1*rop_2
            nf1 = lop_0*rop_1 + lop_1*rop_3
            nf2 = lop_2*rop_0 + lop_3*rop_2
            nf3 = lop_2*rop_1 + lop_3*rop_3
            n_0 = 0 if nf0 == 0 else 1/(nf0.real**2 + nf0.imag**2)
            n_1 = 0 if nf1 == 0 else 1/(nf1.real**2 + nf1.imag**2)
            n_2 = 0 if nf2 == 0 else 1/(nf2.real**2 + nf2.imag**2)
            n_3 = 0 if nf3 == 0 else 1/(nf3.real**2 + nf3.imag**2)

            # jhwr element: lop @ (diag-normalised residual) @ rop, keeping only
            # the [0] (XX) entry. The incoming residual is already weighted; the
            # normalisation factor is applied here, matching imul(res, normf).
            s_0 = res[0]*n_0
            s_1 = res[1]*n_1
            s_2 = res[2]*n_2
            s_3 = res[3]*n_3

            mm_0 = s_0*rop_0 + s_1*rop_2
            mm_2 = s_2*rop_0 + s_3*rop_2
            r_0 = lop_0*mm_0 + lop_1*mm_2

            gc_0 = gain[0].conjugate()
            drv_00 = -1j*gc_0
            upd_00 = (drv_00*r_0).real

            # jhwj element: the normalisation is folded into the weights.
            # NOTE: rop is effectively transposed (rop[2] used as rop_01)
            # relative to lop, matching the row-major kronecker convention.
            w_0 = n_0 * w[0]
            w_1 = n_1 * w[1]
            w_2 = n_2 * w[2]
            w_3 = n_3 * w[3]

            jh_00 = lop_0 * rop_0
            jh_01 = lop_0 * rop_2
            jh_02 = lop_1 * rop_0
            jh_03 = lop_1 * rop_2

            j_00 = jh_00.conjugate()
            j_01 = jh_01.conjugate()
            j_02 = jh_02.conjugate()
            j_03 = jh_03.conjugate()

            jhwj_00 = jh_00*w_0*j_00 + jh_01*w_1*j_01 + \
                jh_02*w_2*j_02 + jh_03*w_3*j_03

            return (
                acc[0] + upd_00,
                acc[1] + jhwj_00.real,
            )

    else:
        raise ValueError("Crosshand phase can only be solved for with four "
                         "correlation data.")

    return factories.qcjit(impl)


@njit(**JIT_OPTIONS)
def crosshand_params_to_gains(
    params,
    gains
):

    n_time, n_freq, n_ant, n_dir, n_corr = gains.shape

    for t in range(n_time):
        for f in range(n_freq):
            for a in range(n_ant):
                for d in range(n_dir):

                    g = gains[t, f, a, d]
                    p = params[t, f, a, d]

                    g[0] = np.exp(1j*p[0])
                    g[-1] = 1
