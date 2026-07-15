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
                                              apply_gain_flags_to_gains)
from quartical.gains.general.convenience import get_extents
import quartical.gains.general.factories as factories
from quartical.gains.general.accumulation import build_jhj_jhr_impl
from quartical.gains.general.solver_ops import compute_update  # noqa


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
        scalar = meta_inputs.scalar
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

            if scalar and corr_mode != 1:
                scalar_jhj_jhr(native_imdry)

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

    # The accumulation loop itself is shared between kernels - only the hooks
    # below (the per-term maths) are specific to diagonal complex terms. As
    # with the (full) complex kernel, the diagonal residual is simply r - v
    # with no auxiliary values and no per-channel coefficients, so n_resid_aux
    # is zero and there is no stage hook. The jhj element for a diagonal term
    # is shaped like the gains (a flat correlation vector, not a (4, 4) block),
    # so there is no upper/lower triangle to mirror and mirror_factory is None.
    return build_jhj_jhr_impl(
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


def resid_factory(corr_mode):
    """Produce the residual tuple for a diagonal complex term.

    As with the (full) complex term the residual is simply r - v. No auxiliary
    values are appended (n_resid_aux is zero), so the returned flat tuple holds
    only the per-correlation residual values.
    """

    tuple_sub = factories.tuple_sub_factory(corr_mode)

    def impl(r, v):
        return tuple_sub(r, v)

    return factories.qcjit(impl)


def jhwj_jhwr_zeros_factory(corr_mode):
    """Produce the zero jhwr/jhwj accumulator tuple for a given corr mode.

    Unlike the (full) complex kernel, a diagonal term stores jhj with the same
    shape as the gains (a flat correlation vector, not a (4, 4) block). The
    accumulator is a single flat tuple holding the jhwr element(s) followed by
    the jhwj element(s). A flat tuple is used deliberately - returning nested
    tuples from inlined functions inside a prange trips a numba parfor array
    analysis bug.

    The layouts (per corr mode) are:
      - corr 1: (jhr0, jhj0) - jhj is real (w|rop|^2).
      - corr 2: (jhr0, jhr1, jhj0, jhj1) - both jhj entries real.
      - corr 4: (jhr0, jhr3, jhj00, jhj03, jhj33) - the diagonal jhr entries
        and the three distinct diagonal-in-correlation jhj entries; the off-
        diagonal jhr entries stay zero and jhj[2] = conj(jhj[1]) is filled in
        by flush. These are complex.
    """

    if corr_mode.literal_value == 4:
        def impl(invec):
            z = invec[0]*0
            return z, z, z, z, z
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

    For a diagonal term the jhj element is a flat correlation vector (see
    jhwj_jhwr_zeros_factory). In the 4 correlation case only the diagonal jhr
    entries and three distinct jhj entries are accumulated; jhj[2] is the
    conjugate of jhj[1] (conjugation commutes with summation, so conjugating
    the accumulated sum once here is bit-identical to conjugating each
    contribution).
    """

    if corr_mode.literal_value == 4:
        def impl(jhr, jhj, acc):

            jhr[0] += acc[0]
            jhr[3] += acc[1]

            jhj[0] += acc[2]
            jhj[1] += acc[3]
            jhj[2] += acc[3].conjugate()
            jhj[3] += acc[4]
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
    jhwj_jhwr_zeros_factory).

    The signature follows the unified elem contract of the shared accumulation
    loop (see accumulation.py). Diagonal complex terms have no chain rule
    beyond the operators themselves, so the gain and aux arguments are unused -
    the compiler eliminates them entirely after inlining. The 1 and 2
    correlation cases are identical to the (full) complex kernel; only the 4
    correlation case differs, because a diagonal term keeps just the diagonal
    (in correlation) entries of jhr and jhj.
    """

    tuple_v1_mul_v2 = factories.tuple_v1_mul_v2_factory(corr_mode)

    if corr_mode.literal_value == 4:
        def impl(lop, rop, w, gain, aux, res, acc):

            l0, l1, l2, l3 = lop[0], lop[1], lop[2], lop[3]
            r0, r1, r2, r3 = rop[0], rop[1], rop[2], rop[3]

            # Off-diagonal weights are effectively zero for a diagonal term.
            w_0, w_3 = w[0], w[3]  # NOTE: XX, YY

            # jhwr = diag(lop @ diag(res_00, res_11) @ rop). The incoming res is
            # the weighted residual; only its diagonal entries contribute (the
            # off-diagonals are dropped, matching the original elem).
            wr_0, wr_3 = res[0], res[3]
            jhr0 = (l0*wr_0)*r0 + (l1*wr_3)*r2
            jhr3 = (l2*wr_0)*r1 + (l3*wr_3)*r3

            # jhwj uses the row-major kronecker product identity (the MS stores
            # correlations XX, XY, YX, YY). With the off-diagonal weights zero,
            # only rows 0 and 3 of the kronecker product survive, and only
            # their columns 0 and 3 are non-zero, so the distinct jhj entries
            # reduce to sums over the two diagonal correlations below.
            tk0_0 = l0*r0  # kron[0, 0]
            tk0_3 = l1*r2  # kron[0, 3]
            tk3_0 = l2*r1  # kron[3, 0]
            tk3_3 = l3*r3  # kron[3, 3]

            jhwj_00 = (tk0_0*w_0)*tk0_0.conjugate() + \
                (tk0_3*w_3)*tk0_3.conjugate()
            jhwj_03 = (tk0_0*w_0)*tk3_0.conjugate() + \
                (tk0_3*w_3)*tk3_3.conjugate()
            jhwj_33 = (tk3_0*w_0)*tk3_0.conjugate() + \
                (tk3_3*w_3)*tk3_3.conjugate()

            return (
                acc[0] + jhr0,
                acc[1] + jhr3,
                acc[2] + jhwj_00,
                acc[3] + jhwj_03,
                acc[4] + jhwj_33,
            )
    elif corr_mode.literal_value == 2:
        def impl(lop, rop, w, gain, aux, res, acc):

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
        def impl(lop, rop, w, gain, aux, res, acc):

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


@njit(**JIT_OPTIONS)
def scalar_jhj_jhr(solver_imdry):
    """This manipulates the entries of jhj and jhr to be scalar."""

    # NOTE: This differes from the generic implmenentation in generics.py.

    jhj = solver_imdry.jhj
    jhr = solver_imdry.jhr

    n_tint, n_fint, n_ant, n_dir, n_corr = jhj.shape

    for t in range(n_tint):
        for f in range(n_fint):
            for a in range(n_ant):
                for d in range(n_dir):

                    jhr_sel = jhr[t, f, a, d]
                    jhj_sel = jhj[t, f, a, d]

                    # Sum to a single scalar element.
                    for p in range(1, n_corr):
                        jhr_sel[0] += jhr_sel[p]
                        jhr_sel[p] = 0
                        jhj_sel[0] += jhj_sel[p]
                        jhj_sel[p] = 0

                    # Repopluate appropriate zeroed values from scalar sum.
                    jhr_sel[-1] = jhr_sel[0]
                    jhj_sel[-1] = jhj_sel[0]
