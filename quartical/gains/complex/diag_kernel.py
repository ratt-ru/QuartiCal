# -*- coding: utf-8 -*-
import numpy as np
from numba import njit
from numba.extending import overload
from quartical.utils.numba import (coerce_literal,
                                   JIT_OPTIONS,
                                   PARALLEL_JIT_OPTIONS)
from quartical.gains.general.flagging import apply_gain_flags_to_gains
import quartical.gains.general.factories as factories
from quartical.gains.general.solver_components import build_jhj_jhr_impl
from quartical.gains.general.solver_loop import (build_gain_solver_impl,
                                                 identity_dims)
from quartical.gains.general.residuals import standard_residual_factory


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

    # The outer solver loop is shared between non-parameterised kernels - only
    # the hooks below are specific to diagonal complex terms. A diagonal term
    # stores jhj gain-shaped (so it passes the identity dims helper), supports
    # scalar mode via its own collapse_to_scalar_jhj_jhr, and references its
    # gains after
    # solving. The shared loop is inlined into the module-local trampoline
    # below rather than returned directly. This gives diag_complex a private
    # on-disk cache namespace - see the cache correctness constraint in
    # solver_components.py.
    shared_impl = build_gain_solver_impl(
        get_jhj_dims=identity_dims,
        compute_jhj_jhr=compute_jhj_jhr,
        collapse_to_scalar_jhj_jhr=collapse_to_scalar_jhj_jhr,
        scalar_error_message=None,
        finalize_update=finalize_update,
        reference_gains=reference_gains,
    )

    def impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        corr_mode
    ):
        return shared_impl(
            ms_inputs,
            mapping_inputs,
            chain_inputs,
            meta_inputs,
            corr_mode
        )

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
    # with no per-channel coefficients, so there is no compute_channel_coeffs
    # hook (the accumulate hook receives an empty channel_coeffs tuple). The
    # jhj element for a diagonal term is shaped like the gains (a flat
    # correlation vector, not a (4, 4) block), so there is no upper/lower
    # triangle to mirror and mirror_jhj_factory is None.
    # The shared loop is inlined into the module-local trampoline below
    # rather than returned directly. This gives diag_complex a private on-disk
    # cache namespace - see the cache correctness constraint in
    # solver_components.py.
    shared_impl = build_jhj_jhr_impl(
        corr_mode=corr_mode,
        row_weights_type=row_weights_type,
        accumulate_jhj_jhr_factory=accumulate_jhj_jhr_factory,
        zero_jhj_jhr_factory=zero_jhj_jhr_factory,
        flush_jhj_jhr_factory=flush_jhj_jhr_factory,
        compute_residual_factory=standard_residual_factory,
        compute_channel_coeffs_factory=None,
        mirror_jhj_factory=None,
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


def zero_jhj_jhr_factory(corr_mode):
    """Produce the zero jhwj/jhwr accumulator tuple for a given corr mode.

    Unlike the (full) complex kernel, a diagonal term stores jhj with the same
    shape as the gains (a flat correlation vector, not a (4, 4) block). The
    accumulator is a single flat tuple holding the jhwj element(s) followed by
    the jhwr element(s). A flat tuple is used deliberately - returning nested
    tuples from inlined functions inside a prange trips a numba parfor array
    analysis bug.

    The layouts (per corr mode) are:
      - corr 1: (jhj0, jhr0) - jhj is real (w|rop|^2), jhr complex.
      - corr 2: (jhj0, jhj1, jhr0, jhr1) - both jhj entries real, jhr complex.
      - corr 4: (jhj00, jhj03, jhj33, jhr0, jhr3) - the three distinct
        diagonal-in-correlation jhj entries and the diagonal jhr entries; the
        off-diagonal jhr entries stay zero and jhj[2] = conj(jhj[1]) is filled
        in by flush. These are complex.
    """

    if corr_mode.literal_value == 4:
        def impl(invec):
            z = invec[0]*0
            return z, z, z, z, z
    elif corr_mode.literal_value == 2:
        def impl(invec):
            z = invec[0]*0
            zr = invec[0].real*0
            return zr, zr, z, z
    elif corr_mode.literal_value == 1:
        def impl(invec):
            return invec[0].real*0, invec[0]*0
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def flush_jhj_jhr_factory(corr_mode):
    """Add a register-accumulated jhwr/jhwj accumulator into the arrays.

    For a diagonal term the jhj element is a flat correlation vector (see
    zero_jhj_jhr_factory). In the 4 correlation case only the diagonal jhr
    entries and three distinct jhj entries are accumulated; jhj[2] is the
    conjugate of jhj[1] (conjugation commutes with summation, so conjugating
    the accumulated sum once here is bit-identical to conjugating each
    contribution).
    """

    if corr_mode.literal_value == 4:
        def impl(jhj, jhr, jhj_jhr):

            jhj[0] += jhj_jhr[0]
            jhj[1] += jhj_jhr[1]
            jhj[2] += jhj_jhr[1].conjugate()
            jhj[3] += jhj_jhr[2]

            jhr[0] += jhj_jhr[3]
            jhr[3] += jhj_jhr[4]
    elif corr_mode.literal_value == 2:
        def impl(jhj, jhr, jhj_jhr):

            jhj[0] += jhj_jhr[0]
            jhj[1] += jhj_jhr[1]

            jhr[0] += jhj_jhr[2]
            jhr[1] += jhj_jhr[3]
    elif corr_mode.literal_value == 1:
        def impl(jhj, jhr, jhj_jhr):

            jhj[0] += jhj_jhr[0]

            jhr[0] += jhj_jhr[1]
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def accumulate_jhj_jhr_factory(corr_mode):
    """Accumulate a jhwr/jhwj element into a register-resident accumulator.

    All inputs and the returned accumulator are tuples (register-resident
    values) - the accumulator is only flushed to memory by flush_jhj_jhr.
    The accumulator is a single flat tuple (jhwj followed by jhwr - see
    zero_jhj_jhr_factory).

    The signature follows the unified accumulate_jhj_jhr contract of the shared
    accumulation loop (see solver_components.py). Diagonal complex terms have
    no chain rule beyond the operators themselves, so the gain and
    channel_coeffs arguments are unused - the compiler eliminates them entirely
    after inlining. The 1 and 2 correlation cases are identical to the (full)
    complex kernel; only the 4 correlation case differs, because a diagonal
    term keeps just the diagonal (in correlation) entries of jhr and jhj.
    """

    tuple_v1_mul_v2 = factories.tuple_v1_mul_v2_factory(corr_mode)

    if corr_mode.literal_value == 4:
        def impl(lop, rop, w, gain, channel_coeffs, wres, jhj_jhr):

            l0, l1, l2, l3 = lop[0], lop[1], lop[2], lop[3]
            r0, r1, r2, r3 = rop[0], rop[1], rop[2], rop[3]

            # Off-diagonal weights are effectively zero for a diagonal term.
            w_0, w_3 = w[0], w[3]  # NOTE: XX, YY

            # jhwr = diag(lop @ diag(res_00, res_11) @ rop). The incoming wres
            # is the weighted residual; only its diagonal entries contribute -
            # the off-diagonals are dropped.
            wr_0, wr_3 = wres[0], wres[3]
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
                jhj_jhr[0] + jhwj_00,
                jhj_jhr[1] + jhwj_03,
                jhj_jhr[2] + jhwj_33,
                jhj_jhr[3] + jhr0,
                jhj_jhr[4] + jhr3,
            )
    elif corr_mode.literal_value == 2:
        def impl(lop, rop, w, gain, channel_coeffs, wres, jhj_jhr):

            # Accumulate an element of jhwr.
            upd = tuple_v1_mul_v2(wres, rop)

            # Accumulate an element of jhwj: w|rop|^2, which is real.
            jh_00, jh_11 = rop[0], rop[1]

            return (
                jhj_jhr[0] + w[0]*(jh_00.real*jh_00.real +
                               jh_00.imag*jh_00.imag),
                jhj_jhr[1] + w[1]*(jh_11.real*jh_11.real +
                               jh_11.imag*jh_11.imag),
                jhj_jhr[2] + upd[0],
                jhj_jhr[3] + upd[1],
            )
    elif corr_mode.literal_value == 1:
        def impl(lop, rop, w, gain, channel_coeffs, wres, jhj_jhr):

            # Accumulate an element of jhwr.
            upd = tuple_v1_mul_v2(wres, rop)

            # Accumulate an element of jhwj: w|rop|^2, which is real.
            jh_00 = rop[0]

            return (
                jhj_jhr[0] + w[0]*(jh_00.real*jh_00.real +
                               jh_00.imag*jh_00.imag),
                jhj_jhr[1] + upd[0],
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
def collapse_to_scalar_jhj_jhr(solver_imdry):
    """Sum jhj and jhr over correlation to give a scalar solve.

    This exists separately from generics.scalar_jhj_jhr because the two act on
    differently shaped jhj arrays. A diagonal term stores jhj with the same
    shape as its gains - a flat correlation vector per solution element, hence
    the identity_dims helper - so collapsing it is a sum along the correlation
    axis, broadcast back afterwards. The generic routine is for parameterised
    terms, whose jhj element is a real (n_param, n_param) block; it indexes
    jhj_sel[p0, p1] and folds the halves of that block together using
    values_per_correlation as the stride, which has no meaning here and would
    not even index correctly against a one-dimensional element.

    diag_complex is the only non-parameterised term supporting a scalar solve
    (complex and leakage pass ``collapse_to_scalar_jhj_jhr=None``), so this is
    the sole caller.

    Args:
        solver_imdry: The native intermediaries holding jhj and jhr.
    """

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

                    # Repopulate appropriate zeroed values from scalar sum.
                    jhr_sel[-1] = jhr_sel[0]
                    jhj_sel[-1] = jhj_sel[0]
