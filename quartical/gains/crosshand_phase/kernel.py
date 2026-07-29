# -*- coding: utf-8 -*-
import numpy as np
from numba import njit
from numba.extending import overload
from quartical.utils.numba import (coerce_literal,
                                   JIT_OPTIONS,
                                   PARALLEL_JIT_OPTIONS)
import quartical.gains.general.factories as factories
from quartical.gains.general.solver_components import build_jhj_jhr_impl
from quartical.gains.general.solver_loop import build_param_solver_impl
# Crosshand phase's residual is amplitude-normalised in exactly the same way
# as the phase term (r_i*|v_i|/|r_i| - v_i), so it reuses phase's residual hook
# rather than duplicating it.
from quartical.gains.phase.kernel import compute_residual_factory


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

    # The outer solver loop is shared between parameterised kernels - only the
    # hooks below are specific to crosshand phase terms. Crosshand phase solves
    # on the gain grid, does not support scalar mode, needs no referencing
    # stage, and enters/leaves no scaled solver basis (no pre/post-solve
    # stages). The shared loop is inlined into the module-local trampoline
    # below rather than returned directly. This gives crosshand_phase a private
    # on-disk cache namespace - see the cache correctness constraint in
    # solver_components.py.
    shared_impl = build_param_solver_impl(
        pre_solve=None,
        compute_jhj_jhr=compute_jhj_jhr,
        params_per_corr=None,
        scalar_error_message=(
            "Scalar mode not supported for crosshand phase terms."
        ),
        finalize_update=finalize_update,
        numbness=1e9,
        identity_params=identity_params,
        reference_params=None,
        post_solve=None,
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
    # below (the per-term maths) are specific to crosshand phase terms. The
    # loop body is phase's verbatim, so crosshand reuses phase's amplitude-
    # normalised residual hook. Crosshand solves a single parameter, so its jhj
    # is (1, 1) and the mirror hook is a no-op (mirror_jhj_factory is None).
    # There are no per-channel coefficients, so there is no
    # compute_channel_coeffs hook. The shared loop is inlined into the
    # module-local trampoline below rather than returned directly. This gives
    # crosshand_phase a private on-disk cache namespace - see the cache
    # correctness constraint in solver_components.py.
    shared_impl = build_jhj_jhr_impl(
        corr_mode=corr_mode,
        row_weights_type=row_weights_type,
        accumulate_jhj_jhr_factory=accumulate_jhj_jhr_factory,
        zero_jhj_jhr_factory=zero_jhj_jhr_factory,
        flush_jhj_jhr_factory=flush_jhj_jhr_factory,
        compute_residual_factory=compute_residual_factory,
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
    param_to_gain = param_to_gain_factory(corr_mode)

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


def zero_jhj_jhr_factory(corr_mode):
    """Produce the zero jhj/jhr accumulator tuple for a given corr mode.

    Crosshand phase solves a single parameter, so the accumulator is a flat
    tuple holding the single (1, 1) jhj element followed by the one real jhr
    entry: (jhj00, jhr0). The reference element is a jhr slice, whose dtype
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


def flush_jhj_jhr_factory(corr_mode):
    """Add a register-accumulated jhr/jhj accumulator into the arrays.

    Crosshand phase's jhj is (1, 1), so there is no upper triangle to mirror -
    the mirror hook is a no-op (see nb_compute_jhj_jhr).
    """

    if corr_mode.literal_value == 4:
        def impl(jhj, jhr, jhj_jhr):

            jhj[0, 0] += jhj_jhr[0]

            jhr[0] += jhj_jhr[1]
    else:
        raise ValueError("Crosshand phase can only be solved for with four "
                         "correlation data.")

    return factories.qcjit(impl)


def accumulate_jhj_jhr_factory(corr_mode):
    """Accumulate a jhr/jhj element into a register-resident accumulator.

    All inputs and the returned accumulator are tuples (register-resident
    values) - the accumulator is only flushed to memory by flush_jhj_jhr.
    The accumulator is a flat tuple (jhj00, jhr0) - see zero_jhj_jhr_factory.

    The signature follows the unified accumulate_jhj_jhr contract of the shared
    accumulation loop (see solver_components.py). The crosshand chain rule uses
    the active-term gain (drv = -1j*conj(g)), so the gain argument is consumed.
    The channel_coeffs argument is empty (crosshand phase has no
    compute_channel_coeffs hook) and unused; the normalisation applied to the
    residual is recomputed here from the operators rather than being passed in
    from compute_residual.

    Unlike phase, crosshand keeps the full (2, 2) operator product rather than
    only its diagonal: the derivative is with respect to the single crosshand
    phase and only the [0] (XX) component of lop @ (normalised residual) @ rop
    is retained for jhr, while jhj sums all four elements of the first row of
    the row-major kronecker product.
    """

    if corr_mode.literal_value == 4:
        def impl(lop, rop, w, gain, channel_coeffs, wres, jhj_jhr):

            lop_0, lop_1, lop_2, lop_3 = lop[0], lop[1], lop[2], lop[3]
            rop_0, rop_1, rop_2, rop_3 = rop[0], rop[1], rop[2], rop[3]

            # Row-major Kronecker entries of J^H, matching a_kron_bt: rop
            # enters transposed, which is why rop_2 supplies column 1. The
            # derivative is with respect to a single crosshand phase, so only
            # row 0 of the (4, 4) product is needed - but all four
            # correlations of the residual contribute to it.
            jh_00 = lop_0*rop_0
            jh_01 = lop_0*rop_2
            jh_02 = lop_1*rop_0
            jh_03 = lop_1*rop_2

            # The J^H element for each correlation: the entries of lop @ rop.
            # Correlation 0 is Kronecker row 0 contracted over its
            # diagonal-correlation columns; the other three would need rows
            # 1-3, which this hook never forms, so they are written out from
            # lop/rop directly. Each normalises its own residual entry and
            # weight by the reciprocal squared modulus, guarded against a zero
            # element.
            jh_0 = jh_00 + jh_03
            jh_1 = lop_0*rop_1 + lop_1*rop_3
            jh_2 = lop_2*rop_0 + lop_3*rop_2
            jh_3 = lop_2*rop_1 + lop_3*rop_3
            n_0 = 0 if jh_0 == 0 else 1/(jh_0.real**2 + jh_0.imag**2)
            n_1 = 0 if jh_1 == 0 else 1/(jh_1.real**2 + jh_1.imag**2)
            n_2 = 0 if jh_2 == 0 else 1/(jh_2.real**2 + jh_2.imag**2)
            n_3 = 0 if jh_3 == 0 else 1/(jh_3.real**2 + jh_3.imag**2)

            nres_0 = wres[0]*n_0
            nres_1 = wres[1]*n_1
            nres_2 = wres[2]*n_2
            nres_3 = wres[3]*n_3

            # jhwr = J^H W r: Kronecker row 0 contracted with the normalised,
            # already weighted residual. Only the [0] (XX) entry is retained.
            r_0 = jh_00*nres_0 + jh_01*nres_1 + jh_02*nres_2 + jh_03*nres_3

            gc_0 = gain[0].conjugate()
            drv_00 = -1j*gc_0
            upd_00 = (drv_00*r_0).real

            # jhwj = J^H W J, with the normalisation folded into the weights.
            w_0 = n_0 * w[0]
            w_1 = n_1 * w[1]
            w_2 = n_2 * w[2]
            w_3 = n_3 * w[3]

            j_00 = jh_00.conjugate()
            j_01 = jh_01.conjugate()
            j_02 = jh_02.conjugate()
            j_03 = jh_03.conjugate()

            jhwj_00 = jh_00*w_0*j_00 + jh_01*w_1*j_01 + \
                jh_02*w_2*j_02 + jh_03*w_3*j_03

            return (
                jhj_jhr[0] + jhwj_00.real,
                jhj_jhr[1] + upd_00,
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
