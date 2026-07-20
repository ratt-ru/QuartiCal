# -*- coding: utf-8 -*-
import numpy as np
from numba import njit
from numba.extending import overload
from quartical.utils.numba import (coerce_literal,
                                   JIT_OPTIONS,
                                   PARALLEL_JIT_OPTIONS)
import quartical.gains.general.factories as factories
from quartical.gains.general.accumulation import build_jhj_jhr_impl
from quartical.gains.general.solver_loop import build_param_solver_impl
from quartical.gains.general.solver_ops import compute_update  # noqa


def get_identity_params(corr_mode):

    if corr_mode.literal_value in (2, 4):
        return np.ones((2,), dtype=np.float64)
    elif corr_mode.literal_value == 1:
        return np.ones((1,), dtype=np.float64)
    else:
        raise ValueError("Unsupported number of correlations.")


@njit(**JIT_OPTIONS)
def amplitude_solver(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    return amplitude_solver_impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        corr_mode
    )


def amplitude_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    raise NotImplementedError


@overload(amplitude_solver_impl, jit_options=JIT_OPTIONS)
def nb_amplitude_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):

    coerce_literal(nb_amplitude_solver_impl, ["corr_mode"])

    identity_params = get_identity_params(corr_mode)

    # The outer solver loop is shared between parameterised kernels - only the
    # hooks below are specific to amplitude terms. Amplitude solves on the gain
    # grid, supports scalar mode (one parameter per correlation), needs no
    # referencing stage, and enters/leaves no scaled solver basis (no
    # pre/post-solve stages). The shared loop is inlined into the module-local
    # trampoline below rather than returned directly. This gives amplitude a
    # private on-disk cache namespace - see the cache correctness constraint in
    # accumulation.py.
    shared_impl = build_param_solver_impl(
        solve_on_param_grid=False,
        pre_solve=None,
        compute_jhj_jhr=compute_jhj_jhr,
        params_per_corr=1,
        scalar_error_message=None,
        finalize_update=finalize_update,
        numbness=1e-6,
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
    # below (the per-term maths) are specific to amplitude terms. Amplitude's
    # residual fully normalises out the model amplitude before weighting, so
    # the elem consumes no auxiliary values (n_resid_aux is 0) and there is no
    # stage hook. The elem also ignores the active-term gain: amplitude's
    # parameter-to-gain map is the identity, so its chain-rule derivative is
    # one and the gain never enters the maths.
    # The shared loop is inlined into the module-local trampoline below
    # rather than returned directly. This gives amplitude a private on-disk
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
        mirror_factory=mirror_jhj_factory,
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
                            p[:] = 1
                            set_identity(g)
                        else:
                            upd /= 2
                            p += upd
                            param_to_gain(p, g)

    return impl


def param_to_gain_factory(corr_mode):

    if corr_mode.literal_value == 4:
        def impl(params, gain):
            gain[0] = params[0]
            gain[3] = params[1]
    elif corr_mode.literal_value == 2:
        def impl(params, gain):
            gain[0] = params[0]
            gain[1] = params[1]
    elif corr_mode.literal_value == 1:
        def impl(params, gain):
            gain[0] = params[0]
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def resid_factory(corr_mode):
    """Produce the amplitude-normalised residual for an amplitude term.

    The amplitude solver normalises the data to the model amplitude before
    forming the residual: the per-correlation factor is normf_i = |r_i| / |v_i|
    (zero where v_i is zero, matching absv1_idiv_absv2), and the residual is
    normf_i*v_i - v_i. This reproduces the original array-buffer kernel exactly
    (absv1_idiv_absv2(r, v, r); r *= v; r -= v). The elem hook consumes no
    auxiliary values, so n_resid_aux is 0 and the returned flat tuple holds
    only the residual values.
    """

    tuple_normf = factories.tuple_normf_factory(corr_mode)

    if corr_mode.literal_value == 4:
        def impl(r, v):
            f0, f1, f2, f3 = tuple_normf(r, v)
            return (
                f0*v[0] - v[0],
                f1*v[1] - v[1],
                f2*v[2] - v[2],
                f3*v[3] - v[3],
            )
    elif corr_mode.literal_value == 2:
        def impl(r, v):
            f0, f1 = tuple_normf(r, v)
            return (
                f0*v[0] - v[0],
                f1*v[1] - v[1],
            )
    elif corr_mode.literal_value == 1:
        def impl(r, v):
            f0, = tuple_normf(r, v)
            return (
                f0*v[0] - v[0],
            )
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def jhwj_jhwr_zeros_factory(corr_mode):
    """Produce the zero jhr/jhj accumulator tuple for a given corr mode.

    The accumulator is a single flat tuple holding the (real) jhr entries
    followed by the upper triangle of the (n_param, n_param) real jhj element
    in row-major order. For the 2 and 4 correlation cases n_param is 2, giving
    (jhr0, jhr1, jhj00, jhj01, jhj11); for the single correlation case n_param
    is 1, giving (jhr0, jhj00). The reference element is a jhr slice, whose
    dtype is real, so every accumulator value is a real zero.
    """

    if corr_mode.literal_value in (2, 4):
        def impl(invec):
            z = invec[0]*0
            return z, z, z, z, z
    elif corr_mode.literal_value == 1:
        def impl(invec):
            z = invec[0]*0
            return z, z
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def flush_jhwj_jhwr_factory(corr_mode):
    """Add a register-accumulated jhr/jhj accumulator into the arrays.

    For the 2 and 4 correlation cases only the upper triangle of the (2, 2)
    jhj element is accumulated (jhj[0, 1]); the lower triangle is filled in by
    mirror_jhj once per solution interval. In the 2 correlation case jhj[0, 1]
    is always zero, so the mirror is effectively a no-op there.
    """

    if corr_mode.literal_value in (2, 4):
        def impl(jhr, jhj, acc):

            jhr[0] += acc[0]
            jhr[1] += acc[1]

            jhj[0, 0] += acc[2]
            jhj[0, 1] += acc[3]
            jhj[1, 1] += acc[4]
    elif corr_mode.literal_value == 1:
        def impl(jhr, jhj, acc):

            jhr[0] += acc[0]

            jhj[0, 0] += acc[1]
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def mirror_jhj_factory(corr_mode):
    """Fill in the lower triangle of the per-interval (n_param, n_param) jhj.

    Accumulation in compute_jhwj_jhwr_elem only writes the upper triangle of
    each real, symmetric jhj element. The lower triangle is a straight copy
    (jhj is real) done once per solution interval rather than once per
    visibility. This is a no-op in the single parameter (single correlation)
    case, where jhj is (1, 1).
    """

    if corr_mode.literal_value in (2, 4):
        def impl(jhj_tifi):
            n_ant, n_gdir = jhj_tifi.shape[:2]
            for a in range(n_ant):
                for d in range(n_gdir):
                    jhj_tifi[a, d, 1, 0] = jhj_tifi[a, d, 0, 1]
    else:
        def impl(jhj_tifi):
            pass

    return factories.qcjit(impl)


def compute_jhwj_jhwr_elem_factory(corr_mode):
    """Accumulate a jhr/jhj element into a register-resident accumulator.

    All inputs and the returned accumulator are tuples (register-resident
    values) - the accumulator is only flushed to memory by flush_jhwj_jhwr.
    The accumulator is a single flat tuple (jhr entries followed by the upper
    triangle of the real jhj element - see jhwj_jhwr_zeros_factory).

    The signature follows the unified elem contract of the shared accumulation
    loop (see accumulation.py). Amplitude's parameter-to-gain map is the
    identity, so its chain-rule derivative is one and the gain argument is not
    consumed. The aux argument is empty (the residual hook fully normalises the
    residual, so there are no auxiliary values) and is likewise ignored. The
    residual arrives already normalised and weighted, so this elem applies no
    further normalisation - it only forms the operator products.
    """

    if corr_mode.literal_value == 4:
        def impl(lop, rop, w, gain, aux, res, acc):

            lop_0, lop_1, lop_2, lop_3 = lop[0], lop[1], lop[2], lop[3]
            rop_0, rop_1, rop_2, rop_3 = rop[0], rop[1], rop[2], rop[3]

            # jhwr element: lop @ (diag(residual) @ rop), keeping the diagonal.
            # The off-diagonal residual entries carry zero weight (the array
            # kernel zeroed res[1]/res[2]), so only res[0] and res[3] are used.
            res_0 = res[0]
            res_3 = res[3]
            o0 = res_0*rop_0
            o1 = res_0*rop_1
            o2 = res_3*rop_2
            o3 = res_3*rop_3
            r_0 = lop_0*o0 + lop_1*o2
            r_3 = lop_2*o1 + lop_3*o3

            # jhwj element (upper triangle).
            # NOTE: rop is effectively transposed (rop_1 <-> rop_2) relative
            # to lop, matching the row-major kronecker convention.
            jh_00 = lop_0 * rop_0
            jh_03 = lop_1 * rop_2

            j_00 = jh_00.conjugate()
            j_03 = jh_03.conjugate()

            jh_30 = lop_2 * rop_1
            jh_33 = lop_3 * rop_3

            j_30 = jh_30.conjugate()
            j_33 = jh_33.conjugate()

            w_0 = w[0]
            w_3 = w[3]

            jhwj_00 = jh_00*w_0*j_00 + jh_03*w_3*j_03
            jhwj_03 = jh_00*w_0*j_30 + jh_03*w_3*j_33
            jhwj_33 = jh_30*w_0*j_30 + jh_33*w_3*j_33

            return (
                acc[0] + r_0.real,
                acc[1] + r_3.real,
                acc[2] + jhwj_00.real,
                acc[3] + jhwj_03.real,
                acc[4] + jhwj_33.real,
            )

    elif corr_mode.literal_value == 2:
        def impl(lop, rop, w, gain, aux, res, acc):

            rop_0, rop_1 = rop[0], rop[1]

            # jhwr element (diagonal).
            r_0 = res[0]*rop_0
            r_1 = res[1]*rop_1

            # jhwj element (diagonal, real). The off-diagonal (acc[3]) is left
            # untouched, matching the array kernel which never sets it.
            jhj_00 = (rop_0*w[0]*rop_0.conjugate()).real
            jhj_11 = (rop_1*w[1]*rop_1.conjugate()).real

            return (
                acc[0] + r_0.real,
                acc[1] + r_1.real,
                acc[2] + jhj_00,
                acc[3],
                acc[4] + jhj_11,
            )

    elif corr_mode.literal_value == 1:
        def impl(lop, rop, w, gain, aux, res, acc):

            rop_0 = rop[0]

            # jhwr element.
            r_0 = res[0]*rop_0

            # jhwj element (real).
            jhj_00 = (rop_0*w[0]*rop_0.conjugate()).real

            return (
                acc[0] + r_0.real,
                acc[1] + jhj_00,
            )
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


@njit(**JIT_OPTIONS)
def amplitude_params_to_gains(
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

                    g[0] = p[0]

                    if n_corr > 1:
                        g[-1] = p[-1]
