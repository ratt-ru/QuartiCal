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
from quartical.gains.general.parameters import get_identity_params
from quartical.gains.general.residuals import standard_residual_factory
from quartical.gains.general.accumulator import (
    triangular_accumulator_factories
)


PARAMS_PER_CORR = None

accumulator = triangular_accumulator_factories(PARAMS_PER_CORR)


@njit(**JIT_OPTIONS)
def rm_solver(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    return rm_solver_impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        corr_mode
    )


def rm_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    raise NotImplementedError


@overload(rm_solver_impl, jit_options=JIT_OPTIONS)
def nb_rm_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):

    coerce_literal(nb_rm_solver_impl, ["corr_mode"])

    identity_params = get_identity_params(corr_mode, PARAMS_PER_CORR)

    # The outer solver loop is shared between parameterised kernels - only the
    # hooks below are specific to rotation measure terms. Rotation measure
    # solves on the parameter grid, does not support scalar mode, needs no
    # referencing stage, and enters/leaves no scaled solver basis (no
    # pre/post-solve stages).
    # Inlined into the trampoline below for a private cache namespace.
    shared_impl = build_param_solver_impl(
        pre_solve=None,
        compute_jhj_jhr=compute_jhj_jhr,
        params_per_corr=PARAMS_PER_CORR,
        scalar_error_message=(
            "Scalar mode not supported for rotation measure terms."
        ),
        finalize_update=finalize_update,
        # NB: 1e9 disables the trend (divergence) flagging, and enabling it for
        # this term can cause problems - the accumulate hook below reads its
        # derivative out of the per-channel gain, which hard flagging
        # overwrites with the identity while the coarser parameter interval
        # keeps its non-zero rm. See the linearisation-point note in
        # solver_components.py.
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
    # below (the per-term maths) are specific to rotation measure terms. Like
    # rotation, it constrains a real angle, so it takes the standard residual
    # (r - v). Unlike rotation, the rotation angle is
    # frequency dependent (beta = lambda_sq*rm), so a per-channel lambda_sq
    # coefficient is supplied by the compute_channel_coeffs hook as the
    # channel_coeffs tuple consumed by the accumulate hook. Rotation measure
    # solves a single parameter, so its jhj is (1, 1) and the mirror hook is a
    # no-op.
    # Inlined into the trampoline below for a private cache namespace.
    shared_impl = build_jhj_jhr_impl(
        corr_mode=corr_mode,
        row_weights_type=row_weights_type,
        accumulate_jhj_jhr_factory=accumulate_jhj_jhr_factory,
        zero_jhj_jhr_factory=accumulator.zero,
        flush_jhj_jhr_factory=accumulator.flush,
        compute_residual_factory=standard_residual_factory,
        compute_channel_coeffs_factory=compute_channel_coeffs_factory,
        mirror_jhj_factory=accumulator.mirror,
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

            param_freq_map = mapping_inputs.param_freq_maps[active_term]
            dir_map = mapping_inputs.dir_maps[active_term]

            # The per-channel lambda squared drives the frequency dependence
            # of the rotation angle when mapping parameters back onto gains.
            # This is negligible work relative to the loop below.
            chan_freqs = ms_inputs.CHAN_FREQ
            lambda_sq = (299792458/chan_freqs)**2

            update = native_imdry.update

            update /= 2
            params += update

            n_time, n_freq, n_ant, n_dir, _ = gain.shape

            if dd_term:
                for pd in pinned_directions:
                    params[..., pd, :] = 0

            for t in range(n_time):
                for f in range(n_freq):
                    lsq = lambda_sq[f]
                    for a in range(n_ant):
                        for d in range(n_dir):

                            f_m = param_freq_map[f]
                            d_m = dir_map[d]
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


def compute_channel_coeffs_factory(corr_mode):
    """Produce the per-channel lambda squared coefficient for the shared loop.

    Rotation measure's rotation angle is frequency dependent,
    beta = lambda_sq*rm with lambda_sq = (c/chan_freq)**2, so differentiating
    the model with respect to the parameter introduces the per-channel factor
    lambda_sq. The compute_channel_coeffs hook computes this once per channel
    and returns it as a single-element flat tuple (lsq,) which is the
    channel_coeffs tuple passed to the accumulate hook.
    """

    def impl(ms_inputs, meta_inputs, f):
        chan_freq = ms_inputs.CHAN_FREQ
        lsq = (299792458/chan_freq[f])**2
        return (lsq,)

    return factories.qcjit(impl)


def accumulate_jhj_jhr_factory(corr_mode):
    """Accumulate a jhj/jhr element into a register-resident accumulator.

    All inputs and the returned accumulator are tuples (register-resident
    values) - the accumulator is only flushed to memory by flush_jhj_jhr.
    The accumulator is a flat tuple (jhj00, jhr0) - see
    general/accumulator.py.

    The signature follows the unified accumulate_jhj_jhr contract of the shared
    accumulation loop (see solver_components.py). This is rotation's accumulate
    hook with the per-channel lambda squared factor folded into the derivative.
    The active-term gain IS the rotation matrix [cos, -sin; sin, cos]
    (row-major XX, XY, YX, YY) with argument beta = lambda_sq*rm, so cos_beta =
    gain[0].real and sin_beta = gain[2].real are read straight out of the gain
    tuple rather than costing two transcendental calls per visibility. The
    derivative of the model with respect to rm carries the extra lambda_sq
    factor from the chain rule; lambda_sq is supplied per channel by the
    compute_channel_coeffs hook as channel_coeffs[0].

    Reading the gain means this hook linearises about the gain rather than
    about the parameters. The two agree wherever the solve can reach, but not
    by construction: set_identity overwrites a hard-flagged gain element with
    the identity, leaving cos_beta = 1 and sin_beta = 0 whatever the parameters
    hold. Rotation measure is one of the terms with no structural protection
    here, since its gain is evaluated per channel while its parameter is solved
    on a coarser frequency grid - see the linearisation-point note in
    solver_components.py.

    No (4, 4) kronecker temp is formed: the four column contractions dhjh_j of
    a_kron_bt(lop, rop) are inlined symbolically from lop/rop. jhr is
    dh . (lop @ wres @ rop) and jhj sums w_j * |dhjh_j|^2 over the four
    correlations; both carry the lambda_sq (jhr) and lambda_sq**2 (jhj) factors
    through dh.
    """

    tuple_v1_mul_v2 = factories.tuple_v1_mul_v2_factory(corr_mode)

    if corr_mode.literal_value == 4:
        def impl(lop, rop, w, gain, channel_coeffs, wres, jhj_jhr):

            lsq = channel_coeffs[0]

            lop_0, lop_1, lop_2, lop_3 = lop[0], lop[1], lop[2], lop[3]
            rop_0, rop_1, rop_2, rop_3 = rop[0], rop[1], rop[2], rop[3]
            w_0, w_1, w_2, w_3 = w[0], w[1], w[2], w[3]

            # jhwr element: r = lop @ (wres @ rop), where wres is the weighted
            # residual.
            r_0, r_1, r_2, r_3 = tuple_v1_mul_v2(
                lop, tuple_v1_mul_v2(wres, rop)
            )

            # Derivative of the rotation matrix wrt rm, read straight from the
            # gain and scaled by the per-channel lambda squared (chain rule):
            # d/drm [cos, -sin; sin, cos](lambda_sq*rm) =
            #   lambda_sq*[-sin, -cos; cos, -sin] (row-major XX, XY, YX, YY).
            cos_beta = gain[0].real
            sin_beta = gain[2].real

            dh_0 = -lsq*sin_beta
            dh_1 = -lsq*cos_beta
            dh_2 = lsq*cos_beta
            dh_3 = -lsq*sin_beta

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
                jhj_jhr[0] + jhj_00,
                jhj_jhr[1] + upd,
            )

    else:
        raise ValueError("Rotation measure can only be solved for with four "
                         "correlation data.")

    return factories.qcjit(impl)


@njit(**JIT_OPTIONS)
def rm_params_to_gains(
    params,
    gains,
    lambda_sq,
    param_freq_map
):

    n_time, n_freq, n_ant, n_dir, n_corr = gains.shape

    for t in range(n_time):
        for f in range(n_freq):
            lsq = lambda_sq[f]
            f_m = param_freq_map[f]
            for a in range(n_ant):
                for d in range(n_dir):

                    g = gains[t, f, a, d]
                    rm = params[t, f_m, a, d, 0]

                    beta = lsq*rm

                    cos_beta = np.cos(beta)
                    sin_beta = np.sin(beta)

                    g[0] = cos_beta
                    g[1] = -sin_beta
                    g[2] = sin_beta
                    g[3] = cos_beta
