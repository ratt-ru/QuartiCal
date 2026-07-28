# -*- coding: utf-8 -*-
import numpy as np
from numba import njit
from numba.extending import overload
from quartical.utils.numba import (
    coerce_literal,
    JIT_OPTIONS,
    PARALLEL_JIT_OPTIONS
)
from quartical.gains.general.flagging import (
    apply_gain_flags_to_gains,
    apply_param_flags_to_params
)
import quartical.gains.general.factories as factories
from quartical.gains.general.solver_components import build_jhj_jhr_impl
from quartical.gains.general.solver_loop import build_param_solver_impl


def get_identity_params(corr_mode):

    if corr_mode.literal_value in (2, 4):
        return np.zeros((4,), dtype=np.float64)
    elif corr_mode.literal_value == 1:
        return np.zeros((2,), dtype=np.float64)
    else:
        raise ValueError("Unsupported number of correlations.")


@njit(**JIT_OPTIONS)
def delay_and_offset_solver(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    return delay_and_offset_solver_impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        corr_mode
    )


def delay_and_offset_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    raise NotImplementedError


# We actually solve for D' = (D(nu_min + nu_max))/2. This helps avoid
# numerical issues, but requires some scaling of the parameters. Only the delay
# entries (the strided [..., 1::2] slots) are rescaled; the offset entries are
# left untouched. The solver enters this scaled basis before the loop
# (pre_solve) and leaves it afterwards (post_solve); both hooks are
# module-local qcjit closures that fetch their inputs from the standardised
# hook arguments.
@factories.qcjit
def pre_solve(ms_inputs, chain_inputs, meta_inputs):
    active_params = chain_inputs.params[meta_inputs.active_term]
    mid_freq = (ms_inputs.MIN_FREQ + ms_inputs.MAX_FREQ) / 2
    active_params[..., 1::2] *= mid_freq


@factories.qcjit
def post_solve(ms_inputs, chain_inputs, meta_inputs, native_imdry):
    active_params = chain_inputs.params[meta_inputs.active_term]
    mid_freq = (ms_inputs.MIN_FREQ + ms_inputs.MAX_FREQ) / 2
    # Undo rescaling so that quantities are in native units.
    active_params[..., 1::2] /= mid_freq
    native_imdry.jhj[..., 1::2] *= mid_freq ** 2


@overload(delay_and_offset_solver_impl, jit_options=JIT_OPTIONS)
def nb_delay_and_offset_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):

    coerce_literal(nb_delay_and_offset_solver_impl, ["corr_mode"])

    identity_params = get_identity_params(corr_mode)

    # The outer solver loop is shared between parameterised kernels - only the
    # hooks below are specific to delay_and_offset terms. It solves on the
    # parameter grid, supports scalar mode (two parameters per correlation - a
    # delay and an offset), has a referencing stage, and enters/leaves a scaled
    # solver basis (the pre/post-solve stages rescale its delay parameters).
    # The shared loop is inlined into the module-local trampoline below rather
    # than returned directly. This gives delay_and_offset a private on-disk
    # cache namespace - see the cache correctness constraint in
    # solver_components.py.
    shared_impl = build_param_solver_impl(
        pre_solve=pre_solve,
        compute_jhj_jhr=compute_jhj_jhr,
        params_per_corr=2,
        scalar_error_message=None,
        finalize_update=finalize_update,
        numbness=1e9,
        identity_params=identity_params,
        reference_params=reference_params,
        post_solve=post_solve,
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
    # below (the per-term maths) are specific to delay_and_offset. Its residual
    # normalises out amplitude exactly as delay's does. Like delay, it
    # differentiates a frequency-dependent exponent, so it carries the same
    # per-channel coefficient computed by the compute_channel_coeffs hook; that
    # coefficient is the channel_coeffs tuple consumed by the accumulate hook.
    # The offset parameter adds a second (frequency-independent) parameter per
    # correlation, so the accumulator carries a 4-parameter jhj/jhr instead of
    # delay's 2-parameter one. The shared loop is inlined into the module-local
    # trampoline below rather than returned directly. This gives
    # delay_and_offset a private on-disk cache namespace - see the cache
    # correctness constraint in solver_components.py.
    shared_impl = build_jhj_jhr_impl(
        corr_mode=corr_mode,
        row_weights_type=row_weights_type,
        accumulate_jhj_jhr_factory=accumulate_jhj_jhr_factory,
        zero_jhj_jhr_factory=zero_jhj_jhr_factory,
        flush_jhj_jhr_factory=flush_jhj_jhr_factory,
        compute_residual_factory=compute_residual_factory,
        compute_channel_coeffs_factory=compute_channel_coeffs_factory,
        mirror_jhj_factory=mirror_jhj_factory,
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

    if corr_mode.literal_value in (1, 2, 4):
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

            update = native_imdry.update

            update /= 2
            params += update

            n_time, n_freq, n_ant, n_dir, _ = gain.shape

            if dd_term:
                for pd in pinned_directions:
                    params[..., pd, :] = 0

            chan_freq = ms_inputs.CHAN_FREQ
            cf_mid = (ms_inputs.MIN_FREQ + ms_inputs.MAX_FREQ) / 2

            for t in range(n_time):
                for f in range(n_freq):
                    f_m = param_freq_map[f]
                    coeff = 2 * np.pi * (chan_freq[f]/cf_mid - 1)
                    for a in range(n_ant):
                        for d in range(n_dir):

                            d_m = dir_map[d]
                            g = gain[t, f, a, d]
                            fl = gain_flags[t, f, a, d]
                            p = params[t, f_m, a, d_m]

                            if fl == 1:
                                set_identity(g)
                            else:
                                param_to_gain(p, coeff, g)
    else:
        raise ValueError("Unsupported number of correlations.")

    return impl


def param_to_gain_factory(corr_mode):

    if corr_mode.literal_value == 4:
        def impl(params, coeff, gain):
            gain[0] = np.exp(1j * (coeff * params[1] + params[0]))
            gain[3] = np.exp(1j * (coeff * params[3] + params[2]))
    elif corr_mode.literal_value == 2:
        def impl(params, coeff, gain):
            gain[0] = np.exp(1j * (coeff * params[1] + params[0]))
            gain[1] = np.exp(1j * (coeff * params[3] + params[2]))
    elif corr_mode.literal_value == 1:
        def impl(params, coeff, gain):
            gain[0] = np.exp(1j * (coeff * params[1] + params[0]))
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def compute_channel_coeffs_factory(corr_mode):
    """Produce the per-channel delay coefficient tuple for the shared loop.

    delay_and_offset solves for a frequency-dependent exponent (the delay
    parameter) plus a frequency-independent offset. Differentiating the model
    with respect to the delay parameter introduces the same per-channel
    coefficient as the delay term, coeff = 2*pi*(chan_freq[f]/cf_mid - 1),
    where cf_mid is the midpoint of the band (the same rescaling the solver
    applies to the delay parameters). The compute_channel_coeffs hook computes
    this once per channel and returns it as a single-element flat tuple
    (coeff,) which is the channel_coeffs tuple passed to the accumulate hook.
    """

    def impl(ms_inputs, meta_inputs, f):
        chan_freq = ms_inputs.CHAN_FREQ
        cf_mid = (ms_inputs.MIN_FREQ + ms_inputs.MAX_FREQ) / 2
        coeff = 2 * np.pi * (chan_freq[f] / cf_mid - 1)
        return (coeff,)

    return factories.qcjit(impl)


def compute_residual_factory(corr_mode):
    """Produce the amplitude-normalised residual for a delay_and_offset term.

    The residual is identical to delay's (and phase's): it normalises out
    amplitude before forming the residual. The per-correlation factor is
    normf_i = |v_i| / |r_i| (zero where r_i is zero, matching
    absv1_idiv_absv2), and the residual is r_i*normf_i - v_i, returned as the
    per-correlation residual tuple.
    """

    tuple_normf = factories.tuple_normf_factory(corr_mode)

    if corr_mode.literal_value == 4:
        def impl(r, v):
            f0, f1, f2, f3 = tuple_normf(v, r)
            return (
                r[0]*f0 - v[0],
                r[1]*f1 - v[1],
                r[2]*f2 - v[2],
                r[3]*f3 - v[3],
            )
    elif corr_mode.literal_value == 2:
        def impl(r, v):
            f0, f1 = tuple_normf(v, r)
            return (
                r[0]*f0 - v[0],
                r[1]*f1 - v[1],
            )
    elif corr_mode.literal_value == 1:
        def impl(r, v):
            f0, = tuple_normf(v, r)
            return (
                r[0]*f0 - v[0],
            )
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def zero_jhj_jhr_factory(corr_mode):
    """Produce the zero jhj/jhr accumulator tuple for a given corr mode.

    The accumulator is a single flat tuple holding the upper triangle of the
    (n_param, n_param) real jhj element in row-major order followed by the
    (real) jhr entries. For the 2 and 4 correlation cases n_param is 4, giving
    10 upper-triangle jhj entries and 4 jhr entries (14 slots). For the single
    correlation case n_param is 2, giving 3 upper-triangle jhj entries and 2
    jhr entries (5 slots). The reference element is a jhr slice, whose dtype is
    real, so every accumulator value is a real zero.
    """

    if corr_mode.literal_value in (2, 4):
        def impl(invec):
            z = invec[0]*0
            return (
                z, z, z, z,
                z, z, z, z, z, z, z, z, z, z,
            )
    elif corr_mode.literal_value == 1:
        def impl(invec):
            z = invec[0]*0
            return z, z, z, z, z
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def flush_jhj_jhr_factory(corr_mode):
    """Add a register-accumulated jhj/jhr accumulator into the arrays.

    The accumulator holds the upper triangle of the (n_param, n_param) jhj
    element in row-major order followed by the jhr entries. The lower triangle
    is filled in by mirror_jhj once per solution interval. For the 2
    correlation case the cross-correlation jhj entries (0, 2), (0, 3), (1, 2),
    (1, 3) are always zero, so mirroring them is a harmless no-op there.
    """

    if corr_mode.literal_value in (2, 4):
        def impl(jhj, jhr, jhj_jhr):

            jhj[0, 0] += jhj_jhr[0]
            jhj[0, 1] += jhj_jhr[1]
            jhj[0, 2] += jhj_jhr[2]
            jhj[0, 3] += jhj_jhr[3]
            jhj[1, 1] += jhj_jhr[4]
            jhj[1, 2] += jhj_jhr[5]
            jhj[1, 3] += jhj_jhr[6]
            jhj[2, 2] += jhj_jhr[7]
            jhj[2, 3] += jhj_jhr[8]
            jhj[3, 3] += jhj_jhr[9]

            jhr[0] += jhj_jhr[10]
            jhr[1] += jhj_jhr[11]
            jhr[2] += jhj_jhr[12]
            jhr[3] += jhj_jhr[13]
    elif corr_mode.literal_value == 1:
        def impl(jhj, jhr, jhj_jhr):

            jhj[0, 0] += jhj_jhr[0]
            jhj[0, 1] += jhj_jhr[1]
            jhj[1, 1] += jhj_jhr[2]

            jhr[0] += jhj_jhr[3]
            jhr[1] += jhj_jhr[4]
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def mirror_jhj_factory(corr_mode):
    """Fill in the lower triangle of the per-interval (n_param, n_param) jhj.

    Accumulation in accumulate_jhj_jhr only writes the upper triangle of
    each real, symmetric jhj element. The lower triangle is a straight copy
    (jhj is real) done once per solution interval rather than once per
    visibility. For the 2 and 4 correlation cases jhj is (4, 4); for the single
    correlation case jhj is (2, 2).
    """

    if corr_mode.literal_value in (2, 4):
        def impl(jhj_tifi):
            n_ant, n_gdir = jhj_tifi.shape[:2]
            for a in range(n_ant):
                for d in range(n_gdir):
                    jhj = jhj_tifi[a, d]
                    jhj[1, 0] = jhj[0, 1]
                    jhj[2, 0] = jhj[0, 2]
                    jhj[2, 1] = jhj[1, 2]
                    jhj[3, 0] = jhj[0, 3]
                    jhj[3, 1] = jhj[1, 3]
                    jhj[3, 2] = jhj[2, 3]
    elif corr_mode.literal_value == 1:
        def impl(jhj_tifi):
            n_ant, n_gdir = jhj_tifi.shape[:2]
            for a in range(n_ant):
                for d in range(n_gdir):
                    jhj_tifi[a, d, 1, 0] = jhj_tifi[a, d, 0, 1]
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def accumulate_jhj_jhr_factory(corr_mode):
    """Accumulate a jhj/jhr element into a register-resident accumulator.

    All inputs and the returned accumulator are tuples (register-resident
    values) - the accumulator is only flushed to memory by flush_jhj_jhr. The
    accumulator is a single flat tuple (the upper triangle of the real jhj
    element followed by jhr entries - see zero_jhj_jhr_factory).

    The signature follows the unified accumulate_jhj_jhr contract of the shared
    accumulation loop (see solver_components.py). The chain rule uses the
    active-term gain (drv = -1j*conj(g)), so the gain argument is consumed. The
    channel_coeffs argument is the single-element tuple (coeff,) produced by
    the compute_channel_coeffs hook: coeff is at channel_coeffs[0] in every
    corr mode. The offset parameter is frequency independent (its Jacobian
    carries no coeff), while the delay parameter's Jacobian carries the coeff.
    This yields the interleaved (offset, delay) accumulator layout, with jhj
    entries scaled by 1, coeff or coeff**2 depending on which parameter pair
    the entry couples. The normalisation applied to the residual is recomputed
    here from the operators rather than being passed in from compute_residual.
    """

    if corr_mode.literal_value == 4:
        def impl(lop, rop, w, gain, channel_coeffs, wres, jhj_jhr):

            coeff = channel_coeffs[0]
            coeffsq = coeff*coeff

            lop_0, lop_1, lop_2, lop_3 = lop[0], lop[1], lop[2], lop[3]
            rop_0, rop_1, rop_2, rop_3 = rop[0], rop[1], rop[2], rop[3]

            # Normalisation factor: 1/|lop @ rop|^2 on the diagonal, with the
            # same zero guard as the array kernel. The off-diagonal residual
            # entries carry zero weight, so only the diagonal of the (2, 2)
            # product is required.
            m0 = lop_0*rop_0 + lop_1*rop_2
            m3 = lop_2*rop_1 + lop_3*rop_3
            n_0 = 0 if m0 == 0 else 1/(m0.real**2 + m0.imag**2)
            n_3 = 0 if m3 == 0 else 1/(m3.real**2 + m3.imag**2)

            # jhwr element: lop @ (diag(normalised residual) @ rop), keeping
            # the diagonal. The off-diagonal residual entries are dropped by
            # only forming res_0 and res_3 (i.e. zero weight off-diagonal).
            res_0 = wres[0]*n_0
            res_3 = wres[3]*n_3
            o0 = res_0*rop_0
            o1 = res_0*rop_1
            o2 = res_3*rop_2
            o3 = res_3*rop_3
            r_0 = lop_0*o0 + lop_1*o2
            r_3 = lop_2*o1 + lop_3*o3

            g_3 = gain[3]
            gc_0 = gain[0].conjugate()
            gc_3 = gain[3].conjugate()

            drv_00 = -1j*gc_0
            drv_13 = -1j*gc_3

            upd_00 = (drv_00*r_0).real
            upd_11 = (drv_13*r_3).real

            # jhwj element: the normalisation is folded into the weights.
            # NOTE: rop is effectively transposed (rop_1 <-> rop_2) relative
            # to lop, matching the row-major kronecker convention.
            w_0 = n_0 * w[0]
            w_3 = n_3 * w[3]

            jh_00 = lop_0 * rop_0
            jh_03 = lop_1 * rop_2

            j_00 = jh_00.conjugate()
            j_03 = jh_03.conjugate()

            jh_30 = lop_2 * rop_1
            jh_33 = lop_3 * rop_3

            j_30 = jh_30.conjugate()
            j_33 = jh_33.conjugate()

            jhwj_00 = jh_00*w_0*j_00 + jh_03*w_3*j_03
            jhwj_03 = jh_00*w_0*j_30 + jh_03*w_3*j_33
            jhwj_33 = jh_30*w_0*j_30 + jh_33*w_3*j_33

            tmp_0 = jhwj_00.real
            tmp_1 = (jhwj_03*gc_0*g_3).real
            tmp_2 = jhwj_33.real

            # jhj upper triangle in row-major order followed by jhr entries
            # in (offset, delay) order per correlation: the offset Jacobian
            # carries no coeff, the delay Jacobian carries coeff (see
            # zero_jhj_jhr).
            return (
                jhj_jhr[0] + tmp_0,
                jhj_jhr[1] + coeff*tmp_0,
                jhj_jhr[2] + tmp_1,
                jhj_jhr[3] + coeff*tmp_1,
                jhj_jhr[4] + coeffsq*tmp_0,
                jhj_jhr[5] + coeff*tmp_1,
                jhj_jhr[6] + coeffsq*tmp_1,
                jhj_jhr[7] + tmp_2,
                jhj_jhr[8] + coeff*tmp_2,
                jhj_jhr[9] + coeffsq*tmp_2,
                jhj_jhr[10] + upd_00,
                jhj_jhr[11] + coeff*upd_00,
                jhj_jhr[12] + upd_11,
                jhj_jhr[13] + coeff*upd_11,
            )

    elif corr_mode.literal_value == 2:
        def impl(lop, rop, w, gain, channel_coeffs, wres, jhj_jhr):

            coeff = channel_coeffs[0]
            coeffsq = coeff*coeff

            rop_0, rop_1 = rop[0], rop[1]

            # Normalisation factor: 1/|rop|^2 per corr, zero-guarded.
            n_0 = 0 if rop_0 == 0 else 1/(rop_0.real**2 + rop_0.imag**2)
            n_1 = 0 if rop_1 == 0 else 1/(rop_1.real**2 + rop_1.imag**2)

            # jhwr element (diagonal only).
            r_0 = wres[0]*n_0*rop_0
            r_1 = wres[1]*n_1*rop_1

            gc_0 = gain[0].conjugate()
            gc_1 = gain[1].conjugate()

            drv_00 = -1j*gc_0
            drv_23 = -1j*gc_1

            upd_00 = (drv_00*r_0).real
            upd_11 = (drv_23*r_1).real

            # jhwj element (block diagonal, real). The cross-correlation jhj
            # entries (jhj_jhr[2], jhj_jhr[3], jhj_jhr[5], jhj_jhr[6]) are
            # never set, so they stay at the zero the accumulator was
            # initialised with.
            jhj_00 = (rop_0*n_0*w[0]*rop_0.conjugate()).real
            jhj_11 = (rop_1*n_1*w[1]*rop_1.conjugate()).real

            return (
                jhj_jhr[0] + jhj_00,
                jhj_jhr[1] + coeff*jhj_00,
                jhj_jhr[2],
                jhj_jhr[3],
                jhj_jhr[4] + coeffsq*jhj_00,
                jhj_jhr[5],
                jhj_jhr[6],
                jhj_jhr[7] + jhj_11,
                jhj_jhr[8] + coeff*jhj_11,
                jhj_jhr[9] + coeffsq*jhj_11,
                jhj_jhr[10] + upd_00,
                jhj_jhr[11] + coeff*upd_00,
                jhj_jhr[12] + upd_11,
                jhj_jhr[13] + coeff*upd_11,
            )

    elif corr_mode.literal_value == 1:
        def impl(lop, rop, w, gain, channel_coeffs, wres, jhj_jhr):

            coeff = channel_coeffs[0]
            coeffsq = coeff*coeff

            rop_0 = rop[0]

            # Normalisation factor: 1/|rop|^2, zero-guarded.
            n_0 = 0 if rop_0 == 0 else 1/(rop_0.real**2 + rop_0.imag**2)

            # jhwr element.
            r_0 = wres[0]*n_0*rop_0

            gc_0 = gain[0].conjugate()
            drv_00 = -1j*gc_0
            upd_00 = (drv_00*r_0).real

            # jhwj element (real).
            jhj_00 = (rop_0*n_0*w[0]*rop_0.conjugate()).real

            return (
                jhj_jhr[0] + jhj_00,
                jhj_jhr[1] + coeff*jhj_00,
                jhj_jhr[2] + coeffsq*jhj_00,
                jhj_jhr[3] + upd_00,
                jhj_jhr[4] + coeff*upd_00,
            )
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


@njit(**JIT_OPTIONS)
def delay_and_offset_params_to_gains(
    params,
    gains,
    chan_freq,
    min_freq,
    max_freq,
    param_freq_map,
    rescaled=False
):

    n_time, n_freq, n_ant, n_dir, n_corr = gains.shape

    cf_mid = (min_freq + max_freq) / 2

    if rescaled:
        offset = 1.0
        denom = cf_mid
    else:
        offset = cf_mid
        denom = 1.0

    for t in range(n_time):
        for f in range(n_freq):
            f_m = param_freq_map[f]
            coeff = 2 * np.pi * (chan_freq[f] / denom - offset)
            for a in range(n_ant):
                for d in range(n_dir):

                    g = gains[t, f, a, d]
                    p = params[t, f_m, a, d]

                    g[0] = np.exp(1j*(coeff*p[1] + p[0]))

                    if n_corr > 1:
                        g[-1] = np.exp(1j*(coeff*p[3] + p[2]))


@njit(**JIT_OPTIONS)
def reference_params(ms_inputs, mapping_inputs, chain_inputs, meta_inputs):

    chan_freq = ms_inputs.CHAN_FREQ

    active_term = meta_inputs.active_term
    ref_ant = meta_inputs.reference_antenna

    gains = chain_inputs.gains[active_term]
    gain_flags = chain_inputs.gain_flags[active_term]
    params = chain_inputs.params[active_term]
    param_flags = chain_inputs.param_flags[active_term]

    param_freq_map = mapping_inputs.param_freq_maps[active_term]

    n_ti, n_fi, n_ant, n_dir, n_corr = params.shape

    ref_params = params[:, :, ref_ant: ref_ant + 1, :, :].copy()

    for t in range(n_ti):
        for f in range(n_fi):
            for a in range(n_ant):
                for d in range(n_dir):

                    p = params[t, f, a, d]
                    rp = ref_params[t, f, 0, d]

                    if param_flags[t, f, a, d] == 1:
                        continue
                    else:
                        p -= rp

    delay_and_offset_params_to_gains(
        params,
        gains,
        chan_freq,
        ms_inputs.MIN_FREQ,
        ms_inputs.MAX_FREQ,
        param_freq_map,
        rescaled=True
    )

    # Referencing may move flagged gains/params from identity.
    apply_param_flags_to_params(param_flags, params, 0)
    apply_gain_flags_to_gains(gain_flags, gains)
