# -*- coding: utf-8 -*-
import numpy as np
from numba import njit
from numba.extending import overload
from quartical.utils.numba import (coerce_literal,
                                   JIT_OPTIONS,
                                   PARALLEL_JIT_OPTIONS)
from quartical.gains.general.flagging import (apply_gain_flags_to_gains,
                                              apply_param_flags_to_params)
import quartical.gains.general.factories as factories
from quartical.gains.general.solver_components import build_jhj_jhr_impl
from quartical.gains.general.solver_loop import build_param_solver_impl
from quartical.gains.general.parameters import get_identity_params
from quartical.gains.general.residuals import phase_only_residual_factory
from quartical.gains.general.accumulator import (
    triangular_accumulator_factories
)


PARAMS_PER_CORR = 2

accumulator = triangular_accumulator_factories(PARAMS_PER_CORR)


@njit(**JIT_OPTIONS)
def tec_and_offset_solver(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    return tec_and_offset_solver_impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        corr_mode
    )


def tec_and_offset_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    raise NotImplementedError


# The delay/tec family enters a scaled solver basis before the loop and leaves
# it afterwards; these two module-local qcjit hooks fetch their inputs from the
# standardised hook arguments and apply (respectively undo) the parameter
# rescaling and the zero-mean offset correction.
@factories.qcjit
def pre_solve(ms_inputs, chain_inputs, meta_inputs):
    active_params = chain_inputs.params[meta_inputs.active_term]
    active_param_flags = chain_inputs.param_flags[meta_inputs.active_term]

    # We actually solve for TEC' = TEC/bandwidth. This helps avoid
    # numerical issues, but requires some scaling of the parameters.
    min_freq = ms_inputs.MIN_FREQ
    max_freq = ms_inputs.MAX_FREQ
    bandwidth = max_freq - min_freq
    active_params[..., 1::2] /= bandwidth

    # This alters the offset parameter to be consistent with the zero mean
    # corrections used in the solver. QuartiCal now removes this factor
    # when returning from this solver.
    apply_zero_mean_correction(
        min_freq, max_freq, active_params, active_param_flags
    )


@factories.qcjit
def post_solve(ms_inputs, chain_inputs, meta_inputs, native_imdry):
    active_params = chain_inputs.params[meta_inputs.active_term]
    active_param_flags = chain_inputs.param_flags[meta_inputs.active_term]
    min_freq = ms_inputs.MIN_FREQ
    max_freq = ms_inputs.MAX_FREQ
    bandwidth = max_freq - min_freq

    # Undo rescaling so that quantities are in native units.
    active_params[..., 1::2] *= bandwidth
    native_imdry.jhj[..., 1::2, 1::2] /= bandwidth ** 2

    # This alters the offset parameter to be consistent with the zero mean
    # corrections used in the solver. QuartiCal now removes this factor
    # when returning from this solver.
    apply_zero_mean_correction(
        min_freq, max_freq, active_params, active_param_flags, inverse=True
    )


@overload(tec_and_offset_solver_impl, jit_options=JIT_OPTIONS)
def nb_tec_and_offset_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):

    coerce_literal(nb_tec_and_offset_solver_impl, ["corr_mode"])

    identity_params = get_identity_params(corr_mode, PARAMS_PER_CORR)

    # The outer solver loop is shared between parameterised kernels - only the
    # hooks below are specific to tec_and_offset terms. It solves on the
    # parameter grid, supports scalar mode (two parameters per correlation - a
    # TEC and an offset), has a referencing stage, and enters/leaves a scaled
    # solver basis (the pre/post-solve stages rescale its TEC parameters and
    # apply the zero-mean correction). The shared loop is inlined into the
    # module-local trampoline below rather than returned directly. This gives
    # tec_and_offset a private on-disk cache namespace - see the cache
    # correctness constraint in solver_components.py.
    shared_impl = build_param_solver_impl(
        pre_solve=pre_solve,
        compute_jhj_jhr=compute_jhj_jhr,
        params_per_corr=PARAMS_PER_CORR,
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
    # below (the per-term maths) are specific to tec_and_offset. Its residual
    # normalises out amplitude exactly as delay's does. Like delay, it
    # differentiates a frequency-dependent exponent, so it carries a
    # per-channel coefficient computed by the compute_channel_coeffs hook; that
    # coefficient is the channel_coeffs tuple consumed by the accumulate hook.
    # The TEC exponent differs from delay's: its coefficient has the form
    # 2*pi*(bandwidth/chan_freq[f] + offset) with offset = log(cf_min/cf_max).
    # The offset parameter adds a second (frequency-independent) parameter per
    # correlation, so the accumulator carries a 4-parameter jhj/jhr. Aside from
    # the coefficient formula, every hook here is identical to
    # delay_and_offset's. The shared loop is inlined into the module-local
    # trampoline below rather than returned directly. This gives tec_and_offset
    # a private on-disk cache namespace - see the cache correctness constraint
    # in solver_components.py.
    shared_impl = build_jhj_jhr_impl(
        corr_mode=corr_mode,
        row_weights_type=row_weights_type,
        accumulate_jhj_jhr_factory=accumulate_jhj_jhr_factory,
        zero_jhj_jhr_factory=accumulator.zero,
        flush_jhj_jhr_factory=accumulator.flush,
        compute_residual_factory=phase_only_residual_factory,
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
            cf_min = ms_inputs.MIN_FREQ
            cf_max = ms_inputs.MAX_FREQ
            bandwidth = cf_max - cf_min
            offset = np.log(cf_min/cf_max)

            for t in range(n_time):
                for f in range(n_freq):
                    f_m = param_freq_map[f]
                    coeff = 2 * np.pi * (bandwidth / chan_freq[f] + offset)
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
    """Produce the per-channel TEC coefficient tuple for the shared loop.

    tec_and_offset solves for a frequency-dependent exponent (the TEC
    parameter) plus a frequency-independent offset. Differentiating the model
    with respect to the TEC parameter introduces a per-channel coefficient
    coeff = 2*pi*(bandwidth/chan_freq[f] + offset), where bandwidth is
    cf_max - cf_min and offset = log(cf_min/cf_max) (the same rescaling the
    solver applies to the TEC parameters). The compute_channel_coeffs hook
    computes this once per channel and returns it as a single-element flat
    tuple (coeff,) which is the channel_coeffs tuple passed to the accumulate
    hook.
    """

    def impl(ms_inputs, meta_inputs, f):
        chan_freq = ms_inputs.CHAN_FREQ
        cf_min = ms_inputs.MIN_FREQ
        cf_max = ms_inputs.MAX_FREQ
        bandwidth = cf_max - cf_min
        offset = np.log(cf_min / cf_max)
        coeff = 2 * np.pi * (bandwidth / chan_freq[f] + offset)
        return (coeff,)

    return factories.qcjit(impl)


def accumulate_jhj_jhr_factory(corr_mode):
    """Accumulate a jhj/jhr element into a register-resident accumulator.

    All inputs and the returned accumulator are tuples (register-resident
    values) - the accumulator is only flushed to memory by flush_jhj_jhr. The
    accumulator is a single flat tuple (the upper triangle of the real jhj
    element followed by the jhr entries - see general/accumulator.py).

    The signature follows the unified accumulate_jhj_jhr contract of the shared
    accumulation loop (see solver_components.py). The chain rule uses the
    active-term gain (drv = -1j*conj(g)), so the gain argument is consumed. The
    channel_coeffs argument is the single-element tuple (coeff,) produced by
    the compute_channel_coeffs hook: coeff is at channel_coeffs[0] in every
    corr mode. The offset parameter is frequency independent (its Jacobian
    carries no coeff), while the TEC parameter's Jacobian carries the coeff.
    This yields the interleaved (offset, TEC) accumulator layout, with jhj
    entries scaled by 1, coeff or coeff**2 depending on which parameter pair
    the entry couples. The maths here is identical to delay_and_offset's
    accumulate hook - only the value of coeff (supplied by the
    compute_channel_coeffs hook) differs. The normalisation applied to the
    residual is recomputed here from the operators rather than being passed in
    from compute_residual.
    """

    if corr_mode.literal_value == 4:
        def impl(lop, rop, w, gain, channel_coeffs, wres, jhj_jhr):

            coeff = channel_coeffs[0]
            coeffsq = coeff*coeff

            lop_0, lop_1, lop_2, lop_3 = lop[0], lop[1], lop[2], lop[3]
            rop_0, rop_1, rop_2, rop_3 = rop[0], rop[1], rop[2], rop[3]

            # Row-major Kronecker entries of J^H, matching a_kron_bt: rop
            # enters transposed, which is why rop_1 and rop_2 appear swapped
            # relative to lop. The off-diagonal residual entries carry zero
            # weight, so rows 0 and 3 are needed in columns 0 and 3 only.
            jh_00 = lop_0*rop_0
            jh_03 = lop_1*rop_2
            jh_30 = lop_2*rop_1
            jh_33 = lop_3*rop_3

            # The J^H element for a correlation is its Kronecker row contracted
            # over those two columns. Both the residual and the weights are
            # normalised by its reciprocal squared modulus, guarded against a
            # zero element.
            jh_0 = jh_00 + jh_03
            jh_3 = jh_30 + jh_33
            n_0 = 0 if jh_0 == 0 else 1/(jh_0.real**2 + jh_0.imag**2)
            n_3 = 0 if jh_3 == 0 else 1/(jh_3.real**2 + jh_3.imag**2)

            nres_0 = wres[0]*n_0
            nres_3 = wres[3]*n_3

            # jhwr = J^H W r: each Kronecker row contracted with the
            # normalised, already weighted residual.
            r_0 = jh_00*nres_0 + jh_03*nres_3
            r_3 = jh_30*nres_0 + jh_33*nres_3

            g_3 = gain[3]
            gc_0 = gain[0].conjugate()
            gc_3 = gain[3].conjugate()

            drv_00 = -1j*gc_0
            drv_13 = -1j*gc_3

            upd_00 = (drv_00*r_0).real
            upd_11 = (drv_13*r_3).real

            # jhwj = J^H W J, with the normalisation folded into the weights.
            w_0 = n_0 * w[0]
            w_3 = n_3 * w[3]

            j_00 = jh_00.conjugate()
            j_03 = jh_03.conjugate()
            j_30 = jh_30.conjugate()
            j_33 = jh_33.conjugate()

            jhwj_00 = jh_00*w_0*j_00 + jh_03*w_3*j_03
            jhwj_03 = jh_00*w_0*j_30 + jh_03*w_3*j_33
            jhwj_33 = jh_30*w_0*j_30 + jh_33*w_3*j_33

            tmp_0 = jhwj_00.real
            tmp_1 = (jhwj_03*gc_0*g_3).real
            tmp_2 = jhwj_33.real

            # jhj upper triangle in row-major order followed by jhr entries in
            # (offset, TEC) order per correlation: the offset Jacobian carries
            # no coeff, the TEC Jacobian carries coeff (see zero_jhj_jhr).
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
def tec_and_offset_params_to_gains(
    params,
    gains,
    chan_freq,
    min_freq,
    max_freq,
    param_freq_map,
    rescaled=False
):

    n_time, n_freq, n_ant, n_dir, n_corr = gains.shape

    cf_min = min_freq
    cf_max = max_freq
    bandwidth = cf_max - cf_min

    # The log term absorbs the leading factor of -1, inverting the fraction.
    if rescaled:
        offset = np.log(cf_min/cf_max)
        numerator = bandwidth
    else:
        offset = np.log(cf_min/cf_max)/bandwidth
        numerator = 1.0

    for t in range(n_time):
        for f in range(n_freq):
            f_m = param_freq_map[f]
            coeff = 2 * np.pi * (numerator / chan_freq[f] + offset)
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

    tec_and_offset_params_to_gains(
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


@njit(**JIT_OPTIONS)
def apply_zero_mean_correction(
    min_freq, max_freq, params, param_flags, inverse=False
):
    sign = -1 if inverse else 1
    # Set the starting value of the offset to be consistent with the
    # zero-mean correction factor. This is important if we are loading
    # a term.
    tec_factor = np.log(min_freq/max_freq)/(max_freq - min_freq)
    params[..., 0::2] += -sign * 2 * np.pi * tec_factor * params[..., 1::2]

    # Ensure that the values of flagged parameters remain zero.
    apply_param_flags_to_params(param_flags, params, 0)
