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
from quartical.gains.general.accumulation import build_jhj_jhr_impl
from quartical.gains.general.solver_loop import build_param_solver_impl
from quartical.gains.general.solver_ops import compute_update  # noqa


def get_identity_params(corr_mode):

    if corr_mode.literal_value in (2, 4):
        return np.zeros((4,), dtype=np.float64)
    elif corr_mode.literal_value == 1:
        return np.zeros((2,), dtype=np.float64)
    else:
        raise ValueError("Unsupported number of correlations.")


@njit(**JIT_OPTIONS)
def delay_and_tec_solver(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    return delay_and_tec_solver_impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        corr_mode
    )


def delay_and_tec_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    raise NotImplementedError


# The delay/tec family enters a scaled solver basis before the loop and leaves
# it afterwards; these two module-local qcjit hooks fetch their inputs from the
# standardised hook arguments, exactly mirroring the inline rescaling the loop
# body previously performed.
def pre_solve(ms_inputs, chain_inputs, meta_inputs):
    active_params = chain_inputs.params[meta_inputs.active_term]

    # We actually solve for D' = (D(nu_min + nu_max))/2. This helps avoid
    # numerical issues, but requires some scaling of the parameters.
    # We actually solve for TEC' = TEC/bandwidth. This helps avoid
    # numerical issues, but requires some scaling of the parameters.
    min_freq = ms_inputs.MIN_FREQ
    max_freq = ms_inputs.MAX_FREQ
    mid_freq = (min_freq + max_freq) / 2
    active_params[..., 1::2] *= mid_freq
    bandwidth = max_freq - min_freq
    active_params[..., 0::2] /= bandwidth


pre_solve = factories.qcjit(pre_solve)


def post_solve(ms_inputs, chain_inputs, meta_inputs, native_imdry):
    active_params = chain_inputs.params[meta_inputs.active_term]
    min_freq = ms_inputs.MIN_FREQ
    max_freq = ms_inputs.MAX_FREQ
    mid_freq = (min_freq + max_freq) / 2
    bandwidth = max_freq - min_freq

    # Undo rescaling so that quantities are in native units.
    active_params[..., 1::2] /= mid_freq
    native_imdry.jhj[..., 1::2] *= mid_freq ** 2
    active_params[..., 0::2] *= bandwidth
    native_imdry.jhj[..., 0::2, 0::2] /= bandwidth ** 2


post_solve = factories.qcjit(post_solve)


@overload(delay_and_tec_solver_impl, jit_options=JIT_OPTIONS)
def nb_delay_and_tec_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):

    coerce_literal(nb_delay_and_tec_solver_impl, ["corr_mode"])

    identity_params = get_identity_params(corr_mode)

    # The outer solver loop is shared between parameterised kernels - only the
    # hooks below are specific to delay_and_tec terms. It solves on the
    # parameter grid, supports scalar mode (two parameters per correlation - a
    # delay and a TEC), has a referencing stage, and enters/leaves a scaled
    # solver basis (the pre/post-solve stages rescale its delay and TEC
    # parameters). The shared loop is inlined into the module-local trampoline
    # below rather than returned directly. This gives delay_and_tec a private
    # on-disk cache namespace - see the cache correctness constraint in
    # accumulation.py.
    shared_impl = build_param_solver_impl(
        True,
        pre_solve,
        compute_jhj_jhr,
        2,
        None,
        finalize_update,
        1e9,
        identity_params,
        reference_params,
        post_solve,
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
    # below (the per-term maths) are specific to delay_and_tec. Its residual
    # normalises out amplitude exactly as delay's does, so its residual hook
    # appends a per-correlation normalisation factor as auxiliary values
    # (n_resid_aux is n_corr). It differentiates a frequency-dependent exponent
    # with respect to two parameters (a delay and a TEC), so the stage hook
    # carries TWO per-channel coefficients; both are concatenated onto the
    # residual's auxiliary values to form the aux tuple consumed by the elem.
    # There is no offset parameter here (both parameters are frequency
    # dependent), so the accumulator carries a 4-parameter jhj/jhr like
    # delay_and_offset, but every entry is scaled by a product of the delay and
    # TEC coefficients rather than by a single coefficient.
    #
    # The shared loop is inlined into the module-local trampoline below rather
    # than returned directly. This gives delay_and_tec a private on-disk cache
    # namespace - see the cache correctness constraint in accumulation.py.
    shared_impl = build_jhj_jhr_impl(
        corr_mode,
        row_weights_type,
        elem_factory=compute_jhwj_jhwr_elem_factory,
        acc_zeros_factory=jhwj_jhwr_zeros_factory,
        flush_factory=flush_jhwj_jhwr_factory,
        resid_factory=resid_factory,
        n_resid_aux=corr_mode.literal_value,
        stage_factory=stage_factory,
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
            cf_mid = (cf_min + cf_max) / 2
            bandwidth = cf_max - cf_min
            tec_offset = np.log(cf_min / cf_max)

            for t in range(n_time):
                for f in range(n_freq):
                    f_m = param_freq_map[f]
                    delay_coeff = \
                        2 * np.pi * (chan_freq[f]/cf_mid - 1)
                    tec_coeff = \
                        2 * np.pi * (bandwidth/chan_freq[f] + tec_offset)
                    for a in range(n_ant):
                        for d in range(n_dir):

                            d_m = dir_map[d]
                            g = gain[t, f, a, d]
                            fl = gain_flags[t, f, a, d]
                            p = params[t, f_m, a, d_m]

                            if fl == 1:
                                set_identity(g)
                            else:
                                param_to_gain(p, delay_coeff, tec_coeff, g)
    else:
        raise ValueError("Unsupported number of correlations.")

    return impl


def param_to_gain_factory(corr_mode):

    if corr_mode.literal_value == 4:
        def impl(params, delay_coeff, tec_coeff, gain):
            gain[0] = np.exp(
                1j * (delay_coeff * params[1] + tec_coeff * params[0])
            )
            gain[3] = np.exp(
                1j * (delay_coeff * params[3] + tec_coeff * params[2])
            )
    elif corr_mode.literal_value == 2:
        def impl(params, delay_coeff, tec_coeff, gain):
            gain[0] = np.exp(
                1j * (delay_coeff * params[1] + tec_coeff * params[0])
            )
            gain[1] = np.exp(
                1j * (delay_coeff * params[3] + tec_coeff * params[2])
            )
    elif corr_mode.literal_value == 1:
        def impl(params, delay_coeff, tec_coeff, gain):
            gain[0] = np.exp(
                1j * (delay_coeff * params[1] + tec_coeff * params[0])
            )
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def stage_factory(corr_mode):
    """Produce the per-channel delay and TEC coefficient tuple.

    delay_and_tec solves for two frequency-dependent exponents: a delay and a
    TEC. Differentiating the model with respect to each parameter introduces a
    distinct per-channel coefficient. The delay coefficient is
    delay_coeff = 2*pi*(chan_freq[f]/cf_mid - 1) with cf_mid the band midpoint;
    the TEC coefficient is tec_coeff = 2*pi*(bandwidth/chan_freq[f] + offset)
    with bandwidth = cf_max - cf_min and offset = log(cf_min/cf_max) (matching
    the rescaling the solver applies to each parameter). The stage hook computes
    both once per channel and returns them as a flat tuple
    (delay_coeff, tec_coeff); the shared loop concatenates it onto the
    residual's auxiliary values to form the aux tuple passed to the elem.
    """

    def impl(ms_inputs, meta_inputs, f):
        chan_freq = ms_inputs.CHAN_FREQ
        cf_min = ms_inputs.MIN_FREQ
        cf_max = ms_inputs.MAX_FREQ
        cf_mid = (cf_min + cf_max) / 2
        bandwidth = cf_max - cf_min
        tec_offset = np.log(cf_min / cf_max)
        delay_coeff = 2 * np.pi * (chan_freq[f] / cf_mid - 1)
        tec_coeff = 2 * np.pi * (bandwidth / chan_freq[f] + tec_offset)
        return (delay_coeff, tec_coeff)

    return factories.qcjit(impl)


def resid_factory(corr_mode):
    """Produce the amplitude-normalised residual for a delay_and_tec term.

    The residual is identical to delay's (and phase's): it normalises out
    amplitude before forming the residual. The per-correlation factor is
    normf_i = |v_i| / |r_i| (zero where r_i is zero, matching
    absv1_idiv_absv2), and the residual is r_i*normf_i - v_i. The normf values
    are appended as auxiliary values (n_resid_aux = n_corr) so the returned flat
    tuple is (residual..., normf...). The elem hook recomputes its own
    operator-based normalisation, so it does not actually consume these normf
    auxiliary values - they are retained only to keep the residual/aux contract
    of the shared loop uniform across terms.
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
                f0, f1, f2, f3,
            )
    elif corr_mode.literal_value == 2:
        def impl(r, v):
            f0, f1 = tuple_normf(v, r)
            return (
                r[0]*f0 - v[0],
                r[1]*f1 - v[1],
                f0, f1,
            )
    elif corr_mode.literal_value == 1:
        def impl(r, v):
            f0, = tuple_normf(v, r)
            return (
                r[0]*f0 - v[0],
                f0,
            )
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def jhwj_jhwr_zeros_factory(corr_mode):
    """Produce the zero jhr/jhj accumulator tuple for a given corr mode.

    The accumulator is a single flat tuple holding the (real) jhr entries
    followed by the upper triangle of the (n_param, n_param) real jhj element in
    row-major order. For the 2 and 4 correlation cases n_param is 4, giving 4
    jhr entries and 10 upper-triangle jhj entries (14 slots). For the single
    correlation case n_param is 2, giving 2 jhr entries and 3 upper-triangle jhj
    entries (5 slots). The reference element is a jhr slice, whose dtype is
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


def flush_jhwj_jhwr_factory(corr_mode):
    """Add a register-accumulated jhr/jhj accumulator into the arrays.

    The accumulator holds the jhr entries first, then the upper triangle of the
    (n_param, n_param) jhj element in row-major order. The lower triangle is
    filled in by mirror_jhj once per solution interval. For the 2 correlation
    case the cross-correlation jhj entries (0, 2), (0, 3), (1, 2), (1, 3) are
    always zero, so mirroring them is a harmless no-op there.
    """

    if corr_mode.literal_value in (2, 4):
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
    elif corr_mode.literal_value == 1:
        def impl(jhr, jhj, acc):

            jhr[0] += acc[0]
            jhr[1] += acc[1]

            jhj[0, 0] += acc[2]
            jhj[0, 1] += acc[3]
            jhj[1, 1] += acc[4]
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def mirror_jhj_factory(corr_mode):
    """Fill in the lower triangle of the per-interval (n_param, n_param) jhj.

    Accumulation in compute_jhwj_jhwr_elem only writes the upper triangle of
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


def compute_jhwj_jhwr_elem_factory(corr_mode):
    """Accumulate a jhr/jhj element into a register-resident accumulator.

    All inputs and the returned accumulator are tuples (register-resident
    values) - the accumulator is only flushed to memory by flush_jhwj_jhwr. The
    accumulator is a single flat tuple (jhr entries followed by the upper
    triangle of the real jhj element - see jhwj_jhwr_zeros_factory).

    The signature follows the unified elem contract of the shared accumulation
    loop (see accumulation.py). The chain rule uses the active-term gain
    (drv = -1j*conj(g)), so the gain argument is consumed. The aux argument is
    the flat tuple (normf..., delay_coeff, tec_coeff) built by concatenating the
    residual hook's normf values with the stage hook's two per-channel
    coefficients. Its layout is:

        corr 4: aux = (normf0, normf1, normf2, normf3, delay_coeff, tec_coeff)
        corr 2: aux = (normf0, normf1, delay_coeff, tec_coeff)
        corr 1: aux = (normf0, delay_coeff, tec_coeff)

    so delay_coeff is at aux[n_corr] and tec_coeff at aux[n_corr + 1]. Both
    parameters are frequency dependent, so unlike the offset-pair kernels every
    jhj entry carries a product of the two coefficients: the (tec, tec) entry is
    scaled by tec_coeff**2, the (delay, delay) entry by delay_coeff**2, and the
    (tec, delay) cross entries by delay_coeff*tec_coeff. The per-correlation
    parameter order is (tec, delay), matching param_to_gain and the array
    kernel. The normf values are ignored - this elem recomputes its own
    operator-based normalisation, exactly as the original array-buffer kernel
    did.
    """

    if corr_mode.literal_value == 4:
        def impl(lop, rop, w, gain, aux, res, acc):

            delay_coeff = aux[4]
            tec_coeff = aux[5]
            dcsq = delay_coeff*delay_coeff
            tcsq = tec_coeff*tec_coeff
            dctc = delay_coeff*tec_coeff

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
            res_0 = res[0]*n_0
            res_3 = res[3]*n_3
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

            # jhr entries in (tec, delay) order per correlation: the tec
            # Jacobian carries tec_coeff, the delay Jacobian carries
            # delay_coeff. jhj upper triangle in row-major order (see
            # jhwj_jhwr_zeros); the (delay0, tec3) entry equals (tec0, delay3)
            # by symmetry, so both carry dctc*tmp_1.
            return (
                acc[0] + tec_coeff*upd_00,
                acc[1] + delay_coeff*upd_00,
                acc[2] + tec_coeff*upd_11,
                acc[3] + delay_coeff*upd_11,
                acc[4] + tcsq*tmp_0,
                acc[5] + dctc*tmp_0,
                acc[6] + tcsq*tmp_1,
                acc[7] + dctc*tmp_1,
                acc[8] + dcsq*tmp_0,
                acc[9] + dctc*tmp_1,
                acc[10] + dcsq*tmp_1,
                acc[11] + tcsq*tmp_2,
                acc[12] + dctc*tmp_2,
                acc[13] + dcsq*tmp_2,
            )

    elif corr_mode.literal_value == 2:
        def impl(lop, rop, w, gain, aux, res, acc):

            delay_coeff = aux[2]
            tec_coeff = aux[3]
            dcsq = delay_coeff*delay_coeff
            tcsq = tec_coeff*tec_coeff
            dctc = delay_coeff*tec_coeff

            rop_0, rop_1 = rop[0], rop[1]

            # Normalisation factor: 1/|rop|^2 per corr, zero-guarded.
            n_0 = 0 if rop_0 == 0 else 1/(rop_0.real**2 + rop_0.imag**2)
            n_1 = 0 if rop_1 == 0 else 1/(rop_1.real**2 + rop_1.imag**2)

            # jhwr element (diagonal only).
            r_0 = res[0]*n_0*rop_0
            r_1 = res[1]*n_1*rop_1

            gc_0 = gain[0].conjugate()
            gc_1 = gain[1].conjugate()

            drv_00 = -1j*gc_0
            drv_23 = -1j*gc_1

            upd_00 = (drv_00*r_0).real
            upd_11 = (drv_23*r_1).real

            # jhwj element (block diagonal, real). The cross-correlation jhj
            # entries (acc[6], acc[7], acc[9], acc[10]) are left untouched,
            # matching the array kernel which never sets them.
            jhj_00 = (rop_0*n_0*w[0]*rop_0.conjugate()).real
            jhj_11 = (rop_1*n_1*w[1]*rop_1.conjugate()).real

            return (
                acc[0] + tec_coeff*upd_00,
                acc[1] + delay_coeff*upd_00,
                acc[2] + tec_coeff*upd_11,
                acc[3] + delay_coeff*upd_11,
                acc[4] + tcsq*jhj_00,
                acc[5] + dctc*jhj_00,
                acc[6],
                acc[7],
                acc[8] + dcsq*jhj_00,
                acc[9],
                acc[10],
                acc[11] + tcsq*jhj_11,
                acc[12] + dctc*jhj_11,
                acc[13] + dcsq*jhj_11,
            )

    elif corr_mode.literal_value == 1:
        def impl(lop, rop, w, gain, aux, res, acc):

            delay_coeff = aux[1]
            tec_coeff = aux[2]
            dcsq = delay_coeff*delay_coeff
            tcsq = tec_coeff*tec_coeff
            dctc = delay_coeff*tec_coeff

            rop_0 = rop[0]

            # Normalisation factor: 1/|rop|^2, zero-guarded.
            n_0 = 0 if rop_0 == 0 else 1/(rop_0.real**2 + rop_0.imag**2)

            # jhwr element.
            r_0 = res[0]*n_0*rop_0

            gc_0 = gain[0].conjugate()
            drv_00 = -1j*gc_0
            upd_00 = (drv_00*r_0).real

            # jhwj element (real).
            jhj_00 = (rop_0*n_0*w[0]*rop_0.conjugate()).real

            return (
                acc[0] + tec_coeff*upd_00,
                acc[1] + delay_coeff*upd_00,
                acc[2] + tcsq*jhj_00,
                acc[3] + dctc*jhj_00,
                acc[4] + dcsq*jhj_00,
            )
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


@njit(**JIT_OPTIONS)
def delay_and_tec_params_to_gains(
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
    cf_mid = (cf_min + cf_max) / 2

    # DELAY
    if rescaled:
        delay_offset = 1.0
        delay_denom = cf_mid
    else:
        delay_offset = cf_mid
        delay_denom = 1.0

    bandwidth = cf_max - cf_min

    # TEC
    if rescaled:
        tec_offset = np.log(cf_min/cf_max)
        tec_numerator = bandwidth
    else:
        tec_offset = np.log(cf_min/cf_max)/bandwidth
        tec_numerator = 1.0

    for t in range(n_time):
        for f in range(n_freq):
            f_m = param_freq_map[f]
            # Coeff associated with the the delay.
            delay_coeff = \
                2 * np.pi * (chan_freq[f] / delay_denom - delay_offset)
            # Coeff associated with the the tec.
            tec_coeff = \
                2 * np.pi * (tec_numerator / chan_freq[f] + tec_offset)
            for a in range(n_ant):
                for d in range(n_dir):

                    g = gains[t, f, a, d]
                    p = params[t, f_m, a, d]

                    g[0] = np.exp(1j*(delay_coeff*p[1] + tec_coeff*p[0]))

                    if n_corr > 1:
                        g[-1] = np.exp(1j*(delay_coeff*p[3] + tec_coeff*p[2]))


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

    delay_and_tec_params_to_gains(
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
