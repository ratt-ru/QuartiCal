# -*- coding: utf-8 -*-
import numpy as np
from numba import njit
from numba.typed import List
from numba.extending import overload
from quartical.utils.numba import (coerce_literal,
                                   JIT_OPTIONS,
                                   PARALLEL_JIT_OPTIONS)
from quartical.gains.general.generics import (native_intermediaries,
                                              upsampled_itermediaries,
                                              per_array_jhj_jhr,
                                              resample_solints,
                                              downsample_jhj_jhr,
                                              invert_gains)
from quartical.gains.general.flagging import (flag_intermediaries,
                                              update_gain_flags,
                                              finalize_gain_flags,
                                              apply_gain_flags_to_flag_col,
                                              update_param_flags)
from quartical.gains.general.convenience import get_extents
import quartical.gains.general.factories as factories
from quartical.gains.general.solver_components import build_jhj_jhr_impl
from quartical.gains.general.solver_components import compute_update
# The null-V residual is a plain r - v (no amplitude normalisation), which is
# exactly the complex term's residual hook.
from quartical.gains.complex.kernel import compute_residual_factory
# The accumulator/flush hooks are identical to the crosshand phase term's -
# both solve a single parameter with a (1, 1) jhj element.
from quartical.gains.crosshand_phase.kernel import (
    zero_jhj_jhr_factory,
    flush_jhj_jhr_factory
)


def get_identity_params(corr_mode):

    if corr_mode.literal_value == 4:
        return np.zeros((1,), dtype=np.float64)
    else:
        raise ValueError("Unsupported number of correlations.")


@njit(**JIT_OPTIONS)
def null_v_crosshand_phase_solver(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    return null_v_crosshand_phase_solver_impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        corr_mode
    )


def null_v_crosshand_phase_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    raise NotImplementedError


@overload(null_v_crosshand_phase_solver_impl, jit_options=JIT_OPTIONS)
def nb_null_v_crosshand_phase_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):

    coerce_literal(nb_null_v_crosshand_phase_solver_impl, ["corr_mode"])

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

        inverse_gains = List()
        for gain_term in gains:
            inverse_gains.append(np.empty_like(gain_term))
        invert_gains(gains, inverse_gains, corr_mode)

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
                inverse_gains,
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

            # The parameters/gains are correct but we are solving for the
            # inverse so we update the inverse term here.
            inverse_gains[active_term][:] = active_gain.conj()

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
    inverse_gains,
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
    inverse_gains,
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    upsampled_imdry,
    extents,
    corr_mode
):

    coerce_literal(nb_compute_jhj_jhr, ["corr_mode"])

    # The null-V kernel solves on the INVERSE of the chain: it inverts the
    # gains, reverses the chain/mappings, and treats the observed data as the
    # "model" that the reversed inverse chain corrupts towards zero (the
    # residual base is zero, so r = -v). It also applies NO amplitude
    # normalisation and NO weights to the residual. All of these differences
    # are input transformations rather than loop-body changes, so the shared
    # accumulation loop can be reused by forging the input namedtuples here
    # (cheap views - no data is copied) and dispatching to the shared-loop
    # trampoline below. The runtime namedtuple classes are captured from the
    # numba types at overload time so the forged instances can be constructed
    # inside the jitted impl.
    ms_inputs_cls = ms_inputs.instance_class
    mapping_inputs_cls = mapping_inputs.instance_class
    chain_inputs_cls = chain_inputs.instance_class
    meta_inputs_cls = meta_inputs.instance_class

    def impl(
        inverse_gains,
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        upsampled_imdry,
        extents,
        corr_mode
    ):

        data = ms_inputs.DATA
        n_row, n_chan, n_corr = data.shape
        n_dir = ms_inputs.MODEL_DATA.shape[2]

        # The observed data becomes the per-direction "model" (the original
        # kernel used data[row, f] for every direction); the data becomes
        # zero (r = -v); the weights become unity and the row weights None,
        # reproducing the original kernel's completely unweighted residual.
        # All three are zero-copy broadcast views.
        forged_model = np.broadcast_to(
            np.expand_dims(data, 2), (n_row, n_chan, n_dir, n_corr)
        )
        forged_data = np.broadcast_to(
            np.zeros((1, 1, n_corr), dtype=data.dtype), data.shape
        )
        forged_weights = np.broadcast_to(
            np.ones((1, 1, n_corr), dtype=ms_inputs.WEIGHT.dtype),
            ms_inputs.WEIGHT.shape
        )

        forged_ms_inputs = ms_inputs_cls(
            MODEL_DATA=forged_model,
            DATA=forged_data,
            ANTENNA1=ms_inputs.ANTENNA1,
            ANTENNA2=ms_inputs.ANTENNA2,
            WEIGHT=forged_weights,
            FLAG=ms_inputs.FLAG,
            ROW_MAP=ms_inputs.ROW_MAP,
            ROW_WEIGHTS=None,
            TIME=ms_inputs.TIME,
        )

        # Reverse the (inverse) gains and their mappings; the active term
        # index is likewise reversed. The param mappings are not used by the
        # shared loop and pass through unchanged.
        forged_mapping_inputs = mapping_inputs_cls(
            time_bins=mapping_inputs.time_bins,
            time_maps=mapping_inputs.time_maps[::-1],
            freq_maps=mapping_inputs.freq_maps[::-1],
            dir_maps=mapping_inputs.dir_maps[::-1],
            param_time_bins=mapping_inputs.param_time_bins,
            param_time_maps=mapping_inputs.param_time_maps,
            param_freq_maps=mapping_inputs.param_freq_maps,
        )

        forged_chain_inputs = chain_inputs_cls(
            gains=inverse_gains[::-1],
            gain_flags=chain_inputs.gain_flags,
            params=chain_inputs.params,
            param_flags=chain_inputs.param_flags,
        )

        forged_meta_inputs = meta_inputs_cls(
            iters=meta_inputs.iters,
            active_term=len(inverse_gains) - meta_inputs.active_term - 1,
            stop_frac=meta_inputs.stop_frac,
            stop_crit=meta_inputs.stop_crit,
            threads=meta_inputs.threads,
            robust=meta_inputs.robust,
            reference_antenna=meta_inputs.reference_antenna,
            scalar=meta_inputs.scalar,
            dd_term=meta_inputs.dd_term,
            pinned_directions=meta_inputs.pinned_directions,
            solve_per=meta_inputs.solve_per,
        )

        _shared_compute_jhj_jhr(
            forged_ms_inputs,
            forged_mapping_inputs,
            forged_chain_inputs,
            forged_meta_inputs,
            upsampled_imdry,
            extents,
            corr_mode
        )
        return
    return impl


def _shared_compute_jhj_jhr(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    upsampled_imdry,
    extents,
    corr_mode
):
    return NotImplementedError


@overload(_shared_compute_jhj_jhr, jit_options=PARALLEL_JIT_OPTIONS)
def nb_shared_compute_jhj_jhr(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    upsampled_imdry,
    extents,
    corr_mode
):

    coerce_literal(nb_shared_compute_jhj_jhr, ["corr_mode"])

    # We want to dispatch based on this field so we need its type. The forged
    # inputs always carry None row weights (see nb_compute_jhj_jhr above).
    row_weights_idx = ms_inputs.fields.index('ROW_WEIGHTS')
    row_weights_type = ms_inputs[row_weights_idx]

    # The accumulation loop itself is shared between kernels - only the hooks
    # below (the per-term maths) are specific to the null-V crosshand term.
    # The residual is a plain r - v (complex's residual hook, no auxiliary
    # values); the accumulator/flush hooks are crosshand phase's (single
    # parameter, (1, 1) jhj, so the mirror hook is a no-op); the accumulate
    # hook is the null-V projection defined below. There are no per-channel
    # coefficients, so there is no compute_channel_coeffs hook.
    # The shared loop is inlined into the module-local trampoline below
    # rather than returned directly. This gives the null-V crosshand term a
    # private on-disk cache namespace - see the cache correctness constraint
    # in solver_components.py.
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
                            # NOTE: Halving the update absolutely required.
                            p -= 0.5*upd  # Flip sign for non-inverse solution.
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


def accumulate_jhj_jhr_factory(corr_mode):
    """Accumulate a jhr/jhj element into a register-resident accumulator.

    All inputs and the returned accumulator are tuples (register-resident
    values) - the accumulator is only flushed to memory by flush_jhj_jhr.
    The accumulator is a flat tuple (jhj00, jhr0) - see zero_jhj_jhr_factory
    in the crosshand phase kernel, from which both the zeros and flush hooks
    are imported.

    The signature follows the unified accumulate_jhj_jhr contract of the shared
    accumulation loop (see solver_components.py). The chain rule uses the
    active-term gain (drv = -1j*conj(g)), so the gain argument is consumed; the
    channel_coeffs argument is empty (crosshand has no compute_channel_coeffs
    hook) and unused. The incoming residual is r = -v (zero data, plain
    subtraction) and is UNWEIGHTED - the forged unit weights in
    nb_compute_jhj_jhr guarantee this.

    The residual is first projected onto the V-nulling combination
    (v_res = -0.5j*r_1 + 0.5j*r_2, wres -> [0, 0.5j*v_res, -0.5j*v_res, 0]);
    jhr keeps only the [0] (XX) entry of lop @ wres @ rop, while jhj is
    |0.5j*(jh_01 - jh_02)|^2 - only the two cross entries of the first row of
    the row-major kronecker product survive the projection.
    """

    if corr_mode.literal_value == 4:
        def impl(lop, rop, w, gain, channel_coeffs, wres, jhj_jhr):

            # Project the residual onto the V-nulling combination.
            r_1 = wres[1]
            r_2 = wres[2]

            v_res = -0.5j*r_1 + 0.5j*r_2

            s_1 = 0.5j*v_res
            s_2 = -0.5j*v_res

            # jhwr element: lop @ [[0, s_1], [s_2, 0]] @ rop, keeping only
            # the [0] (XX) entry.
            mm_0 = s_1*rop[2]
            mm_2 = s_2*rop[0]

            r_0 = lop[0]*mm_0 + lop[1]*mm_2

            gc_0 = gain[0].conjugate()

            drv_00 = -1j*gc_0

            upd_00 = (drv_00*r_0).real

            # jhwj element: no weights are applied (see the docstring).
            # NOTE: rop is effectively transposed (rop[2] used as rop_01)
            # relative to lop, matching the row-major kronecker convention.
            jh_01 = lop[0]*rop[2]
            jh_02 = lop[1]*rop[0]

            jh_v = 0.5j*jh_01 - 0.5j*jh_02
            j_v = jh_v.conjugate()

            jhj_v = jh_v*j_v

            return (
                jhj_jhr[0] + jhj_v.real,
                jhj_jhr[1] + upd_00,
            )

    else:
        raise ValueError("Crosshand phase can only be solved for with four "
                         "correlation data.")

    return factories.qcjit(impl)
