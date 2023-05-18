# -*- coding: utf-8 -*-
import numpy as np
from numba import generated_jit, jit
from quartical.utils.numba import coerce_literal
from quartical.gains.general.generics import (native_intermediaries,
                                              upsampled_itermediaries,
                                              per_array_jhj_jhr,
                                              resample_solints,
                                              downsample_jhj_jhr)
from quartical.gains.general.flagging import (flag_intermediaries,
                                              update_gain_flags,
                                              finalize_gain_flags,
                                              apply_gain_flags,
                                              update_param_flags)
from quartical.gains.general.convenience import get_extents
from quartical.gains.delay.kernel import (compute_jhj_jhr,
                                          compute_update,
                                          finalize_update)


def get_identity_params(corr_mode):

    if corr_mode.literal_value in (2, 4):
        return np.zeros((4,), dtype=np.float64)
    elif corr_mode.literal_value == 1:
        return np.zeros((2,), dtype=np.float64)
    else:
        raise ValueError("Unsupported number of correlations.")


@generated_jit(
    nopython=True,
    fastmath=True,
    parallel=False,
    cache=True,
    nogil=True
)
def tec_solver(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):

    # NOTE: This just reuses delay solver functionality.

    coerce_literal(tec_solver, ["corr_mode"])

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
        dd_term = meta_inputs.dd_term
        n_thread = meta_inputs.threads

        active_gain = gains[active_term]
        active_gain_flags = gain_flags[active_term]
        active_params = chain_inputs.params[active_term]

        # Set up some intemediaries used for flagging. TODO: Move?
        km1_gain = active_gain.copy()
        km1_abs2_diffs = np.zeros_like(active_gain_flags, dtype=np.float64)
        abs2_diffs_trend = np.zeros_like(active_gain_flags, dtype=np.float64)
        flag_imdry = \
            flag_intermediaries(km1_gain, km1_abs2_diffs, abs2_diffs_trend)

        # Set up some intemediaries used for solving.
        real_dtype = active_gain.real.dtype
        param_shape = active_params.shape

        active_t_map_g = mapping_inputs.time_maps[active_term]
        active_f_map_p = mapping_inputs.param_freq_maps[active_term]

        # Create more work to do in paralllel when needed, else no-op.
        resampler = resample_solints(active_t_map_g, param_shape, n_thread)

        # Determine the starts and stops of the rows and channels associated
        # with each solution interval.
        extents = get_extents(resampler.upsample_t_map, active_f_map_p)

        upsample_shape = resampler.upsample_shape
        upsampled_jhj = np.empty(upsample_shape + (upsample_shape[-1],),
                                 dtype=real_dtype)
        upsampled_jhr = np.empty(upsample_shape, dtype=real_dtype)
        jhj = upsampled_jhj[:param_shape[0]]
        jhr = upsampled_jhr[:param_shape[0]]
        update = np.zeros(param_shape, dtype=real_dtype)

        upsampled_imdry = upsampled_itermediaries(upsampled_jhj, upsampled_jhr)
        native_imdry = native_intermediaries(jhj, jhr, update)

        scaled_icf = ms_inputs.CHAN_FREQ.copy()  # Don't mutate.
        min_freq = np.min(scaled_icf)
        scaled_icf = min_freq/scaled_icf  # Scale freqs to avoid precision.
        scaled_icf *= 2*np.pi  # Introduce 2pi here - neglect everywhere else.
        active_params[..., 1::2] /= min_freq  # Scale consistently with freq.

        for loop_idx in range(max_iter):

            compute_jhj_jhr(
                ms_inputs,
                mapping_inputs,
                chain_inputs,
                meta_inputs,
                upsampled_imdry,
                extents,
                scaled_icf,
                corr_mode
            )

            if resampler.active:
                downsample_jhj_jhr(upsampled_imdry, resampler.downsample_t_map)

            if solve_per == "array":
                per_array_jhj_jhr(native_imdry)

            compute_update(native_imdry, corr_mode)

            finalize_update(
                mapping_inputs,
                chain_inputs,
                meta_inputs,
                native_imdry,
                loop_idx,
                scaled_icf,
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
            apply_gain_flags(
                ms_inputs,
                mapping_inputs,
                chain_inputs,
                meta_inputs
            )

        active_params[..., 1::2] *= min_freq  # Undo scaling for SI units.
        native_imdry.jhj[..., 1::2] /= min_freq ** 2

        return native_imdry.jhj, loop_idx + 1, conv_perc

    return impl


@jit(
    nopython=True,
    fastmath=True,
    parallel=False,
    cache=True,
    nogil=True
)
def tec_params_to_gains(
    params,
    gains,
    chan_freq,
    param_freq_map
):

    n_time, n_freq, n_ant, n_dir, n_corr = gains.shape

    for t in range(n_time):
        for f in range(n_freq):
            icf = 1/chan_freq[f]
            f_m = param_freq_map[f]
            for a in range(n_ant):
                for d in range(n_dir):

                    g = gains[t, f, a, d]
                    p = params[t, f_m, a, d]

                    g[0] = np.exp(1j*2*np.pi*(icf*p[1] + p[0]))

                    if n_corr > 1:
                        g[-1] = np.exp(1j*2*np.pi*(icf*p[3] + p[2]))
