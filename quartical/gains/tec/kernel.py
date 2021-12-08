# -*- coding: utf-8 -*-
import numpy as np
from numba import generated_jit
from quartical.utils.numba import coerce_literal
from quartical.gains.general.generics import (compute_residual,
                                              per_array_jhj_jhr)
from quartical.gains.general.flagging import (update_gain_flags,
                                              finalize_gain_flags,
                                              apply_gain_flags,
                                              gain_flags_to_param_flags)
from quartical.gains.delay.kernel import (compute_jhj_jhr,
                                          compute_update,
                                          finalize_update)
from collections import namedtuple


# This can be done without a named tuple now. TODO: Add unpacking to
# constructor.
stat_fields = {"conv_iters": np.int64,
               "conv_perc": np.float64}

term_conv_info = namedtuple("term_conv_info", " ".join(stat_fields.keys()))

tec_args = namedtuple(
    "tec_args",
    (
        "params",
        "param_flags",
        "chan_freqs",
        "t_bin_arr"
    )
)


@generated_jit(nopython=True,
               fastmath=True,
               parallel=False,
               cache=True,
               nogil=True)
def tec_solver(base_args, term_args, meta_args, corr_mode):

    # NOTE: This just reuses delay solver functionality.

    coerce_literal(tec_solver, ["corr_mode"])

    def impl(base_args, term_args, meta_args, corr_mode):

        model = base_args.model
        data = base_args.data
        a1 = base_args.a1
        a2 = base_args.a2
        weights = base_args.weights
        flags = base_args.flags
        t_map_arr = base_args.t_map_arr
        t_map_arr_g = t_map_arr[0]
        f_map_arr = base_args.f_map_arr
        f_map_arr_g = f_map_arr[0]  # Gain mappings.
        f_map_arr_p = f_map_arr[1]  # Parameter mappings.
        d_map_arr = base_args.d_map_arr
        gains = base_args.gains
        gain_flags = base_args.gain_flags
        row_map = base_args.row_map
        row_weights = base_args.row_weights

        stop_frac = meta_args.stop_frac
        stop_crit = meta_args.stop_crit
        active_term = meta_args.active_term
        iters = meta_args.iters
        solve_per = meta_args.solve_per

        active_params = term_args.params[active_term]  # Params for this term.
        active_param_flags = term_args.param_flags[active_term]
        t_bin_arr = term_args.t_bin_arr
        chan_freqs = term_args.chan_freqs.copy()  # Don't mutate orginal.
        min_freq = np.min(chan_freqs)
        inv_chan_freqs = min_freq/chan_freqs  # Scale freqs to avoid precision.
        active_params[..., 1::2] /= min_freq  # Scale consistently with freq.

        n_term = len(gains)

        active_gain = gains[active_term]
        active_gain_flags = gain_flags[active_term]

        dd_term = np.any(d_map_arr[active_term])

        # Set up some intemediaries used for flagging.
        last_gain = active_gain.copy()
        km1_abs2_diffs = np.zeros_like(active_gain_flags, dtype=np.float64)
        abs2_diffs_trend = np.zeros_like(active_gain_flags, dtype=np.float64)
        cnv_perc = 0.

        real_dtype = gains[active_term].real.dtype

        pshape = active_params.shape
        jhj = np.empty(pshape + (pshape[-1],), dtype=real_dtype)
        jhr = np.empty(pshape, dtype=real_dtype)
        update = np.zeros_like(jhr)

        for i in range(iters):

            if dd_term or n_term > 1:
                residual = compute_residual(data,
                                            model,
                                            gains,
                                            a1,
                                            a2,
                                            t_map_arr_g,
                                            f_map_arr_g,
                                            d_map_arr,
                                            row_map,
                                            row_weights,
                                            corr_mode)
            else:
                residual = data

            compute_jhj_jhr(jhj,
                            jhr,
                            model,
                            gains,
                            residual,
                            a1,
                            a2,
                            weights,
                            flags,
                            t_map_arr_g,
                            f_map_arr_g,
                            f_map_arr_p,
                            d_map_arr,
                            inv_chan_freqs,
                            row_map,
                            row_weights,
                            active_term,
                            corr_mode)

            if solve_per == "array":
                per_array_jhj_jhr(jhj, jhr)

            compute_update(update,
                           jhj,
                           jhr,
                           corr_mode)

            finalize_update(update,
                            active_params,
                            active_gain,
                            active_gain_flags,
                            inv_chan_freqs,
                            f_map_arr[:, :, active_term],
                            d_map_arr[active_term, :],
                            corr_mode)

            # Check for gain convergence. Produced as a side effect of
            # flagging. The converged percentage is based on unflagged
            # intervals.
            cnv_perc = update_gain_flags(active_gain,
                                         last_gain,
                                         active_gain_flags,
                                         km1_abs2_diffs,
                                         abs2_diffs_trend,
                                         stop_crit,
                                         corr_mode,
                                         i)

            if not dd_term:
                apply_gain_flags(active_gain_flags,
                                 flags,
                                 active_term,
                                 a1,
                                 a2,
                                 t_map_arr_g,
                                 f_map_arr_g)

            # Don't update the last gain if converged/on final iteration.
            if (cnv_perc >= stop_frac) or (i == iters - 1):
                break
            else:
                last_gain[:] = active_gain

        # NOTE: Removes soft flags and flags points which have bad trends.
        finalize_gain_flags(active_gain,
                            active_gain_flags,
                            abs2_diffs_trend,
                            corr_mode)

        # Propagate gain flags to parameter flags. TODO: Verify that this
        # is adequate. Do we need to consider setting the identity params.
        gain_flags_to_param_flags(active_gain_flags,
                                  active_param_flags,
                                  t_bin_arr[:, :, active_term],
                                  f_map_arr[:, :, active_term],
                                  d_map_arr)

        # Call this one last time to ensure points flagged by finialize are
        # propagated (in the DI case).
        if not dd_term:
            apply_gain_flags(active_gain_flags,
                             flags,
                             active_term,
                             a1,
                             a2,
                             t_map_arr_g,
                             f_map_arr_g)

        active_params[..., 1::2] *= min_freq

        return jhj, term_conv_info(i + 1, cnv_perc)

    return impl
