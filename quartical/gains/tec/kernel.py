# -*- coding: utf-8 -*-
import numpy as np
from numba import generated_jit
from quartical.utils.numba import coerce_literal
from quartical.gains.general.generics import (compute_residual,
                                              compute_convergence,
                                              per_array_jhj_jhr)
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
        t_map_arr = base_args.t_map_arr[0]  # Don't need time param mappings.
        f_map_arr_g = base_args.f_map_arr[0]  # Gain mappings.
        f_map_arr_p = base_args.f_map_arr[1]  # Parameter mappings.
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

        params = term_args.params[active_term]  # Params for this term.
        t_bin_arr = term_args.t_bin_arr[0]  # Don't need time param mappings.
        chan_freqs = term_args.chan_freqs.copy()  # Don't mutate orginal.
        min_freq = np.min(chan_freqs)
        inv_chan_freqs = min_freq/chan_freqs  # Scale freqs to avoid precision.
        params[..., 1::2] /= min_freq  # Scale consistently with freq.

        n_term = len(gains)

        active_gain = gains[active_term]

        dd_term = np.any(d_map_arr[active_term])

        last_gain = active_gain.copy()

        cnv_perc = 0.

        real_dtype = gains[active_term].real.dtype

        pshape = params.shape
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
                                            t_map_arr,
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
                            t_map_arr,
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
                            params,
                            gains[active_term],
                            inv_chan_freqs,
                            t_bin_arr,
                            f_map_arr_p,
                            d_map_arr,
                            dd_term,
                            active_term,
                            corr_mode)

            # Check for gain convergence. TODO: This can be affected by the
            # weights. Currently unsure how or why, but using unity weights
            # leads to monotonic convergence in all solution intervals.

            cnv_perc = compute_convergence(gains[active_term][:],
                                           last_gain,
                                           stop_crit)

            last_gain[:] = gains[active_term][:]

            if cnv_perc >= stop_frac:
                break

        params[..., 1::2] *= min_freq

        return jhj, term_conv_info(i + 1, cnv_perc)

    return impl
