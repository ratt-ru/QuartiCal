# -*- coding: utf-8 -*-
import numpy as np
from numba import generated_jit
from quartical.utils.numba import coerce_literal
from quartical.gains.general.generics import (compute_residual_solver,
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


flag_intermediaries = namedtuple(
    "flag_intermediaries",
    (
        "km1_gain",
        "km1_abs2_diffs",
        "abs2_diffs_trend"
    )

)


solver_intermediaries = namedtuple(
    "solver_intermediaries",
    (
        "jhj",
        "jhr",
        "residual",
        "update"
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

        data = base_args.data
        gains = base_args.gains
        gain_flags = base_args.gain_flags

        active_term = meta_args.active_term
        max_iter = meta_args.iters
        solve_per = meta_args.solve_per
        dd_term = meta_args.dd_term

        active_gain = gains[active_term]
        active_gain_flags = gain_flags[active_term]
        active_params = term_args.params[active_term]

        # Set up some intemediaries used for flagging. TODO: Move?
        km1_gain = active_gain.copy()
        km1_abs2_diffs = np.zeros_like(active_gain_flags, dtype=np.float64)
        abs2_diffs_trend = np.zeros_like(active_gain_flags, dtype=np.float64)
        flag_imdry = \
            flag_intermediaries(km1_gain, km1_abs2_diffs, abs2_diffs_trend)

        # Set up some intemediaries used for solving. TODO: Move?
        real_dtype = active_gain.real.dtype
        pshape = active_params.shape
        jhj = np.empty(pshape + (pshape[-1],), dtype=real_dtype)
        jhr = np.empty(pshape, dtype=real_dtype)
        residual = np.empty_like(data) if dd_term else data
        update = np.zeros_like(jhr)
        solver_imdry = solver_intermediaries(jhj, jhr, residual, update)

        scaled_icf = term_args.chan_freqs.copy()  # Don't mutate.
        min_freq = np.min(scaled_icf)
        scaled_icf = min_freq/scaled_icf  # Scale freqs to avoid precision.
        active_params[..., 1::2] /= min_freq  # Scale consistently with freq.

        for loop_idx in range(max_iter):

            if dd_term or len(gains) > 1:
                compute_residual_solver(base_args,
                                        solver_imdry,
                                        corr_mode)

            compute_jhj_jhr(base_args,
                            term_args,
                            meta_args,
                            solver_imdry,
                            scaled_icf,
                            corr_mode)

            if solve_per == "array":
                per_array_jhj_jhr(solver_imdry)

            compute_update(solver_imdry,
                           corr_mode)

            finalize_update(base_args,
                            term_args,
                            meta_args,
                            solver_imdry,
                            scaled_icf,
                            loop_idx,
                            corr_mode)

            # Check for gain convergence. Produced as a side effect of
            # flagging. The converged percentage is based on unflagged
            # intervals.
            conv_perc = update_gain_flags(base_args,
                                          term_args,
                                          meta_args,
                                          flag_imdry,
                                          loop_idx,
                                          corr_mode)

            if conv_perc > meta_args.stop_frac:
                break

        # NOTE: Removes soft flags and flags points which have bad trends.
        finalize_gain_flags(base_args,
                            meta_args,
                            flag_imdry,
                            corr_mode)

        # Propagate gain flags to parameter flags. TODO: Verify that this
        # is adequate. Do we need to consider setting the identity params.
        gain_flags_to_param_flags(base_args,
                                  term_args,
                                  meta_args)

        # Call this one last time to ensure points flagged by finialize are
        # propagated (in the DI case).
        if not dd_term:
            apply_gain_flags(base_args,
                             meta_args)

        active_params[..., 1::2] *= min_freq  # Undo scaling for SI units.

        return jhj, term_conv_info(loop_idx + 1, conv_perc)

    return impl
