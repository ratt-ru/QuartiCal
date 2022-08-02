# -*- coding: utf-8 -*-
import numpy as np
from numba import jit, generated_jit
from collections import namedtuple
from quartical.gains.general.convenience import get_row
import quartical.gains.general.factories as factories
from quartical.utils.numba import coerce_literal


flag_intermediaries = namedtuple(
    "flag_intermediaries",
    (
        "km1_gain",
        "km1_abs2_diffs",
        "abs2_diffs_trend"
    )

)


def init_gain_flags(term_shape, term_ind, **kwargs):
    """Initialise the gain flags for a term using the various mappings."""

    flag_col = kwargs["flags"]
    ant1_col = kwargs["a1"]
    ant2_col = kwargs["a2"]
    t_map_arr = kwargs["t_map_arr"][0]
    f_map_arr = kwargs["f_map_arr"][0]
    row_map = kwargs.get("row_map", None)

    return _init_flags(term_shape, term_ind, flag_col, ant1_col, ant2_col,
                       t_map_arr, f_map_arr, row_map)


def init_param_flags(term_shape, term_ind, **kwargs):
    """Initialise the param flags for a term using the various mappings."""

    flag_col = kwargs["flags"]
    ant1_col = kwargs["a1"]
    ant2_col = kwargs["a2"]
    t_map_arr = kwargs["t_map_arr"][1]
    f_map_arr = kwargs["f_map_arr"][1]
    row_map = kwargs.get("row_map", None)

    return _init_flags(term_shape, term_ind, flag_col, ant1_col, ant2_col,
                       t_map_arr, f_map_arr, row_map)


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def _init_flags(term_shape, term_ind, flag_col, ant1_col, ant2_col,
                t_map_arr, f_map_arr, row_map):
    """Initialise the flags for a term using the various mappings."""

    flags = np.ones(term_shape[:-1], dtype=np.int8)
    _, _, _, n_dir, _ = term_shape

    n_row = t_map_arr.shape[0]
    n_chan = f_map_arr.shape[0]

    for row_ind in range(n_row):
        ti = t_map_arr[row_ind, term_ind]

        # NOTE: The following handles the BDA case where an element in the
        # time map may be backed by a different row in the data.
        row = get_row(row_ind, row_map)
        a1, a2 = ant1_col[row], ant2_col[row]

        for f in range(n_chan):
            fi = f_map_arr[f, term_ind]
            flag = flag_col[row, f]
            for d in range(n_dir):
                flags[ti, fi, a1, d] &= flag
                flags[ti, fi, a2, d] &= flag

    return flags


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def update_gain_flags(base_args, term_args, meta_args, flag_imdry, loop_idx,
                      corr_mode, numbness=1e-6):
    """Update the current state of the gain flags.

    Uses the current (km0) and previous (km1) gains to identify diverging
    soultions. This implements trendy flagging - see overleaf document.
    TODO: Add link.

    Args:
        gain: A (ti, fi, a, d, c) array of gain values.
        km1_gain: A (ti, fi, a, d, c) array of gain values at prev iteration.
        gain_flags: A (ti, fi, a, d) array of flag values.
        km1_abs2_diffs: A (ti, fi, a, d) itemediary array to store the
            previous absolute difference in the gains.
        ab2_diffs_trends: A (ti, fi, a, d) itemediary array to store the
            accumulated trend values of the differences between absolute
            differences of the gains.
        critera: A float value below which a gain is considered converged.
        corr_mode: An int which controls how we handle coreelations.
        iteration: An int containing the iteration number.
    """

    coerce_literal(update_gain_flags, ['corr_mode'])

    set_identity = factories.set_identity_factory(corr_mode)

    def impl(
        base_args,
        term_args,
        meta_args,
        flag_imdry,
        loop_idx,
        corr_mode,
        numbness=1e-6
    ):

        active_term = meta_args.active_term
        max_iter = meta_args.iters
        stop_frac = meta_args.stop_frac
        stop_crit2 = meta_args.stop_crit**2

        gain = base_args.gains[active_term]
        gain_flags = base_args.gain_flags[active_term]

        km1_gain = flag_imdry.km1_gain
        km1_abs2_diffs = flag_imdry.km1_abs2_diffs
        abs2_diffs_trend = flag_imdry.abs2_diffs_trend

        n_tint, n_fint, n_ant, n_dir, n_corr = gain.shape

        n_cnvgd = 0
        n_flagged = 0

        for ti in range(n_tint):
            for fi in range(n_fint):
                for a in range(n_ant):
                    for d in range(n_dir):

                        # We can skip points which are already hard flagged.
                        if gain_flags[ti, fi, a, d] == 1:
                            n_flagged += 1
                            continue

                        # Relative difference: (|g_k-1 - g_k|/|g_k-1|)^2.
                        km1_abs2 = 0
                        km0_abs2_diff = 0

                        for c in range(n_corr):

                            km0_g = gain[ti, fi, a, d, c]
                            km1_g = km1_gain[ti, fi, a, d, c]

                            diff = km1_g - km0_g

                            km1_abs2 += km1_g.real**2 + km1_g.imag**2
                            km0_abs2_diff += diff.real**2 + diff.imag**2

                        if km1_abs2 == 0:  # TODO: Precaution, not ideal.
                            gain_flags[ti, fi, a, d] = 1
                            set_identity(gain[ti, fi, a, d])
                            n_flagged += 1
                            continue

                        # Grab absolute difference squared at k-1 and update.
                        km1_abs2_diff = km1_abs2_diffs[ti, fi, a, d]
                        km1_abs2_diffs[ti, fi, a, d] = km0_abs2_diff

                        # We cannot flag on the first few iterations.
                        if loop_idx < 2:
                            continue

                        # Grab trend at k-1 and update.
                        km1_trend = abs2_diffs_trend[ti, fi, a, d]
                        km0_trend = km1_trend + km0_abs2_diff - km1_abs2_diff

                        abs2_diffs_trend[ti, fi, a, d] = km0_trend

                        # This if-else ladder aims to do the following:
                        # 1) If a point has converged, ensure it is unflagged.
                        # 2) If a point is strictly converging, it should have
                        #    no flags. Note we allow a small epsilon of
                        #    "numbness" - this is important if our initial
                        #    estimate is very close to the solution.
                        # 3) If a point strictly diverging, it should be soft
                        #    flagged. If it continues to diverge (twice in a
                        #    row) it should be hard flagged and reset.

                        if km0_abs2_diff/km1_abs2 < stop_crit2:
                            # Unflag points which converged.
                            gain_flags[ti, fi, a, d] = 0
                            n_cnvgd += 1
                        elif km0_trend < km1_trend < numbness:
                            gain_flags[ti, fi, a, d] = 0
                        elif km0_trend > km1_trend > numbness:
                            gain_flags[ti, fi, a, d] = \
                                1 if gain_flags[ti, fi, a, d] else -1

                        if gain_flags[ti, fi, a, d] == 1:
                            n_flagged += 1
                            set_identity(gain[ti, fi, a, d])

        n_solvable = (n_tint*n_fint*n_ant*n_dir - n_flagged)

        if n_solvable:
            conv_perc = n_cnvgd/n_solvable
        else:
            conv_perc = 0.

        # Update the k-1 gain if not converged/on final iteration.
        if (conv_perc < stop_frac) and (loop_idx < max_iter - 1):
            km1_gain[:] = gain

        return conv_perc

    return impl


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def finalize_gain_flags(base_args, meta_args, flag_imdry, corr_mode):
    """Removes soft flags and flags points which failed to converge.

    Given the gains, assosciated gain flags and the trend of abosolute
    differences, remove soft flags which were never hardened and hard flag
    points which have positive trend values. This corresponds to points
    which have bad solutions when convergence/maximum iterations are reached.

    Args:
        gain: A (ti, fi, a, d, c) array of gain values.
        gain_flags: A (ti, fi, a, d) array of flag values.
        ab2_diffs_trends: An array containing the accumulated trend values of
            the absolute difference between gains at each iteration. Positive
            values correspond to points which are nowhere near convergence.
    """

    set_identity = factories.set_identity_factory(corr_mode)

    def impl(base_args, meta_args, flag_imdry, corr_mode):

        active_term = meta_args.active_term

        gain = base_args.gains[active_term]
        gain_flags = base_args.gain_flags[active_term]

        abs2_diffs_trend = flag_imdry.abs2_diffs_trend

        n_tint, n_fint, n_ant, n_dir = gain_flags.shape

        for ti in range(n_tint):
            for fi in range(n_fint):
                for a in range(n_ant):
                    for d in range(n_dir):
                        if abs2_diffs_trend[ti, fi, a, d] > 1e-6:
                            gain_flags[ti, fi, a, d] = 1
                            set_identity(gain[ti, fi, a, d])
                        elif gain_flags[ti, fi, a, d] == -1:
                            gain_flags[ti, fi, a, d] = 0

    return impl


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def update_param_flags(base_args, term_args, meta_args, identity_params):
    """Propagate gain flags into parameter flags.

    Given the gain flags, parameter flags and the relevant mappings, propagate
    gain flags into parameter flags. NOTE: This may not be the best approach.
    We could flag on the parameters directly but this is difficult due to
    having a variable set of identitiy paramters and no reason to believe that
    the parameters live on the same scale.

    Args:
        gain_flags: A (gti, gfi, a, d) array of gain flags.
        param_flags: A (pti, pfi, a, d) array of paramter flag values.
        t_bin_arr: A (2, n_utime, n_term) array of utime to solint mappings.
        f_map_arr: A (2, n_ufreq, n_term) array of ufreq to solint mappings.
        d_map_arr: A (n_term, n_dir) array of direction mappings.
        """

    def impl(base_args, term_args, meta_args, identity_params):

        active_term = meta_args.active_term

        # NOTE: We don't yet let params and gains have different direction
        # maps but this will eventually be neccessary.
        t_bin_arr = term_args.t_bin_arr[:, :, active_term]
        f_map_arr = base_args.f_map_arr[:, :, active_term]

        gain_flags = base_args.gain_flags[active_term]
        param_flags = term_args.param_flags[active_term]
        params = term_args.params[active_term]

        _, _, n_ant, n_dir = gain_flags.shape

        param_flags[:] = 1

        for gt, pt in zip(t_bin_arr[0], t_bin_arr[1]):
            for gf, pf in zip(f_map_arr[0], f_map_arr[1]):
                for a in range(n_ant):
                    for d in range(n_dir):

                        flag = gain_flags[gt, gf, a, d] == 1
                        param_flags[pt, pf, a, d] &= flag

        n_tint, n_fint, n_ant, n_dir = param_flags.shape

        for ti in range(n_tint):
            for fi in range(n_fint):
                for a in range(n_ant):
                    for d in range(n_dir):
                        if param_flags[ti, fi, a, d] == 1:
                            params[ti, fi, a, d] = identity_params

    return impl


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def apply_gain_flags(base_args, meta_args):
    """Apply gain_flags to flag_col."""

    active_term = meta_args.active_term

    gain_flags = base_args.gain_flags[active_term]
    flag_col = base_args.flags
    ant1_col = base_args.a1
    ant2_col = base_args.a2

    # Select out just the mappings we need.
    t_map_arr = base_args.t_map_arr[0, :, active_term]
    f_map_arr = base_args.f_map_arr[0, :, active_term]

    n_row, n_chan = flag_col.shape

    for row in range(n_row):
        a1, a2 = ant1_col[row], ant2_col[row]
        t_m = t_map_arr[row]
        for f in range(n_chan):
            f_m = f_map_arr[f]

            # NOTE: We only care about the DI case for now.
            flag_col[row, f] |= gain_flags[t_m, f_m, a1, 0] == 1
            flag_col[row, f] |= gain_flags[t_m, f_m, a2, 0] == 1
