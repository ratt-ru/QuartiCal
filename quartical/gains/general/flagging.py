# -*- coding: utf-8 -*-
import numpy as np
from numba import jit, generated_jit
import quartical.gains.general.factories as factories
from quartical.utils.numba import coerce_literal


def init_gain_flags(term_shape, term_ind, **kwargs):
    """Initialise the gain flags for a term using the various mappings."""

    flag_col = kwargs["flags"]
    ant1_col = kwargs["a1"]
    ant2_col = kwargs["a2"]
    t_map_arr = kwargs["t_map_arr"][0]
    f_map_arr = kwargs["f_map_arr"][0]

    return _init_flags(term_shape, term_ind, flag_col, ant1_col, ant2_col,
                       t_map_arr, f_map_arr)


def init_param_flags(term_shape, term_ind, **kwargs):
    """Initialise the gain flags for a term using the various mappings."""

    flag_col = kwargs["flags"]
    ant1_col = kwargs["a1"]
    ant2_col = kwargs["a2"]
    t_map_arr = kwargs["t_map_arr"][1]
    f_map_arr = kwargs["f_map_arr"][1]

    return _init_flags(term_shape, term_ind, flag_col, ant1_col, ant2_col,
                       t_map_arr, f_map_arr)


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def _init_flags(term_shape, term_ind, flag_col, ant1_col, ant2_col,
                t_map_arr, f_map_arr):
    """Initialise the flags for a term using the various mappings."""

    flags = np.ones(term_shape[:-1], dtype=np.int8)
    _, _, _, n_dir, _ = term_shape

    n_row, n_chan = flag_col.shape

    for row in range(n_row):
        a1, a2 = ant1_col[row], ant2_col[row]
        ti = t_map_arr[row, term_ind]
        for f in range(n_chan):
            fi = f_map_arr[f, term_ind]
            flag = flag_col[row, f]
            for d in range(n_dir):
                flags[ti, fi, a1, d] &= flag
                flags[ti, fi, a2, d] &= flag

    return flags


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def update_gain_flags(gain, km1_gain, gain_flags, km1_abs2_diffs,
                      abs2_diffs_trend, criteria, corr_mode, iteration):
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

    def impl(gain, km1_gain, gain_flags, km1_abs2_diffs,
             abs2_diffs_trend, criteria, corr_mode, iteration):

        n_tint, n_fint, n_ant, n_dir, n_corr = gain.shape

        criteria_sq = criteria**2
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
                        if iteration < 2:
                            continue

                        # Grab trend at k-1 and update.
                        km1_trend = abs2_diffs_trend[ti, fi, a, d]
                        km0_trend = km1_trend + km0_abs2_diff - km1_abs2_diff

                        abs2_diffs_trend[ti, fi, a, d] = km0_trend

                        # This if-else ladder aims to do the following:
                        # 1) If a point has converged, ensure it is unflagged.
                        # 2) If a point is strictly converging, it should have
                        #    no flags.
                        # 3) If a point strictly diverging, it should be soft
                        #    flagged. If it continues to diverge (twice in a
                        #    row) it should be hard flagged and reset.

                        if km0_abs2_diff/km1_abs2 < criteria_sq:
                            # Unflag points which converged.
                            gain_flags[ti, fi, a, d] = 0
                            n_cnvgd += 1
                        elif km0_trend < km1_trend < 0:
                            gain_flags[ti, fi, a, d] = 0
                        elif km0_trend > km1_trend > 0:
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

        return conv_perc

    return impl


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def finalize_gain_flags(gain, gain_flags, abs2_diffs_trend, mode):
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

    set_identity = factories.set_identity_factory(mode)

    def impl(gain, gain_flags, abs2_diffs_trend, mode):

        n_tint, n_fint, n_ant, n_dir = gain_flags.shape

        for ti in range(n_tint):
            for fi in range(n_fint):
                for a in range(n_ant):
                    for d in range(n_dir):
                        if abs2_diffs_trend[ti, fi, a, d] > 0:
                            gain_flags[ti, fi, a, d] = 1
                            set_identity(gain[ti, fi, a, d])
                        elif gain_flags[ti, fi, a, d] == -1:
                            gain_flags[ti, fi, a, d] = 0

    return impl


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def gain_flags_to_param_flags(gain_flags, param_flags, t_bin_arr, f_map_arr,
                              d_map_arr):
    """Propagate gain flags into parameter flags.

    Given the gain flags, parameter flags and the relevant mappings, propagate
    gain flags into parameter flags. NOTE: This may not be the best approach.
    We could flag on the parameters directly but this is difficult due to
    having a variable set of identitiy paramters and no reasol to believe that
    the parameters live on the same scale.

    Args:
        gain_flags: A (gti, gfi, a, d) array of gain flags.
        param_flags: A (pti, pfi, a, d) array of paramter flag values.
        t_bin_arr: A (2, n_utime, n_term) array of utime to solint mappings.
        f_map_arr: A (2, n_ufreq, n_term) array of ufreq to solint mappings.
        d_map_arr: A (n_term, n_dir) array of direction mappings.
        """

    def impl(gain_flags, param_flags, t_bin_arr, f_map_arr, d_map_arr):

        _, _, n_ant, n_dir = gain_flags.shape

        param_flags[:] = 1

        for gt, pt in zip(t_bin_arr[0], t_bin_arr[1]):
            for gf, pf in zip(f_map_arr[0], f_map_arr[1]):
                for a in range(n_ant):
                    for d in range(n_dir):

                        flag = gain_flags[gt, gf, a, d] == 1
                        param_flags[pt, pf, a, d] &= flag

    return impl


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def apply_gain_flags(gain_flags, flag_col, term_ind, ant1_col, ant2_col,
                     t_map_arr, f_map_arr):
    """Apply gain_flags to flag_col."""

    _, _, _, n_dir = gain_flags.shape

    n_row, n_chan = flag_col.shape

    for row in range(n_row):
        a1, a2 = ant1_col[row], ant2_col[row]
        ti = t_map_arr[row, term_ind]
        for f in range(n_chan):
            fi = f_map_arr[f, term_ind]
            for d in range(1):  # NOTE: Only apply propagate DI flags.
                flag_col[row, f] |= gain_flags[ti, fi, a1, d] == 1
                flag_col[row, f] |= gain_flags[ti, fi, a2, d] == 1
