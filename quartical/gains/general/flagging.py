# -*- coding: utf-8 -*-
import numpy as np
from numba import jit


def init_gain_flags(term_shape, term_ind, **kwargs):
    """Initialise the gain flags for a term using the various mappings."""

    flag_col = kwargs["flags"]
    ant1_col = kwargs["a1"]
    ant2_col = kwargs["a2"]
    t_map_arr = kwargs["t_map_arr"]
    f_map_arr = kwargs["f_map_arr"]
    row_map = kwargs["row_map"]

    return _init_gain_flags(term_shape, term_ind, flag_col, ant1_col, ant2_col,
                            t_map_arr, f_map_arr, row_map)


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def _init_gain_flags(term_shape, term_ind, flag_col, ant1_col, ant2_col,
                     t_map_arr, f_map_arr, row_map):
    """Initialise the gain flags for a term using the various mappings."""

    # TODO: Consider what happens in the parameterised case.

    gain_flags = np.ones(term_shape[:-1], dtype=np.int8)
    _, _, _, n_dir, _ = term_shape

    n_row, n_chan = flag_col.shape

    for row in range(n_row):
        a1, a2 = ant1_col[row], ant2_col[row]
        ti = t_map_arr[0, row, term_ind]
        for f in range(n_chan):
            fi = f_map_arr[0, f, term_ind]
            flag = flag_col[row, f]
            for d in range(n_dir):
                gain_flags[ti, fi, a1, d] &= flag
                gain_flags[ti, fi, a2, d] &= flag

    return gain_flags


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def update_gain_flags(gain, last_gain, gain_flags, rel_diffs, criteria,
                      initial=False):

    n_tint, n_fint, n_ant, n_dir, n_corr = gain.shape

    criteria_sq = criteria**2
    n_cnvgd = 0
    n_flagged = 0

    for ti in range(n_tint):
        for fi in range(n_fint):
            for a in range(n_ant):
                for d in range(n_dir):

                    # If the gain is hard flagged, we can skip these checks.
                    if gain_flags[ti, fi, a, d] == 1:
                        n_flagged += 1
                        continue

                    gain_abs2 = 0
                    gain_diff_abs2 = 0

                    for c in range(n_corr):

                        gsel = gain[ti, fi, a, d, c]
                        lgsel = last_gain[ti, fi, a, d, c]

                        diff = lgsel - gsel

                        gain_abs2 += gsel.real**2 + gsel.imag**2
                        gain_diff_abs2 += diff.real**2 + diff.imag**2

                    if gain_abs2 == 0:  # TODO: Precaution, not ideal.
                        gain_flags[ti, fi, a, d] = 1
                        n_flagged += 1
                        continue

                    new_rel_diff = gain_diff_abs2/gain_abs2
                    old_rel_diff = rel_diffs[ti, fi, a, d]
                    rel_diffs[ti, fi, a, d] = new_rel_diff

                    # If initial is set, we don't want to flag on this run.
                    if initial:
                        continue

                    # This nasty if-else ladder aims to do the following:
                    # 1) If a point has converged, ensure it is unflagged.
                    # 2) If a point is has failed to converge by the final
                    #    iteration or has not converged when the stopping
                    #    criteria is reached, it should be flagged.
                    # 3) If a point is diverging, it should be soft flagged
                    #    if this is the first time. It should be hard flagged
                    #    if this is the second time. All soft flags should be
                    #    discarded on the final iteration if they were not
                    #    hardened due to divergence.

                    if new_rel_diff < criteria_sq:
                        # Unflag points which converged.
                        gain_flags[ti, fi, a, d] = 0
                        n_cnvgd += 1
                    # elif final:
                    #     # Flag points which didn't converge.
                    #     gain_flags[ti, fi, a, d] = 1
                    #     n_flagged += 1
                    elif old_rel_diff < new_rel_diff:
                        # Soft flag antennas which have a diverging rel_diff.
                        # Points which are soft flagged twice are hard flagged.
                        if gain_flags[ti, fi, a, d] == -1:
                            gain_flags[ti, fi, a, d] = 1
                            n_flagged += 1
                        else:
                            gain_flags[ti, fi, a, d] = -1
                    else:
                        # If the point took a good step, remove (soft) flags.
                        gain_flags[ti, fi, a, d] = 0

    n_solvable = (n_tint*n_fint*n_ant*n_dir - n_flagged)

    if n_solvable:
        conv_perc = n_cnvgd/n_solvable
    else:
        conv_perc = 0.

    return conv_perc
