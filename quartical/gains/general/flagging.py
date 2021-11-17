# -*- coding: utf-8 -*-
import numpy as np
from numba import prange, generated_jit, jit
from quartical.utils.numba import coerce_literal
import quartical.gains.general.factories as factories


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def update_gain_flags(gain, last_gain, gain_flags, rel_diffs, criteria,
                      initial=False, final=False):

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

                    if gain_abs2 == 0:  # Missing data. TODO: Catch elsewhere.
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
                    elif final:
                        # Flag points which didn't converge.
                        gain_flags[ti, fi, a, d] = 1
                        n_flagged += 1
                    elif old_rel_diff < new_rel_diff:
                        # Soft flag antennas which have a diverging rel_diff.
                        # Points which are soft flagged twice are hard flagged.
                        if gain_flags[ti, fi, a, d] == -1:
                            gain_flags[ti, fi, a, d] = 1
                            n_flagged += 1
                        elif final:
                            # Discard soft flags raised on final call.
                            gain_flags[ti, fi, a, d] = 0
                        else:
                            gain_flags[ti, fi, a, d] = -1

    n_solvable = (n_tint*n_fint*n_ant*n_dir - n_flagged)

    if n_solvable:
        conv_perc = n_cnvgd/n_solvable
    else:
        conv_perc = 0.

    return conv_perc
