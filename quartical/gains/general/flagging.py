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
    t_map_arr = kwargs["t_map_arr"]
    f_map_arr = kwargs["f_map_arr"]

    return _init_gain_flags(term_shape, term_ind, flag_col, ant1_col, ant2_col,
                            t_map_arr, f_map_arr)


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def _init_gain_flags(term_shape, term_ind, flag_col, ant1_col, ant2_col,
                     t_map_arr, f_map_arr):
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


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def update_gain_flags(gain, last_gain, gain_flags, rel_diffs, criteria,
                      corr_mode, initial=False):

    coerce_literal(update_gain_flags, ['corr_mode'])

    set_identity = factories.set_identity_factory(corr_mode)

    def impl(gain, last_gain, gain_flags, rel_diffs, criteria,
             corr_mode, initial=False):

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
                            set_identity(gain[ti, fi, a, d])
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
                        # 2) If a point is diverging, it should be soft
                        #    flagged. If it continues to diverge (twice in a
                        #    row) it should be hard flagged and reset.
                        # 3) If a point was soft flagged due to divergence but
                        #    then takes a step towards convergence, remove its
                        #    its soft flag.

                        if new_rel_diff < criteria_sq:
                            # Unflag points which converged.
                            gain_flags[ti, fi, a, d] = 0
                            n_cnvgd += 1
                        elif old_rel_diff < new_rel_diff:
                            # Soft flag antennas which have a diverging
                            # rel_diff. Points which are soft flagged twice
                            # in a row are hard flagged. Reset flagged points.
                            if gain_flags[ti, fi, a, d] == -1:
                                gain_flags[ti, fi, a, d] = 1
                                set_identity(gain[ti, fi, a, d])
                                n_flagged += 1
                            else:
                                gain_flags[ti, fi, a, d] = -1
                        else:
                            # Point started to converge, remove (soft) flags.
                            gain_flags[ti, fi, a, d] = 0

        n_solvable = (n_tint*n_fint*n_ant*n_dir - n_flagged)

        if n_solvable:
            conv_perc = n_cnvgd/n_solvable
        else:
            conv_perc = 0.

        return conv_perc

    return impl


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def finalize_gain_flags(gain_flags):
    """Removes soft flags which were raised on the final iteration."""

    n_tint, n_fint, n_ant, n_dir = gain_flags.shape

    for ti in range(n_tint):
        for fi in range(n_fint):
            for a in range(n_ant):
                for d in range(n_dir):
                    if gain_flags[ti, fi, a, d] == -1:
                        gain_flags[ti, fi, a, d] = 0


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
