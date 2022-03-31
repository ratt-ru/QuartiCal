import numpy as np
from numba import jit


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def compute_chisq(resid_arr, weights):

    n_row, n_chan, n_corr = resid_arr.shape

    chisq = np.zeros((n_row, n_chan), dtype=resid_arr.real.dtype)

    for r in range(n_row):
        for f in range(n_chan):
            for c in range(n_corr):

                r_rfc = resid_arr[r, f, c]
                r_rfc_conj = r_rfc.conjugate()
                w_rfc = weights[r, f, c]

                chisq[r, f] += (r_rfc_conj * w_rfc * r_rfc).real

    return chisq


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def compute_bl_mad_and_med(chisq, flags, a1, a2, n_ant):

    mad_and_med = np.zeros((2, n_ant, n_ant), dtype=chisq.dtype)

    for a in range(n_ant):
        for b in range(a + 1, n_ant):

            bl_sel = \
                np.where(((a1 == a) & (a2 == b)) | ((a1 == b) & (a2 == a)))

            if not bl_sel[0].size:  # Missing baseline.
                continue

            bl_chisq = chisq[bl_sel].flatten()
            bl_flags = flags[bl_sel].flatten()  # No correlation axis.

            unflagged_sel = np.where(bl_flags == 0)

            if unflagged_sel[0].size:
                bl_median = np.median(bl_chisq[unflagged_sel])
                bl_mad = np.median(np.abs(bl_chisq - bl_median)[unflagged_sel])
                mad_and_med[0, a, b] = bl_mad
                mad_and_med[1, a, b] = bl_median

    return mad_and_med


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def compute_gbl_mad_and_med(chisq, flags):

    gbl_chisq = chisq.flatten()
    unflagged_sel = np.where(flags.flatten() == 0)

    if unflagged_sel[0].size:  # We have unflagged data.
        gbl_median = np.median(gbl_chisq[unflagged_sel])
        gbl_mad = np.median(np.abs(gbl_chisq - gbl_median)[unflagged_sel])
    else:
        gbl_median = 0
        gbl_mad = 0

    return np.array((gbl_mad, gbl_median)).astype(chisq.dtype)


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def compute_mad_flags(chisq, gbl_mad_and_med, bl_mad_and_med, ant1, ant2,
                      gbl_threshold, bl_threshold, max_deviation):

    flags = np.zeros_like(chisq, dtype=np.int8)

    gbl_mad, gbl_med = gbl_mad_and_med
    bl_mad, bl_med = bl_mad_and_med

    if gbl_mad == 0:  # Indicates that all data was flagged.
        flags[:] = 1
        return flags

    scale_factor = 1.4826

    gbl_std = scale_factor * gbl_mad  # MAD to standard deviation.
    gbl_cutoff = gbl_threshold * gbl_std

    # This is the MAD estimate of the baseline MAD estimates. We want to use
    # this to find bad baselines.
    valid_bl_mad = bl_mad.flatten()[bl_mad.flatten() > 0]
    valid_bl_mad_median = np.median(valid_bl_mad)
    bl_mad_mad = np.median(np.abs(valid_bl_mad - valid_bl_mad_median))

    n_row, n_chan = chisq.shape

    for row in range(n_row):

        a1_m, a2_m = ant1[row], ant2[row]

        mad_pq = bl_mad[a1_m, a2_m]
        med_pq = bl_med[a1_m, a2_m]

        bl_mad_deviation = np.abs(mad_pq - valid_bl_mad_median)/bl_mad_mad

        if bl_mad_deviation > max_deviation:
            flags[row] = 1
            continue

        bl_std = scale_factor * mad_pq
        bl_cutoff = bl_threshold * bl_std

        for chan in range(n_chan):

            if np.abs(chisq[row, chan] - med_pq) > bl_cutoff:
                flags[row, chan] = 1

            if np.abs(chisq[row, chan] - gbl_med) > gbl_cutoff:
                flags[row, chan] = 1

    return flags
