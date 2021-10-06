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
def compute_bl_mad(chisq, flags, a1, a2, n_ant):

    mad_per_bl = np.zeros((1, n_ant, n_ant), dtype=chisq.dtype)

    for a in range(n_ant):
        for b in range(a + 1, n_ant):

            bl_sel = \
                np.where(((a1 == a) & (a2 == b)) | ((a1 == b) & (a2 == a)))

            if not bl_sel[0].size:  # Missing baseline.
                mad_per_bl[0, a, b] = 0
                continue

            bl_chisq = chisq[bl_sel].flatten()
            bl_flags = flags[bl_sel].flatten()  # No correlation axis.

            unflagged_sel = np.where(bl_flags == 0)

            if unflagged_sel[0].size:
                bl_median = np.median(bl_chisq[unflagged_sel])
                bl_mad = np.median(np.abs(bl_chisq - bl_median)[unflagged_sel])
                mad_per_bl[0, a, b] = bl_mad
            else:
                mad_per_bl[0, a, b] = 0

    return mad_per_bl


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def compute_gbl_mad(chisq, flags):

    gbl_chisq = chisq.flatten()
    unflagged_sel = np.where(flags.flatten() == 0)

    gbl_median = np.median(gbl_chisq[unflagged_sel])
    gbl_mad = np.median(np.abs(gbl_chisq - gbl_median)[unflagged_sel])

    return gbl_mad


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def compute_mad_flags(chisq, gbl_mad_est, bl_mad_est, ant1, ant2,
                      gbl_threshold, bl_threshold, max_deviation):

    flags = np.zeros_like(chisq, dtype=np.int8)

    scale_factor = 1.4826

    gbl_std = scale_factor * gbl_mad_est  # MAD to standard deviation.

    gbl_cutoff = gbl_threshold * gbl_std

    gbl_std_threshold = max_deviation * gbl_std

    n_row, n_chan = chisq.shape

    for row in range(n_row):

        a1_m, a2_m = ant1[row], ant2[row]

        bl_std = scale_factor * bl_mad_est[0, a1_m, a2_m]

        if bl_std > gbl_std_threshold:
            flags[row] = 1
            continue

        bl_cutoff = bl_threshold * bl_std

        cutoff = min(gbl_cutoff, bl_cutoff)

        for chan in range(n_chan):

            if chisq[row, chan] > cutoff:
                flags[row, chan] = 1

    return flags
