import numpy as np
from numba import jit


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def compute_whitened_residual(resid_arr, weights):

    n_row, n_chan, n_corr = resid_arr.shape

    wres = np.zeros((n_row, n_chan, n_corr), dtype=resid_arr.dtype)

    for r in range(n_row):
        for f in range(n_chan):
            for c in range(n_corr):

                r_rfc = resid_arr[r, f, c]
                w_rfc = weights[r, f, c]

                wres[r, f, c] = np.sqrt(w_rfc) * r_rfc

    return wres


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def compute_bl_mad_and_med(wres, flags, a1, a2, n_ant):

    n_corr = wres.shape[-1]

    # TODO: Make this a bl dim array, with computed indices.
    mad_and_med = np.zeros((1, 1, n_ant, n_ant, n_corr, 2), dtype=wres.dtype)

    for a in range(n_ant):
        for b in range(a + 1, n_ant):

            bl_sel = \
                np.where(((a1 == a) & (a2 == b)) | ((a1 == b) & (a2 == a)))[0]

            if not bl_sel.size:  # Missing baseline.
                continue

            bl_flags = flags[bl_sel].ravel()
            unflagged_sel = np.where(bl_flags == 0)

            if unflagged_sel[0].size:  # Not fully flagged.
                for c in range(n_corr):

                    bl_wres = wres[bl_sel, :, c].ravel()

                    bl_median = np.median(bl_wres[unflagged_sel])
                    bl_mad = \
                        np.median(np.abs(bl_wres - bl_median)[unflagged_sel])
                    mad_and_med[0, 0, a, b, c, 0] = bl_mad
                    mad_and_med[0, 0, a, b, c, 1] = bl_median

    return mad_and_med


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def compute_gbl_mad_and_med(wres, flags):

    n_corr = wres.shape[-1]

    mad_and_med = np.zeros((1, 1, n_corr, 2), dtype=wres.dtype)

    unflagged_sel = np.where(flags.ravel() == 0)

    if unflagged_sel[0].size:  # We have unflagged data.
        for c in range(n_corr):
            gbl_wres = wres[..., c].ravel()[unflagged_sel]

            gbl_median = np.median(gbl_wres)
            gbl_mad = np.median(np.abs(gbl_wres - gbl_median))

            mad_and_med[0, 0, c, 0] = gbl_mad
            mad_and_med[0, 0, c, 1] = gbl_median

    return mad_and_med


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def compute_mad_flags(
    wres,
    gbl_mad_and_med,
    bl_mad_and_med,
    ant1,
    ant2,
    gbl_threshold,
    bl_threshold,
    max_deviation
):

    n_row, n_chan, n_corr = wres.shape

    flags = np.zeros((n_row, n_chan), dtype=np.int8)

    scale_factor = 1.4826

    for corr in range(n_corr):

        gbl_mad = gbl_mad_and_med[0, 0, corr, 0]
        gbl_med = gbl_mad_and_med[0, 0, corr, 1]
        bl_mad = bl_mad_and_med[0, 0, :, :, corr, 0]
        bl_med = bl_mad_and_med[0, 0, :, :, corr, 1]

        if gbl_mad == 0:  # Indicates that all data was flagged.
            flags[:] = 1
            continue

        gbl_std = scale_factor * gbl_mad  # MAD to standard deviation.
        gbl_cutoff = gbl_threshold * gbl_std

        # This is the MAD estimate of the baseline MAD estimates. We want to
        # use this to find bad baselines.
        valid_bl_mad = bl_mad.flatten()[bl_mad.flatten() > 0]
        valid_bl_mad_median = np.median(valid_bl_mad)
        bl_mad_mad = np.median(np.abs(valid_bl_mad - valid_bl_mad_median))

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

                if np.abs(wres[row, chan, corr] - med_pq) > bl_cutoff:
                    flags[row, chan] = 1

                if np.abs(wres[row, chan, corr] - gbl_med) > gbl_cutoff:
                    flags[row, chan] = 1

    return flags
