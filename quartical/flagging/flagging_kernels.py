import numpy as np
from numba import jit


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def madmax(resid_arr, weights, flags, a1, a2, n_ant):

    median_per_bl = np.zeros((1, n_ant, n_ant), dtype=resid_arr.real.dtype)

    for a in range(n_ant):
        for b in range(a + 1, n_ant):

            # This should catch conjugate points. We don't care, as we take the
            # absolute value anyway.

            bl_sel = \
                np.where(((a1 == a) & (a2 == b)) | ((a1 == b) & (a2 == a)))

            if not bl_sel[0].size:  # Missing baseline.
                median_per_bl[0, a, b] = np.inf
                continue

            bl_resid = resid_arr[bl_sel].flatten()
            bl_weights = weights[bl_sel].flatten()
            bl_flags = flags[bl_sel].flatten()

            abs_bl_resid = \
                np.sqrt((bl_resid.conj() * bl_weights * bl_resid).real)

            unflagged_sel = np.where(bl_flags == 0)

            if unflagged_sel[0].size:
                median_per_bl[0, a, b] = np.median(abs_bl_resid[unflagged_sel])
            else:
                median_per_bl[0, a, b] = np.inf

    return median_per_bl


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def threshold(resid_arr, weights, mad_ests, med_mad_ests, flags, a1, a2):

    n_row, n_chan, n_corr = resid_arr.shape

    bad_values = np.zeros(resid_arr.shape, dtype=np.bool_)

    sigma = 1.4826

    for row in range(n_row):

        a1_m, a2_m = a1[row], a2[row]
        thr = min(mad_ests[0, a1_m, a2_m]/sigma, med_mad_ests[0]/sigma)

        for chan in range(n_chan):
            for corr in range(n_corr):

                r = resid_arr[row, chan, corr]
                r_conj = r.conjugate()
                w = weights[row, chan, corr]

                bad_values[row, chan, corr] = \
                    np.sqrt((r_conj * w * r).real) > thr

    return bad_values
