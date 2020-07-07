import numpy as np
from numba import jit


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def madmax(resid_arr, flags, a1, a2, n_ant):

    n_row, n_chan, n_corr = resid_arr.shape

    median_per_bl = np.zeros((1, n_ant, n_ant), dtype=resid_arr.real.dtype)

    for a in range(n_ant):
        for b in range(a, n_ant):

            # This should catch conjugate points. We don't care, as we take the
            # absolue value anyway.

            bl_sel = \
                np.where(((a1 == a) & (a2 == b)) | ((a1 == b) & (a2 == a)))

            bl_resid = resid_arr[bl_sel].flatten()

            abs_bl_resid = np.sqrt(bl_resid.real**2 + bl_resid.imag**2)

            unflagged_sel = np.where(flags[bl_sel].flatten() == 0)

            if unflagged_sel[0].size == 0:
                median_per_bl[0, a, b] = np.inf
            else:
                median_per_bl[0, a, b] = np.median(abs_bl_resid[unflagged_sel])

    return median_per_bl


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def threshold(resid_arr, mad_ests, med_mad_ests, flags, a1, a2):

    n_row, n_chan, n_corr = resid_arr.shape

    bad_values = np.zeros(resid_arr.shape, dtype=np.bool_)

    sigma = 1.4826

    for row in range(n_row):

        a1_m, a2_m = a1[row], a2[row]
        thr = min(mad_ests[0, a1_m, a2_m]/sigma, med_mad_ests[0]/sigma)

        for chan in range(n_chan):
            for corr in range(n_corr):

                r_val = resid_arr[row, chan, corr]

                bad_values[row, chan, corr] = \
                    np.sqrt(r_val.real**2 + r_val.imag**2) > thr

    return bad_values
