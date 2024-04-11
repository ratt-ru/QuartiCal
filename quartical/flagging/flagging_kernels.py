import numpy as np
from numba import njit
from quartical.utils.numba import JIT_OPTIONS


@njit(**JIT_OPTIONS)
def get_bl_ids(a1, a2, n_ant):
    return a1*(2*n_ant - a1 - 1)//2 + a2


@njit(**JIT_OPTIONS)
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


@njit(**JIT_OPTIONS)
def compute_bl_mad_and_med(wres, flags, a1, a2, n_ant):

    n_corr = wres.shape[-1]
    n_bl_w_autos = (n_ant * (n_ant + 1))//2

    bl_ids = get_bl_ids(a1, a2, n_ant)

    # Leading dims represent time and chan axes.
    mad_and_med = np.zeros((1, 1, n_bl_w_autos, n_corr, 2), dtype=wres.dtype)

    for bl_id in range(n_bl_w_autos):

        bl_sel = np.where(bl_ids == bl_id)[0]

        if not bl_sel.size:  # Missing baseline.
            continue

        bl_flags = flags[bl_sel].ravel()

        unflagged_sel = np.where(bl_flags != 1)[0]

        if unflagged_sel.size:  # Not fully flagged.
            for c in range(n_corr):

                # Unflagged elements of baseline whitened residuals.
                # NOTE: https://github.com/numba/numba/issues/8999.
                bl_wres = wres[bl_sel][:, :, c].ravel()[unflagged_sel]

                bl_med = np.median(bl_wres)
                bl_mad = np.median(np.abs(bl_wres - bl_med))
                mad_and_med[0, 0, bl_id, c, 0] = bl_mad
                mad_and_med[0, 0, bl_id, c, 1] = bl_med

    return mad_and_med


@njit(**JIT_OPTIONS)
def compute_gbl_mad_and_med(wres, flags):

    n_corr = wres.shape[-1]

    # Leading dims represent time and chan axes.
    mad_and_med = np.zeros((1, 1, n_corr, 2), dtype=wres.dtype)

    unflagged_sel = np.where(flags.ravel() != 1)

    if unflagged_sel[0].size:  # We have unflagged data.
        for c in range(n_corr):
            gbl_wres = wres[..., c].ravel()[unflagged_sel]

            gbl_med = np.median(gbl_wres)
            gbl_mad = np.median(np.abs(gbl_wres - gbl_med))

            mad_and_med[0, 0, c, 0] = gbl_mad
            mad_and_med[0, 0, c, 1] = gbl_med

    return mad_and_med


@njit(**JIT_OPTIONS)
def compute_mad_flags(
    wres,
    gbl_mad_and_med_real,
    gbl_mad_and_med_imag,
    bl_mad_and_med_real,
    bl_mad_and_med_imag,
    ant1,
    ant2,
    gbl_threshold,
    bl_threshold,
    max_deviation,
    corr_sel,
    n_ant
):

    n_row, n_chan, n_corr = wres.shape

    bl_ids = get_bl_ids(ant1, ant2, n_ant)

    flags = np.zeros((n_row, n_chan), dtype=np.int8)

    scale_factor = 1.4826

    bl_threshold2 = bl_threshold ** 2 or np.inf
    gbl_threshold2 = gbl_threshold ** 2 or np.inf
    max_deviation2 = max_deviation ** 2 or np.inf

    for corr in corr_sel:

        gbl_mad_real = gbl_mad_and_med_real[0, 0, corr, 0] * scale_factor
        gbl_med_real = gbl_mad_and_med_real[0, 0, corr, 1]
        gbl_mad_imag = gbl_mad_and_med_imag[0, 0, corr, 0] * scale_factor
        gbl_med_imag = gbl_mad_and_med_imag[0, 0, corr, 1]
        bl_mad_real = bl_mad_and_med_real[0, 0, :, corr, 0] * scale_factor
        bl_med_real = bl_mad_and_med_real[0, 0, :, corr, 1]
        bl_mad_imag = bl_mad_and_med_imag[0, 0, :, corr, 0] * scale_factor
        bl_med_imag = bl_mad_and_med_imag[0, 0, :, corr, 1]

        # Catch fully flagged chunks i.e. which have no estimates.
        if (gbl_mad_real == 0) or (gbl_mad_imag == 0):
            flags[:] = 1
            continue

        # The per-baseline MAD values should be relatively unbiased estimates
        # of the standard deviation of the whitened residuals. As such, we may
        # want to flag out baselines with MAD values which point to
        # inconsistent statistics.

        bl_mad_mean_re = np.mean(bl_mad_real[bl_mad_real > 0])
        bl_mad_mean_im = np.mean(bl_mad_imag[bl_mad_imag > 0])
        bl_mad_std_re = np.std(bl_mad_real[bl_mad_real > 0])
        bl_mad_std_im = np.std(bl_mad_imag[bl_mad_imag > 0])

        for row in range(n_row):

            a1_m, a2_m = ant1[row], ant2[row]

            # Do not attempt to flag autocorrelations.
            if a1_m == a2_m:
                continue

            bl_id = bl_ids[row]

            mad_pq_re = bl_mad_real[bl_id]
            med_pq_re = bl_med_real[bl_id]

            mad_pq_im = bl_mad_imag[bl_id]
            med_pq_im = bl_med_imag[bl_id]

            # No/bad mad estimates for this baseline. Likely already flagged.
            if (mad_pq_re == 0) or (mad_pq_im == 0):
                flags[row] = 1
                continue

            bl_mean_dev_re = \
                np.abs(mad_pq_re - bl_mad_mean_re)/bl_mad_std_re
            bl_mean_dev_im = \
                np.abs(mad_pq_im - bl_mad_mean_im)/bl_mad_std_im

            if (bl_mean_dev_re**2 + bl_mean_dev_im**2) > max_deviation2:
                flags[row] = 1
                continue

            for chan in range(n_chan):

                bl_dev_re = \
                    np.abs(wres[row, chan, corr] - med_pq_re)/mad_pq_re
                bl_dev_im = \
                    np.abs(wres[row, chan, corr] - med_pq_im)/mad_pq_im

                if (bl_dev_re**2 + bl_dev_im**2) > bl_threshold2:
                    flags[row, chan] = 1

                gbl_dev_re = \
                    np.abs(wres[row, chan, corr] - gbl_med_real)/gbl_mad_real
                gbl_dev_im = \
                    np.abs(wres[row, chan, corr] - gbl_med_imag)/gbl_mad_imag

                if (gbl_dev_re**2 + gbl_dev_im**2) > gbl_threshold2:
                    flags[row, chan] = 1

    return flags
