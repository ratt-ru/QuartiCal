import numpy as np
from numba import jit


@jit(nopython=True, fastmath=False, parallel=False, cache=True, nogil=True)
def estimate_noise_kernel(data, flags, a1, a2, n_ant):
    """Estimates the noise in a data-like array by forming channel differences.

    Given a data-like array and its assosciated flags, produces an overall
    error estimate for the data and an error estimate per channel. These
    estimates are made by summing the errors of the contributing terms in
    quadrature and normalising by the number of valid data points.

    Note that this implementation is NOT thread-safe.

    Args:
        data: An array containing data (or the residual).
        flags: An array containing bitflags.
        a1: An array of antenna values.
        a2: An array of antenna values.
        n_ant: Integer number of antennas.

    Returns:
        noise_est: A float corresponding to the inverse of the variance.
        inv_var_per_chan: An array containing the inverse variance per channel.
    """

    n_rows, n_chan, n_corr = data.shape

    chan_diff_sqrd = np.zeros((n_chan, n_ant), dtype=np.float32)
    valid_counts = np.zeros((n_chan, n_ant), dtype=np.uint32)

    if n_chan == 1:
        return 1., np.ones((1, n_chan), dtype=np.float32)

    for row in range(n_rows):

        a1_m, a2_m = a1[row], a2[row]

        for f in range(1, n_chan):

            f_row_a = flags[row, f]
            f_row_b = flags[row, f - 1]

            fa00 = f_row_a[0]
            fa01 = f_row_a[1]
            fa10 = f_row_a[2]
            fa11 = f_row_a[3]

            fb00 = f_row_b[0]
            fb01 = f_row_b[1]
            fb10 = f_row_b[2]
            fb11 = f_row_b[3]

            f00 = ~((fa00 != 0) | (fb00 != 0))
            f01 = ~((fa01 != 0) | (fb01 != 0))
            f10 = ~((fa10 != 0) | (fb10 != 0))
            f11 = ~((fa11 != 0) | (fb11 != 0))

            d_row_a = data[row, f]
            d_row_b = data[row, f - 1]

            da00 = d_row_a[0]
            da01 = d_row_a[1]
            da10 = d_row_a[2]
            da11 = d_row_a[3]

            db00 = d_row_b[0]
            db01 = d_row_b[1]
            db10 = d_row_b[2]
            db11 = d_row_b[3]

            d00 = (da00 - db00)*f00
            d01 = (da01 - db01)*f01
            d10 = (da10 - db10)*f10
            d11 = (da11 - db11)*f11

            net_d = (d00*d00.conjugate() + d01*d01.conjugate() +
                     d10*d10.conjugate() + d11*d11.conjugate()).real

            chan_diff_sqrd[f, a1_m] += net_d
            chan_diff_sqrd[f, a2_m] += net_d

            net_flags = f00 + f01 + f10 + f11

            valid_counts[f, a1_m] += net_flags
            valid_counts[f, a2_m] += net_flags

    chan_diff_sqrd[0, :] = chan_diff_sqrd[1, :]
    valid_counts[0, :] = valid_counts[1, :]

    # Normalise by the number of contributing noise terms. This should be 4 -
    # 2 per contributing complex value, 2 contributing values ber difference.

    chan_diff_sqrd /= 4

    inv_var = valid_counts.sum() / chan_diff_sqrd.sum()
    noise_est = np.float32(np.sqrt(1 / inv_var))
    inv_var_per_chan = valid_counts.sum(axis=1) / chan_diff_sqrd.sum(axis=1)

    # Isolated but valid channels may end up with no noise estimate at all.
    # These are assumed to be equal to the overall inverse variance.
    valid_chans = (flags == 0).sum(axis=2).sum(axis=0) != 0
    inv_var_per_chan[valid_chans & ~np.isfinite(inv_var_per_chan)] = inv_var

    inv_var_per_chan[~np.isfinite(inv_var_per_chan)] = 0

    return noise_est, inv_var_per_chan.reshape((1, -1))


@jit(nopython=True, fastmath=False, parallel=False, cache=True, nogil=True)
def interval_stats_kernel(flags, a1, a2, n_ant):

    n_rows, n_chan, n_corr = flags.shape

    chan_diff_sqrd = np.zeros((n_chan, n_ant), dtype=np.float32)
    valid_counts = np.zeros((n_chan, n_ant), dtype=np.uint32)

    if n_chan == 1:
        return 1., np.ones((1, n_chan), dtype=np.float32)

    for row in range(n_rows):

        a1_m, a2_m = a1[row], a2[row]

        for f in range(1, n_chan):

            f_row_a = flags[row, f]
            f_row_b = flags[row, f - 1]

            fa00 = f_row_a[0]
            fa01 = f_row_a[1]
            fa10 = f_row_a[2]
            fa11 = f_row_a[3]

            fb00 = f_row_b[0]
            fb01 = f_row_b[1]
            fb10 = f_row_b[2]
            fb11 = f_row_b[3]

            f00 = ~((fa00 != 0) | (fb00 != 0))
            f01 = ~((fa01 != 0) | (fb01 != 0))
            f10 = ~((fa10 != 0) | (fb10 != 0))
            f11 = ~((fa11 != 0) | (fb11 != 0))

            d_row_a = data[row, f]
            d_row_b = data[row, f - 1]

            da00 = d_row_a[0]
            da01 = d_row_a[1]
            da10 = d_row_a[2]
            da11 = d_row_a[3]

            db00 = d_row_b[0]
            db01 = d_row_b[1]
            db10 = d_row_b[2]
            db11 = d_row_b[3]

            d00 = (da00 - db00)*f00
            d01 = (da01 - db01)*f01
            d10 = (da10 - db10)*f10
            d11 = (da11 - db11)*f11

            net_d = (d00*d00.conjugate() + d01*d01.conjugate() +
                     d10*d10.conjugate() + d11*d11.conjugate()).real

            chan_diff_sqrd[f, a1_m] += net_d
            chan_diff_sqrd[f, a2_m] += net_d

            net_flags = f00 + f01 + f10 + f11

            valid_counts[f, a1_m] += net_flags
            valid_counts[f, a2_m] += net_flags

    chan_diff_sqrd[0, :] = chan_diff_sqrd[1, :]
    valid_counts[0, :] = valid_counts[1, :]

    # Normalise by the number of contributing noise terms. This should be 4 -
    # 2 per contributing complex value, 2 contributing values ber difference.

    chan_diff_sqrd /= 4

    inv_var = valid_counts.sum() / chan_diff_sqrd.sum()
    noise_est = np.float32(np.sqrt(1 / inv_var))
    inv_var_per_chan = valid_counts.sum(axis=1) / chan_diff_sqrd.sum(axis=1)

    # Isolated but valid channels may end up with no noise estimate at all.
    # These are assumed to be equal to the overall inverse variance.
    valid_chans = (flags == 0).sum(axis=2).sum(axis=0) != 0
    inv_var_per_chan[valid_chans & ~np.isfinite(inv_var_per_chan)] = inv_var

    inv_var_per_chan[~np.isfinite(inv_var_per_chan)] = 0

    return noise_est, inv_var_per_chan.reshape((1, -1))
