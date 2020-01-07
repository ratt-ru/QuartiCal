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

    # We cannot make noise estimates from a single channel - we instead return
    # ones for the noise estimate and per-channel inverse of the variance.
    # TODO: Can this be moved out of the function?

    if n_chan == 1:
        return np.array(1, dtype=np.float32).reshape((1, -1)), \
               np.ones((1, n_chan), dtype=np.float32)

    for row in range(n_rows):

        a1_m, a2_m = a1[row], a2[row]

        for f in range(1, n_chan):

            f_row_a = flags[row, f]
            f_row_b = flags[row, f - 1]

            d_row_a = data[row, f]
            d_row_b = data[row, f - 1]

            net_d = 0
            net_fl = 0

            for c in range(n_corr):

                fla = f_row_a[c]
                flb = f_row_b[c]

                diff_fl = ~((fla != 0) | (flb != 0))

                da = d_row_a[c]
                db = d_row_b[c]

                diff_d = (da - db)*diff_fl

                net_d += (diff_d*diff_d.conjugate()).real
                net_fl += diff_fl

            chan_diff_sqrd[f, a1_m] += net_d
            chan_diff_sqrd[f, a2_m] += net_d

            valid_counts[f, a1_m] += net_fl
            valid_counts[f, a2_m] += net_fl

    chan_diff_sqrd[0, :] = chan_diff_sqrd[1, :]
    valid_counts[0, :] = valid_counts[1, :]

    # Normalise by the number of contributing noise terms. This should be 4 -
    # 2 per contributing complex value, 2 contributing values per difference.
    # We do not include a correction for correlation here as it is absorbed
    # by the flag counts.

    chan_diff_sqrd /= 4

    # TODO: Consider returning here and doing the last bit with map_blocks/
    # blockwise. This is just to make it a little neater.

    inv_var = valid_counts.sum() / chan_diff_sqrd.sum()
    noise_est = np.float32(np.sqrt(1 / inv_var))

    inv_var_per_chan = valid_counts.sum(axis=1) / chan_diff_sqrd.sum(axis=1)

    # Isolated but valid channels may end up with no noise estimate at all.
    # These are assumed to be equal to the overall inverse variance.
    valid_chans = (flags == 0).sum(axis=2).sum(axis=0) != 0
    inv_var_per_chan[valid_chans & ~np.isfinite(inv_var_per_chan)] = inv_var

    inv_var_per_chan[~np.isfinite(inv_var_per_chan)] = 0

    return np.array(noise_est).reshape((1, -1)), \
           inv_var_per_chan.reshape((1, -1))


@jit(nopython=True, fastmath=False, parallel=False, cache=True, nogil=True)
def logical_and_intervals(in_arr, ant1_col, ant2_col, t_map, f_map, n_ti, n_fi, n_ant):
    """Compute a logical and per interval.

    Given an rowlike array of boolean values, uses the time and frequency
    mappings to reduce them into a per interval value.

    Args:
        in_arr (np.ndarray): An 2D array of boolean values.
        t_map (np.ndarray): A 1D array of time mappings.
        f_map (np.ndarray): A 1D array of freq mappings.
        n_ti (int): Number of time intervals.
        n_fi (int): Number of frequency intervals.

    Returns:
        out_arr: (np.ndarray) A 2D array of per-interval boolean values.
    """

    n_row = t_map.shape[0]
    n_chan = f_map.shape[0]

    out_arr = np.ones((n_ti[0], n_fi[0], n_ant), dtype=np.bool_)

    for row in range(n_row):

        t_m = t_map[row]
        a1_m = ant1_col[row]
        a2_m = ant2_col[row]

        for chan in range(n_chan):

            f_m = f_map[chan]

            out_arr[t_m, f_m, a1_m] &= in_arr[row, chan]
            out_arr[t_m, f_m, a2_m] &= in_arr[row, chan]

    return out_arr


@jit(nopython=True, fastmath=False, parallel=False, cache=True, nogil=True)
def accumulate_intervals(in_arr, t_map, f_map, n_ti, n_fi):
    """Compute a sum per interval.

    Given an rowlike array of values, uses the time and frequency
    mappings to reduce them into a per interval value.

    Args:
        in_arr (np.ndarray): A 2D array of values.
        t_map (np.ndarray): A 1D array of time mappings.
        f_map (np.ndarray): A 1D array of freq mappings.
        n_ti (int): Number of time intervals.
        n_fi (int): Number of frequency intervals.

    Returns:
        out_arr: (np.ndarray) A 2D array of per-interval values.
    """

    n_row = t_map.shape[0]
    n_chan = f_map.shape[0]

    out_arr = np.zeros((n_ti, n_fi), dtype=in_arr.dtype)

    for row in range(n_row):

        t_m = t_map[row]

        for chan in range(n_chan):

            f_m = f_map[chan]

            out_arr[t_m, f_m] += in_arr[row, chan]

    return out_arr
