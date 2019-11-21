from cubicalv2.statistics.stat_kernels import (estimate_noise_kernel,
                                               accumulate_intervals,
                                               logical_and_intervals)
import dask.array as da
import numpy as np
import xarray


def create_data_stats_xds(utime_val, n_chan, n_ant, n_chunks):
    """Set up a stats xarray dataset and define its coordinates."""

    stats_xds = xarray.Dataset(
        coords={"ant": ("ant", da.arange(n_ant, dtype=np.int16)),
                "time": ("time", utime_val),
                "chan": ("chan", da.arange(n_chan, dtype=np.int16)),
                "chunk": ("chunk", da.arange(n_chunks, dtype=np.int16))})

    return stats_xds


def create_gain_stats_xds(n_tint, n_fint, n_ant, n_dir, n_corr, name, ind):
    """Set up a stats xarray dataset and define its coordinates."""

    stats_xds = xarray.Dataset(
        coords={"ant": ("ant", da.arange(n_ant, dtype=np.int16)),
                "time_int": ("time_int", da.arange(n_tint, dtype=np.int16)),
                "freq_int": ("freq_int", da.arange(n_fint, dtype=np.int16)),
                "dir": ("dir", da.arange(n_dir, dtype=np.int16)),
                "corr": ("corr", da.arange(n_corr, dtype=np.int16))},
        attrs={"name": "{}-{}".format(name, ind)})

    return stats_xds


def assign_noise_estimates(stats_xds, data_col, cubical_bitflags, ant1_col,
                           ant2_col, n_ant):
    """Wrapper and unpacker for the Numba noise estimator code.

    Uses blockwise and the numba kernel function to produce a noise estimate
    and inverse variance per channel per chunk of data_col.

    Args:
        data_col: A chunked dask array containing data (or the residual).
        cubical_bitflags: An chunked dask array containing bitflags.
        ant1_col: A chunked dask array of antenna values.
        ant2_col: A chunked dask array of antenna values.
        n_ant: Integer number of antennas.

    Returns:
        noise_est: Graph which produces noise estimates.
        inv_var_per_chan: Graph which produces inverse variance per channel.
    """

    noise_tuple = da.blockwise(
        estimate_noise_kernel, ("rowlike", "chan"),
        data_col, ("rowlike", "chan", "corr"),
        cubical_bitflags, ("rowlike", "chan", "corr"),
        ant1_col, ("rowlike",),
        ant2_col, ("rowlike",),
        n_ant, None,
        adjust_chunks={"rowlike": 1},
        concatenate=True,
        dtype=np.float32,
        align_arrays=False,
        meta=np.empty((0, 0), dtype=np.float32)
    )

    # The following unpacks values from the noise tuple of (noise_estimate,
    # inv_var_per_chan). Noise estimate is embedded in a 2D array in order
    # to make these blockwise calls less complicated - the channel dimension
    # is not meaningful and we immediately squeeze it out.

    noise_est = da.blockwise(
        lambda nt: nt[0], ("rowlike", "chan"),
        noise_tuple, ("rowlike", "chan"),
        adjust_chunks={"rowlike": 1,
                       "chan": 1},
        dtype=np.float32).squeeze()

    inv_var_per_chan = da.blockwise(
        lambda nt: nt[1], ("rowlike", "chan"),
        noise_tuple, ("rowlike", "chan"),
        dtype=np.float32)

    updated_stats_xds = stats_xds.assign(
        {"inv_var": (("chunk", "chan"), inv_var_per_chan),
         "noise_est": (("chunk",), noise_est)})

    return updated_stats_xds


def assign_tf_statistics(stats_xds, cubical_bitflags, ant1_col,
                         ant2_col, time_ind, n_time_ind, n_ant, n_chunk,
                         n_chan, chunk_spec):

    unflagged = cubical_bitflags == 0

    rows_unflagged = unflagged.map_blocks(np.sum, axis=(1, 2),
                                          chunks=(unflagged.chunks[0],),
                                          drop_axis=(1, 2))

    eqs_per_ant = da.map_blocks(sum_eqs_per_ant, rows_unflagged, ant1_col,
                                ant2_col, n_ant, dtype=np.int64,
                                new_axis=1,
                                chunks=((1,)*n_chunk, (n_ant,)))

    eqs_per_tf = da.blockwise(sum_eqs_per_tf, ("rowlike", "chan"),
                              unflagged, ("rowlike", "chan", "corr"),
                              time_ind, ("rowlike",),
                              n_time_ind, ("rowlike",),
                              dtype=np.int64,
                              concatenate=True,
                              align_arrays=False,
                              adjust_chunks={"rowlike": tuple(chunk_spec)})

    tf_norm_factor = da.map_blocks(block_reciprocal,
                                   eqs_per_tf, dtype=np.float64)

    total_eqs = da.map_blocks(lambda x: np.atleast_1d(np.sum(x)),
                              eqs_per_tf, dtype=np.int64,
                              drop_axis=1,
                              chunks=(1,))

    total_norm_factor = da.map_blocks(block_reciprocal,
                                      total_eqs, dtype=np.float64)

    modified_stats_xds = \
        stats_xds.assign({"eqs_per_ant": (("chunk", "ant"), eqs_per_ant),
                          "eqs_per_tf": (("time", "chan"), eqs_per_tf),
                          "tf_norm_factor": (("time", "chan"), tf_norm_factor),
                          "tot_norm_factor": (("chunk",), total_norm_factor)})

    return modified_stats_xds


def assign_interval_statistics(stats_xds, cubical_bitflags, ant1_col,
                               ant2_col, t_map_arr, f_map_arr, n_t_int,
                               n_f_int, t_int, f_int, gain_terms, chunk_spec):

    flags = (cubical_bitflags != 0).all(axis=-1)
    n_chan = cubical_bitflags.shape[1]

    for term_ind, term_name in enumerate(gain_terms):

        ti_chunk_dims = tuple(c//t_int + bool(c % t_int) for c in chunk_spec)

        fi_chunk_dims = (n_chan//f_int + bool(n_chan % f_int),)

        # TODO: I actually need to have the time and frequency solution
        # interval axes on the xds at this point. Otherwise it will be
        # impossible to asasign this onto the xds.

        interval_flags = \
            da.blockwise(logical_and_intervals, ("rowlike", "chan"),
                         flags, ("rowlike", "chan"),
                         t_map_arr[:, term_ind], ("rowlike",),
                         f_map_arr[:, term_ind], ("rowlike",),
                         n_t_int, ("rowlike",),
                         n_f_int, ("rowlike",),
                         dtype=np.int64,
                         concatenate=True,
                         align_arrays=False,
                         adjust_chunks={"rowlike": ti_chunk_dims,
                                        "chan": fi_chunk_dims})

        # TODO: This is likely incorrect.

        missing_intervals = da.map_blocks(
            lambda x: np.sum(x, keepdims=True)/x.size,
            interval_flags, dtype=np.int32)



    # print(missing_intervals.compute())
    # print(interval_flags.compute())

    return stats_xds


def sum_eqs_per_ant(rows_unflagged, ant1_col, ant2_col, n_ant):

    eqs_per_ant = np.zeros((1, n_ant), dtype=np.int64)

    np.add.at(eqs_per_ant[0, :], ant1_col, rows_unflagged)
    np.add.at(eqs_per_ant[0, :], ant2_col, rows_unflagged)

    return 2*eqs_per_ant  # The conjugate points double the eqs.


def sum_eqs_per_tf(unflagged, time_ind, n_time_ind):

    _, n_chan, _ = unflagged.shape

    eqs_per_tf = np.zeros((n_time_ind.item(), n_chan), dtype=np.int64)

    np.add.at(eqs_per_tf, time_ind, unflagged.sum(axis=-1))

    return 4*eqs_per_tf  # Conjugate points + each row contributes to 2 ants.


def block_reciprocal(in_arr):

    with np.errstate(divide='ignore'):
        out_arr = np.where(in_arr != 0, 1./in_arr, 0)

    return out_arr


# def difference_chan_unflags(flags):

#     unflags_diff = flags != 0

#     unflags_diff[:, 1:, :] = unflags_diff[:, 1:, :] | unflags_diff[:, :-1, :]
#     unflags_diff[:, 0, :] = unflags_diff[:, 1, :]

#     return unflags_diff


# def difference_chan_data(data, n_utime):

#     n_row, n_chan, n_corr = data.shape

#     data_diff = np.empty(n_utime, n_chan, dtype=data.real.dtype)

#     data_diff[:, 1:, :] =