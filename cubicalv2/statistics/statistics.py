from cubicalv2.statistics.stat_kernels import (estimate_noise_kernel,
                                               accumulate_intervals,
                                               logical_and_intervals,
                                               column_to_tfadc,
                                               column_to_tfac)
from cubicalv2.utils.maths import cabs2
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


def create_gain_stats_xds(n_tint, n_fint, n_ant, n_dir, n_corr, n_chunk, name,
                          ind):
    """Set up a stats xarray dataset and define its coordinates."""

    stats_xds = xarray.Dataset(
        coords={"ant": ("ant", da.arange(n_ant, dtype=np.int16)),
                "time_int": ("time_int", da.arange(n_tint, dtype=np.int16)),
                "freq_int": ("freq_int", da.arange(n_fint, dtype=np.int16)),
                "dir": ("dir", da.arange(n_dir, dtype=np.int16)),
                "corr": ("corr", da.arange(n_corr, dtype=np.int16)),
                "chunk": ("chunk", da.arange(n_chunk, dtype=np.int16))},
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
        dtype=np.float32).squeeze(axis=1)

    inv_var_per_chan = da.blockwise(
        lambda nt: nt[1], ("rowlike", "chan"),
        noise_tuple, ("rowlike", "chan"),
        dtype=np.float32)

    updated_stats_xds = stats_xds.assign(
        {"inv_var": (("chunk", "chan"), inv_var_per_chan),
         "noise_est": (("chunk",), noise_est)})

    return updated_stats_xds


def assign_tf_stats(stats_xds, cubical_bitflags, ant1_col,
                    ant2_col, time_ind, n_time_ind, n_ant, n_chunk,
                    n_chan, chunk_spec):

    # Get all the unflagged points.

    unflagged = cubical_bitflags == 0

    # Compute the number of unflagged points per row. Note that this includes
    # a summation over channel and correlation - version 1 did not have a
    # a correlation axis in the flags but I believe we should.
    rows_unflagged = unflagged.map_blocks(np.sum, axis=(1, 2),
                                          chunks=(unflagged.chunks[0],),
                                          drop_axis=(1, 2))

    # Determine the number of equations per antenna by summing the appropriate
    # values from the per-row unflagged values.
    eqs_per_ant = da.map_blocks(sum_eqs_per_ant, rows_unflagged, ant1_col,
                                ant2_col, n_ant, dtype=np.int64,
                                new_axis=1,
                                chunks=((1,)*n_chunk, (n_ant,)))

    # Determine the number of equations per time-frequency slot.
    eqs_per_tf = da.blockwise(sum_eqs_per_tf, ("rowlike", "chan"),
                              unflagged, ("rowlike", "chan", "corr"),
                              time_ind, ("rowlike",),
                              n_time_ind, ("rowlike",),
                              dtype=np.int64,
                              concatenate=True,
                              align_arrays=False,
                              adjust_chunks={"rowlike": tuple(chunk_spec)})

    # Determine the normalisation factor as the reciprocal of the equations
    # per time-frequency bin.
    tf_norm_factor = da.map_blocks(block_reciprocal,
                                   eqs_per_tf, dtype=np.float64)

    # Compute the total number of equations per chunk.
    total_eqs = da.map_blocks(lambda x: np.atleast_1d(np.sum(x)),
                              eqs_per_tf, dtype=np.int64,
                              drop_axis=1,
                              chunks=(1,))

    # Compute the overall normalisation factor.
    total_norm_factor = da.map_blocks(block_reciprocal,
                                      total_eqs, dtype=np.float64)

    # Assign the relevant values to the xds.
    modified_stats_xds = \
        stats_xds.assign({"eqs_per_ant": (("chunk", "ant"), eqs_per_ant),
                          "eqs_per_tf": (("time", "chan"), eqs_per_tf),
                          "tf_norm_factor": (("time", "chan"), tf_norm_factor),
                          "tot_norm_factor": (("chunk",), total_norm_factor)})

    return modified_stats_xds


def assign_model_stats(stats_xds, model_col, cubical_bitflags, ant1_col,
                       ant2_col, utime_ind, n_utime, n_ant, n_chunk,
                       n_chan, n_dir, chunk_spec):

    # Get all the unflagged points.

    unflagged = cubical_bitflags == 0

    abs_sqrd_model = model_col.map_blocks(cabs2)

    abs_sqrd_model_tfadc = \
        da.blockwise(column_to_tfadc, ("rowlike", "chan", "ant", "dir", "corr"),
                     abs_sqrd_model, ("rowlike", "chan", "dir", "corr"),
                     ant1_col, ("rowlike",),
                     ant2_col, ("rowlike",),
                     utime_ind, ("rowlike",),
                     n_utime, ("rowlike",),
                     n_ant, None,
                     dtype=model_col.dtype,
                     concatenate=True,
                     align_arrays=False,
                     new_axes={"ant": n_ant},
                     adjust_chunks={"rowlike": chunk_spec})

    abs_sqrd_model_tfad = \
        abs_sqrd_model_tfadc.map_blocks(np.sum, axis=4, drop_axis=4)

    unflagged_tfac = \
        da.blockwise(column_to_tfac, ("rowlike", "chan", "ant", "corr"),
                     unflagged, ("rowlike", "chan", "corr"),
                     ant1_col, ("rowlike",),
                     ant2_col, ("rowlike",),
                     utime_ind, ("rowlike",),
                     n_utime, ("rowlike",),
                     n_ant, None,
                     dtype=unflagged.dtype,
                     concatenate=True,
                     align_arrays=False,
                     new_axes={"ant": n_ant},
                     adjust_chunks={"rowlike": chunk_spec})

    unflagged_tfa = unflagged_tfac.map_blocks(np.sum, axis=3, drop_axis=3)

    avg_abs_sqrd_model = \
        abs_sqrd_model_tfad.map_blocks(silent_divide, unflagged_tfa[..., None])


def assign_interval_stats(gain_xds, fullres_bitflags, ant1_col,
                          ant2_col, t_map, f_map, t_int_per_chunk,
                          f_int_per_chunk, ti_chunks, fi_chunks):

    flagged = (fullres_bitflags != 0).all(axis=-1)
    n_ant = gain_xds.dims["ant"]

    # This creates a (n_t_int, n_f_int, n_ant) boolean array which indicates
    # which antennas (and consquently gain solutions) are missing.

    ant_missing_per_int = \
        da.blockwise(logical_and_intervals, ("rowlike", "chan", "ant"),
                     flagged, ("rowlike", "chan"),
                     ant1_col, ("rowlike",),
                     ant2_col, ("rowlike",),
                     t_map, ("rowlike",),
                     f_map, ("rowlike",),
                     t_int_per_chunk, ("rowlike",),
                     f_int_per_chunk, ("rowlike",),
                     n_ant, None,
                     dtype=np.bool,
                     concatenate=True,
                     align_arrays=False,
                     new_axes={"ant": n_ant},
                     adjust_chunks={"rowlike": ti_chunks,
                                    "chan": fi_chunks[0]})

    missing_fraction = \
        da.map_blocks(lambda x: np.atleast_1d(np.sum(x)/x.size),
                      ant_missing_per_int,
                      chunks=(1,),
                      drop_axis=(1, 2),
                      dtype=np.int32)

    updated_gain_xds = gain_xds.assign(
        {"missing_fraction": (("chunk",), missing_fraction)})

    return updated_gain_xds, ant_missing_per_int


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


def silent_divide(in_arr1, in_arr2):

    with np.errstate(divide='ignore', invalid='ignore'):
        out_arr = np.where(in_arr2 != 0, in_arr1/in_arr2, 0)

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