from cubicalv2.statistics.stat_kernels import estimate_noise_kernel
import dask.array as da
import numpy as np
import xarray


def estimate_noise(data_col, cubical_bitflags, ant1_col, ant2_col, n_ant):
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
    # inv_var_per_chan). Noise estiamte is a float so we are forced to make
    # it an array whilst unpacking.
    # noise_est = da.blockwise(
    #     lambda nt: np.array(nt[0], ndmin=2), ("rowlike", "chan"),
    #     noise_tuple, ("rowlike", "chan"),
    #     adjust_chunks={"chan": 1},
    #     dtype=np.float32
    # )

    noise_est = da.blockwise(
        lambda nt: nt[0][0], ("rowlike",),
        noise_tuple, ("rowlike", "chan"),
        adjust_chunks={"rowlike": 1},
        dtype=np.float32
    )

    inv_var_per_chan = da.blockwise(
        lambda nt: nt[1], ("rowlike", "chan"),
        noise_tuple, ("rowlike", "chan"),
        dtype=np.float32
    )

    return noise_est, inv_var_per_chan


def compute_interval_statistics(stats_xds, cubical_bitflags, ant1_col,
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
        out_arr = np.where(in_arr != 0, 1/in_arr, 0)

    return out_arr