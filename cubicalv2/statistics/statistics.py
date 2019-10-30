from cubicalv2.statistics.stat_kernels import estimate_noise_kernel
import dask.array as da
import numpy as np


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
    noise_est = da.blockwise(
        lambda nt: np.array(nt[0], ndmin=2), ("rowlike", "chan"),
        noise_tuple, ("rowlike", "chan"),
        adjust_chunks={"chan": 1},
        dtype=np.float32
    )

    inv_var_per_chan = da.blockwise(
        lambda nt: nt[1], ("rowlike", "chan"),
        noise_tuple, ("rowlike", "chan"),
        dtype=np.float32
    )

    return noise_est, inv_var_per_chan
