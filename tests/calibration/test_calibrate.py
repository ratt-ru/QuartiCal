import pytest
from quartical.data_handling.ms_handler import read_xds_list
from quartical.calibration.calibrate import (make_t_mappings,
                                             make_f_mappings,
                                             make_d_mappings)
from argparse import Namespace
import dask.array as da
import numpy as np
from numpy.testing import assert_array_equal


@pytest.fixture(scope="module")
def opts(base_opts):

    # Don't overwrite base config - instead create a new Namespace and update.

    options = Namespace(**vars(base_opts))

    options._model_columns = ["MODEL_DATA"]

    return options


@pytest.fixture(scope="module")
def _read_xds_list(opts):

    return read_xds_list(opts)


# ------------------------------make_t_mappings--------------------------------


@pytest.mark.parametrize("time_chunk", [33, 60])
def test_t_mappings(time_int, time_chunk, opts):
    """Test construction of time mappings for different chunks/intervals."""

    opts.G_time_interval = time_int  # Setting time interval on first gain.
    opts.B_time_interval = time_int*2  # Setting time interval on second gain.

    n_time = 100  # Total number of unique times to consider.
    n_bl = 351  # 27 antenna array - VLA-like.

    time_stamps = da.arange(100, chunks=time_chunk)

    time_col = da.map_blocks(
        np.repeat,
        time_stamps,
        n_bl,
        chunks=(tuple(c*n_bl for c in time_stamps.chunks[0]),))

    utime_ind = da.map_blocks(
        lambda a, **kwargs: np.unique(a, **kwargs)[1],
        time_col,
        return_inverse=True)

    da_t_maps = make_t_mappings(utime_ind, opts)

    # Set up and compute numpy values to test against.

    t_ints = [getattr(opts, t + "_time_interval") or n_time
              for t in opts.solver_gain_terms]

    np_t_maps = np.array(list(map(lambda ti: utime_ind//ti, t_ints))).T

    assert_array_equal(da_t_maps, np_t_maps)


# ------------------------------make_f_mappings--------------------------------


@pytest.mark.parametrize("freq_chunk", [30, 64])
def test_f_mappings(freq_int, freq_chunk, opts):
    """Test construction of freq mappings for different chunks/intervals."""

    opts.G_time_interval = freq_int  # Setting time interval on first gain.
    opts.B_time_interval = freq_int*2  # Setting time interval on second gain.

    n_freq = 64  # Total number of channels to consider.

    chan_freqs = da.arange(n_freq, chunks=freq_chunk)

    da_f_maps = make_f_mappings(chan_freqs, opts)

    # Set up and compute numpy values to test against.

    f_ints = [getattr(opts, t + "_freq_interval") or n_freq
              for t in opts.solver_gain_terms]

    np_f_maps = [np.array(list(map(lambda fi: np.arange(nf)//fi, f_ints))).T
                 for nf in chan_freqs.chunks[0]]
    np_f_maps = np.concatenate(np_f_maps, axis=0)

    assert_array_equal(da_f_maps, np_f_maps)


# ------------------------------make_d_mappings--------------------------------


@pytest.mark.parametrize("n_dir", [2, 10])
@pytest.mark.parametrize("has_dd_term", [False, True])
def test_d_mappings(n_dir, has_dd_term, opts):
    """Test construction of direction mappings for different n_dir."""

    opts.B_direction_dependent = has_dd_term

    d_maps = make_d_mappings(n_dir, opts)  # Not a dask array.

    # Set up and compute numpy values to test against.

    dd_terms = [getattr(opts, t + "_direction_dependent")
                for t in opts.solver_gain_terms]

    np_d_maps = np.array(list(map(lambda dd: np.arange(n_dir)*dd, dd_terms)))

    assert_array_equal(d_maps, np_d_maps)


# ----------------------------make_gain_xds_list-------------------------------
