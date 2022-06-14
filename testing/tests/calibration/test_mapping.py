from copy import deepcopy
import pytest
import dask.array as da
import numpy as np
from numpy.testing import assert_array_equal
from quartical.config.internal import yield_from
from quartical.calibration.mapping import (make_t_binnings,
                                           make_f_mappings,
                                           make_d_mappings)


@pytest.fixture(scope="module")
def opts(base_opts, time_int, freq_int):

    # Don't overwrite base config - instead duplicate and update.

    _opts = deepcopy(base_opts)

    _opts.G.time_interval = time_int
    _opts.B.time_interval = 2*time_int
    _opts.G.freq_interval = freq_int
    _opts.B.freq_interval = 2*freq_int

    return _opts


# ------------------------------make_t_binnings--------------------------------


@pytest.mark.parametrize("time_chunk", [33, 60])
def test_t_binnings(time_chunk, chain_opts):
    """Test construction of time mappings for different chunks/intervals."""

    n_time = 100  # Total number of unique times to consider.
    n_bl = 351  # 27 antenna array - VLA-like.
    interval = 8

    time_stamps = da.arange(100, chunks=time_chunk)

    utime_per_chunk = da.from_array(time_stamps.chunks[0],
                                    chunks=(1,))

    time_col = da.map_blocks(
        np.repeat,
        time_stamps,
        n_bl,
        chunks=(tuple(c*n_bl for c in time_stamps.chunks[0]),))

    interval_col = da.ones_like(time_col)*interval

    utime_loc = da.map_blocks(
        lambda a, **kwargs: np.unique(a, **kwargs)[1],
        time_col,
        return_index=True,
        chunks=time_stamps.chunks)

    utime_intervals = da.map_blocks(
        lambda arr, inds: arr[inds],
        interval_col,
        utime_loc,
        chunks=utime_loc.chunks,
        dtype=np.float64)

    utime_scan_numbers = da.zeros_like(utime_intervals, dtype=np.int32)

    # TODO: Should also check parameter mappings.
    da_t_bins = make_t_binnings(utime_per_chunk,
                                utime_intervals,
                                utime_scan_numbers,
                                chain_opts)[0, ...]

    t_ints = [ti or n_time
              for _, ti in yield_from(chain_opts, "time_interval")]

    for block_ind in range(da_t_bins.npartitions):
        binning = da_t_bins.blocks[block_ind].compute()
        ivl_col = interval_col.blocks[block_ind].compute()
        for g_ind, t_int in enumerate(t_ints):
            if isinstance(t_int, float):
                sol_widths = np.zeros(np.max(binning[:, g_ind] + 1))
                for ivl_ind, target in enumerate(binning[:, g_ind]):
                    sol_widths[target] += ivl_col[ivl_ind]
                assert all(sol_widths[:-1] >= t_int)
            else:
                assert all(np.unique(binning[:, g_ind],
                                     return_counts=True)[1] <= t_int)

# ------------------------------make_t_mappings--------------------------------

# TODO: This test has been removed temporarily because the binner does the
# majority of this work now. All the mapping code does is select out to the
# appropriate resolution.

# @pytest.mark.parametrize("time_chunk", [33, 60])
# def test_t_mappings(time_int, time_chunk, mapping_opts):
#     """Test construction of time mappings for different chunks/intervals."""

#     opts = mapping_opts
#     opts.G_time_interval = time_int  # Setting time interval on first gain.
#     opts.B_time_interval = time_int*2  # Setting time interval on second
#                                          gain.

#     n_time = 100  # Total number of unique times to consider.
#     n_bl = 351  # 27 antenna array - VLA-like.

#     time_stamps = da.arange(100, chunks=time_chunk)

#     time_col = da.map_blocks(
#         np.repeat,
#         time_stamps,
#         n_bl,
#         chunks=(tuple(c*n_bl for c in time_stamps.chunks[0]),))

#     utime_ind = da.map_blocks(
#         lambda a, **kwargs: np.unique(a, **kwargs)[1],
#         time_col,
#         return_inverse=True)

#     da_t_maps = make_t_mappings(utime_ind, opts)

#     # Set up and compute numpy values to test against.

#     t_ints = [getattr(opts, t + "_time_interval") or n_time
#               for t in opts.solver.terms]

#     np_t_maps = np.array(list(map(lambda ti: utime_ind//ti, t_ints))).T

#     assert_array_equal(da_t_maps, np_t_maps)


# ------------------------------make_f_mappings--------------------------------


@pytest.mark.parametrize("freq_chunk", [30, 64])
def test_f_mappings(freq_chunk, chain_opts):
    """Test construction of freq mappings for different chunks/intervals."""

    n_freq = 64  # Total number of channels to consider.

    chan_freqs = da.arange(n_freq, chunks=freq_chunk)
    chan_widths = da.ones(n_freq, chunks=freq_chunk)*7

    # Pull out just the gain mapping, not the parameter mapping. TODO: Should
    # also check parameter mappings.
    da_f_maps = make_f_mappings(chan_freqs, chan_widths, chain_opts)[0, ...]

    # Set up and compute numpy values to test against.

    f_ints = [fi or n_freq
              for _, fi in yield_from(chain_opts, "freq_interval")]

    for block_ind in range(da_f_maps.npartitions):
        f_map = da_f_maps.blocks[block_ind].compute()
        chan_width = chan_widths.blocks[block_ind].compute()
        for g_ind, f_int in enumerate(f_ints):
            if isinstance(f_int, float):
                sol_widths = np.zeros(np.max(f_map[:, g_ind] + 1))
                for ivl_ind, target in enumerate(f_map[:, g_ind]):
                    sol_widths[target] += chan_width[ivl_ind]
                assert all(sol_widths[:-1] >= f_int)
            else:
                assert all(np.unique(f_map[:, g_ind],
                                     return_counts=True)[1] <= f_int)

# ------------------------------make_d_mappings--------------------------------


@pytest.mark.parametrize("n_dir", [2, 10])
@pytest.mark.parametrize("has_dd_term", [False, True])
def test_d_mappings(n_dir, has_dd_term, chain_opts):
    """Test construction of direction mappings for different n_dir."""

    chain_opts.B.direction_dependent = has_dd_term

    d_maps = make_d_mappings(n_dir, chain_opts)  # Not a dask array.

    # Set up and compute numpy values to test against.

    dd_terms = [dd for _, dd in yield_from(chain_opts, "direction_dependent")]

    np_d_maps = np.array(list(map(lambda dd: np.arange(n_dir)*dd, dd_terms)))

    assert_array_equal(d_maps, np_d_maps)

# -----------------------------------------------------------------------------
