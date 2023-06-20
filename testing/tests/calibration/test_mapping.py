from copy import deepcopy
import pytest
import dask.array as da
import numpy as np
from numpy.testing import assert_array_equal


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
def test_t_binnings(time_chunk, chain):
    """Test construction of time mappings for different chunks/intervals."""

    n_time = 100  # Total number of unique times to consider.
    n_bl = 351  # 27 antenna array - VLA-like.
    interval = 8

    time_stamps = da.arange(n_time, chunks=time_chunk)

    time_col = da.map_blocks(
        np.repeat,
        time_stamps,
        n_bl,
        chunks=(tuple(c*n_bl for c in time_stamps.chunks[0]),)
    )

    interval_col = da.ones_like(time_col)*interval
    scan_col = da.zeros_like(time_col)  # TODO: Scan boundary case?

    # TODO: Should also check parameter mappings.
    for term in chain:

        t_int = term.time_interval or n_time

        da_t_bins = term.make_time_bins(
            time_col,
            interval_col,
            scan_col,
            t_int,
            term.respect_scan_boundaries,
            chunks=time_stamps.chunks
        )

        for block_ind in range(da_t_bins.npartitions):
            binning = da_t_bins.blocks[block_ind].compute()
            ivl_col = interval_col.blocks[block_ind].compute()

            if isinstance(term.time_interval, float):
                sol_widths = np.zeros(np.max(binning) + 1)
                for ivl_ind, target in enumerate(binning):
                    sol_widths[target] += ivl_col[ivl_ind]
                assert all(sol_widths[:-1] >= t_int)
            else:
                assert all(np.unique(binning, return_counts=True)[1] <= t_int)

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
def test_f_mappings(freq_chunk, chain):
    """Test construction of freq mappings for different chunks/intervals."""

    n_freq = 64  # Total number of channels to consider.

    chan_freqs = da.arange(n_freq, chunks=freq_chunk)
    chan_widths = da.ones(n_freq, chunks=freq_chunk)*7

    # TODO: Should also check parameter mappings.
    for term in chain:

        f_int = term.freq_interval or n_freq

        da_f_maps = term.make_freq_map(
            chan_freqs,
            chan_widths,
            term.freq_interval
        )

        # Set up and compute numpy values to test against.

        for block_ind in range(da_f_maps.npartitions):
            f_map = da_f_maps.blocks[block_ind].compute()
            chan_width = chan_widths.blocks[block_ind].compute()
            if isinstance(f_int, float):
                sol_widths = np.zeros(np.max(f_map) + 1)
                for ivl_ind, target in enumerate(f_map):
                    sol_widths[target] += chan_width[ivl_ind]
                assert all(sol_widths[:-1] >= f_int)
            else:
                assert all(np.unique(f_map, return_counts=True)[1] <= f_int)

# ------------------------------make_d_mappings--------------------------------


@pytest.mark.parametrize("n_dir", [2, 10])
@pytest.mark.parametrize("has_dd_term", [False, True])
def test_d_mappings(n_dir, has_dd_term, chain):
    """Test construction of direction mappings for different n_dir."""

    # Make second term in the chain direction dependent.
    chain[1].direction_dependent = has_dd_term

    for term in chain:

        d_map = term.make_dir_map(n_dir, term.direction_dependent)

        # Set up and compute numpy values to test against.

        np_d_map = np.array(np.arange(n_dir)*term.direction_dependent)

        assert_array_equal(d_map, np_d_map)

# -----------------------------------------------------------------------------
