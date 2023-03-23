import dask.array as da
import numpy as np
from uuid import uuid4
from quartical.utils.dask import blockwise_unique
from quartical.gains import TERM_TYPES
from quartical.config.internal import yield_from


def get_array_items(arr, inds):
    return arr[inds]


def make_t_maps(data_xds_list, chain_opts):
    """Figure out how timeslots map to solution interval bins.

    Args:
        data_xds_list: A list of xarray.Dataset objects contatining MS data.
        chain_opts: A Chain config object.

    Returns:
        t_bin_list: A list of dask.Arrays mapping unique times map to
            solution intervals.
        t_map_list: A list of dask.Arrays mapping row to solution intervals.
    """

    t_bin_list = []
    t_map_list = []

    for xds in data_xds_list:

        if hasattr(xds, "UPSAMPLED_TIME"):  # We are dealing with BDA.
            time_col = xds.UPSAMPLED_TIME.data
            interval_col = xds.UPSAMPLED_INTERVAL.data
        else:
            time_col = xds.TIME.data
            interval_col = xds.INTERVAL.data

        # Convert the time column data into indices. Chunks is expected to
        # be a tuple of tuples.
        utime_chunks = xds.UTIME_CHUNKS
        _, utime_loc, utime_ind = blockwise_unique(time_col,
                                                   chunks=(utime_chunks,),
                                                   return_index=True,
                                                   return_inverse=True)

        # Assosciate each unique time with an interval. This assumes that
        # all rows at a given time have the same interval as the
        # alternative is madness.
        utime_intervals = da.map_blocks(
            get_array_items,
            interval_col,
            utime_loc,
            chunks=utime_loc.chunks,
            dtype=np.float64)

        if "SCAN_NUMBER" in xds.data_vars.keys():
            utime_scan_numbers = da.map_blocks(
                get_array_items,
                xds.SCAN_NUMBER.data,
                utime_loc,
                chunks=utime_loc.chunks,
                dtype=np.int32)
        else:
            utime_scan_numbers = da.zeros_like(
                utime_intervals,
                dtype=np.int32,
                name="scan_number-" + uuid4().hex
            )

        # Daskify the chunks per array - these are already known from the
        # initial chunking step.
        utime_per_chunk = da.from_array(utime_chunks,
                                        chunks=(1,),
                                        name="utpc-" + uuid4().hex)

        t_bin_arr = make_t_binnings(utime_per_chunk,
                                    utime_intervals,
                                    utime_scan_numbers,
                                    chain_opts)

        t_map_arr = make_t_mappings(utime_ind, t_bin_arr)
        t_bin_list.append(t_bin_arr)
        t_map_list.append(t_map_arr)

    return t_bin_list, t_map_list


def make_t_binnings(
    utime_per_chunk, utime_intervals, utime_scan_numbers, chain_opts
):
    """Figure out how timeslots map to solution interval bins.

    Args:
        utime_per_chunk: dask.Array for number of utimes per chunk.
        utime_intervals: dask.Array of intervals assoscaited with each utime.
        chain_opts: A Chain config object.
    Returns:
        t_bin_arr: A dask.Array of binnings per gain term.
    """

    term_t_bins = []

    for _, tt, ti, rsb in yield_from(
        chain_opts, ("type", "time_interval", "respect_scan_boundaries")
    ):

        term_t_bin = da.map_blocks(
            TERM_TYPES[tt].make_t_bins,
            utime_per_chunk,
            utime_intervals,
            utime_scan_numbers,
            ti or np.inf,  # Or handles zero.
            rsb,
            chunks=(2, utime_intervals.chunks[0]),
            new_axis=0,
            dtype=np.int32,
            name="tbins-" + uuid4().hex)

        term_t_bins.append(term_t_bin)

    t_bin_arr = da.stack(term_t_bins, axis=2).rechunk({2: len(term_t_bins)})

    return t_bin_arr


def make_t_mappings(utime_ind, t_bin_arr):
    """Convert unique time indices into solution interval mapping."""

    _, _, n_term = t_bin_arr.shape

    t_map_arr = da.blockwise(
        _make_t_mappings, ("param", "rowlike", "term"),
        t_bin_arr, ("param", "rowlike", "term"),
        utime_ind, ("rowlike",),
        adjust_chunks={"rowlike": utime_ind.chunks[0]},
        dtype=np.int32,
        align_arrays=False,
        name="tmaps-" + uuid4().hex)

    return t_map_arr


def _make_t_mappings(t_bin_arr, utime_ind):

    n_map, _, n_term = t_bin_arr.shape
    n_row = utime_ind.size

    t_map_arr = np.empty((n_map, n_row, n_term), dtype=np.int32)

    for i in range(n_map):
        t_map_arr[i, :, :] = t_bin_arr[i, utime_ind]

    return t_map_arr


def make_f_maps(data_xds_list, chain_opts):
    """Figure out how channels map to solution interval bins.

    Args:
        data_xds_list: A list of xarray.Dataset objects contatining MS data.
        chain_opts: A Chain config object.
    Returns:
        f_map_list: A list of dask.Arrays mapping channel to solution interval.
    """

    f_map_list = []

    for xds in data_xds_list:

        chan_freqs = xds.CHAN_FREQ.data
        chan_widths = xds.CHAN_WIDTH.data
        f_map_arr = make_f_mappings(chan_freqs, chan_widths, chain_opts)
        f_map_list.append(f_map_arr)

    return f_map_list


def make_f_mappings(chan_freqs, chan_widths, chain_opts):
    """Generate channel to solution interval mapping."""

    n_chan = chan_freqs.size

    term_f_maps = []

    for _, tt, fi in yield_from(chain_opts, ("type", "freq_interval")):

        term_f_map = da.map_blocks(
            TERM_TYPES[tt].make_f_maps,
            chan_freqs,
            chan_widths,
            fi or n_chan,  # Or handles the zero case.
            chunks=(2, chan_freqs.chunks[0],),
            new_axis=0,
            dtype=np.int32,
            name="fmaps-" + uuid4().hex)

        term_f_maps.append(term_f_map)

    f_map_arr = da.stack(term_f_maps, axis=2).rechunk({2: len(term_f_maps)})

    return f_map_arr


def make_d_maps(data_xds_list, chain_opts):
    """Figure out how directions map against the gain terms.

    Args:
        data_xds_list: A list of xarray.Dataset objects contatining MS data.
        chain_opts: A Chain config object.
    Returns:
        d_map_list: A list of dask.Arrays mapping direction to term.
    """

    d_map_list = []

    for xds in data_xds_list:

        n_dir = xds.dims["dir"]
        d_map_arr = make_d_mappings(n_dir, chain_opts)

        d_map_list.append(d_map_arr)

    return d_map_list


def make_d_mappings(n_dir, chain_opts):
    """Generate direction to solution interval mapping."""

    # Get direction dependence for all terms.
    dd_terms = [dd for _, dd in yield_from(chain_opts, "direction_dependent")]

    # Generate a mapping between model directions gain directions.
    d_map_arr = (np.arange(n_dir, dtype=np.int32)[:, None] * dd_terms).T

    return d_map_arr
