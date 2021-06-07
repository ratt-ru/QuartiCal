import dask.array as da
import numpy as np
from uuid import uuid4
from quartical.utils.dask import blockwise_unique
from quartical.calibration.gain_types import term_types


def get_array_items(arr, inds):
    return arr[inds]


def make_t_maps(data_xds_list, opts):
    """Figure out how timeslots map to solution interval bins.

    Args:
        data_xds_list: A list of xarray.Dataset objects contatining MS data.
        opts: Namespace object of global options.
    Returns:
        t_bin_list: A list of dask.Arrays mapping unique times map to
            solution intervals.
        t_map_list: A list of dask.Arrays mapping row to solution intervals.
    """

    t_bin_list = []
    t_map_list = []

    for xds in data_xds_list:

        if opts.input_ms_is_bda:
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

        # Daskify the chunks per array - these are already known from the
        # initial chunking step.
        utime_per_chunk = da.from_array(utime_chunks,
                                        chunks=(1,),
                                        name="utpc-" + uuid4().hex)

        t_bin_arr = make_t_binnings(utime_per_chunk, utime_intervals, opts)
        t_map_arr = make_t_mappings(utime_ind, t_bin_arr)
        t_bin_list.append(t_bin_arr)
        t_map_list.append(t_map_arr)

    return t_bin_list, t_map_list


def make_t_binnings(utime_per_chunk, utime_intervals, opts):
    """Figure out how timeslots map to solution interval bins.

    Args:
        utime_per_chunk: dask.Array for number of utimes per chunk.
        utime_intervals: dask.Array of intervals assoscaited with each utime.
        opts: Namespace object of global options.
    Returns:
        t_bin_arr: A dask.Array of binnings per gain term.
    """

    terms = opts.solver_gain_terms
    n_term = len(terms)

    term_t_bins = []

    for term in terms:
        # Get frequency intervals. Or handles the zero case.
        t_int = getattr(opts, term + "_time_interval") or np.inf
        term_type = getattr(opts, term + "_type") or np.inf

        # Generate a mapping between time at data resolution and
        # time intervals.

        term_t_bin = da.map_blocks(
            _make_t_binnings,  # WRONG! Needs to be term dependent.
            utime_per_chunk,
            utime_intervals,
            t_int,
            chunks=(2, utime_intervals.chunks[0]),
            new_axis=0,
            dtype=np.int32,
            name="tbins-" + uuid4().hex)

        term_t_bins.append(term_t_bin)

    t_bin_arr = da.stack(term_t_bins, axis=2).rechunk({2: n_term})

    return t_bin_arr


def _make_t_binnings(n_utime, utime_intervals, t_int):
    """Internals of the time binner."""

    tbin_arr = np.empty((2, utime_intervals.size), dtype=np.int32)

    if isinstance(t_int, float):
        net_ivl = 0
        bin_num = 0
        for i, ivl in enumerate(utime_intervals):
            tbin_arr[:, i] = bin_num
            net_ivl += ivl
            if net_ivl >= t_int:
                net_ivl = 0
                bin_num += 1
    else:
        tbin_arr[:, :] = np.floor_divide(np.arange(n_utime),
                                         t_int,
                                         dtype=np.int32)

    return tbin_arr


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


def make_f_maps(data_xds_list, opts):
    """Figure out how channels map to solution interval bins.

    Args:
        data_xds_list: A list of xarray.Dataset objects contatining MS data.
        opts: Namespace object of global options.
    Returns:
        f_map_list: A list of dask.Arrays mapping channel to solution interval.
    """

    f_map_list = []

    for xds in data_xds_list:

        chan_freqs = xds.CHAN_FREQ.data
        chan_widths = xds.CHAN_WIDTH.data
        f_map_arr = make_f_mappings(chan_freqs, chan_widths, opts)
        f_map_list.append(f_map_arr)

    return f_map_list


def make_f_mappings(chan_freqs, chan_widths, opts):
    """Generate channel to solution interval mapping."""

    terms = opts.solver_gain_terms
    n_term = len(terms)
    n_chan = chan_freqs.size

    term_f_maps = []

    for term in terms:
        # Get frequency intervals. Or handles the zero case.
        f_int = getattr(opts, term + "_freq_interval") or n_chan
        term_type = getattr(opts, term + "_type") or n_chan

        # Generate a mapping between frequency at data resolution and
        # frequency intervals.

        term_f_map = da.map_blocks(
            term_types[term_type].make_f_maps,
            chan_freqs,
            chan_widths,
            f_int,
            chunks=(2, chan_freqs.chunks[0],),
            new_axis=0,
            dtype=np.int32,
            name="fmaps-" + uuid4().hex)

        term_f_maps.append(term_f_map)

    f_map_arr = da.stack(term_f_maps, axis=2).rechunk({2: n_term})

    return f_map_arr


def make_d_maps(data_xds_list, opts):
    """Figure out how directions map against the gain terms.

    Args:
        data_xds_list: A list of xarray.Dataset objects contatining MS data.
        opts: Namespace object of global options.
    Returns:
        d_map_list: A list of dask.Arrays mapping direction to term.
    """

    d_map_list = []

    for xds in data_xds_list:

        n_dir = xds.dims["dir"]
        d_map_arr = make_d_mappings(n_dir, opts)

        d_map_list.append(d_map_arr)

    return d_map_list


def make_d_mappings(n_dir, opts):
    """Generate direction to solution interval mapping."""

    terms = opts.solver_gain_terms

    # Get direction dependence for all terms. Or handles the zero case.
    dd_terms = [getattr(opts, term + "_direction_dependent") for term in terms]

    # Generate a mapping between model directions gain directions.

    d_map_arr = (np.arange(n_dir, dtype=np.int32)[:, None] * dd_terms).T

    return d_map_arr
