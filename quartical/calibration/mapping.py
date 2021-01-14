import dask.array as da
import numpy as np
from uuid import uuid4
from daskms.optimisation import inlined_array
from quartical.utils.dask import blockwise_unique


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
            lambda arr, inds: arr[inds],
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

    # Get time intervals for all terms. Or handles the zero case.
    t_ints = \
        [getattr(opts, term + "_time_interval") or np.inf for term in terms]

    t_bin_arr = da.map_blocks(
        _make_t_binnings,
        utime_per_chunk,
        utime_intervals,
        t_ints,
        chunks=(utime_intervals.chunks[0], (n_term,)),
        new_axis=1,
        dtype=np.int32,
        name="tbins-" + uuid4().hex)

    return t_bin_arr


def _make_t_binnings(n_utime, utime_intervals, t_ints):
    """Internals of the time binner."""

    tbins = np.empty((utime_intervals.size, len(t_ints)), dtype=np.int32)

    for tn, t_int in enumerate(t_ints):
        if isinstance(t_int, float):
            bins = np.empty_like(utime_intervals, dtype=np.int32)
            net_ivl = 0
            bin_num = 0
            for i, ivl in enumerate(utime_intervals):
                bins[i] = bin_num
                net_ivl += ivl
                if net_ivl >= t_int:
                    net_ivl = 0
                    bin_num += 1
            tbins[:, tn] = bins

        else:
            tbins[:, tn] = np.floor_divide(np.arange(n_utime),
                                           t_int,
                                           dtype=np.int32)

    return tbins


def make_t_mappings(utime_ind, t_bin_arr):
    """Convert unique time indices into solution interval mapping."""

    _, n_term = t_bin_arr.shape

    t_map_arr = da.blockwise(
        lambda arr, inds: arr[inds], ("rowlike", "term"),
        t_bin_arr, ("rowlike", "term"),
        utime_ind, ("rowlike",),
        adjust_chunks=(utime_ind.chunks[0], (n_term,)),
        dtype=np.int32,
        align_arrays=False,
        name="tmaps-" + uuid4().hex)

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

    # Get frequency intervals for all terms. Or handles the zero case.
    f_ints = \
        [getattr(opts, term + "_freq_interval") or n_chan for term in terms]

    # Generate a mapping between frequency at data resolution and
    # frequency intervals.

    f_map_arr = da.map_blocks(
        _make_f_mappings,
        chan_freqs,
        chan_widths,
        f_ints,
        chunks=(chan_freqs.chunks[0], (n_term,)),
        new_axis=1,
        dtype=np.int32,
        name="fmaps-" + uuid4().hex)

    f_map_arr = inlined_array(f_map_arr)

    return f_map_arr


def _make_f_mappings(chan_freqs, chan_widths, f_ints):
    """Internals of the frequency interval mapper."""

    n_chan = chan_freqs.size
    n_term = len(f_ints)

    f_map_arr = np.empty((n_chan, n_term), dtype=np.int32)

    for fn, f_int in enumerate(f_ints):
        if isinstance(f_int, float):
            net_ivl = 0
            bin_num = 0
            for i, ivl in enumerate(chan_widths):
                f_map_arr[i, fn] = bin_num
                net_ivl += ivl
                if net_ivl >= f_int:
                    net_ivl = 0
                    bin_num += 1
        else:
            f_map_arr[:, fn] = np.arange(n_chan)//f_int

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
