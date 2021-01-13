# -*- coding: utf-8 -*-
import numpy as np
import dask.array as da
from daskms.optimisation import inlined_array
from quartical.kernels.generics import (compute_residual,
                                        compute_corrected_residual)
from quartical.statistics.statistics import (assign_interval_stats,
                                             assign_post_solve_chisq,
                                             assign_presolve_data_stats,)
from quartical.flagging.flagging import (set_bitflag,
                                         compute_mad_flags)
from quartical.calibration.constructor import construct_solver
from quartical.utils.dask import blockwise_unique
from uuid import uuid4
from loguru import logger  # noqa
from collections import namedtuple
import xarray


# The following supresses the egregious numba pending deprecation warnings.
# TODO: Make sure that the code doesn't break when they finally decprecate
# reflected lists.
from numba.core.errors import NumbaDeprecationWarning
from numba.core.errors import NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


gain_shape_tup = namedtuple("gain_shape_tup",
                            "n_t_int n_f_int n_ant n_dir n_corr")
chunk_spec_tup = namedtuple("chunk_spec_tup",
                            "tchunk fchunk achunk dchunk cchunk")
dstat_dims_tup = namedtuple("dstat_dims_tup",
                            "n_utime n_chan n_ant n_t_chunk n_f_chunk")


def dask_residual(data, model, a1, a2, t_map_arr, f_map_arr, d_map_arr,
                  row_map, row_weights, corr_mode, *gains):
    """Thin wrapper to handle an unknown number of input gains."""

    return compute_residual(data, model, gains, a1, a2, t_map_arr,
                            f_map_arr, d_map_arr, row_map, row_weights,
                            corr_mode)


def dask_corrected_residual(residual, a1, a2, t_map_arr, f_map_arr,
                            d_map_arr, row_map, row_weights, corr_mode,
                            *gains):
    """Thin wrapper to handle an unknown number of input gains."""

    return compute_corrected_residual(residual, gains, a1, a2, t_map_arr,
                                      f_map_arr, d_map_arr, row_map,
                                      row_weights, corr_mode)


def add_calibration_graph(data_xds_list, col_kwrds, opts):
    """Given data graph and options, adds the steps necessary for calibration.

    Extends the data graph with the steps necessary to perform gain
    calibration and in accordance with the options Namespace.

    Args:
        data_xds_list: A list of xarray data sets/graphs providing input data.
        col_kwrds: A dictionary containing column keywords.
        opts: A Namespace object containing all necessary configuration.

    Returns:
        A dictionary of lists containing graphs which prodcuce a gain array
        per gain term per xarray dataset.
    """

    # Calibrate per xds. This list will likely consist of an xds per SPW, per
    # scan. This behaviour can be changed.

    corr_mode = opts.input_ms_correlation_mode

    gain_xds_dict = {name: [] for name in opts.solver_gain_terms}
    data_stats_xds_list = []
    post_cal_data_xds_list = []

    for xds_ind, xds in enumerate(data_xds_list):

        # Unpack the data on the xds into variables with understandable names.
        ant1_col = xds.ANTENNA1.data
        ant2_col = xds.ANTENNA2.data

        if opts.input_ms_is_bda:
            time_col = xds.UPSAMPLED_TIME.data
            interval_col = xds.UPSAMPLED_INTERVAL.data
        else:
            time_col = xds.TIME.data
            interval_col = xds.INTERVAL.data

        bitflag_col = xds.BITFLAG.data
        weight_col = xds.WEIGHT.data
        data_bitflags = xds.DATA_BITFLAGS.data
        chan_freqs = xds.CHAN_FREQ.data
        chan_widths = xds.CHAN_WIDTH.data

        # Convert the time column data into indices. Chunks is expected to be a
        # tuple of tuples.
        utime_chunks = xds.UTIME_CHUNKS
        _, utime_loc, utime_ind = blockwise_unique(time_col,
                                                   chunks=(utime_chunks,),
                                                   return_index=True,
                                                   return_inverse=True)

        # Assosciate each unique time with an interval. This assumes that all
        # rows at a given time have the same interval as the alternative is
        # madness.
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

        # Set up some values relating to problem dimensions.
        n_row, n_chan, n_ant, n_dir, n_corr = \
            [xds.dims[d] for d in ["row", "chan", "ant", "dir", "corr"]]

        n_t_chunk, n_f_chunk = [len(xds.chunks[d]) for d in ["row", "chan"]]

        # Create and populate xds for statisics at data resolution. Returns
        # some useful arrays required for future computations. TODO: I really
        # dislike this layer. Consider revising.

        data_stats_xds, unflagged_tfac, avg_abs_sqrd_model = \
            assign_presolve_data_stats(xds, utime_ind, utime_per_chunk)

        # Construct arrays containing mappings between data resolution and
        # solution intervals per term.
        t_bin_arr = make_t_binnings(utime_per_chunk, utime_intervals, opts)
        t_map_arr = make_t_mappings(utime_ind, t_bin_arr)
        f_map_arr = make_f_mappings(chan_freqs, chan_widths, opts)
        d_map_arr = make_d_mappings(n_dir, opts)

        # Generate an xds per gain term - these conveniently store dimension
        # info. We can assign results to them later.

        gain_xds_list = make_gain_xds_list(xds, t_map_arr, f_map_arr, opts)

        # Update the gain xds with relevant interval statistics. Used to be
        # very expensive - has been improved.

        gain_xds_list, empty_intervals = \
            assign_interval_stats(gain_xds_list,
                                  data_stats_xds,
                                  unflagged_tfac,
                                  avg_abs_sqrd_model,
                                  utime_per_chunk,
                                  t_bin_arr,
                                  f_map_arr,
                                  opts)

        # This has been fixed - this now constructs a custom graph which
        # preserves gain chunking. It also somewhat simplifies future work
        # as we now have a blueprint for pulling values out of the solver
        # layer. Note that we reuse the variable name gain_xds_list to keep
        # things succinct. TODO: Just pass in xds and unpack internally?

        gain_xds_list = construct_solver(xds,
                                         t_map_arr,
                                         f_map_arr,
                                         d_map_arr,
                                         corr_mode,
                                         gain_xds_list,
                                         opts)

        # This has been improved substantially but likely still needs work.

        for ind, term in enumerate(opts.solver_gain_terms):

            gain_xds_dict[term].append(gain_xds_list[ind])

        # TODO: I want to remove this code if possible - I think V2 should
        # split solve and apply into two separate tasks. This is problematic
        # in the direction dependent case, as it necessiates recomputing the
        # model visibilities (expensive).

        visibility_products = make_visibiltiy_output(xds,
                                                     gain_xds_list,
                                                     t_map_arr,
                                                     f_map_arr,
                                                     d_map_arr,
                                                     opts)

        residuals = visibility_products["residual"]

        # --------------------------------MADMAX-------------------------------
        # This is the madmax flagging step which is not always enabled. TODO:
        # This likely needs to be moved into the solver. Note that this use of
        # set bitflags is likely to break the distributed scheduler.

        if opts.flags_mad_enable:
            mad_flags = compute_mad_flags(residuals,
                                          bitflag_col,
                                          ant1_col,
                                          ant2_col,
                                          n_ant,
                                          n_t_chunk,
                                          opts)

            data_bitflags = set_bitflag(data_bitflags, "MAD", mad_flags)

        # ---------------------------------------------------------------------

        data_stats_xds = assign_post_solve_chisq(data_stats_xds,
                                                 residuals,
                                                 weight_col,
                                                 ant1_col,
                                                 ant2_col,
                                                 utime_ind,
                                                 utime_per_chunk,
                                                 utime_chunks)

        data_stats_xds_list.append(data_stats_xds)

        # Add quantities required elsewhere to the xds and mark certain columns
        # for saving. TODO: Consider whether and where this could be moved.

        ms_outputs = {}
        write_cols = []

        if opts.output_visibility_product:
            n_vis_prod = len(opts.output_visibility_product)
            itr = zip(opts.output_column, opts.output_visibility_product)
            ms_outputs.update({cn: (xds.DATA.dims, visibility_products[vn])
                              for cn, vn in itr})

            write_cols.extend(opts.output_column[:n_vis_prod])

        ms_outputs["CUBI_BITFLAG"] = (xds.BITFLAG.dims, data_bitflags)

        updated_xds = xds.assign(ms_outputs)
        updated_xds.WRITE_COLS.extend(write_cols)

        post_cal_data_xds_list.append(updated_xds)

    # Return the resulting graphs for the gains and updated xds.
    return gain_xds_dict, post_cal_data_xds_list


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


def make_d_mappings(n_dir, opts):
    """Generate direction to solution interval mapping."""

    terms = opts.solver_gain_terms

    # Get direction dependence for all terms. Or handles the zero case.
    dd_terms = [getattr(opts, term + "_direction_dependent") for term in terms]

    # Generate a mapping between model directions gain directions.

    d_map_arr = (np.arange(n_dir, dtype=np.int32)[:, None] * dd_terms).T

    return d_map_arr


def make_gain_xds_list(data_xds, t_map_arr, f_map_arr, opts):
    """Returns a list of xarray.Dataset objects describing the gain terms.

    For a given input xds containing data, creates an xarray.Dataset object
    per term which describes the term's dimensions.

    Args:
        data_xds: xarray.Dataset object containing input data.
        opts: Namepsace object containing global config.

    Returns:
        gain_xds_list: A list of xarray.Dataset objects describing the gain
            terms assosciated with the data_xds.
    """

    gain_xds_list = []

    tipc_per_term = da.map_blocks(lambda arr: arr[-1:, :] + 1,
                                  t_map_arr,
                                  chunks=((1,)*t_map_arr.numblocks[0],
                                          t_map_arr.chunks[1]))

    fipc_per_term = da.map_blocks(lambda arr: arr[-1:, :] + 1,
                                  f_map_arr,
                                  chunks=((1,)*f_map_arr.numblocks[0],
                                          f_map_arr.chunks[1]))

    tipc_per_term, fipc_per_term = da.compute(tipc_per_term, fipc_per_term)

    for term_ind, term in enumerate(opts.solver_gain_terms):

        dd_term = getattr(opts, "{}_direction_dependent".format(term))
        term_type = getattr(opts, "{}_type".format(term))

        utime_chunks = data_xds.UTIME_CHUNKS
        freq_chunks = data_xds.chunks["chan"]

        n_t_chunk = len(utime_chunks)
        n_f_chunk = len(freq_chunks)

        n_chan, n_ant, n_dir, n_corr = \
            [data_xds.dims[d] for d in ["chan", "ant", "dir", "corr"]]

        n_t_int_per_chunk = tuple(map(int, tipc_per_term[:, term_ind]))

        n_t_int = np.sum(n_t_int_per_chunk)

        n_f_int_per_chunk = tuple(map(int, fipc_per_term[:, term_ind]))

        n_f_int = np.sum(n_f_int_per_chunk)

        # Determine the per-chunk gain shapes from the time and frequency
        # intervals per chunk. Note that this uses the number of
        # correlations in the measurement set. TODO: This should depend
        # on the solver mode.

        chunk_spec = chunk_spec_tup(n_t_int_per_chunk,
                                    n_f_int_per_chunk,
                                    (n_ant,),
                                    (n_dir if dd_term else 1,),
                                    (n_corr,))

        # Stored fields which identify the data with which this gain is
        # assosciated.
        id_fields = {f: data_xds.attrs[f]
                     for f in ["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"]}

        # Set up an xarray.Dataset describing the gain term.
        gain_xds = xarray.Dataset(
            coords={"time_int": ("time_int",
                                 np.arange(n_t_int, dtype=np.int32)),
                    "freq_int": ("freq_int",
                                 np.arange(n_f_int, dtype=np.int32)),
                    "ant": ("ant",
                            np.arange(n_ant, dtype=np.int32)),
                    "dir": ("dir",
                            np.arange(n_dir if dd_term else 1,
                                      dtype=np.int32)),
                    "corr": ("corr",
                             np.arange(n_corr, dtype=np.int32)),
                    "t_chunk": ("t_chunk",
                                np.arange(n_t_chunk, dtype=np.int32)),
                    "f_chunk": ("f_chunk",
                                np.arange(n_f_chunk, dtype=np.int32))},
            attrs={"NAME": term,
                   "CHUNK_SPEC": chunk_spec,
                   "TYPE": term_type,
                   **id_fields})

        gain_xds_list.append(gain_xds)

    return gain_xds_list


def make_visibiltiy_output(data_xds, gain_xds_list, t_map_arr, f_map_arr,
                           d_map_arr, opts):
    """Creates dask arrays for possible visibility outputs.

    Given and xds containing data and its assosciated gains, produces
    dask.Array objects containing the possible visibility outputs.

    Args:
        data_xds: An xarray.Dataset object containing input data.
        gain_xds_list: A list containing gain xarray.Dataset objects.
        t_map_arr: A dask.Array object of time mappings.
        f_map_arr: A dask.Array object of frequency mappings.
        d_map_arr: A dask.Array object of direction mappings.
        opts: A Namespace object containing all necessary configuration.

    Returns:
        A dictionary of lists containing graphs which prodcuce a gain array
        per gain term per xarray dataset.

    """

    data_col = data_xds.DATA.data
    model_col = data_xds.MODEL_DATA.data
    ant1_col = data_xds.ANTENNA1.data
    ant2_col = data_xds.ANTENNA2.data

    row_map = data_xds.ROW_MAP.data if opts.input_ms_is_bda else None
    row_weights = data_xds.ROW_WEIGHTS.data if opts.input_ms_is_bda else None

    gain_schema = ("rowlike", "chan", "ant", "dir", "corr")

    # TODO: For gains with n_dir > 1, we can select out the gains we actually
    # want to correct for.
    gain_list = \
        [x for gxds in gain_xds_list for x in (gxds.gains.data, gain_schema)]

    corr_mode = opts.input_ms_correlation_mode
    is_bda = opts.input_ms_is_bda

    residual = da.blockwise(
        dask_residual, ("rowlike", "chan", "corr"),
        data_col, ("rowlike", "chan", "corr"),
        model_col, ("rowlike", "chan", "dir", "corr"),
        ant1_col, ("rowlike",),
        ant2_col, ("rowlike",),
        t_map_arr, ("rowlike", "term"),
        f_map_arr, ("chan", "term"),
        d_map_arr, None,
        *((row_map, ("rowlike",)) if is_bda else (None, None)),
        *((row_weights, ("rowlike",)) if is_bda else (None, None)),
        corr_mode, None,
        *gain_list,
        dtype=data_col.dtype,
        align_arrays=False,
        concatenate=True,
        adjust_chunks={"rowlike": data_col.chunks[0],
                       "chan": data_col.chunks[1]})

    corrected_residual = da.blockwise(
        dask_corrected_residual, ("rowlike", "chan", "corr"),
        residual, ("rowlike", "chan", "corr"),
        ant1_col, ("rowlike",),
        ant2_col, ("rowlike",),
        t_map_arr, ("rowlike", "term"),
        f_map_arr, ("chan", "term"),
        d_map_arr, None,
        *((row_map, ("rowlike",)) if is_bda else (None, None)),
        *((row_weights, ("rowlike",)) if is_bda else (None, None)),
        corr_mode, None,
        *gain_list,
        dtype=residual.dtype,
        align_arrays=False,
        concatenate=True,
        adjust_chunks={"rowlike": data_col.chunks[0],
                       "chan": data_col.chunks[1]})

    # We can cheat and reuse the corrected residual code - the only difference
    # is whether we supply the residuals or the data.
    corrected_data = da.blockwise(
        dask_corrected_residual, ("rowlike", "chan", "corr"),
        data_col, ("rowlike", "chan", "corr"),
        ant1_col, ("rowlike",),
        ant2_col, ("rowlike",),
        t_map_arr, ("rowlike", "term"),
        f_map_arr, ("chan", "term"),
        d_map_arr, None,
        *((row_map, ("rowlike",)) if is_bda else (None, None)),
        *((row_weights, ("rowlike",)) if is_bda else (None, None)),
        corr_mode, None,
        *gain_list,
        dtype=residual.dtype,
        align_arrays=False,
        concatenate=True,
        adjust_chunks={"rowlike": data_col.chunks[0],
                       "chan": data_col.chunks[1]})

    visibility_outputs = {"residual": residual,
                          "corrected_residual": corrected_residual,
                          "corrected_data": corrected_data}

    return visibility_outputs
