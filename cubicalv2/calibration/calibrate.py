# -*- coding: utf-8 -*-
import numpy as np
import dask.array as da
from cubicalv2.kernels.generics import (compute_residual)
from cubicalv2.statistics.statistics import (assign_interval_stats,
                                             create_gain_stats_xds,
                                             assign_post_solve_chisq,
                                             assign_presolve_data_stats,
                                             create_data_stats_xds)
from cubicalv2.flagging.flagging import (make_bitmask,
                                         initialise_bitflags,
                                         is_set,
                                         set_bitflag,
                                         compute_mad_flags)
from cubicalv2.calibration.constructor import construct_solver
from cubicalv2.weights.weights import initialize_weights
from cubicalv2.utils.dask import blockwise_unique
from itertools import chain, zip_longest
from uuid import uuid4
from loguru import logger  # noqa
from collections import namedtuple


# The following supresses the egregious numba pending deprecation warnings.
# TODO: Make sure that the code doesn't break when they finally decprecate
# reflected lists.
from numba.core.errors import NumbaDeprecationWarning
from numba.core.errors import NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


gain_shape_tup = namedtuple("gain_shape", "n_tint n_fint n_ant n_dir n_corr")
gain_chunk_tup = namedtuple("gain_chunk", "tchunk fchunk achunk dchunk cchunk")


def dask_residual(data, model, a1, a2, t_map_arr, f_map_arr, d_map_arr,
                  corr_mode, *gains):

    gain_list = [g for g in gains]

    return compute_residual(data, model, gain_list, a1, a2, t_map_arr,
                            f_map_arr, d_map_arr, corr_mode)


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

    # In the event that not all input BITFLAGS are required, generate a mask
    # which can be applied to select the appropriate bits.
    bitmask = make_bitmask(col_kwrds, opts)

    gain_xds_dict = {name: [] for name in opts.solver_gain_terms}
    data_stats_xds_list = []
    post_cal_data_xds_list = []

    for xds_ind, xds in enumerate(data_xds_list):

        # Unpack the data on the xds into variables with understandable names.
        # We create copies of arrays we intend to mutate as otherwise we end
        # up implicitly updating the xds.
        data_col = xds.DATA.data.copy()
        model_col = xds.MODEL_DATA.data
        ant1_col = xds.ANTENNA1.data
        ant2_col = xds.ANTENNA2.data
        time_col = xds.TIME.data
        flag_col = xds.FLAG.data
        flag_row_col = xds.FLAG_ROW.data
        bitflag_col = xds.BITFLAG.data
        bitflag_row_col = xds.BITFLAG_ROW.data
        weight_col = initialize_weights(xds, data_col, opts)

        bitflags = initialise_bitflags(data_col,
                                       weight_col,
                                       flag_col,
                                       flag_row_col,
                                       bitflag_col,
                                       bitflag_row_col,
                                       bitmask)   # TODO: Move

        # Selections of the following form generate where operations - this
        # will return a new array and is therefore distribution safe. It might
        # be suboptimal though.

        # If we raised the invalid bitflag, zero those data points.
        data_col[is_set(bitflags, "INVALID")] = 0

        # Anywhere we have a full resolution bitflag, we set the weight to 0.
        weight_col[bitflags] = 0

        # Convert the time column data into indices. Chunks is expected to be a
        # tuple of tuples.
        utime_chunks = xds.UTIME_CHUNKS
        utime_val, utime_ind = blockwise_unique(time_col,
                                                (utime_chunks,),
                                                return_inverse=True)

        # Daskify the chunks per array - these are already known from the
        # initial chunkings step.
        utime_per_chunk = da.from_array(utime_chunks,
                                        chunks=(1,),
                                        name="utpc-" + uuid4().hex)

        # Set up some values relating to problem dimensions.
        n_ant = xds.dims["ant"]
        n_row, n_chan, n_dir, n_corr = model_col.shape
        n_t_chunk, n_f_chunk, _ = data_col.numblocks
        n_term = len(opts.solver_gain_terms)  # Number of gain terms.

        # Initialise some empty containers for mappings/dimensions.
        t_maps = []
        f_maps = []
        d_maps = []
        ti_chunks = {}
        fi_chunks = {}

        # We preserve the gain chunking scheme here - returning multiple arrays
        # in later calls can obliterate chunking information.

        gain_shape_list = []
        gain_chunk_list = []

        # Create and populate xds for statisics at data resolution. Returns
        # some useful arrays required for future computations.
        data_stats_xds = create_data_stats_xds(utime_val,
                                               n_chan,
                                               n_ant,
                                               n_t_chunk,
                                               n_f_chunk)  # TODO: Unchecked

        data_stats_xds, unflagged_tfac, avg_abs_sqrd_model = \
            assign_presolve_data_stats(data_stats_xds,
                                       data_col,
                                       model_col,
                                       weight_col,
                                       bitflags,
                                       ant1_col,
                                       ant2_col,
                                       utime_ind,
                                       utime_per_chunk,
                                       utime_chunks)  # TODO: Unchecked

        for term in opts.solver_gain_terms:

            atomic_t_int = getattr(opts, "{}_time_interval".format(term))
            atomic_f_int = getattr(opts, "{}_freq_interval".format(term))
            dd_term = getattr(opts, "{}_direction_dependent".format(term))

            # Number of time intervals per data chunk. If this is zero,
            # solution interval is the entire axis per chunk.
            if atomic_t_int:
                ti_chunks[term] = tuple(int(np.ceil(nt/atomic_t_int))
                                        for nt in utime_chunks)
            else:
                ti_chunks[term] = tuple(1 for nt in utime_chunks)

            n_tint = np.sum(ti_chunks[term])

            # Number of frequency intervals per data chunk. If this is zero,
            # solution interval is the entire axis per chunk.
            if atomic_f_int:
                fi_chunks[term] = tuple(int(np.ceil(nc/atomic_f_int))
                                        for nc in data_col.chunks[1])
            else:
                fi_chunks[term] = tuple(1 for _ in range(n_f_chunk))

            n_fint = np.sum(fi_chunks[term])

            freqs_per_chunk = da.from_array(data_col.chunks[1],
                                            chunks=(1,),
                                            name="fpc-" + uuid4().hex)

            # Determine the per-chunk gain shapes from the time and frequency
            # intervals per chunk. Note that this uses the number of
            # correlations in the measurement set. TODO: This should depend
            # on the solver mode.

            gain_shape_list.append(gain_shape_tup(
                n_tint, n_fint, n_ant, n_dir if dd_term else 1, n_corr))
            gain_chunk_list.append(gain_chunk_tup(
                ti_chunks[term], fi_chunks[term], (n_ant,),
                (n_dir if dd_term else 1,), (n_corr,)))

            # Generate a mapping between time at data resolution and time
            # intervals. The or handles the 0 (full axis) case.

            t_map = utime_ind.map_blocks(np.floor_divide,
                                         atomic_t_int or n_row,
                                         chunks=utime_ind.chunks,
                                         dtype=np.uint32)
            t_maps.append(t_map)

            # Generate a mapping between frequency at data resolution and
            # frequency intervals. The or handles the 0 (full axis) case.

            f_map = freqs_per_chunk.map_blocks(
                lambda f, f_i: np.array([i//f_i for i in range(f.item())]),
                atomic_f_int or n_chan,
                chunks=(data_col.chunks[1],),
                dtype=np.uint32)
            f_maps.append(f_map)

            # Generate direction mapping - necessary for mixing terms.
            d_maps.append(list(range(n_dir)) if dd_term else [0]*n_dir)

            # Create an xds in which to store the gain values and assosciated
            # interval statistics.

            gain_xds = create_gain_stats_xds(n_tint,
                                             n_fint,
                                             n_ant,
                                             n_dir if dd_term else 1,
                                             n_corr,
                                             n_t_chunk,
                                             n_f_chunk,
                                             term,
                                             xds_ind)   # TODO: Unchecked

            # Update the gain xds with relevant interval statistics. This is
            # INSANELY expensive. TODO: Investigate necessity/improve.

            gain_xds, empty_intervals = \
                assign_interval_stats(gain_xds,
                                      data_stats_xds,
                                      unflagged_tfac,
                                      avg_abs_sqrd_model,
                                      ti_chunks[term],
                                      fi_chunks[term],
                                      atomic_t_int or n_row,
                                      atomic_f_int or n_chan,
                                      utime_per_chunk)    # TODO: Unchecked

            gain_xds_dict[term].append(gain_xds)

        # For each chunk, stack the per-gain mappings into a single array.
        t_map_arr = da.stack(t_maps, axis=1).rechunk({1: n_term})
        f_map_arr = da.stack(f_maps, axis=1).rechunk({1: n_term})
        d_map_arr = np.array(d_maps, dtype=np.uint32)

        # This has been fixed - this now constructs a custom graph which
        # preserves gain chunking. It also somewhat simplifies future work
        # as we now have a blueprint for pulling values out of the solver
        # layer.

        gain_list, conv_perc_list, conv_iter_list = \
            construct_solver(model_col,
                             data_col,
                             ant1_col,
                             ant2_col,
                             weight_col,
                             t_map_arr,
                             f_map_arr,
                             d_map_arr,
                             corr_mode,
                             gain_shape_list,
                             gain_chunk_list,
                             opts)

        # This has been improved substantially but likely still needs work.

        for ind, term in enumerate(opts.solver_gain_terms):

            gain_xds_dict[term][-1] = gain_xds_dict[term][-1].assign(
                {"gains": (("time_int", "freq_int", "ant", "dir", "corr"),
                           gain_list[ind]),
                 "conv_perc": (("t_chunk", "f_chunk"), conv_perc_list[ind]),
                 "conv_iter": (("t_chunk", "f_chunk"), conv_iter_list[ind])})

        # TODO: I want to remove this code if possible - I think V2 should
        # split solve and apply into two separate tasks.

        gain_schema = ("rowlike", "chan", "ant", "dir", "corr")

        gain_list_gen = chain.from_iterable(zip_longest(gain_list, [],
                                            fillvalue=gain_schema))
        gain_list = [x for x in gain_list_gen]

        residuals = da.blockwise(
            dask_residual, ("rowlike", "chan", "corr"),
            data_col, ("rowlike", "chan", "corr"),
            model_col, ("rowlike", "chan", "dir", "corr"),
            ant1_col, ("rowlike",),
            ant2_col, ("rowlike",),
            t_map_arr, ("rowlike", "term"),
            f_map_arr, ("chan", "term"),
            d_map_arr, None,
            corr_mode, None,
            *gain_list,
            dtype=data_col.dtype,
            align_arrays=False,
            concatenate=True,
            adjust_chunks={"rowlike": data_col.chunks[0],
                           "chan": data_col.chunks[1]})

        #######################################################################
        # This is the madmax flagging step which is not always enabled. TODO:
        # This likely needs to be moved into the solver.

        if opts.flags_mad_enable:
            mad_flags = compute_mad_flags(residuals,
                                          bitflag_col,
                                          ant1_col,
                                          ant2_col,
                                          n_ant,
                                          n_t_chunk,
                                          opts)

            bitflags = set_bitflag(bitflags, "MAD", mad_flags)

        #######################################################################

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
        # for saving. TODO: This is VERY rudimentary. Need to be done in
        # accordance with opts.

        updated_xds = \
            xds.assign({opts.output_column: (xds.DATA.dims, residuals),
                        "CUBI_BITFLAG": (xds.BITFLAG.dims, bitflags),
                        "CUBI_MODEL": (xds.DATA.dims,
                                       model_col.sum(axis=2,
                                                     dtype=np.complex64))})
        updated_xds.attrs["WRITE_COLS"] += [opts.output_column]

        post_cal_data_xds_list.append(updated_xds)

    # Return the resulting graphs for the gains and updated xds.
    return gain_xds_dict, post_cal_data_xds_list
