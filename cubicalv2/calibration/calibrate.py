# -*- coding: utf-8 -*-
import numpy as np
import dask.array as da
from cubicalv2.calibration.solver import chain_solver, stat_fields
from cubicalv2.kernels.gjones_chain import update_func_factory, residual_full
from cubicalv2.statistics.statistics import (assign_interval_stats,
                                             create_gain_stats_xds,
                                             assign_post_solve_chisq,
                                             assign_presolve_data_stats,
                                             create_data_stats_xds)
from cubicalv2.flagging.flagging import (make_bitmask,
                                         initialise_fullres_bitflags,
                                         is_set,
                                         set_bitflag,
                                         compute_mad_flags)
from cubicalv2.calibration.constructor import construct_solver
from cubicalv2.weights.weights import initialize_weights
from cubicalv2.utils.dask import blockwise_unique
from itertools import chain, zip_longest
from uuid import uuid4
from operator import getitem
from loguru import logger  # noqa


# The following supresses the egregious numba pending deprecation warnings.
# TODO: Make sure that the code doesn't break when they finally decprecate
# reflected lists.
from numba.errors import NumbaDeprecationWarning
from numba.errors import NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# Defines which solver modes are supported given the number of correlations
# in the measurement set. If a mode is supported, this dictionary contains
# the slice object which will make the measurement set data compatible with
# the solver. TODO: Make it possible to select off diagonal entries/specific
# correlations.

slice_scheme = {1: {"scalar": slice(None)},
                2: {"diag-diag": slice(None),
                    "scalar": slice(0, 1)},
                4: {"full-full": slice(None),
                    "diag-full": slice(None),
                    "diag-diag": slice(None, None, 3),
                    "scalar": slice(0, 1)}}


def dask_residual(data, model, a1, a2, t_map_arr, f_map_arr, d_map_arr,
                  *gains):

    gain_list = [g for g in gains]

    return residual_full(data, model, gain_list, a1, a2, t_map_arr,
                         f_map_arr, d_map_arr)


def initialize_gain(shape):

    dtype = np.complex128

    gain = np.zeros(shape[0], dtype=dtype)
    gain[..., ::3] = 1

    return gain


def shape_maker(t, f, na, nd, nc):
    return np.atleast_2d([t.item(), f.item(), na, nd, nc])


def add_calibration_graph(data_xds, col_kwrds, opts):
    """Given data graph and options, adds the steps necessary for calibration.

    Extends the data graph with the steps necessary to perform gain
    calibration and in accordance with the options Namespace.

    Args:
        data_xds: A list of xarray data sets/graphs providing input data.
        col_kwrds: A dictionary containing column keywords.
        opts: A Namespace object containing all necessary configuration.

    Returns:
        A dictionary of lists containing graphs which prodcuce a gain array
        per gain term per xarray dataset.
    """

    # Calibrate per xds. This list will likely consist of an xds per SPW, per
    # scan. This behaviour can be changed.

    # Figure out how to slice the correlation axis.

    corr_slice = slice_scheme[opts._ms_ncorr].get(opts.solver_mode, None)

    if not isinstance(corr_slice, slice):
        raise ValueError("{} solver mode incompatible with measurement set "
                         "containing {} correlations.".format(
                             opts.solver_mode, opts._ms_ncorr, ))

    # In the event that not all input BITFLAGS are required, generate a mask
    # which can be applied to select the appropriate bits.
    bitmask = make_bitmask(col_kwrds, opts)

    gain_xds_dict = {name: [] for name in opts.solver_gain_terms}
    data_stats_xds_list = []
    post_cal_data_xds_list = []

    for xds_ind, xds in enumerate(data_xds):

        # Unpack the data on the xds into variables with understandable names.
        # We create copies of arrays we intend to mutate as otherwise we end
        # up implicitly updating the xds.
        data_col = xds.DATA.data[..., corr_slice].copy()
        model_col = xds.MODEL_DATA.data[..., corr_slice]
        ant1_col = xds.ANTENNA1.data
        ant2_col = xds.ANTENNA2.data
        time_col = xds.TIME.data
        flag_col = xds.FLAG.data[..., corr_slice]
        flag_row_col = xds.FLAG_ROW.data[..., corr_slice]

        # We immediately apply the bitmask so that we only have the bitflags
        # we intend to use.
        bitflag_col = xds.BITFLAG.data[..., corr_slice] & bitmask
        bitflag_row_col = xds.BITFLAG_ROW.data[..., corr_slice] & bitmask

        utime_chunks = xds.UTIME_CHUNKS

        weight_col = initialize_weights(xds, data_col, corr_slice, opts)

        fullres_bitflags = initialise_fullres_bitflags(data_col,
                                                       weight_col,
                                                       flag_col,
                                                       flag_row_col,
                                                       bitflag_col,
                                                       bitflag_row_col,
                                                       bitmask)

        # If we raised the invalid bitflag, zero those data points.
        data_col[is_set(fullres_bitflags, "INVALID")] = 0

        # Anywhere we have a full resolution bitflag, we set the weight to 0.
        weight_col[fullres_bitflags] = 0

        # Convert the time column data into indices. Chunks is expected to be a
        # tuple of tuples.
        utime_val, utime_ind = blockwise_unique(time_col,
                                                (utime_chunks,),
                                                return_inverse=True)

        # Daskify the chunks per array - these are already known from the
        # initial chunkings step.
        utime_per_chunk = da.from_array(utime_chunks,
                                        chunks=(1,),
                                        name=False)

        # Set up some values relating to problem dimensions.
        n_ant = opts._n_ant
        n_row, n_chan, n_dir, n_corr = model_col.shape
        n_t_chunk = len(data_col.chunks[0])
        n_f_chunk = len(data_col.chunks[1])
        n_term = len(opts.solver_gain_terms)  # Number of gain terms.

        # Initialise some empty containers for mappings/dimensions.
        t_maps = []
        f_maps = []
        d_maps = []
        ti_chunks = {}
        fi_chunks = {}

        # We preserve the gain chunking scheme here - returning multiple arrays
        # in later calls can obliterate chunking information. TODO: Build a
        # custom graph.

        gain_schema = ("rowlike", "chan", "ant", "dir", "corr")
        gain_list = []
        gain_chunks = {}

        # Create and populate xds for statisics at data resolution. Returns
        # some useful arrays required for future computations.
        data_stats_xds = create_data_stats_xds(utime_val,
                                               n_chan,
                                               n_ant,
                                               n_t_chunk,
                                               n_f_chunk)

        data_stats_xds, unflagged_tfac, avg_abs_sqrd_model = \
            assign_presolve_data_stats(data_stats_xds,
                                       data_col,
                                       model_col,
                                       weight_col,
                                       fullres_bitflags,
                                       ant1_col,
                                       ant2_col,
                                       utime_ind,
                                       utime_per_chunk,
                                       utime_chunks)

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

            n_fint = sum(fi_chunks[term])

            # Convert the chunk dimensions into dask arrays.
            t_int_per_chunk = da.from_array(ti_chunks[term],
                                            chunks=(1,),
                                            name=False)
            f_int_per_chunk = da.from_array(fi_chunks[term],
                                            chunks=(1,),
                                            name=False)

            freqs_per_chunk = da.from_array(data_col.chunks[1],
                                            chunks=(1,),
                                            name=False)

            # Determine the per-chunk gain shapes from the time and frequency
            # intervals per chunk. Note that this uses the number of
            # correlations in the measurement set. TODO: This should depend
            # on the solver mode.

            g_shape = da.blockwise(
                shape_maker, ("rowlike", "chan",),
                t_int_per_chunk, ("rowlike",),
                f_int_per_chunk, ("chan",),
                n_ant, None,
                n_dir if dd_term else 1, None,
                opts._ms_ncorr, None,
                dtype=np.int32)

            # Note that while we technically have a frequency chunk per row
            # chunk, we assume uniform frequency chunking to avoid madness.

            gain = da.blockwise(
                initialize_gain, gain_schema,
                g_shape, ("rowlike", "chan"),
                dtype=np.complex128,
                new_axes={"ant": n_ant,
                          "dir": n_dir if dd_term else 1,
                          "corr": opts._ms_ncorr},
                adjust_chunks={"rowlike": ti_chunks[term],
                               "chan": fi_chunks[term]})

            gain_list.append(gain)

            # The chunking of each gain will be lost post-solve due to Dask's
            # lack of support for multiple return. We store the chunking values
            # in a diecitonary so we can later correctly describe the output
            # gains.

            gain_chunks[term] = gain.chunks

            # Create a an array for gain resolution bitflags. These have the
            # same shape as the gains. We use an explicit creation routine
            # to ensure we don't have accidental aliasing.

            gain_flags = da.zeros(gain.shape,
                                  chunks=gain.chunks,
                                  dtype=np.uint8,
                                  name="gflags-" + uuid4().hex)

            gain_list.append(gain_flags)

            # Generate a mapping between time at data resolution and time
            # intervals. The or handles the 0 (full axis) case.

            t_map = utime_ind.map_blocks(np.floor_divide,
                                         atomic_t_int or n_row,
                                         chunks=utime_ind.chunks,
                                         dtype=np.uint32)
            t_maps.append(t_map)

            # Generate a mapping between frequency at data resolution and
            # frequency intervals. The or handles the 0 (full axis) case.
            # This currently presumes that we don't chunk in frequency. BEWARE!
            f_map = freqs_per_chunk.map_blocks(
                lambda f, f_i: np.array([i//f_i for i in range(f.item())]),
                atomic_f_int or n_chan,
                chunks=(data_col.chunks[1],),
                dtype=np.uint32)
            f_maps.append(f_map)

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
                                             xds_ind)

            # Update the gain xds with relevant interval statistics.

            gain_xds, empty_intervals = \
                assign_interval_stats(gain_xds,
                                      data_stats_xds,
                                      unflagged_tfac,
                                      avg_abs_sqrd_model,
                                      ti_chunks[term],
                                      fi_chunks[term],
                                      atomic_t_int or n_row,
                                      atomic_f_int or n_chan,
                                      utime_per_chunk)

            # After computing the stats we need to do some flagging operations/
            # construct the gain flags.

            # Empty intervals corresponds to missing gains.

            gain_flags = set_bitflag(gain_flags, "MISSING", empty_intervals)

            gain_xds_dict[term].append(gain_xds)

        # For each chunk, stack the per-gain mappings into a single array.
        t_map_arr = da.stack(t_maps, axis=1).rechunk({1: n_term})
        f_map_arr = da.stack(f_maps, axis=1).rechunk({1: n_term})
        d_map_arr = np.array(d_maps, dtype=np.uint32)

        # We use a factory function to produce appropriate update functions
        # for use in the solver. TODO: Investigate using generated jit for this
        # purpose.
        compute_jhj_and_jhr, compute_update = \
            update_func_factory(opts.solver_mode)

        # Gains will not report its size or chunks correctly - this is because
        # it is actually returning multiple arrays (somewhat sordid) which we
        # will subseqently unpack and give appropriate dimensions. NB - this
        # call WILL mutate the contents of gain list. This is a necessary evil
        # for now though it may be possible to move the array creation into
        # the numba layer. TODO: It would be more appropriate to construct a
        # custom graph for this - this works but I worry that it is very
        # vulnerable to interface changes.

        gain_list, solver_info = construct_solver(chain_solver,
                                                  model_col,
                                                  data_col,
                                                  ant1_col,
                                                  ant2_col,
                                                  weight_col,
                                                  t_map_arr,
                                                  f_map_arr,
                                                  d_map_arr,
                                                  compute_jhj_and_jhr,
                                                  compute_update,
                                                  *gain_list)

        # Gains are in dask limbo at this point - we have returned a list
        # which is not understood by the array interface. We take the list of
        # gains per chunk and explicitly unpack them using blockwise. We use
        # the stored chunk values to give the resulting gains meaningful
        # shapes.

        stats_dict = {}
        for ind, term in enumerate(opts.solver_gain_terms):

            dd_term = getattr(opts, "{}_direction_dependent".format(term))

            stats_dict[term] = {}

            for field, field_dtype in stat_fields.items():
                stats_dict[term][field] = da.blockwise(
                    lambda d, t, n: np.atleast_2d(getattr(d[t], n)),
                    ("rowlike", "chan"),
                    solver_info, ("rowlike", "chan"),
                    ind, None,
                    field, None,
                    dtype=field_dtype,
                    align_arrays=False)

            gain_xds_dict[term][-1] = \
                gain_xds_dict[term][-1].assign(
                    {"gains": (("time_int", "freq_int", "ant", "dir", "corr"),
                               gain_list[ind]),
                     "conv_perc": (("t_chunk", "f_chunk"),
                                   stats_dict[term]["conv_perc"]),
                     "conv_iters": (("t_chunk", "f_chunk"),
                                    stats_dict[term]["conv_iters"])})

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
            *gain_list,
            dtype=data_col.dtype,
            align_arrays=False,
            concatenate=True,
            adjust_chunks={"rowlike": data_col.chunks[0],
                           "chan": data_col.chunks[1]})

        #######################################################################
        # This is the madmax flagging step which is not always enabled.

        if opts.flags_mad_enable:
            mad_flags = compute_mad_flags(residuals,
                                          bitflag_col,
                                          ant1_col,
                                          ant2_col,
                                          n_ant,
                                          n_t_chunk,
                                          opts)

            fullres_bitflags = set_bitflag(fullres_bitflags, "MAD", mad_flags)

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
            xds.assign({"CUBI_RESIDUAL": (xds.DATA.dims, residuals),
                        "CUBI_BITFLAG": (xds.BITFLAG.dims, fullres_bitflags),
                        "CUBI_MODEL": (xds.DATA.dims,
                                       model_col.sum(axis=2,
                                                     dtype=np.complex64))})
        updated_xds.attrs["WRITE_COLS"] += ["CUBI_RESIDUAL"]

        post_cal_data_xds_list.append(updated_xds)

    # Return the resulting graphs for the gains and updated xds.
    return gain_xds_dict, post_cal_data_xds_list
