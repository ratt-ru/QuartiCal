# -*- coding: utf-8 -*-
import numpy as np
import dask.array as da
from math import ceil
from cubicalv2.calibration.solver import chain_solver
from cubicalv2.kernels.gjones_chain import update_func_factory, residual_full
from cubicalv2.statistics.statistics import (assign_noise_estimates,
                                             assign_tf_statistics,
                                             assign_interval_statistics,
                                             create_data_stats_xds,
                                             create_gain_stats_xds)
from cubicalv2.flagging.flagging import (set_bitflag, unset_bitflag,
                                         make_bitmask, ibfdtype)
from loguru import logger  # noqa
from numba.typed import List
from itertools import chain, repeat
from uuid import uuid4
import xarray

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

    gain = np.zeros(shape, dtype=dtype)
    gain[..., ::3] = 1

    return gain


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

    corr_slice = slice_scheme[opts._ms_ncorr].get(opts.solver_mode, None)

    if not isinstance(corr_slice, slice):
        raise ValueError("{} solver mode incompatible with measurement set "
                         "containing {} correlations.".format(
                             opts.solver_mode, opts._ms_ncorr, ))

    # In the event that not all input BITFLAGS are required, generate a mask
    # which can be applied to select the appropriate bits.
    bitmask = make_bitmask(col_kwrds, opts)

    gains_per_xds = {name: [] for name in opts.solver_gain_terms}
    gain_stats_xds_dict = {name: [] for name in opts.solver_gain_terms}
    data_stats_xds_list = []
    post_cal_xds = []

    for xds_ind, xds in enumerate(data_xds):

        # Unpack the data on the xds into variables with understandable names.
        data_col = xds.DATA.data[..., corr_slice]
        model_col = xds.MODEL_DATA.data[..., corr_slice]
        ant1_col = xds.ANTENNA1.data
        ant2_col = xds.ANTENNA2.data
        time_col = xds.TIME.data
        flag_col = xds.FLAG.data[..., corr_slice]
        flag_row_col = xds.FLAG_ROW.data[..., corr_slice]
        bitflag_col = xds.BITFLAG.data[..., corr_slice] & bitmask
        bitflag_row_col = xds.BITFLAG_ROW.data[..., corr_slice] & bitmask

        n_utime = xds.CHUNK_SPEC

        if opts._unity_weights:
            weight_col = da.ones_like(data_col[:, :1, :], dtype=np.float32)
        else:
            weight_col = xds[opts.input_ms_weight_column].data[..., corr_slice]

        # The following handles the fact that the chosen weight column might
        # not have a frequency axis.

        if weight_col.ndim == 2:
            weight_col = weight_col.map_blocks(
                lambda w: np.expand_dims(w, 1), new_axis=1)

        cubical_bitflags = da.zeros(bitflag_col.shape,
                                    dtype=ibfdtype,
                                    chunks=bitflag_col.chunks,
                                    name="zeros-" + uuid4().hex)
        # If no bitmask is generated, we presume that we are using the
        # conventional FLAG and FLAG_ROW columns. TODO: Consider whether this
        # is safe behaviour.
        if bitmask == 0:
            cubical_bitflags = set_bitflag(cubical_bitflags, "PRIOR",
                                           flag_col)
            cubical_bitflags = set_bitflag(cubical_bitflags, "PRIOR",
                                           flag_row_col)
        else:
            cubical_bitflags = set_bitflag(cubical_bitflags, "PRIOR",
                                           bitflag_col > 0)
            cubical_bitflags = set_bitflag(cubical_bitflags, "PRIOR",
                                           bitflag_row_col > 0)

        # Anywhere we have input data flags (or invalid data), set the weights
        # to zero. Due to the fact np.inf*0 = np.nan (very annoying), we also
        # set the data to zero at these locations. These operations implicitly
        # broadcast the weights to the same frequency dimension as the data.
        # This unavoidable as we want to exploit the weights to effectively
        # flag.

        invalid_points = ~da.isfinite(data_col)
        data_col[invalid_points] = 0  # Set nasty data points to zero.

        cubical_bitflags = set_bitflag(cubical_bitflags,
                                       "INVALID",
                                       invalid_points)

        missing_points = da.logical_or(data_col[..., 0:1] == 0,
                                       data_col[..., 3:4] == 0)
        missing_points = da.logical_or(missing_points, data_col == 0)

        cubical_bitflags = set_bitflag(cubical_bitflags,
                                       "MISSING",
                                       missing_points)

        cubical_bitflags = set_bitflag(cubical_bitflags,
                                       "NULLWGHT",
                                       weight_col == 0)

        weight_col[cubical_bitflags] = 0

        # Convert the time column data into indices.
        utime_tuple = \
            time_col.map_blocks(lambda d: np.unique(d, return_inverse=True),
                                dtype=np.float32, meta=np.empty(0))

        utime_val = utime_tuple.map_blocks(lambda ut: ut[0],
                                           dtype=np.float64,
                                           chunks=(n_utime,))
        utime_ind = utime_tuple.map_blocks(lambda ut: ut[1], dtype=np.int32)

        # Figure out the number of times per chunk.
        utime_per_chunk = \
            utime_ind.map_blocks(lambda f: np.max(f, keepdims=True) + 1,
                                 chunks=(1,),
                                 dtype=utime_ind.dtype)

        # Set up some values relating to problem dimensions.
        n_ant = opts._n_ant
        n_row, n_chan, n_dir, n_corr = model_col.shape
        n_chunks = data_col.numblocks[0]  # Number of chunks in row/time.
        n_term = len(opts.solver_gain_terms)  # Number of gain terms.

        # Initialise some empty lists onto which we can append associated
        # mappings/dimensions.
        t_maps = []
        f_maps = []
        g_shapes = {}
        g_t_chunks = {}
        g_f_chunks = {}
        d_maps = []

        for term in opts.solver_gain_terms:

            atomic_t_int = getattr(opts, "{}_time_interval".format(term))
            atomic_f_int = getattr(opts, "{}_freq_interval".format(term))
            dd_term = getattr(opts, "{}_direction_dependent".format(term))

            # Number of time intervals per data chunk.
            chunk_ti_dims = \
                tuple(nt//atomic_t_int +
                      bool(nt % atomic_t_int) for nt in n_utime)

            g_t_chunks[term] = chunk_ti_dims

            n_tint = np.sum(chunk_ti_dims)

            # Number of frequency intervals per data chunk.
            chunk_fi_dims = \
                tuple(n_chan//atomic_f_int +
                      bool(n_chan % atomic_f_int) for _ in range(n_chunks))

            g_f_chunks[term] = chunk_fi_dims

            n_fint = chunk_fi_dims[0]

            # Convert the chunk dimensions into dask arrays.
            t_int_per_chunk = da.from_array(chunk_ti_dims, chunks=(1,))
            f_int_per_chunk = da.from_array(chunk_fi_dims, chunks=(1,))

            # The or handles intervals specified as zero. These are assumed to
            # be solved aross an entire chunk. n_row is >> number of unique
            # times, but that value is unavaible at this point.
            t_int = da.full_like(utime_per_chunk, atomic_t_int or n_row)
            f_int = da.full_like(utime_per_chunk, atomic_f_int or n_chan)

            freqs_per_chunk = da.full_like(utime_per_chunk, n_chan)

            # Determine the per-chunk gain shapes from the time and frequency
            # intervals per chunk. Note that this uses the number of
            # correlations in the measurement set. TODO: This should depend
            # on the solver mode.
            g_shapes[term] = \
                da.map_blocks(
                    lambda t, f, na=None, nd=None, nc=None:
                        np.array([t.item(), f.item(), na, nd, nc]),
                    t_int_per_chunk,
                    f_int_per_chunk,
                    na=n_ant,
                    nd=n_dir if dd_term else 1,
                    nc=opts._ms_ncorr,
                    meta=np.empty((0, 0, 0, 0, 0), dtype=np.int32),
                    dtype=np.int32)

            # Generate a mapping between time at data resolution and time
            # intervals.
            t_map = utime_ind.map_blocks(
                lambda t, t_i: t//t_i, t_int,
                chunks=utime_ind.chunks,
                dtype=np.uint32)
            t_maps.append(t_map)

            # Generate a mapping between frequency at data resolution and
            # frequency intervals. This currently presumes that we don't chunk
            # in frequency. BEWARE!
            f_map = freqs_per_chunk.map_blocks(
                lambda f, f_i: np.array([i//f_i[0] for i in range(n_chan)]),
                f_int,
                chunks=(n_chan,),
                dtype=np.uint32)
            f_maps.append(f_map)

            d_maps.append(list(range(n_dir)) if dd_term else [0]*n_dir)

            gain_stats_xds = create_gain_stats_xds(n_tint,
                                                   n_fint,
                                                   n_ant,
                                                   n_dir if dd_term else 1,
                                                   n_corr,
                                                   n_chunks)

            gain_stats_xds_dict[term].append(gain_stats_xds)

        # For each chunk, stack the per-gain mappings into a single array.
        t_map_arr = da.stack(t_maps, axis=1).rechunk({1: n_term})
        f_map_arr = da.stack(f_maps, axis=1).rechunk({1: n_term})
        d_map_arr = np.array(d_maps, dtype=np.uint32)

        # We preserve the gain chunking scheme here - returning multiple arrays
        # in later calls can obliterate chunking information.

        gain_schema = ("rowlike", "chan", "ant", "dir", "corr")
        gain_list = []
        gain_chunks = {}

        # Note that while we technically have a frequency chunk per row chunk,
        # we assume uniform frequency chunking to avoid madness.

        for term, shape in g_shapes.items():
            gain = da.blockwise(
                initialize_gain, gain_schema,
                shape, ("rowlike",),
                align_arrays=False,
                dtype=np.complex128,
                new_axes={"chan": n_chan,
                          "ant": n_ant,
                          "dir": n_dir,
                          "corr": opts._ms_ncorr},
                adjust_chunks={"rowlike": g_t_chunks[term],
                               "chan": g_f_chunks[term][0]})
            gain_list.append(gain)
            gain_list.append(gain_schema)
            gain_chunks[term] = gain.chunks



        # We use a factory function to produce appropriate update functions
        # for use in the solver. TODO: Investigate using generated jit for this
        # purpose.
        compute_jhj_and_jhr, compute_update = \
            update_func_factory(opts.solver_mode)

        data_stats_xds = \
            create_data_stats_xds(utime_val, n_chan, n_ant, n_chunks)

        data_stats_xds = assign_noise_estimates(data_stats_xds,
                                                data_col,
                                                cubical_bitflags,
                                                ant1_col,
                                                ant2_col,
                                                n_ant)

        data_stats_xds = assign_tf_statistics(data_stats_xds,
                                              cubical_bitflags,
                                              ant1_col,
                                              ant2_col,
                                              utime_ind,
                                              utime_per_chunk,
                                              n_ant,
                                              n_chunks,
                                              n_chan,
                                              n_utime)

        data_stats_xds = assign_interval_statistics(data_stats_xds,
                                                    cubical_bitflags,
                                                    ant1_col,
                                                    ant2_col,
                                                    t_map_arr,
                                                    f_map_arr,
                                                    t_int_per_chunk,
                                                    f_int_per_chunk,
                                                    atomic_t_int,
                                                    atomic_f_int,
                                                    opts.solver_gain_terms,
                                                    n_utime)

        data_stats_xds_list.append(data_stats_xds)

        # print(stats_xds.tot_norm_factor.compute())

        # Gains will not report its size or chunks correctly - this is because
        # we do not know their shapes during graph construction.
        gains = da.blockwise(
            chain_solver, ("rowlike", "chan", "ant", "dir", "corr"),
            model_col, ("rowlike", "chan", "dir", "corr"),
            data_col, ("rowlike", "chan", "corr"),
            ant1_col, ("rowlike",),
            ant2_col, ("rowlike",),
            weight_col, ("rowlike", "chan", "corr"),
            t_map_arr, ("rowlike", "term"),
            f_map_arr, ("rowlike", "term"),
            d_map_arr, None,
            compute_jhj_and_jhr, None,
            compute_update, None,
            *gain_list,
            concatenate=True,
            dtype=model_col.dtype,
            align_arrays=False,)



        # Gains are in dask limbo at this point - we have returned a list
        # which is not understood by the array interface. We take the list of
        # gains per chunk and explicitly unpack them using map_blocks.

        unpacked_gains = []
        for ind, term in enumerate(opts.solver_gain_terms):
            unpacked_gains.append(
                da.blockwise(lambda g: g[ind], gain_schema,
                             gains, gain_schema,
                             dtype=np.complex128,
                             adjust_chunks={"rowlike": gain_chunks[term][0],
                                            "chan": gain_chunks[term][1]},
                             meta=np.empty((0,0,0,0,0), dtype=np.complex128)))

        print(unpacked_gains)

        gain_zipper = zip(unpacked_gains, repeat(gain_schema, n_term))

        gain_list = list(chain.from_iterable(gain_zipper))

        residuals = da.blockwise(
            dask_residual, ("rowlike", "chan", "corr"),
            data_col, ("rowlike", "chan", "corr"),
            model_col, ("rowlike", "chan", "dir", "corr"),
            ant1_col, ("rowlike",),
            ant2_col, ("rowlike",),
            t_map_arr, ("rowlike", "term"),
            f_map_arr, ("rowlike", "term"),
            d_map_arr, None,
            *gain_list,
            dtype=data_col.dtype,
            align_arrays=False,
            concatenate=True,
            adjust_chunks={"rowlike": data_col.chunks[0],
                           "chan": data_col.chunks[1]})

        # noise_estimate, inv_var_per_chan = estimate_noise(residuals,
        #                                                   cubical_bitflags,
        #                                                   ant1_col,
        #                                                   ant2_col,
        #                                                   n_ant)

        # print(noise_estimate.compute())

        for ind, term in enumerate(opts.solver_gain_terms):
            gains_per_xds[term].append(unpacked_gains[ind])

            print(unpacked_gains[ind].shape)

            gain_stats_xds_dict[term][-1] = \
                gain_stats_xds_dict[term][-1].assign(
                    {"gains": (("time_int", "freq_int", "ant", "dir", "corr"),
                               unpacked_gains[ind])})

        # Add quantities required elsewhere to the xds and mark certain columns
        # for saving.

        updated_xds = \
            xds.assign({"CUBI_RESIDUAL": (xds.DATA.dims, residuals),
                        "CUBI_BITFLAG": (xds.BITFLAG.dims, cubical_bitflags),
                        "CUBI_MODEL": (xds.DATA.dims,
                                       model_col.sum(axis=2,
                                                     dtype=np.complex64))})
        updated_xds.attrs["WRITE_COLS"] += ["CUBI_RESIDUAL"]

        post_cal_xds.append(updated_xds)

    # Return the resulting graphs for the gains and updated xds.
    return gains_per_xds, post_cal_xds
