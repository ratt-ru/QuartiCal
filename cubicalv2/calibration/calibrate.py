# -*- coding: utf-8 -*-
import numpy as np
import dask.array as da
from math import ceil
from cubicalv2.calibration.solver import chain_solver
from cubicalv2.kernels.gjones_chain import update_func_factory
from loguru import logger  # noqa

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


def initialize_gains(*shapes):

    dtype = np.complex128

    gain_tuple = tuple(map(lambda s: np.zeros(s, dtype=dtype), shapes))

    for gain in gain_tuple:
        gain[..., ::3] = 1

    return gain_tuple


def add_calibration_graph(data_xds, opts):
    """Given data graph and options, adds the steps necessary for calibration.

    Extends the data graph with the steps necessary to perform gain
    calibration and in accordance with the options Namespace.

    Args:
        data_xds: A list of xarray data sets/graphs providing input data.
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

    gains_per_xds = {name: [] for name in opts.solver_gain_terms}

    for xds_ind, xds in enumerate(data_xds):

        # Unpack the data on the xds into variables with understandable names.
        data_col = xds.DATA.data[..., corr_slice]
        raw_model_col = xds.MODEL_DATA.data[..., corr_slice]
        ant1_col = xds.ANTENNA1.data
        ant2_col = xds.ANTENNA2.data
        time_col = xds.TIME.data
        flag_col = xds.FLAG.data[..., corr_slice]
        flag_row_col = xds.FLAG_ROW.data
        bitflag_col = xds.BITFLAG.data[..., corr_slice]
        bitflag_row_col = xds.BITFLAG_ROW.data

        data_col = data_col.map_blocks(
            lambda d, f: np.where(f == 1, 0, d), flag_col)
        model_col = raw_model_col.map_blocks(
            lambda m, f: np.where(f == 1, 0, m), flag_col[:, :, None, :])

        # Convert the time column data into indices.
        utime_ind = \
            time_col.map_blocks(lambda d: np.unique(d, return_inverse=True)[1])

        # Figure out the number of times per chunk.
        utime_per_chunk = \
            utime_ind.map_blocks(lambda f: np.max(f, keepdims=True) + 1,
                                 chunks=(1,),
                                 dtype=utime_ind.dtype)

        # Set up some values relating to problem dimensions.
        n_ant = opts._n_ant
        n_row, n_freq, n_dir, _ = model_col.shape
        n_chunks = data_col.numblocks[0]  # Number of chunks in row/time.
        n_term = len(opts.solver_gain_terms)  # Number of gain terms.

        # Initialise some empty lists onto which we can append associated
        # mappings/dimensions.
        t_maps = []
        f_maps = []
        g_shapes = []
        d_maps = []

        for term in opts.solver_gain_terms:

            atomic_t_int = getattr(opts, "{}_time_interval".format(term))
            atomic_f_int = getattr(opts, "{}_freq_interval".format(term))
            dd_term = getattr(opts, "{}_direction_dependent".format(term))

            # The or handles intervals specified as zero. These are assumed to
            # be solved aross an entire chunk. n_row is >> number of unique
            # times, but that value is unavaible at this point.
            t_int = da.full_like(utime_per_chunk, atomic_t_int or n_row)
            f_int = da.full_like(utime_per_chunk, atomic_f_int or n_freq)

            freqs_per_chunk = da.full_like(utime_per_chunk, n_freq)

            # Number of time intervals per data chunk.
            t_int_per_chunk = \
                utime_per_chunk.map_blocks(lambda t, t_i: int(ceil(t/t_i)),
                                           t_int,
                                           chunks=(1,),
                                           dtype=utime_per_chunk.dtype)

            # Number of frequency intervals per data chunk.
            f_int_per_chunk = \
                freqs_per_chunk.map_blocks(lambda f, f_i: int(ceil(f/f_i)),
                                           f_int,
                                           chunks=(1,),
                                           dtype=freqs_per_chunk.dtype)

            # Determine the per-chunk gain shapes from the time and frequency
            # intervals per chunk. Note that this uses the number of
            # correlations in the measurement set. TODO: This should depend
            # on the solver mode.
            g_shape = \
                da.map_blocks(
                    lambda t, f, na=None, nd=None, nc=None:
                        np.array([t, f, na, nd, nc]),
                    t_int_per_chunk,
                    f_int_per_chunk,
                    na=n_ant,
                    nd=n_dir if dd_term else 1,
                    nc=opts._ms_ncorr,
                    meta=np.empty((0, 0, 0, 0, 0), dtype=np.int32),
                    dtype=np.int32)
            g_shapes.append(g_shape)

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
                lambda f, f_i: np.array([i//f_i[0] for i in range(n_freq)]),
                f_int,
                chunks=(n_freq,),
                dtype=np.uint32)
            f_maps.append(f_map)

            d_maps.append(list(range(n_dir)) if dd_term else [0]*n_dir)

        # For each chunk, stack the per-gain mappings into a single array.
        t_map_arr = da.stack(t_maps, axis=1).rechunk({1: n_term})
        f_map_arr = da.stack(f_maps, axis=1).rechunk({1: n_term})
        d_map_arr = np.array(d_maps, dtype=np.uint32)

        # Create a tuple of gain arrays per chunk.
        gain_schema = ("rowlike", "chan", "ant", "dir", "corr")
        gain_args = [initialize_gains, gain_schema]

        for g_shape in g_shapes:
            gain_args.append(g_shape)
            gain_args.append(("rowlike",))

        # We set the time and frequency chunks to nan - this is done to encode
        # the fact that we do not know the shapes of the gains during the
        # graph construction step.
        gain_tuple = da.blockwise(
            *gain_args,
            align_arrays=False,
            dtype=np.complex128,
            new_axes={"chan": n_freq,
                      "ant": n_ant,
                      "dir": n_dir,
                      "corr": opts._ms_ncorr},
            adjust_chunks={"rowlike": (np.nan,)*n_chunks,
                           "chan": (np.nan,)})

        # Initialise the inverse gains. This is basically the same as the
        # gains, but we init them as empty. We init these outside the solver
        # as tuples are notoriously difficult to construct in nopython mode.
        inverse_gain_tuple = gain_tuple.map_blocks(
            lambda gt: tuple(map(np.empty_like, gt)),
            dtype=np.complex128)

        # We use a factory function to produce appropraite update functions
        # for use in the solver. TODO: Investigate using generated jit for this
        # purpose.
        compute_jhj_and_jhr, compute_update = \
            update_func_factory(opts.solver_mode)

        # Gains will not report its size or chunks correctly - this is because
        # we do not know their shapes during graph construction.
        gains = da.blockwise(
            chain_solver, ("rowlike", "chan", "ant", "dir", "corr"),
            model_col, ("rowlike", "chan", "dir", "corr"),
            gain_tuple, ("rowlike", "chan", "ant", "dir", "corr"),
            inverse_gain_tuple, ("rowlike", "chan", "ant", "dir", "corr"),
            data_col, ("rowlike", "chan", "corr"),
            ant1_col, ("rowlike",),
            ant2_col, ("rowlike",),
            t_map_arr, ("rowlike", "term"),
            f_map_arr, ("rowlike", "term"),
            d_map_arr, None,
            compute_jhj_and_jhr, None,
            compute_update, None,
            concatenate=True,
            dtype=model_col.dtype,
            align_arrays=False,)

        for ind, term in enumerate(opts.solver_gain_terms):
            gains_per_xds[term].append(gains.map_blocks(
                lambda g, i=None: g[i], i=ind, dtype=np.complex128))

        # This is an example of updating the contents of and xds. TODO: Use
        # this for writing out data.
        # bitflag_col = da.ones_like(bitflag_col)*10
        # data_xds[xds_ind] = \
        #     xds.assign({"BITFLAG": (xds.BITFLAG.dims, bitflag_col)})

    # Return the resulting graphs for the gains and updated xds.
    return gains_per_xds
