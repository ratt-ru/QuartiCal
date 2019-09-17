# -*- coding: utf-8 -*-
import numpy as np
import dask.array as da
from math import ceil
from cubicalv2.calibration.solver import solver, chain_solver
from cubicalv2.kernels.gjones_chain import update_func_factory
from loguru import logger  # noqa
from numba.typed import List

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


def initialize_gains(gains):

    gains[:, :, :, :] = np.eye(2).ravel()

    return gains


def combine(*mappings):

    out = List()

    for m in mappings:
        out.append(m)

    return out


def calibrate(data_xds, opts):

    # Calibrate per xds. This list will likely consist of an xds per SPW per
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
        model_col = xds.MODEL_DATA.data[..., corr_slice]
        ant1_col = xds.ANTENNA1.data
        ant2_col = xds.ANTENNA2.data
        time_col = xds.TIME.data
        flag_col = xds.FLAG.data[..., corr_slice]
        flag_row_col = xds.FLAG_ROW.data
        bitflag_col = xds.BITFLAG.data[..., corr_slice]
        bitflag_row_col = xds.BITFLAG_ROW.data

        # Convert the time column data into indices.
        utime_ind = \
            time_col.map_blocks(lambda d: np.unique(d, return_inverse=True)[1])

        # Figure out the number of times per chunk.
        utime_per_chunk = \
            utime_ind.map_blocks(lambda f: np.max(f, keepdims=True) + 1,
                                 chunks=(1,),
                                 dtype=utime_ind.dtype)

        # Time/frquency solution intervals. These will ultimately live on opts.
        # This needs to be modified to work for a Jones chain. I want to crate
        # time and frequency mapppings per gain term.

        n_ant = opts._n_ant
        n_dir = model_col.shape[-2]
        n_row = model_col.shape[0]
        n_freq = model_col.shape[1]

        t_maps = []
        f_maps = []
        g_shapes = []

        for term in opts.solver_gain_terms:

            atomic_t_int = getattr(opts, "{}_time_interval".format(term))
            atomic_f_int = getattr(opts, "{}_freq_interval".format(term))

            # The or handles intervals specified as zero. These are assumed to
            # be solved aross an entire chunk. n_row is >> number of unique
            # times, but that value is unavaible at this point.
            t_int = da.full_like(utime_per_chunk, atomic_t_int or n_row)
            f_int = da.full_like(utime_per_chunk, atomic_f_int or n_freq)

            freqs_per_chunk = da.full_like(utime_per_chunk, model_col.shape[1])

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

            # Determine the per-chunk gain shapes from the time a frequency
            # intervals per chunk. Note that this used the number of
            # correlations in the measurement set. TODO: This should depend
            # on the solver mode.
            g_shape = \
                da.map_blocks(
                    lambda t, f, na=None, nd=None, nc=None: (t, f, na, nd, nc),
                    t_int_per_chunk,
                    f_int_per_chunk,
                    na=n_ant,
                    nd=n_dir,
                    nc=opts._ms_ncorr,
                    meta=np.empty((0, 0, 0, 0, 0), dtype=np.int32),
                    dtype=np.int32)
            g_shapes.append(g_shape)

            # Generate a mapping between frequency at data resolution and
            # frequency intervals. This currently presumes that we don't chunk
            # in frequency. BEWARE!
            f_map = freqs_per_chunk.map_blocks(
                lambda f, f_i: np.array([i//f_i[0] for i in range(n_freq)]),
                f_int,
                chunks=(n_freq,),
                dtype=np.uint32)
            f_maps.append(f_map)

            # Generate a mapping between time at data resolution and time
            # intervals.
            t_map = utime_ind.map_blocks(
                lambda t, t_i: t//t_i, t_int,
                chunks=utime_ind.chunks,
                dtype=np.uint32)
            t_maps.append(t_map)

        # For each chunk, create a numba typed list containing the per-gain
        # information. This is all done using the combine function.

        t_map_args = [combine, ("rowlike",)]

        for t_map in t_maps:
            t_map_args.append(t_map)
            t_map_args.append(("rowlike",))

        # t_map_list = da.blockwise(*t_map_args,
        #                           align_arrays=False,
        #                           dtype=np.int32)

        t_map_list = da.stack(t_maps, axis=1).rechunk({1: len(t_maps)})

        f_map_args = [combine, ("rowlike",)]

        for f_map in f_maps:
            f_map_args.append(f_map)
            f_map_args.append(("rowlike",))

        # f_map_list = da.blockwise(*f_map_args,
        #                           align_arrays=False,
        #                           dtype=np.int32)

        f_map_list = da.stack(f_maps, axis=1).rechunk({1: len(f_maps)})

        g_shape_args = [combine, ("rowlike",)]

        for g_shape in g_shapes:
            g_shape_args.append(g_shape)
            g_shape_args.append(("rowlike",))

        g_shape_list = da.blockwise(*g_shape_args,
                                    align_arrays=False,
                                    dtype=np.int32)

        compute_jhj_and_jhr, compute_update = \
            update_func_factory(opts.solver_mode)

        # Gains will not report its size or chunks correctly - this is because
        # we do not know their shapes in advance.
        gains = da.blockwise(
            chain_solver, ("rowlike", "chan", "ant", "dir", "corr"),
            model_col, ("rowlike", "chan", "dir", "corr"),
            g_shape_list, ("rowlike",),
            data_col, ("rowlike", "chan", "corr"),
            ant1_col, ("rowlike",),
            ant2_col, ("rowlike",),
            t_map_list, ("rowlike", "term"),
            f_map_list, ("rowlike", "term"),
            compute_jhj_and_jhr, None,
            compute_update, None,
            concatenate=True,
            new_axes={"ant": n_ant},
            dtype=model_col.dtype,
            align_arrays=False,)

        for ind, term in enumerate(opts.solver_gain_terms):
            gains_per_xds[term].append(gains.map_blocks(
                lambda g, i=None: g[i], i=ind, dtype=np.complex128))

        # This is an example of updating the contents of and xds. TODO: Use
        # this for writing out data.
        bitflag_col = da.ones_like(bitflag_col)*10
        data_xds[xds_ind] = \
            xds.assign({"BITFLAG": (xds.BITFLAG.dims, bitflag_col)})

    # Return the resulting graphs for the gains and updated xds.
    return gains_per_xds, data_xds
