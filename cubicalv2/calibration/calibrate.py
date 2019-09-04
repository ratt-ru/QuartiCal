# -*- coding: utf-8 -*-
import numpy as np
import dask.array as da
from math import ceil
from cubicalv2.calibration.solver import solver
from cubicalv2.kernels.gjones import update_func_factory
from loguru import logger  # noqa
# from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler
# from dask.diagnostics import visualize
# from dask.optimization import fuse

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


def calibrate(data_xds, opts):

    # Calibrate per xds. This list will likely consist of an xds per SPW per
    # scan. This behaviour can be changed.

    corr_slice = slice_scheme[opts._ms_ncorr].get(opts.solver_mode, None)

    if not isinstance(corr_slice, slice):
        raise ValueError("{} solver mode incompatible with measurement set "
                         "containing {} correlations.".format(
                             opts.solver_mode, opts._ms_ncorr, ))

    gains_per_xds = []

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
        atomic_t_int = 1
        atomic_f_int = 1

        t_int = da.full_like(utime_per_chunk, atomic_t_int)
        f_int = da.full_like(utime_per_chunk, atomic_f_int)
        n_ant = opts._n_ant
        n_dir = model_col.shape[-2]

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

        # These values need to be computed early as they are needed to
        # create the gain matrix.
        n_t_int, n_f_int = da.compute(t_int_per_chunk, f_int_per_chunk)

        np_t_int_per_chunk = n_t_int if isinstance(n_t_int, int) \
            else tuple(n_t_int)
        n_t_int = n_t_int if isinstance(n_t_int, int) else np.sum(n_t_int)

        n_f_int = n_f_int if isinstance(n_f_int, int) else n_f_int[0]

        # Create and initialise the gain array. Dask makes this a two-step
        # process.
        gains = da.empty([n_t_int, n_f_int, n_ant, n_dir, 4],
                         dtype=np.complex128,
                         chunks=(np_t_int_per_chunk, -1, -1, -1, -1))

        gains = da.map_blocks(initialize_gains, gains, dtype=gains.dtype)

        # Generate a mapping between frequency at data resolution and frequency
        # intervals.
        freq_mapping = \
            freqs_per_chunk.map_blocks(
                lambda f, f_i: np.array([i//f_i[0] for i in range(f[0])]),
                f_int,
                chunks=(model_col.shape[1],),
                dtype=np.uint32)

        # Generate a mapping between time at data resolution and time
        # intervals.
        time_mapping = \
            utime_ind.map_blocks(
                lambda t, t_i: t//t_i, t_int,
                chunks=utime_ind.chunks,
                dtype=np.uint32)

        compute_jhj_and_jhr, compute_update = \
            update_func_factory(opts.solver_mode)

        gains = da.blockwise(solver, ("rowlike", "chan", "ant", "dir", "corr"),
                             model_col, ("rowlike", "chan", "dir", "corr"),
                             gains, ("rowlike", "chan", "ant", "dir", "corr"),
                             data_col, ("rowlike", "chan", "corr"),
                             ant1_col, ("rowlike",),
                             ant2_col, ("rowlike",),
                             time_mapping, ("rowlike",),
                             freq_mapping, ("rowlike",),
                             compute_jhj_and_jhr, None,
                             compute_update, None,
                             adjust_chunks={"rowlike": np_t_int_per_chunk,
                                            "chan": atomic_f_int},
                             dtype=model_col.dtype,
                             align_arrays=False)

        # Append the per-xds gains to a list.
        gains_per_xds.append(gains)

        # This is an example of updateing the contents of and xds. TODO: Use
        # this for writing out data.
        bitflag_col = da.ones_like(bitflag_col)*10
        data_xds[xds_ind] = \
            xds.assign({"BITFLAG": (xds.BITFLAG.dims, bitflag_col)})

    # Return the resulting graphs for the gains and updated xds.
    return gains_per_xds, data_xds

    # gains_per_xds[0].visualize("graph.pdf", optimize_graph=True)

    # with Profiler() as prof, \
    #      ResourceProfiler(dt=0.25) as rprof, \
    #      CacheProfiler() as cprof:

    #      out = da.compute(gains_per_xds)

    # visualize([prof, rprof, cprof])
