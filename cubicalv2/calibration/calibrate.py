# -*- coding: utf-8 -*-
import numpy as np
import dask.array as da
from math import ceil
from numba import jit, prange
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler
from dask.diagnostics import visualize

def initialize_gains(gains):

    gains[:, :, :] = np.eye(2).ravel()

    return gains

@jit(nopython=True, fastmath=True, parallel=False, cache=False, nogil=True)
def compute_jhj_and_jhr(model, gains, residual, a1, a2, t_map, f_map):

    n_rows, n_chan, _ = model.shape

    jhr = np.zeros_like(gains)
    jhj = np.zeros_like(gains)

    for row in prange(n_rows):
        for f in range(n_chan):

            t_m, f_m, a1_m, a2_m = t_map[row], f_map[f], a1[row], a2[row]

            m = model[row, f]
            ga = gains[t_m, f_m, a1_m]
            gb = gains[t_m, f_m, a2_m]
            r = residual[row, f]

            g00, g01, g10, g11 = gb[0], gb[1], gb[2], gb[3]
            mh00, mh01, mh10, mh11 = m[0].conjugate(), m[2].conjugate(), m[1].conjugate(), m[3].conjugate()
            r00, r01, r10, r11 = r[0], r[1], r[2], r[3]

            jh00 = (g00*mh00 + g01*mh10)
            jh01 = (g00*mh01 + g01*mh11)
            jh10 = (g10*mh00 + g11*mh10)
            jh11 = (g10*mh01 + g11*mh11)

            j00 = jh00.conjugate()
            j01 = jh10.conjugate()
            j10 = jh01.conjugate()
            j11 = jh11.conjugate()

            jhr[t_m, f_m, a1_m, 0] += (r00*jh00 + r01*jh10)
            jhr[t_m, f_m, a1_m, 1] += (r00*jh01 + r01*jh11)
            jhr[t_m, f_m, a1_m, 2] += (r10*jh00 + r11*jh10)
            jhr[t_m, f_m, a1_m, 3] += (r10*jh01 + r11*jh11)

            jhj[t_m, f_m, a1_m, 0] += (j00*jh00 + j01*jh10)
            jhj[t_m, f_m, a1_m, 1] += (j00*jh01 + j01*jh11)
            jhj[t_m, f_m, a1_m, 2] += (j10*jh00 + j11*jh10)
            jhj[t_m, f_m, a1_m, 3] += (j10*jh01 + j11*jh11)

            g00, g01, g10, g11 = ga[0], ga[1], ga[2], ga[3]
            m00, m01, m10, m11 = m[0], m[1], m[2], m[3]
            r00, r01, r10, r11 = r[0].conjugate(), r[2].conjugate(), r[1].conjugate(), r[3].conjugate()

            jh00 = (g00*m00 + g01*m10)
            jh01 = (g00*m01 + g01*m11)
            jh10 = (g10*m00 + g11*m10)
            jh11 = (g10*m01 + g11*m11)

            j00 = jh00.conjugate()
            j01 = jh10.conjugate()
            j10 = jh01.conjugate()
            j11 = jh11.conjugate()

            jhr[t_m, f_m, a2_m, 0] += (r00*jh00 + r01*jh10)
            jhr[t_m, f_m, a2_m, 1] += (r00*jh01 + r01*jh11)
            jhr[t_m, f_m, a2_m, 2] += (r10*jh00 + r11*jh10)
            jhr[t_m, f_m, a2_m, 3] += (r10*jh01 + r11*jh11)

            jhj[t_m, f_m, a2_m, 0] += (j00*jh00 + j01*jh10)
            jhj[t_m, f_m, a2_m, 1] += (j00*jh01 + j01*jh11)
            jhj[t_m, f_m, a2_m, 2] += (j10*jh00 + j11*jh10)
            jhj[t_m, f_m, a2_m, 3] += (j10*jh01 + j11*jh11)

    return jhj, jhr

@jit(nopython=True, fastmath=True, parallel=False, cache=False, nogil=True)
def compute_update(jhj_and_jhr):

    jhj, jhr = jhj_and_jhr

    n_tint, n_fint, n_ant, n_corr = jhj.shape

    update = np.empty_like(jhr)

    for t in range(n_tint):
        for f in range(n_fint):
            for a in range(n_ant):

                jhj00 = jhj[t, f, a, 0]
                jhj01 = jhj[t, f, a, 1]
                jhj10 = jhj[t, f, a, 2]
                jhj11 = jhj[t, f, a, 3]

                det = (jhj00*jhj11 - jhj01*jhj10)

                if det == 0:
                    jhjinv00 = 0
                    jhjinv01 = 0
                    jhjinv10 = 0
                    jhjinv11 = 0
                else:
                    jhjinv00 = jhj11/det
                    jhjinv01 = -jhj01/det
                    jhjinv10 = -jhj10/det
                    jhjinv11 = jhj00/det

                jhr00 = jhr[t, f, a, 0]
                jhr01 = jhr[t, f, a, 1]
                jhr10 = jhr[t, f, a, 2]
                jhr11 = jhr[t, f, a, 3]

                update[t, f, a, 0] = (jhr00*jhjinv00 + jhr01*jhjinv10)
                update[t, f, a, 1] = (jhr00*jhjinv01 + jhr01*jhjinv11)
                update[t, f, a, 2] = (jhr10*jhjinv00 + jhr11*jhjinv10)
                update[t, f, a, 3] = (jhr10*jhjinv01 + jhr11*jhjinv11)

    return update


def calibrate(data_xds, opts):

    # Calibrate per xds. This list will likely consist of an xds per SPW per
    # scan. This behaviour can be changed.

    gains_per_xds = []

    for xds in data_xds:

        # Unpack the data on the xds into variables with understandable names.
        data_col = xds.DATA.data
        model_col = xds.MODEL_DATA.data
        ant1_col = xds.ANTENNA1.data
        ant2_col = xds.ANTENNA2.data
        time_col = xds.TIME.data

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
        gains = da.empty([n_t_int, n_f_int, n_ant, 4],
                         dtype=np.complex128,
                         chunks=(np_t_int_per_chunk, -1, -1, -1))

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

        for i in range(10):
            # Compute the jhj and jhr components of the GN/CubiCal update.
            jhj_and_jhr = \
                da.blockwise(
                    compute_jhj_and_jhr, ("rowlike", "chan", "ant", "corr"),
                    model_col, ("rowlike", "chan", "corr"),
                    gains, ("rowlike", "chan", "ant", "corr"),
                    data_col, ("rowlike", "chan", "corr"),
                    ant1_col, ("rowlike",),
                    ant2_col, ("rowlike",),
                    time_mapping, ("rowlike",),
                    freq_mapping, ("rowlike",),
                    adjust_chunks={"rowlike": np_t_int_per_chunk,
                                   "chan": atomic_f_int},
                    dtype=model_col.dtype,
                    align_arrays=False)

            # Combine jhj^(-1) and jhr into a gain update.
            upd = da.blockwise(
                    compute_update, ("rowlike", "chan", "ant", "corr"),
                    jhj_and_jhr, ("rowlike", "chan", "ant", "corr"),
                    dtype=gains.dtype,
                    align_arrays=False)

            # Update the gains.
            gains = (gains + upd)/2

        # Append the per-xds gains to a list.
        gains_per_xds.append(gains)

    # Call compute on the resulting graph.
    da.compute(gains_per_xds)

    # gains_per_xds[0].visualize("graph.pdf")

    # with Profiler() as prof, \
    #      ResourceProfiler(dt=0.25) as rprof, \
    #      CacheProfiler() as cprof:

    #      out = da.compute(gains_per_xds)

    # visualize([prof, rprof, cprof])
