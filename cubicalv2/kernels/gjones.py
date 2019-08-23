# -*- coding: utf-8 -*-
import numpy as np
from numba import jit, prange


class ModeError(Exception):
    """Raised when solver mode is not understood."""
    pass


def update_func_factory(mode):

    if mode == "full-full":
        return jhj_and_jhr_full, update_full
    elif mode == "diag-full":
        raise NotImplementedError("diag-full solver mode not yet supported.")
    elif mode == "diag-diag":
        raise NotImplementedError("diag-diag solver mode not yet supported.")
    else:
        raise ModeError("Undefined calibration mode.")


@jit(nopython=True, fastmath=True, parallel=False, cache=False, nogil=True)
def jhj_and_jhr_full(model, gains, residual, a1, a2, t_map, f_map):

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
            mh00, mh01, mh10, mh11 = (m[0].conjugate(), m[2].conjugate(),
                                      m[1].conjugate(), m[3].conjugate())
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
            r00, r01, r10, r11 = (r[0].conjugate(), r[2].conjugate(),
                                  r[1].conjugate(), r[3].conjugate())

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
def update_full(jhj, jhr):

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
