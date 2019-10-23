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


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def invert_gains(gain_list, inverse_gain_list):

    for gain_ind, gain in enumerate(gain_list):

        n_tint, n_fint, n_ant, n_dir, n_corr = gain.shape

        inverse_gain = inverse_gain_list[gain_ind]

        for t in range(n_tint):
            for f in range(n_fint):
                for a in range(n_ant):
                    for d in range(n_dir):

                        g00 = gain[t, f, a, d, 0]
                        g01 = gain[t, f, a, d, 1]
                        g10 = gain[t, f, a, d, 2]
                        g11 = gain[t, f, a, d, 3]

                        det = (g00*g11 - g01*g10)

                        if np.abs(det) < 1e-6:
                            inverse_gain[t, f, a, d, 0] = 0
                            inverse_gain[t, f, a, d, 1] = 0
                            inverse_gain[t, f, a, d, 2] = 0
                            inverse_gain[t, f, a, d, 3] = 0
                        else:
                            inverse_gain[t, f, a, d, 0] = g11/det
                            inverse_gain[t, f, a, d, 1] = -g01/det
                            inverse_gain[t, f, a, d, 2] = -g10/det
                            inverse_gain[t, f, a, d, 3] = g00/det


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def jhj_and_jhr_full(model, gain_list, inverse_gain_list, residual, a1, a2,
                     weights, t_map_arr, f_map_arr, d_map_arr, active_term):

    n_rows, n_chan, n_dir, n_corr = model.shape
    n_out_dir = gain_list[active_term].shape[3]

    jhr = np.zeros_like(gain_list[active_term])
    jhj = np.zeros_like(gain_list[active_term])

    tmp_jh_p = np.zeros((n_out_dir, n_corr), dtype=jhj.dtype)
    tmp_jh_q = np.zeros((n_out_dir, n_corr), dtype=jhj.dtype)

    n_gains = len(gain_list)

    for row in prange(n_rows):

        a1_m, a2_m = a1[row], a2[row]

        for f in range(n_chan):

            r = residual[row, f]
            w = weights[row, f]  # Consider a map?

            w0 = w[0]
            # w1 = w[1]  # We do not use the off diagonal weights.
            # w2 = w[2]  # We do not use the off diagonal weights.
            w3 = w[3]

            tmp_jh_p[:, :] = 0
            tmp_jh_q[:, :] = 0

            for d in range(n_dir):

                out_d = d_map_arr[active_term, d]

                r00 = w0*r[0]
                r01 = w0*r[1]
                r10 = w3*r[2]
                r11 = w3*r[3]

                rh00 = r00.conjugate()
                rh01 = r10.conjugate()
                rh10 = r01.conjugate()
                rh11 = r11.conjugate()

                m = model[row, f, d]

                m00 = m[0]
                m01 = m[1]
                m10 = m[2]
                m11 = m[3]

                mh00 = m[0].conjugate()
                mh01 = m[2].conjugate()
                mh10 = m[1].conjugate()
                mh11 = m[3].conjugate()

                for g in range(n_gains - 1, -1, -1):

                    d_m = d_map_arr[g, d]  # Broadcast dir.
                    t_m = t_map_arr[row, g]
                    f_m = f_map_arr[f, g]
                    gb = gain_list[g][t_m, f_m, a2_m, d_m]

                    g00 = gb[0]
                    g01 = gb[1]
                    g10 = gb[2]
                    g11 = gb[3]

                    jh00 = (g00*mh00 + g01*mh10)
                    jh01 = (g00*mh01 + g01*mh11)
                    jh10 = (g10*mh00 + g11*mh10)
                    jh11 = (g10*mh01 + g11*mh11)

                    mh00 = jh00
                    mh01 = jh01
                    mh10 = jh10
                    mh11 = jh11

                for g in range(n_gains - 1, active_term, -1):

                    d_m = d_map_arr[g, d]  # Broadcast dir.
                    t_m = t_map_arr[row, g]
                    f_m = f_map_arr[f, g]
                    ga = gain_list[g][t_m, f_m, a1_m, d_m]

                    gh00 = ga[0].conjugate()
                    gh01 = ga[2].conjugate()
                    gh10 = ga[1].conjugate()
                    gh11 = ga[3].conjugate()

                    jh00 = (mh00*gh00 + mh01*gh10)
                    jh01 = (mh00*gh01 + mh01*gh11)
                    jh10 = (mh10*gh00 + mh11*gh10)
                    jh11 = (mh10*gh01 + mh11*gh11)

                    mh00 = jh00
                    mh01 = jh01
                    mh10 = jh10
                    mh11 = jh11

                for g in range(active_term):

                    d_m = d_map_arr[g, d]  # Broadcast dir.
                    t_m = t_map_arr[row, g]
                    f_m = f_map_arr[f, g]
                    gai = inverse_gain_list[g][t_m, f_m, a1_m, d_m]

                    ginv00 = gai[0]
                    ginv01 = gai[1]
                    ginv10 = gai[2]
                    ginv11 = gai[3]

                    jhr00 = (ginv00*r00 + ginv01*r10)
                    jhr01 = (ginv00*r01 + ginv01*r11)
                    jhr10 = (ginv10*r00 + ginv11*r10)
                    jhr11 = (ginv10*r01 + ginv11*r11)

                    r00 = jhr00
                    r01 = jhr01
                    r10 = jhr10
                    r11 = jhr11

                t_m = t_map_arr[row, active_term]
                f_m = f_map_arr[f, active_term]

                jhr[t_m, f_m, a1_m, out_d, 0] += (r00*jh00 + r01*jh10)
                jhr[t_m, f_m, a1_m, out_d, 1] += (r00*jh01 + r01*jh11)
                jhr[t_m, f_m, a1_m, out_d, 2] += (r10*jh00 + r11*jh10)
                jhr[t_m, f_m, a1_m, out_d, 3] += (r10*jh01 + r11*jh11)

                tmp_jh_p[out_d, 0] += jh00
                tmp_jh_p[out_d, 1] += jh01
                tmp_jh_p[out_d, 2] += jh10
                tmp_jh_p[out_d, 3] += jh11

                for g in range(n_gains-1, -1, -1):

                    d_m = d_map_arr[g, d]  # Broadcast dir.
                    t_m = t_map_arr[row, g]
                    f_m = f_map_arr[f, g]
                    ga = gain_list[g][t_m, f_m, a1_m, d_m]

                    g00 = ga[0]
                    g01 = ga[1]
                    g10 = ga[2]
                    g11 = ga[3]

                    jh00 = (g00*m00 + g01*m10)
                    jh01 = (g00*m01 + g01*m11)
                    jh10 = (g10*m00 + g11*m10)
                    jh11 = (g10*m01 + g11*m11)

                    m00 = jh00
                    m01 = jh01
                    m10 = jh10
                    m11 = jh11

                for g in range(n_gains - 1, active_term, -1):

                    d_m = d_map_arr[g, d]  # Broadcast dir.
                    t_m = t_map_arr[row, g]
                    f_m = f_map_arr[f, g]
                    gb = gain_list[g][t_m, f_m, a2_m, d_m]

                    gh00 = gb[0].conjugate()
                    gh01 = gb[2].conjugate()
                    gh10 = gb[1].conjugate()
                    gh11 = gb[3].conjugate()

                    jh00 = (m00*gh00 + m01*gh10)
                    jh01 = (m00*gh01 + m01*gh11)
                    jh10 = (m10*gh00 + m11*gh10)
                    jh11 = (m10*gh01 + m11*gh11)

                    m00 = jh00
                    m01 = jh01
                    m10 = jh10
                    m11 = jh11

                for g in range(active_term):

                    d_m = d_map_arr[g, d]  # Broadcast dir.
                    t_m = t_map_arr[row, g]
                    f_m = f_map_arr[f, g]
                    gbi = inverse_gain_list[g][t_m, f_m, a2_m, d_m]

                    ginv00 = gbi[0]
                    ginv01 = gbi[1]
                    ginv10 = gbi[2]
                    ginv11 = gbi[3]

                    jhr00 = (ginv00*rh00 + ginv01*rh10)
                    jhr01 = (ginv00*rh01 + ginv01*rh11)
                    jhr10 = (ginv10*rh00 + ginv11*rh10)
                    jhr11 = (ginv10*rh01 + ginv11*rh11)

                    rh00 = jhr00
                    rh01 = jhr01
                    rh10 = jhr10
                    rh11 = jhr11

                t_m = t_map_arr[row, active_term]
                f_m = f_map_arr[f, active_term]

                jhr[t_m, f_m, a2_m, out_d, 0] += (rh00*jh00 + rh01*jh10)
                jhr[t_m, f_m, a2_m, out_d, 1] += (rh00*jh01 + rh01*jh11)
                jhr[t_m, f_m, a2_m, out_d, 2] += (rh10*jh00 + rh11*jh10)
                jhr[t_m, f_m, a2_m, out_d, 3] += (rh10*jh01 + rh11*jh11)

                tmp_jh_q[out_d, 0] += jh00
                tmp_jh_q[out_d, 1] += jh01
                tmp_jh_q[out_d, 2] += jh10
                tmp_jh_q[out_d, 3] += jh11

            for d in range(n_out_dir):

                jh00 = tmp_jh_p[d, 0]
                jh01 = tmp_jh_p[d, 1]
                jh10 = tmp_jh_p[d, 2]
                jh11 = tmp_jh_p[d, 3]

                j00 = jh00.conjugate()
                j01 = jh10.conjugate()
                j10 = jh01.conjugate()
                j11 = jh11.conjugate()

                jhj[t_m, f_m, a1_m, d, 0] += (j00*w0*jh00 + j01*w3*jh10)
                jhj[t_m, f_m, a1_m, d, 1] += (j00*w0*jh01 + j01*w3*jh11)
                jhj[t_m, f_m, a1_m, d, 2] += (j10*w0*jh00 + j11*w3*jh10)
                jhj[t_m, f_m, a1_m, d, 3] += (j10*w0*jh01 + j11*w3*jh11)

                jh00 = tmp_jh_q[d, 0]
                jh01 = tmp_jh_q[d, 1]
                jh10 = tmp_jh_q[d, 2]
                jh11 = tmp_jh_q[d, 3]

                j00 = jh00.conjugate()
                j01 = jh10.conjugate()
                j10 = jh01.conjugate()
                j11 = jh11.conjugate()

                jhj[t_m, f_m, a2_m, d, 0] += (j00*w0*jh00 + j01*w3*jh10)
                jhj[t_m, f_m, a2_m, d, 1] += (j00*w0*jh01 + j01*w3*jh11)
                jhj[t_m, f_m, a2_m, d, 2] += (j10*w0*jh00 + j11*w3*jh10)
                jhj[t_m, f_m, a2_m, d, 3] += (j10*w0*jh01 + j11*w3*jh11)

    return jhj, jhr


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def update_full(jhj, jhr):

    n_tint, n_fint, n_ant, n_dir, n_corr = jhj.shape

    update = np.empty_like(jhr)

    for t in range(n_tint):
        for f in range(n_fint):
            for a in range(n_ant):
                for d in range(n_dir):

                    jhj00 = jhj[t, f, a, d, 0]
                    jhj01 = jhj[t, f, a, d, 1]
                    jhj10 = jhj[t, f, a, d, 2]
                    jhj11 = jhj[t, f, a, d, 3]

                    det = (jhj00*jhj11 - jhj01*jhj10)

                    if det.real < 1e-6:
                        jhjinv00 = 0
                        jhjinv01 = 0
                        jhjinv10 = 0
                        jhjinv11 = 0
                    else:
                        jhjinv00 = jhj11/det
                        jhjinv01 = -jhj01/det
                        jhjinv10 = -jhj10/det
                        jhjinv11 = jhj00/det

                    jhr00 = jhr[t, f, a, d, 0]
                    jhr01 = jhr[t, f, a, d, 1]
                    jhr10 = jhr[t, f, a, d, 2]
                    jhr11 = jhr[t, f, a, d, 3]

                    update[t, f, a, d, 0] = (jhr00*jhjinv00 + jhr01*jhjinv10)
                    update[t, f, a, d, 1] = (jhr00*jhjinv01 + jhr01*jhjinv11)
                    update[t, f, a, d, 2] = (jhr10*jhjinv00 + jhr11*jhjinv10)
                    update[t, f, a, d, 3] = (jhr10*jhjinv01 + jhr11*jhjinv11)

    return update


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def residual_full(data, model, gain_list, a1, a2, t_map_arr, f_map_arr,
                  d_map_arr):

    residual = data.copy()

    n_rows, n_chan, n_dir, _ = model.shape
    n_gains = len(gain_list)

    for row in prange(n_rows):

        a1_m, a2_m = a1[row], a2[row]

        for f in range(n_chan):
            for d in range(n_dir):

                m00 = model[row, f, d, 0]
                m01 = model[row, f, d, 1]
                m10 = model[row, f, d, 2]
                m11 = model[row, f, d, 3]

                for g in range(n_gains-1, -1, -1):

                    d_m = d_map_arr[g, d]  # Broadcast dir.
                    t_m = t_map_arr[row, g]
                    f_m = f_map_arr[f, g]

                    gain = gain_list[g]

                    g00 = gain[t_m, f_m, a1_m, d_m, 0]
                    g01 = gain[t_m, f_m, a1_m, d_m, 1]
                    g10 = gain[t_m, f_m, a1_m, d_m, 2]
                    g11 = gain[t_m, f_m, a1_m, d_m, 3]

                    gh00 = gain[t_m, f_m, a2_m, d_m, 0].conjugate()
                    gh01 = gain[t_m, f_m, a2_m, d_m, 2].conjugate()
                    gh10 = gain[t_m, f_m, a2_m, d_m, 1].conjugate()
                    gh11 = gain[t_m, f_m, a2_m, d_m, 3].conjugate()

                    gm00 = (g00*m00 + g01*m10)
                    gm01 = (g00*m01 + g01*m11)
                    gm10 = (g10*m00 + g11*m10)
                    gm11 = (g10*m01 + g11*m11)

                    r00 = (gm00*gh00 + gm01*gh10)
                    r01 = (gm00*gh01 + gm01*gh11)
                    r10 = (gm10*gh00 + gm11*gh10)
                    r11 = (gm10*gh01 + gm11*gh11)

                    m00 = r00
                    m01 = r01
                    m10 = r10
                    m11 = r11

                residual[row, f, 0] -= r00
                residual[row, f, 1] -= r01
                residual[row, f, 2] -= r10
                residual[row, f, 3] -= r11

    return residual