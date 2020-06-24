# -*- coding: utf-8 -*-
import numpy as np
from numba import jit, prange, literally
from numba.extending import overload
from quartical.kernels.generics import (invert_gains,
                                        compute_residual,
                                        compute_convergence)
from quartical.kernels.complex import compute_jhj_jhr
from collections import namedtuple


# This can be done without a named tuple now. TODO: Add unpacking to
# constructor.
stat_fields = {"conv_iters": np.int64,
               "conv_perc": np.float64}

term_conv_info = namedtuple("term_conv_info", " ".join(stat_fields.keys()))


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def kalman_solver(model, data, a1, a2, weights, t_map_arr, f_map_arr,
                  d_map_arr, corr_mode, active_term, inverse_gain_list,
                  gains, flags):

    n_tint, n_fint, n_ant, n_dir, n_corr = gains[active_term].shape

    # invert_gains(gains, inverse_gain_list, literally(corr_mode))

    dd_term = n_dir > 1

    last_gain = gains[active_term].copy()

    cnv_perc = 0.

    jhj = np.empty((1, n_fint, n_ant, n_dir, n_corr), dtype=last_gain.dtype)
    jhr = np.empty((1, n_fint, n_ant, n_dir, n_corr), dtype=last_gain.dtype)
    g_update = \
        np.empty((1, n_fint, n_ant, n_dir, n_corr), dtype=last_gain.dtype)
    p_update = \
        np.empty((1, n_fint, n_ant, n_dir, n_corr), dtype=last_gain.real.dtype)
    Q = 0.01*np.ones_like(gains[active_term], dtype=last_gain.real.dtype)
    P = np.ones_like(gains[active_term], dtype=last_gain.real.dtype)
    P = P + Q

    for i in range(n_tint):

        sel = np.where(t_map_arr[:, active_term] == i)[0]

        residual = compute_residual(data[sel, :, :],
                                    model[sel, :, :, :],
                                    [gains[active_term][i:i+1, :, :, :, :]],
                                    a1[sel],
                                    a2[sel],
                                    np.zeros_like(t_map_arr[sel]),
                                    f_map_arr,
                                    d_map_arr,
                                    literally(corr_mode))

        compute_jhj_jhr(jhj,
                        jhr,
                        model[sel, :, :, :],
                        [gains[active_term][i:i+1, :, :, :, :]],
                        inverse_gain_list,
                        residual,
                        a1[sel],
                        a2[sel],
                        weights[sel, :, :],
                        np.zeros_like(t_map_arr[sel]),
                        f_map_arr,
                        d_map_arr,
                        active_term,
                        literally(corr_mode))

        # print(P[:, 16, 0, 0, 0])

        compute_update(g_update,
                       p_update,
                       jhj,
                       jhr,
                       P[i:i+1, :, :, :, :])

        gains[active_term][i:i+1, :, :, :, :] += 0.5*g_update
        P[i:i+1, :, :, :, :] -= 0.5*p_update

        if i < n_tint - 1:
            gains[active_term][i+1, :, :, :, :] = \
                gains[active_term][i, :, :, :, :]
            P[i+1, :, :, :, :] = P[i, :, :, :, :] + Q[0]

        # Check for gain convergence. TODO: This can be affected by the
        # weights. Currently unsure how or why, but using unity weights
        # leads to monotonic convergence in all solution intervals.

        cnv_perc = 0

        last_gain[:] = gains[active_term][:]

        if cnv_perc > 0.99:
            break

    return term_conv_info(i, cnv_perc)


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def compute_update(g_update, p_update, jhj, jhr, P):

    n_tint, n_fint, n_ant, n_dir, n_corr = jhj.shape

    for t in range(n_tint):
        for f in range(n_fint):
            for a in range(n_ant):
                for d in range(n_dir):

                    jhj00 = jhj[t, f, a, d, 0]
                    jhj11 = jhj[t, f, a, d, 1]

                    P00 = P[t, f, a, d, 0]
                    P11 = P[t, f, a, d, 1]

                    det = (jhj00*jhj11)

                    if det.real < 1e-6:
                        jhjinv00 = 0
                        jhjinv11 = 0
                    else:

                        P00inv = 1/P00
                        P11inv = 1/P11

                        jhjinv00 = 1/(P00inv + jhj00)
                        jhjinv11 = 1/(P11inv + jhj11)

                    jhr00 = jhr[t, f, a, d, 0]
                    jhr11 = jhr[t, f, a, d, 1]

                    # Component of Woodbury Matrix Identity.
                    wb00 = P00*(1-jhj00*jhjinv00)
                    wb11 = P11*(1-jhj11*jhjinv11)

                    g_update[t, f, a, d, 0] = wb00*jhr00
                    g_update[t, f, a, d, 1] = wb11*jhr11

                    p_update[t, f, a, d, 0] = (wb00*jhj00*P00).real
                    p_update[t, f, a, d, 1] = (wb11*jhj11*P11).real

    return
