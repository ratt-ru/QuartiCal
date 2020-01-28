# -*- coding: utf-8 -*-
from collections import namedtuple
from numba import jit, types, typed
from cubicalv2.kernels.gjones_chain import invert_gains
from cubicalv2.kernels.gjones_chain import (residual_full,
                                            compute_convergence)
import numpy as np


conv_info = namedtuple("conv_info", "conv_iters dummy")


@jit(nopython=True, fastmath=True, parallel=False, cache=False, nogil=True)
def chain_solver(model, data, a1, a2, weights, t_map_arr, f_map_arr,
                 d_map_arr, compute_jhj_and_jhr, compute_update, *gain_list):

    gain_list = [g for g in gain_list[::2]]
    gain_flag_list = [gf for gf in gain_list[1::2]]
    inverse_gain_list = [np.empty_like(g) for g in gain_list]

    for gain_ind in range(len(gain_list)):

        n_tint, t_fint, n_ant, n_dir, n_corr = gain_list[gain_ind].shape

        invert_gains(gain_list, inverse_gain_list)

        dd_term = n_dir > 1

        last_gain = gain_list[gain_ind].copy()

        cnv_perc = 0.

        for i in range(20):

            if dd_term:
                residual = residual_full(data, model, gain_list, a1, a2,
                                         t_map_arr, f_map_arr, d_map_arr)
            else:
                residual = data

            jhj, jhr = compute_jhj_and_jhr(model,
                                           gain_list,
                                           inverse_gain_list,
                                           residual,
                                           a1,
                                           a2,
                                           weights,
                                           t_map_arr,
                                           f_map_arr,
                                           d_map_arr,
                                           gain_ind)

            update = compute_update(jhj, jhr)

            if dd_term:
                gain_list[gain_ind][:] = gain_list[gain_ind][:] + update/2
            elif i % 2 == 0:
                gain_list[gain_ind][:] = update
            else:
                gain_list[gain_ind][:] = (gain_list[gain_ind] + update)/2

            # Check for gain convergence. TODO: This can be affected by the
            # weights. Currently unsure how or why, but using unity weights
            # leads to monotonic convergence in all solution intervals.

            cnv_perc = compute_convergence(gain_list[gain_ind][:], last_gain)

            last_gain[:] = gain_list[gain_ind][:]

            if cnv_perc > 0.99:
                break

    return gain_list, conv_info(i, 10)
