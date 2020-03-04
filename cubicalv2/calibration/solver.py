# -*- coding: utf-8 -*-
from collections import namedtuple
from numba import jit, literally
from cubicalv2.kernels.gjones_chain import (invert_gains,
                                            compute_update,
                                            compute_jhj_jhr,
                                            compute_residual,
                                            compute_convergence)
import numpy as np


stat_fields = {"conv_iters": np.int64,
               "conv_perc": np.float64}

term_conv_info = namedtuple("term_conv_info", " ".join(stat_fields.keys()))


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def chain_solver(model, data, a1, a2, weights, t_map_arr, f_map_arr,
                 d_map_arr, mode, *input_list):

    gain_list = [g for g in input_list[::2]]
    gain_flag_list = [gf for gf in input_list[1::2]]
    inverse_gain_list = [np.empty_like(g) for g in gain_list]

    info_dict = dict()

    for gain_ind in range(len(gain_list)):

        n_tint, t_fint, n_ant, n_dir, n_corr = gain_list[gain_ind].shape

        invert_gains(gain_list, inverse_gain_list, literally(mode))

        dd_term = n_dir > 1

        last_gain = gain_list[gain_ind].copy()

        cnv_perc = 0.

        for i in range(20):

            if dd_term:
                residual = compute_residual(data, model, gain_list, a1, a2,
                                            t_map_arr, f_map_arr, d_map_arr,
                                            literally(mode))
            else:
                residual = data

            jhj, jhr = compute_jhj_jhr(model,
                                       gain_list,
                                       inverse_gain_list,
                                       residual,
                                       a1,
                                       a2,
                                       weights,
                                       t_map_arr,
                                       f_map_arr,
                                       d_map_arr,
                                       gain_ind,
                                       literally(mode))

            update = compute_update(jhj, jhr, literally(mode))

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

        info_dict[gain_ind] = term_conv_info(i, cnv_perc)

    return gain_list, info_dict
