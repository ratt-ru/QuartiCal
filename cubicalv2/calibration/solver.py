# -*- coding: utf-8 -*-
from collections import namedtuple
from numba import jit, literally
from cubicalv2.kernels.kernel_dispatcher import compute_jhj_jhr, compute_update
from cubicalv2.kernels.generics import (invert_gains,
                                        compute_residual,
                                        compute_convergence)
import numpy as np


stat_fields = {"conv_iters": np.int64,
               "conv_perc": np.float64}

term_conv_info = namedtuple("term_conv_info", " ".join(stat_fields.keys()))


def solver_wrapper(model, data, a1, a2, weights, t_map_arr, f_map_arr,
                   d_map_arr, corr_mode, *input_list):

    gain_list = [g.gains for g in input_list]
    gain_flag_list = [g.flags for g in input_list]
    parameter_list = [g.parms for g in input_list]
    inverse_gain_list = [np.empty_like(g) for g in gain_list]

    info_dict = {}

    for gain_ind in range(len(gain_list)):

        term_type = type(input_list[gain_ind]).__name__

        info_dict[gain_ind] = \
            chain_solver(model, data, a1, a2, weights, t_map_arr, f_map_arr,
                         d_map_arr, corr_mode, gain_list, gain_flag_list,
                         inverse_gain_list, gain_ind, parameter_list[gain_ind],
                         term_type)

    return gain_list, info_dict


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def chain_solver(model, data, a1, a2, weights, t_map_arr, f_map_arr,
                 d_map_arr, corr_mode, gain_list, gain_flag_list,
                 inverse_gain_list, gain_ind, params, term_type):

    n_tint, t_fint, n_ant, n_dir, n_corr = gain_list[gain_ind].shape

    invert_gains(gain_list, inverse_gain_list, literally(corr_mode))

    dd_term = n_dir > 1

    last_gain = gain_list[gain_ind].copy()

    cnv_perc = 0.

    for i in range(20):

        if dd_term:
            residual = compute_residual(data, model, gain_list, a1, a2,
                                        t_map_arr, f_map_arr, d_map_arr,
                                        literally(corr_mode))
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
                                   literally(corr_mode),
                                   literally(term_type))

        update = compute_update(jhj,
                                jhr,
                                literally(corr_mode),
                                literally(term_type))

        # TODO: Make this bit less cludgy.

        if params is not None:
            phases = np.angle(gain_list[gain_ind])
            phases += update/2
            gain_list[gain_ind][:] = np.exp(1j*phases)
        elif dd_term:
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

    return term_conv_info(i, cnv_perc)
