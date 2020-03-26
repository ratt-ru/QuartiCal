# -*- coding: utf-8 -*-
import numpy as np
from cubicalv2.calibration.gain_types import term_solvers


def solver_wrapper(model, data, a1, a2, weights, t_map_arr, f_map_arr,
                   d_map_arr, corr_mode, *input_list):

    gain_list = [g.gains for g in input_list]
    gain_flag_list = [g.flags for g in input_list]
    parameter_list = [g.parms for g in input_list]
    inverse_gain_list = [np.empty_like(g) for g in gain_list]

    info_dict = {}

    for gain_ind in range(len(gain_list)):

        solver = term_solvers[type(input_list[gain_ind]).__name__]

        info_dict[gain_ind] = \
            solver(model, data, a1, a2, weights, t_map_arr, f_map_arr,
                   d_map_arr, corr_mode, gain_list, gain_flag_list,
                   inverse_gain_list, gain_ind, parameter_list[gain_ind])

    return gain_list, info_dict
