# -*- coding: utf-8 -*-
import numpy as np
from cubicalv2.calibration.gain_types import term_solvers
import gc


def solver_wrapper(model, data, a1, a2, weights, t_map_arr, f_map_arr,
                   d_map_arr, corr_mode, *input_list):

    gain_list = input_list[0].gains
    inverse_gain_list = [np.empty_like(g) for g in gain_list]

    info_dict = {}

    for gain_ind, term_tuple in enumerate(input_list):

        solver = term_solvers[type(term_tuple).__name__]

        info_dict[gain_ind] = \
            solver(model, data, a1, a2, weights, t_map_arr, f_map_arr,
                   d_map_arr, corr_mode, gain_ind, inverse_gain_list,
                   **term_tuple._asdict())

    gc.collect()

    return gain_list, info_dict
