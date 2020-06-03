# -*- coding: utf-8 -*-
import numpy as np
from cubicalv2.calibration.gain_types import term_solvers
import gc


def solver_wrapper(model, data, a1, a2, weights, t_map_arr, f_map_arr,
                   d_map_arr, corr_mode, term_spec_list):

    # This is rudimentary - it practice we may have more initialisation code
    # here for setting up parameters etc. TODO: Init actually needs to depend
    # on the term type. We also probably need to consider how to parse
    # **kwargs into the solver for terms requiring ancilliary info.

    gain_list = []
    for term_spec in term_spec_list:
        gain = np.zeros(term_spec.shape, dtype=np.complex128)
        gain[..., (0, -1)] = 1  # Set first and last correlations to 1.
        gain_list.append(gain)

    flag_list = [np.zeros_like(g, dtype=np.uint8) for g in gain_list]
    inverse_gain_list = [np.empty_like(g) for g in gain_list]

    info_dict = {}

    for gain_ind, term_spec in enumerate(term_spec_list):

        solver = term_solvers[term_spec.type]

        info_dict[gain_ind] = \
            solver(model, data, a1, a2, weights, t_map_arr, f_map_arr,
                   d_map_arr, corr_mode, gain_ind, inverse_gain_list,
                   gain_list, flag_list)

    gc.collect()

    return gain_list, info_dict
