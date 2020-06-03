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
    additional_args = []

    for term_spec in term_spec_list:
        gain = np.zeros(term_spec.shape, dtype=np.complex128)
        gain[..., (0, -1)] = 1  # Set first and last correlations to 1.
        gain_list.append(gain)

        # This is a nasty interim hack. TODO: I need to settle on an interface
        # for customising each gain term. This is particularly important for
        # terms which which have differing parameterisations/resolutions or
        # require extra info. This should likely be set up in gain types -
        # each solver should implement and return a dictionary of additional
        # arguments.

        if term_spec.type == "phase":
            additional_args.append(
                {"params": np.zeros_like(gain[..., None, :, :],
                                         dtype=gain.real.dtype)})
        else:
            additional_args.append({})

    flag_list = [np.zeros_like(g, dtype=np.uint8) for g in gain_list]
    inverse_gain_list = [np.empty_like(g) for g in gain_list]

    info_dict = {}

    for gain_ind, term_spec in enumerate(term_spec_list):

        solver = term_solvers[term_spec.type]

        info_dict[gain_ind] = \
            solver(model, data, a1, a2, weights, t_map_arr, f_map_arr,
                   d_map_arr, corr_mode, gain_ind, inverse_gain_list,
                   gain_list, flag_list, **additional_args[gain_ind])

    gc.collect()

    return gain_list, info_dict
