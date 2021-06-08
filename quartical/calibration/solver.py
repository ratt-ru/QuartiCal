# -*- coding: utf-8 -*-
import numpy as np
from quartical.gains import term_solvers
import gc


def solver_wrapper(model, data, a1, a2, weights, t_map_arr, f_map_arr,
                   d_map_arr, corr_mode, term_spec_list, *args, **kwargs):

    # This is rudimentary - it practice we may have more initialisation code
    # here for setting up parameters etc. TODO: Init actually needs to depend
    # on the term type. We also probably need to consider how to parse
    # **kwargs into the solver for terms requiring ancilliary info.

    gain_tup = ()
    additional_args = []
    results_dict = {}

    for term_ind, term_spec in enumerate(term_spec_list):
        gain = np.zeros(term_spec.shape, dtype=np.complex128)
        term_name = term_spec.name

        # Check for initialisation data.
        if f"{term_name}_initial_gain" in kwargs:
            gain[:] = kwargs[f"{term_name}_initial_gain"]
        else:
            gain[..., (0, -1)] = 1  # Set first and last correlations to 1.

        gain_tup += (gain,)

        additional_args.append(dict())

        # These are cludges for the BDA case. Might be possible to make this
        # a litte more elegant.
        if "row_map" in kwargs:
            additional_args[term_ind]["row_map"] = kwargs["row_map"]

        if "row_weights" in kwargs:
            additional_args[term_ind]["row_weights"] = kwargs["row_weights"]

        # If the pshape (parameter shape) is defined, we want to initialise it.
        if term_spec.pshape:
            additional_args[term_ind]["params"] = \
                np.zeros(term_spec.pshape, dtype=gain.real.dtype)

            results_dict[term_spec.name + "-param"] = \
                additional_args[term_ind]["params"]

        # Each solver may have additional args living in the kwargs dict. This
        # will associate them with the relevant term.
        for arg in term_spec.args:
            additional_args[term_ind][arg] = kwargs[arg]

        results_dict[term_spec.name + "-gain"] = gain
        results_dict[term_spec.name + "-conviter"] = 0
        results_dict[term_spec.name + "-convperc"] = 0

    flag_tup = tuple([np.zeros_like(g, dtype=np.uint8) for g in gain_tup])
    inverse_gain_tup = tuple([np.empty_like(g) for g in gain_tup])

    for gain_ind, term_spec in enumerate(term_spec_list):

        solver = term_solvers[term_spec.type]

        info_tup = \
            solver(model, data, a1, a2, weights, t_map_arr, f_map_arr,
                   d_map_arr, corr_mode, gain_ind, inverse_gain_tup,
                   gain_tup, flag_tup, **additional_args[gain_ind])

        results_dict[term_spec.name + "-conviter"] += \
            np.atleast_2d(info_tup.conv_iters)
        results_dict[term_spec.name + "-convperc"] += \
            np.atleast_2d(info_tup.conv_perc)

    gc.collect()

    return results_dict
