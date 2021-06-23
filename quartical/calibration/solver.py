# -*- coding: utf-8 -*-
import numpy as np
from quartical.gains import term_types
import gc


def solver_wrapper(**kwargs):

    gain_tup = ()
    param_tup = ()
    flag_tup = ()
    results_dict = {}

    term_spec_list = kwargs["term_spec_list"]

    for term_spec in term_spec_list:
        gain = np.zeros(term_spec.shape, dtype=np.complex128)
        term_name = term_spec.name

        # Check for initialisation data.
        if f"{term_name}_initial_gain" in kwargs:
            gain[:] = kwargs[f"{term_name}_initial_gain"]
        else:
            gain[..., (0, -1)] = 1  # Set first and last correlations to 1.

        gain_tup += (gain,)
        flag_tup += (np.zeros_like(gain, dtype=np.uint8),)

        # If the pshape (parameter shape) is defined, we want to initialise it.
        # Otherwise, create a dummy array to keep the tuple homogeneous.
        if term_spec.pshape:
            param = np.zeros(term_spec.pshape, dtype=gain.real.dtype)
            param_tup += (param,)
            results_dict[term_spec.name + "-param"] = param
        else:
            param_tup += (np.empty((0,)*6, dtype=gain.real.dtype),)

        results_dict[term_spec.name + "-gain"] = gain
        results_dict[term_spec.name + "-conviter"] = 0
        results_dict[term_spec.name + "-convperc"] = 0

    kwargs["gains"] = gain_tup
    kwargs["flags"] = flag_tup
    kwargs["inverse_gains"] = tuple([np.empty_like(g) for g in gain_tup])
    kwargs["params"] = param_tup

    for gain_ind, term_spec in enumerate(term_spec_list):

        term_type = term_types[term_spec.type]

        solver = term_type.solver
        base_args = term_type.base_args
        term_args = term_type.term_args

        _base_args = base_args(**{k: kwargs[k] for k in base_args._fields})
        _term_args = term_args(**{k: kwargs[k] for k in term_args._fields})

        info_tup = solver(_base_args,
                          _term_args,
                          gain_ind,
                          kwargs["corr_mode"])

        results_dict[term_spec.name + "-conviter"] += \
            np.atleast_2d(info_tup.conv_iters)
        results_dict[term_spec.name + "-convperc"] += \
            np.atleast_2d(info_tup.conv_perc)

    gc.collect()

    return results_dict
