# -*- coding: utf-8 -*-
import gc
import numpy as np
from numba import set_num_threads
from collections import namedtuple
from itertools import cycle
from quartical.gains import TERM_TYPES


meta_args_nt = namedtuple(
    "meta_args_nt", (
        "iters",
        "active_term",
        "is_init",
        "stop_frac",
        "stop_crit",
        "solve_per",
        "robust"
        )
    )


def solver_wrapper(term_spec_list, solver_opts, chain_opts, **kwargs):
    """A Python wrapper for the solvers written in Numba.

    This wrapper facilitates getting values in and out of the Numba code and
    creates a dictionary of results which can be understood by the calling
    Dask code.

    Args:
        **kwargs: A dictionary of keyword arguments. All arguments are given
                  as key: value pairs to handle variable input.

    Returns:
        results_dict: A dictionary containing the results of the solvers.
    """

    set_num_threads(solver_opts.threads)

    gain_tup = ()
    param_tup = ()
    flag_tup = ()
    results_dict = {}
    is_initialised = {}

    for (term_name, term_type, term_shape, term_pshape) in term_spec_list:

        gain = np.zeros(term_shape, dtype=np.complex128)

        # Check for initialisation data. TODO: Parameterised terms?
        if f"{term_name}_initial_gain" in kwargs:
            gain[:] = kwargs[f"{term_name}_initial_gain"]
            is_initialised[term_name] = True
        else:
            gain[..., (0, -1)] = 1  # Set first and last correlations to 1.
            is_initialised[term_name] = False

        flag = np.zeros(term_shape, dtype=np.uint8)
        param = np.zeros(term_pshape, dtype=gain.real.dtype)

        gain_tup += (gain,)
        flag_tup += (flag,)
        param_tup += (param,)

        results_dict[f"{term_name}-gain"] = gain
        results_dict[f"{term_name}-flag"] = flag
        results_dict[f"{term_name}-param"] = param
        results_dict[f"{term_name}-conviter"] = np.atleast_2d(0)   # int
        results_dict[f"{term_name}-convperc"] = np.atleast_2d(0.)  # float

    kwargs["gains"] = gain_tup
    kwargs["gain_flags"] = flag_tup
    kwargs["inverse_gains"] = tuple([np.empty_like(g) for g in gain_tup])
    kwargs["params"] = param_tup

    terms = solver_opts.terms
    iter_recipe = solver_opts.iter_recipe
    robust = solver_opts.robust

    # TODO: Analyse the impact of the following. This is necessary if we want
    # to mutate the weights, as we may end up with an unwritable array.
    kwargs["weights"] = np.require(kwargs["weights"], requirements=['W', 'O'])
    results_dict["weights"] = kwargs["weights"]

    for term, iters in zip(cycle(terms), iter_recipe):

        active_term = terms.index(term)
        term_name, term_type, _, _ = term_spec_list[active_term]

        if iters == 0:
            # TODO: Actually compute it in this special case?
            results_dict[f"{term_name}-jhj"] = \
                np.zeros_like(results_dict[f"{term_name}-gain"])
            continue

        term_type_cls = TERM_TYPES[term_type]

        solver = term_type_cls.solver
        base_args_nt = term_type_cls.base_args
        term_args_nt = term_type_cls.term_args

        base_args = \
            base_args_nt(**{k: kwargs[k] for k in base_args_nt._fields})
        term_args = \
            term_args_nt(**{k: kwargs[k] for k in term_args_nt._fields})

        term_opts = getattr(chain_opts, term)

        meta_args = meta_args_nt(iters,
                                 active_term,
                                 is_initialised[term_name],
                                 solver_opts.convergence_fraction,
                                 solver_opts.convergence_criteria,
                                 term_opts.solve_per,
                                 robust)

        jhj, info_tup = solver(base_args,
                               term_args,
                               meta_args,
                               kwargs["corr_mode"])

        # After a solver is run once, it will have been initialised.
        is_initialised[term_name] = True

        if jhj.ndim == 6:
            jhj = jhj[:, :, :, :, range(jhj.shape[-2]), range(jhj.shape[-1])]

        results_dict[f"{term_name}-conviter"] += \
            np.atleast_2d(info_tup.conv_iters)
        results_dict[f"{term_name}-convperc"] += \
            np.atleast_2d(info_tup.conv_perc)
        results_dict[f"{term_name}-jhj"] = jhj

    gc.collect()

    return results_dict
