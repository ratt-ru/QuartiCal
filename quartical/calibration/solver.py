# -*- coding: utf-8 -*-
import gc
import numpy as np
from numba import set_num_threads
from collections import namedtuple
from itertools import cycle
from quartical.gains import TERM_TYPES
from quartical.weights.robust import robust_reweighting
from quartical.gains.general.flagging import init_gain_flags, init_param_flags


meta_args_nt = namedtuple(
    "meta_args_nt", (
        "iters",
        "active_term",
        "stop_frac",
        "stop_crit",
        "threads",
        "dd_term",
        "solve_per",
        "robust"
        )
    )


def solver_wrapper(
    term_spec_list,
    solver_opts,
    chain_opts,
    block_id_arr,
    aux_block_info,
    **kwargs
):
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

    block_id = tuple(block_id_arr.squeeze())

    set_num_threads(solver_opts.threads)
    ref_ant = solver_opts.reference_antenna

    gain_tup = ()
    param_tup = ()
    gain_flags_tup = ()
    param_flags_tup = ()
    results_dict = {}

    for term_ind, term_spec in enumerate(term_spec_list):

        (term_name, term_type, term_shape, term_pshape) = term_spec

        term_type_cls = TERM_TYPES[term_type]
        term_opts = getattr(chain_opts, term_name)

        gain = np.zeros(term_shape, dtype=np.complex128)
        param = np.zeros(term_pshape, dtype=gain.real.dtype)

        # Perform terms specific setup e.g. init gains and params.
        term_type_cls.init_term(
            gain, param, term_ind, term_spec, term_opts, ref_ant, **kwargs
        )

        # Init gain flags by looking for intervals with no data.
        gain_flags = init_gain_flags(term_shape, term_ind, **kwargs)
        param_flags = init_param_flags(term_pshape, term_ind, **kwargs)

        gain_tup += (gain,)
        gain_flags_tup += (gain_flags,)
        param_tup += (param,)
        param_flags_tup += (param_flags,)

        results_dict[f"{term_name}-gain"] = gain
        results_dict[f"{term_name}-gain_flags"] = gain_flags
        results_dict[f"{term_name}-param"] = param
        results_dict[f"{term_name}-param_flags"] = param_flags
        results_dict[f"{term_name}-conviter"] = np.atleast_2d(0)   # int
        results_dict[f"{term_name}-convperc"] = np.atleast_2d(0.)  # float

    kwargs["gains"] = gain_tup
    kwargs["gain_flags"] = gain_flags_tup
    kwargs["inverse_gains"] = tuple([np.empty_like(g) for g in gain_tup])
    kwargs["params"] = param_tup
    kwargs["param_flags"] = param_flags_tup

    terms = solver_opts.terms
    iter_recipe = solver_opts.iter_recipe
    robust = solver_opts.robust

    # NOTE: This is a necessary evil. We do not want to modify the inputs
    # and copying is the best way to ensure that that cannot happen.
    kwargs["weights"] = kwargs["weights"].copy()
    results_dict["weights"] = kwargs["weights"]
    kwargs["flags"] = kwargs["flags"].copy()
    results_dict["flags"] = kwargs["flags"]

    if solver_opts.robust:
        final_epoch = len(iter_recipe) // len(terms)
        etas = np.zeros_like(kwargs["weights"][..., 0])
        icovariance = np.zeros(kwargs["corr_mode"], np.float64)
        dof = 5

    for ind, (term, iters) in enumerate(zip(cycle(terms), iter_recipe)):

        active_term = terms.index(term)
        term_name, term_type, _, _ = term_spec_list[active_term]

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
                                 solver_opts.convergence_fraction,
                                 solver_opts.convergence_criteria,
                                 solver_opts.threads,
                                 term_opts.direction_dependent,
                                 term_opts.solve_per,
                                 robust)

        if iters != 0:
            jhj, info_tup = solver(base_args,
                                   term_args,
                                   meta_args,
                                   kwargs["corr_mode"])
        else:
            # TODO: Actually compute it in this special case?
            jhj = np.zeros_like(results_dict[f"{term_name}-gain"])
            info_tup = (0, 0)

        # If reweighting is enabled, do it when the epoch changes, except
        # for the final epoch - we don't reweight if we won't solve again.
        if solver_opts.robust:
            current_epoch = ind // len(terms)
            next_epoch = (ind + 1) // len(terms)

            if current_epoch != next_epoch and next_epoch != final_epoch:

                dof = robust_reweighting(
                    base_args,
                    meta_args,
                    etas,
                    icovariance,
                    dof,
                    kwargs["corr_mode"])

        # TODO: Ugly hack for larger jhj matrices. Refine.
        if jhj.ndim == 6:
            jhj = jhj[:, :, :, :, range(jhj.shape[-2]), range(jhj.shape[-1])]

        results_dict[f"{term_name}-conviter"] += np.atleast_2d(info_tup[0])
        results_dict[f"{term_name}-convperc"] = np.atleast_2d(info_tup[1])
        results_dict[f"{term_name}-jhj"] = jhj

    gc.collect()

    return results_dict
