# -*- coding: utf-8 -*-
import gc
import numpy as np
from numba import set_num_threads
from collections import namedtuple
from itertools import cycle
from quartical.weights.robust import robust_reweighting
from quartical.gains.general.flagging import init_gain_flags, init_param_flags
from quartical.statistics.stat_kernels import compute_mean_postsolve_chisq
from quartical.statistics.logging import log_chisq


meta_args_nt = namedtuple(
    "meta_args_nt", (
        "iters",
        "active_term",
        "stop_frac",
        "stop_crit",
        "threads",
        "dd_term",
        "pinned_directions",
        "solve_per",
        "robust"
        )
    )


def solver_wrapper(
    term_spec_list,
    solver_opts,
    chain,
    block_id_arr,
    aux_block_info,
    corr_mode,
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
    # NOTE: Currently kwargs, while convenient, may unecessarily maintain
    # references to unused inputs. It may be better unpack them in some way
    # to allow early garbage collection.

    block_id = tuple(block_id_arr.squeeze())

    set_num_threads(solver_opts.threads)
    ref_ant = solver_opts.reference_antenna

    gain_array_tup = ()
    gain_flag_array_tup = ()
    time_bin_tup = ()
    time_map_tup = ()
    freq_map_tup = ()
    dir_map_tup = ()
    param_array_tup = ()
    param_flag_array_tup = ()
    param_time_bin_tup = ()
    param_time_map_tup = ()
    param_freq_map_tup = ()
    results_dict = {}

    for term_ind, term in enumerate(chain):

        term_spec = term_spec_list[term_ind]
        (_, _, term_shape, term_pshape) = term_spec

        gain_array = np.zeros(term_shape, dtype=np.complex128)
        param_array = np.zeros(term_pshape, dtype=gain_array.real.dtype)

        # Perform term specific setup e.g. init gains and params.
        # TODO: Can we streamline these calls?
        term.init_term(
            gain_array,
            param_array,
            term_ind,
            term_spec,
            term,
            ref_ant,
            **kwargs
        )

        time_bins = kwargs.get(f"{term.name}-time-bins")
        time_map = kwargs.get(f"{term.name}-time-map")
        freq_map = kwargs.get(f"{term.name}-freq-map")
        dir_map = kwargs.get(f"{term.name}-dir-map")

        # Init gain flags by looking for intervals with no data.
        gain_flag_array = init_gain_flags(
            term_shape,
            time_map,
            freq_map,
            **kwargs
        )

        param_time_bins = kwargs.get(
            f"{term.name}-param-time-bins",
            np.empty(0, dtype=np.int32)
        )
        param_time_map = kwargs.get(
            f"{term.name}-param-time-map",
            np.empty(0, dtype=np.int32)
        )
        param_freq_map = kwargs.get(
            f"{term.name}-param-freq-map",
            np.empty(0, dtype=np.int32)
        )

        # Init parameter flags by looking for intervals with no data.
        param_flag_array = init_param_flags(
            term_pshape,
            param_time_map,
            param_freq_map,
            **kwargs
        )

        gain_array_tup += (gain_array,)
        gain_flag_array_tup += (gain_flag_array,)
        time_bin_tup += (time_bins,)
        time_map_tup += (time_map,)
        freq_map_tup += (freq_map,)
        dir_map_tup += (dir_map,)
        param_array_tup += (param_array,)
        param_flag_array_tup += (param_flag_array,)
        param_time_bin_tup += (param_time_bins,)
        param_time_map_tup += (param_time_map,)
        param_freq_map_tup += (param_freq_map,)

        results_dict[f"{term.name}-gain"] = gain_array
        results_dict[f"{term.name}-gain_flags"] = gain_flag_array
        results_dict[f"{term.name}-param"] = param_array
        results_dict[f"{term.name}-param_flags"] = param_flag_array
        results_dict[f"{term.name}-conviter"] = np.atleast_2d(0)   # int
        results_dict[f"{term.name}-convperc"] = np.atleast_2d(0.)  # float

    kwargs["gains"] = gain_array_tup
    kwargs["gain_flags"] = gain_flag_array_tup
    kwargs["time_bins"] = time_bin_tup
    kwargs["time_maps"] = time_map_tup
    kwargs["freq_maps"] = freq_map_tup
    kwargs["dir_maps"] = dir_map_tup
    kwargs["params"] = param_array_tup
    kwargs["param_flags"] = param_flag_array_tup
    kwargs["param_time_bins"] = param_time_bin_tup
    kwargs["param_time_maps"] = param_time_map_tup
    kwargs["param_freq_maps"] = param_freq_map_tup

    iter_recipe = solver_opts.iter_recipe

    # NOTE: This is a necessary evil. We do not want to modify the inputs
    # and copying is the best way to ensure that that cannot happen.
    kwargs["WEIGHT"] = kwargs["WEIGHT"].copy()
    results_dict["weights"] = kwargs["WEIGHT"]
    kwargs["FLAG"] = kwargs["FLAG"].copy()
    results_dict["flags"] = kwargs["FLAG"]

    if solver_opts.robust:
        final_epoch = len(iter_recipe) // len(chain)
        etas = np.zeros_like(kwargs["WEIGHT"][..., 0])
        icovariance = np.zeros(corr_mode, np.float64)
        dof = 5

    presolve_chisq = compute_mean_postsolve_chisq(
        kwargs["DATA"],
        kwargs["MODEL_DATA"],
        kwargs["WEIGHT"],
        kwargs["FLAG"],
        kwargs["gains"],
        kwargs["ANTENNA1"],
        kwargs["ANTENNA2"],
        kwargs["time_maps"],
        kwargs["freq_maps"],
        kwargs["dir_maps"],
        kwargs.get("row_map", None),
        kwargs.get("row_weights", None),
        corr_mode
    )

    for ind, (term, iters) in enumerate(zip(cycle(chain), iter_recipe)):

        active_term = chain.index(term)

        base_args = term.base_args(
                **{k: kwargs.get(k, None) for k in term.base_args._fields}
            )
        term_args = term.term_args(
            **{k: kwargs.get(k, None) for k in term.term_args._fields}
        )

        meta_args = meta_args_nt(
            iters,
            active_term,
            solver_opts.convergence_fraction,
            solver_opts.convergence_criteria,
            solver_opts.threads,
            term.direction_dependent,
            term.pinned_directions,
            term.solve_per,
            solver_opts.robust
        )

        if iters != 0:
            jhj, info_tup = term.solver(
                base_args,
                term_args,
                meta_args,
                corr_mode
            )
        else:
            # TODO: Actually compute it in this special case?
            jhj = np.zeros_like(results_dict[f"{term.name}-gain"])
            info_tup = (0, 0)

        # If reweighting is enabled, do it when the epoch changes, except
        # for the final epoch - we don't reweight if we won't solve again.
        if solver_opts.robust:
            current_epoch = ind // len(chain)
            next_epoch = (ind + 1) // len(chain)

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

        results_dict[f"{term.name}-conviter"] += np.atleast_2d(info_tup[0])
        results_dict[f"{term.name}-convperc"] = np.atleast_2d(info_tup[1])
        results_dict[f"{term.name}-jhj"] = jhj

    postsolve_chisq = compute_mean_postsolve_chisq(
        kwargs["DATA"],
        kwargs["MODEL_DATA"],
        kwargs["WEIGHT"],
        kwargs["FLAG"],
        kwargs["gains"],
        kwargs["ANTENNA1"],
        kwargs["ANTENNA2"],
        kwargs["time_maps"],
        kwargs["freq_maps"],
        kwargs["dir_maps"],
        kwargs.get("row_map", None),
        kwargs.get("row_weights", None),
        corr_mode
    )
    log_chisq(presolve_chisq, postsolve_chisq, aux_block_info, block_id)

    results_dict["presolve_chisq"] = presolve_chisq
    results_dict["postsolve_chisq"] = postsolve_chisq

    gc.collect()

    return results_dict
