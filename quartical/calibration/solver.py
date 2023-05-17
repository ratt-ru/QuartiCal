# -*- coding: utf-8 -*-
import gc
import numpy as np
from numba import set_num_threads
from collections import namedtuple
from itertools import cycle
from quartical.weights.robust import robust_reweighting
from quartical.gains.general.flagging import init_flags
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


term_kwarg_defaults = {
    "time_bins": (0, np.int32),
    "time_map": (0, np.int32),
    "freq_map": (0, np.int32),
    "dir_map": (0, np.int32),
    "param_time_bins": (0, np.int32),
    "param_time_map": (0, np.int32),
    "param_freq_map": (0, np.int32),
    "initial_gain": ((0, 0, 0, 0, 0), np.complex128),
    "initial_params": ((0, 0, 0, 0, 0), np.float64)
}


def make_per_term_kwargs(kwargs, chain):

    per_term_kwargs = {}

    for term in chain:

        term_kwargs = {}

        for fld, default in term_kwarg_defaults.items():

            key = f"{term.name}_{fld}"
            term_kwargs[key] = kwargs.pop(key, np.empty(*default))

        per_term_kwargs[term.name] = term_kwargs

    return per_term_kwargs


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

    set_num_threads(solver_opts.threads)  # Set numba threads.

    block_id = tuple(block_id_arr.squeeze())

    ref_ant = solver_opts.reference_antenna
    iter_recipe = solver_opts.iter_recipe

    # Remove ms/data kwargs from the overall kwargs.
    ms_fields = {fld for term in chain for fld in term.ms_inputs._fields}
    ms_kwargs = {k: kwargs.pop(k, None) for k in ms_fields}

    per_term_kwargs = make_per_term_kwargs(kwargs, chain)

    assert len(kwargs) == 0, f"Some kwargs not understood by solver: {kwargs}."

    results_dict = {}

    # NOTE: This is a necessary evil. We do not want to modify the inputs
    # and copying is the best way to ensure that that cannot happen.
    ms_kwargs["WEIGHT"] = ms_kwargs["WEIGHT"].copy()
    results_dict["weights"] = ms_kwargs["WEIGHT"]
    ms_kwargs["FLAG"] = ms_kwargs["FLAG"].copy()
    results_dict["flags"] = ms_kwargs["FLAG"]

    for term_ind, term in enumerate(chain):

        term_spec = term_spec_list[term_ind]
        (_, _, term_shape, term_pshape) = term_spec

        term_kwargs = per_term_kwargs[term.name]

        # Perform term specific setup e.g. init gains and params.
        gain_array, param_array = term.init_term(
            term_spec, ref_ant, ms_kwargs, term_kwargs
        )

        # Init gain flags by looking for intervals with no data.
        gain_flag_array = init_flags(
            term_shape,
            term_kwargs[f"{term.name}_time_map"],
            term_kwargs[f"{term.name}_freq_map"],
            ms_kwargs["FLAG"],
            ms_kwargs["ANTENNA1"],
            ms_kwargs["ANTENNA2"],
            ms_kwargs.get("ROW_MAP", None)
        )

        # Init parameter flags by looking for intervals with no data.
        param_flag_array = init_flags(
            term_pshape,
            term_kwargs[f"{term.name}_param_time_map"],
            term_kwargs[f"{term.name}_param_freq_map"],
            ms_kwargs["FLAG"],
            ms_kwargs["ANTENNA1"],
            ms_kwargs["ANTENNA2"],
            ms_kwargs.get("ROW_MAP", None)
        )

        # Add the quantities which we intend to return to the results dict.
        results_dict[f"{term.name}_gain"] = gain_array
        results_dict[f"{term.name}_gain_flags"] = gain_flag_array
        results_dict[f"{term.name}_param"] = param_array
        results_dict[f"{term.name}_param_flags"] = param_flag_array
        results_dict[f"{term.name}_conviter"] = np.atleast_2d(0)   # int
        results_dict[f"{term.name}_convperc"] = np.atleast_2d(0.)  # float

    # Convert per-term values into appropriately ordered tuples which can be
    # passed into the numba layer. TODO: Changing chain length will result
    # in recompilation. Investigate fixed length tuples.
    time_bin_tup = tuple(
        [v[f"{k}_time_bins"] for k, v in per_term_kwargs.items()]
    )
    time_map_tup = tuple(
        [v[f"{k}_time_map"] for k, v in per_term_kwargs.items()]
    )
    freq_map_tup = tuple(
        [v[f"{k}_freq_map"] for k, v in per_term_kwargs.items()]
    )
    dir_map_tup = tuple(
        [v[f"{k}_dir_map"] for k, v in per_term_kwargs.items()]
    )
    param_time_bin_tup = tuple(
        [v[f"{k}_param_time_bins"] for k, v in per_term_kwargs.items()]
    )
    param_time_map_tup = tuple(
        [v[f"{k}_param_time_map"] for k, v in per_term_kwargs.items()]
    )
    param_freq_map_tup = tuple(
        [v[f"{k}_param_freq_map"] for k, v in per_term_kwargs.items()]
    )
    gain_array_tup = tuple(
        [results_dict[f"{term.name}_gain"] for term in chain]
    )
    gain_flag_array_tup = tuple(
        [results_dict[f"{term.name}_gain_flags"] for term in chain]
    )
    param_array_tup = tuple(
        [results_dict[f"{term.name}_param"] for term in chain]
    )
    param_flag_array_tup = tuple(
        [results_dict[f"{term.name}_param_flags"] for term in chain]
    )

    # Take the tuples above and create a new dictionary for these arguments,
    # now in a form appropriate for the solver calls.
    chain_kwargs = {
        "gains": gain_array_tup,
        "gain_flags": gain_flag_array_tup,
        "params": param_array_tup,
        "param_flags": param_flag_array_tup
    }

    mapping_kwargs = {
        "time_bins": time_bin_tup,
        "time_maps": time_map_tup,
        "freq_maps": freq_map_tup,
        "dir_maps": dir_map_tup,
        "param_time_bins": param_time_bin_tup,
        "param_time_maps": param_time_map_tup,
        "param_freq_maps": param_freq_map_tup
    }

    if solver_opts.robust:
        final_epoch = len(iter_recipe) // len(chain)
        etas = np.zeros_like(ms_kwargs["WEIGHT"][..., 0])
        icovariance = np.zeros(corr_mode, np.float64)
        dof = 5

    presolve_chisq = compute_mean_postsolve_chisq(
        ms_kwargs["DATA"],
        ms_kwargs["MODEL_DATA"],
        ms_kwargs["WEIGHT"],
        ms_kwargs["FLAG"],
        ms_kwargs["ANTENNA1"],
        ms_kwargs["ANTENNA2"],
        ms_kwargs.get("ROW_MAP", None),
        ms_kwargs.get("ROW_WEIGHTS", None),
        chain_kwargs["gains"],
        mapping_kwargs["time_maps"],
        mapping_kwargs["freq_maps"],
        mapping_kwargs["dir_maps"],
        corr_mode
    )

    for ind, (term, iters) in enumerate(zip(cycle(chain), iter_recipe)):

        active_term = chain.index(term)

        ms_fields = term.ms_inputs._fields
        ms_inputs = term.ms_inputs(
            **{k: ms_kwargs.get(k, None) for k in ms_fields}
        )

        mapping_fields = term.mapping_inputs._fields
        mapping_inputs = term.mapping_inputs(
            **{k: mapping_kwargs.get(k, None) for k in mapping_fields}
        )

        chain_fields = term.chain_inputs._fields
        chain_inputs = term.chain_inputs(
            **{k: chain_kwargs.get(k, None) for k in chain_fields}
        )

        meta_inputs = meta_args_nt(
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
                ms_inputs,
                mapping_inputs,
                chain_inputs,
                meta_inputs,
                corr_mode
            )
        else:
            # TODO: Actually compute it in this special case?
            jhj = np.zeros_like(results_dict[f"{term.name}_gain"])
            info_tup = (0, 0)

        # If reweighting is enabled, do it when the epoch changes, except
        # for the final epoch - we don't reweight if we won't solve again.
        if solver_opts.robust:
            current_epoch = ind // len(chain)
            next_epoch = (ind + 1) // len(chain)

            if current_epoch != next_epoch and next_epoch != final_epoch:

                dof = robust_reweighting(
                    ms_inputs,
                    mapping_inputs,
                    chain_inputs,
                    etas,
                    icovariance,
                    dof,
                    corr_mode
                )

        # TODO: Ugly hack for larger jhj matrices. Refine.
        if jhj.ndim == 6:
            jhj = jhj[:, :, :, :, range(jhj.shape[-2]), range(jhj.shape[-1])]

        results_dict[f"{term.name}_conviter"] += np.atleast_2d(info_tup[0])
        results_dict[f"{term.name}_convperc"] = np.atleast_2d(info_tup[1])
        results_dict[f"{term.name}_jhj"] = jhj

    postsolve_chisq = compute_mean_postsolve_chisq(
        ms_kwargs["DATA"],
        ms_kwargs["MODEL_DATA"],
        ms_kwargs["WEIGHT"],
        ms_kwargs["FLAG"],
        ms_kwargs["ANTENNA1"],
        ms_kwargs["ANTENNA2"],
        ms_kwargs.get("ROW_MAP", None),
        ms_kwargs.get("ROW_WEIGHTS", None),
        chain_kwargs["gains"],
        mapping_kwargs["time_maps"],
        mapping_kwargs["freq_maps"],
        mapping_kwargs["dir_maps"],
        corr_mode
    )
    log_chisq(presolve_chisq, postsolve_chisq, aux_block_info, block_id)

    results_dict["presolve_chisq"] = presolve_chisq
    results_dict["postsolve_chisq"] = postsolve_chisq

    gc.collect()

    return results_dict
