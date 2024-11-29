# -*- coding: utf-8 -*-
import gc
import numpy as np
from numba import set_num_threads
from collections import namedtuple
from itertools import cycle
from quartical.weights.robust import robust_reweighting
from quartical.statistics.stat_kernels import compute_mean_postsolve_chisq
from quartical.statistics.logging import log_chisq


meta_args_nt = namedtuple(
    "meta_args_nt", (
        "iters",
        "active_term",
        "stop_frac",
        "stop_crit",
        "threads",
        "robust",
        "reference_antenna",
        "dd_term",
        "pinned_directions",
        "solve_per",
    )
)


def make_mapping_tuple(dictionary, field, default=None):

    default = np.empty(0, dtype=np.int32) if default is None else default

    return tuple(
        [v.get(f"{k}_{field}", default) for k, v in dictionary.items()]
    )


def make_per_term_kwargs(kwargs, chain):

    per_term_kwargs = {}

    for term in chain:

        term_keys = [k for k in kwargs.keys() if k.startswith(f"{term.name}_")]

        term_kwargs = {tk: kwargs.pop(tk) for tk in term_keys}

        per_term_kwargs[term.name] = term_kwargs

    return per_term_kwargs


def solver_wrapper(
    term_spec_list,
    solver_opts,
    chain,
    block_id_arr,
    data_xds_meta,
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

    # Remove ms/data kwargs from the overall kwargs. Note that missing fields
    # will be assigned a value of None. This should only ever apply to the
    # BDA related inputs.
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
        if term.is_parameterized:
            gains, gain_flags, params, param_flags = term.init_term(
                term_spec, ref_ant, ms_kwargs, term_kwargs, meta=data_xds_meta
            )
        else:
            gains, gain_flags = term.init_term(
                term_spec, ref_ant, ms_kwargs, term_kwargs, meta=data_xds_meta
            )
            # Dummy arrays with standard dtypes - aids compilation.
            params = np.empty(term_pshape, dtype=np.float64)
            param_flags = np.empty(term_pshape[:-1], dtype=np.int8)

        # Add the quantities which we intend to return to the results dict.
        results_dict[f"{term.name}_gains"] = gains
        results_dict[f"{term.name}_gain_flags"] = gain_flags
        results_dict[f"{term.name}_params"] = params
        results_dict[f"{term.name}_param_flags"] = param_flags
        results_dict[f"{term.name}_conviter"] = np.atleast_2d(0)   # int
        results_dict[f"{term.name}_convperc"] = np.atleast_2d(0.)  # float

    # Convert per-term values into appropriately ordered tuples which can be
    # passed into the numba layer. TODO: Changing chain length will result
    # in recompilation. Investigate fixed length tuples.
    time_bin_tup = make_mapping_tuple(per_term_kwargs, "time_bins")
    time_map_tup = make_mapping_tuple(per_term_kwargs, "time_map")
    freq_map_tup = make_mapping_tuple(per_term_kwargs, "freq_map")
    dir_map_tup = make_mapping_tuple(per_term_kwargs, "dir_map")
    param_time_bin_tup = make_mapping_tuple(per_term_kwargs, "param_time_bins")
    param_time_map_tup = make_mapping_tuple(per_term_kwargs, "param_time_map")
    param_freq_map_tup = make_mapping_tuple(per_term_kwargs, "param_freq_map")

    gains_tup = tuple([results_dict[f"{term.name}_gains"] for term in chain])
    gain_flags_tup = tuple(
        [results_dict[f"{term.name}_gain_flags"] for term in chain]
    )
    params_tup = tuple([results_dict[f"{term.name}_params"] for term in chain])
    param_flags_tup = tuple(
        [results_dict[f"{term.name}_param_flags"] for term in chain]
    )

    # Take the tuples above and create a new dictionary for these arguments,
    # now in a form appropriate for the solver calls.
    chain_kwargs = {
        "gains": gains_tup,
        "gain_flags": gain_flags_tup,
        "params": params_tup,
        "param_flags": param_flags_tup
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
        dof = 5  # TODO: Expose?

    presolve_chisq = compute_mean_postsolve_chisq(
        ms_kwargs["DATA"],
        ms_kwargs["MODEL_DATA"],
        ms_kwargs["WEIGHT"],
        ms_kwargs["FLAG"],
        ms_kwargs["ANTENNA1"],
        ms_kwargs["ANTENNA2"],
        ms_kwargs["ROW_MAP"],
        ms_kwargs["ROW_WEIGHTS"],
        chain_kwargs["gains"],
        mapping_kwargs["time_maps"],
        mapping_kwargs["freq_maps"],
        mapping_kwargs["dir_maps"],
        corr_mode
    )

    for ind, (term, iters) in enumerate(zip(cycle(chain), iter_recipe)):

        active_term = chain.index(term)
        active_spec = term_spec_list[term_ind]

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
            solver_opts.robust,
            solver_opts.reference_antenna,
            term.direction_dependent,
            term.pinned_directions,
            term.solve_per
        )

        if term.solver:
            jhj, conv_iter, conv_perc = term.solver(
                ms_inputs,
                mapping_inputs,
                chain_inputs,
                meta_inputs,
                corr_mode
            )
        else:
            jhj = np.zeros(getattr(active_spec, "pshape", active_spec.shape))
            conv_iter, conv_perc = 0, 1

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
            jhj = jhj[..., range(jhj.shape[-2]), range(jhj.shape[-1])]

        results_dict[f"{term.name}_conviter"] += np.atleast_2d(conv_iter)
        results_dict[f"{term.name}_convperc"] = np.atleast_2d(conv_perc)
        results_dict[f"{term.name}_jhj"] = jhj

    postsolve_chisq = compute_mean_postsolve_chisq(
        ms_kwargs["DATA"],
        ms_kwargs["MODEL_DATA"],
        ms_kwargs["WEIGHT"],
        ms_kwargs["FLAG"],
        ms_kwargs["ANTENNA1"],
        ms_kwargs["ANTENNA2"],
        ms_kwargs["ROW_MAP"],
        ms_kwargs["ROW_WEIGHTS"],
        chain_kwargs["gains"],
        mapping_kwargs["time_maps"],
        mapping_kwargs["freq_maps"],
        mapping_kwargs["dir_maps"],
        corr_mode
    )

    log_chisq(presolve_chisq, postsolve_chisq, data_xds_meta, block_id)

    results_dict["presolve_chisq"] = presolve_chisq
    results_dict["postsolve_chisq"] = postsolve_chisq

    gc.collect()

    return results_dict
