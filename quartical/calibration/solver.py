# -*- coding: utf-8 -*-
import gc
import numpy as np
from numba import set_num_threads
from collections import namedtuple
from itertools import cycle
from quartical.gains import TERM_TYPES
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

    terms = ()
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

    for term_ind, term_spec in enumerate(term_spec_list):

        (term_name, term_type, term_shape, term_pshape) = term_spec

        term_opts = getattr(chain_opts, term_name)
        gain_obj = TERM_TYPES[term_type](term_name, term_opts)

        gain_array = np.zeros(term_shape, dtype=np.complex128)
        param_array = np.zeros(term_pshape, dtype=gain_array.real.dtype)

        # Perform term specific setup e.g. init gains and params.
        gain_obj.init_term(
            gain_array,
            param_array,
            term_ind,
            term_spec,
            term_opts,
            ref_ant,
            **kwargs
        )

        time_col = kwargs.get("TIME")
        interval_col = kwargs.get("INTERVAL")
        scan_col = kwargs.get("SCAN_NUMBER", np.zeros_like(time_col))

        time_bins = gain_obj._make_time_bins(
            time_col,
            interval_col,
            scan_col,
            term_opts.time_interval,
            term_opts.respect_scan_boundaries
        )

        time_map = gain_obj._make_time_map(
            time_col,
            time_bins
        )

        chan_freqs = kwargs.get("CHAN_FREQ")
        chan_widths = kwargs.get("CHAN_WIDTH")
        freq_interval = gain_obj.freq_interval

        freq_map = gain_obj._make_freq_map(
            chan_freqs,
            chan_widths,
            freq_interval
        )

        # TODO: Should this accept model data as a parameter?
        dir_map = gain_obj._make_dir_map(
            kwargs["MODEL_DATA"].shape[2],
            gain_obj.direction_dependent
        )

        # Init gain flags by looking for intervals with no data.
        gain_flag_array = init_gain_flags(
            term_shape,
            time_map,
            freq_map,
            **kwargs
        )

        if hasattr(gain_obj, "param_axes"):
            param_time_bins = gain_obj._make_param_time_bins(
                time_col,
                interval_col,
                scan_col,
                term_opts.time_interval,
                term_opts.respect_scan_boundaries
            )

            param_time_map = gain_obj._make_param_time_map(
                time_col,
                param_time_bins
            )

            param_freq_map = gain_obj._make_param_freq_map(
                chan_freqs,
                chan_widths,
                freq_interval
            )

            param_flag_array = init_param_flags(
                term_pshape,
                param_time_map,
                param_freq_map,
                **kwargs
            )
        else:
            param_time_bins = np.empty(0, dtype=np.int64)
            param_time_map = np.empty(0, dtype=np.int64)
            param_freq_map = np.empty(0, dtype=np.int64)
            param_flag_array = np.empty(term_pshape[:-1], dtype=np.int64)

        terms += (gain_obj,)
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

        results_dict[f"{term_name}-gain"] = gain_array
        results_dict[f"{term_name}-gain_flags"] = gain_flag_array
        results_dict[f"{term_name}-param"] = param_array
        results_dict[f"{term_name}-param_flags"] = param_flag_array
        results_dict[f"{term_name}-conviter"] = np.atleast_2d(0)   # int
        results_dict[f"{term_name}-convperc"] = np.atleast_2d(0.)  # float

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
    results_dict["WEIGHT"] = kwargs["WEIGHT"]
    kwargs["FLAG"] = kwargs["FLAG"].copy()
    results_dict["FLAG"] = kwargs["FLAG"]

    if solver_opts.robust:
        final_epoch = len(iter_recipe) // len(terms)
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

    for ind, (term, iters) in enumerate(zip(cycle(terms), iter_recipe)):

        active_term = terms.index(term)

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
