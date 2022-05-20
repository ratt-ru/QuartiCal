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
        "is_init",
        "stop_frac",
        "stop_crit",
        "dd_term",
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
    gain_flags_tup = ()
    param_flags_tup = ()
    results_dict = {}
    is_initialised = {}

    for term_ind, term_spec in enumerate(term_spec_list):

        (term_name, term_type, term_shape, term_pshape) = term_spec

        gain = np.zeros(term_shape, dtype=np.complex128)

        # Check for initialisation data. TODO: Parameterised terms?
        if f"{term_name}_initial_gain" in kwargs:
            gain[:] = kwargs[f"{term_name}_initial_gain"]
            is_initialised[term_name] = True
        else:
            gain[..., (0, -1)] = 1  # Set first and last correlations to 1.
            is_initialised[term_name] = False

        # Init gain flags by looking for intervals with no data.
        gain_flags = init_gain_flags(term_shape, term_ind, **kwargs)
        param = np.zeros(term_pshape, dtype=gain.real.dtype)
        param_flags = init_param_flags(term_pshape, term_ind, **kwargs)

        if term_type == "delay":  # TODO: TEMPORARY! Needs to be done per term.
            init_delays(param, gain, term_ind, **kwargs)

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
                                 is_initialised[term_name],
                                 solver_opts.convergence_fraction,
                                 solver_opts.convergence_criteria,
                                 term_opts.direction_dependent,
                                 term_opts.solve_per,
                                 robust)

        if iters != 0:
            jhj, info_tup = solver(base_args,
                                   term_args,
                                   meta_args,
                                   kwargs["corr_mode"])

            # After a solver is run once, it will have been initialised.
            is_initialised[term_name] = True
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
        results_dict[f"{term_name}-convperc"] += np.atleast_2d(info_tup[1])
        results_dict[f"{term_name}-jhj"] = jhj

    gc.collect()

    return results_dict


def init_delays(param, gain, term_ind, **kwargs):

    data = kwargs["data"]  # (row, chan, corr)
    flags = kwargs["flags"]  # (row, chan)
    a1 = kwargs["a1"]
    a2 = kwargs["a2"]
    chan_freq = kwargs["chan_freqs"]
    t_map = kwargs["t_map_arr"][0, :, term_ind]  # time -> sol map for gains.
    refant = 0  # TODO: Make controllable.
    _, n_chan, n_ant, n_dir, n_corr = gain.shape
    n_bl = n_ant * (n_ant - 1) // 2
    # TODO: Make controllable/automate. Check with L.
    pad_factor = int(np.ceil(2 ** 15 / n_chan))

    sel = np.where((a1 == refant) | (a2 == refant))  # All bls with refant.

    # We only need the baselines which include the refant.
    a1 = a1[sel]
    a2 = a2[sel]
    t_map = t_map[sel]
    data = data[sel]
    flags = flags[sel]

    utint = np.unique(t_map)

    for ut in utint:
        sel = np.where(t_map == ut)
        ant_map = \
            np.where(a1[sel] == refant, a2[sel], 0) + \
            np.where(a2[sel] == refant, a1[sel], 0)
        ref_data = np.zeros((n_ant, n_chan, n_corr), dtype=np.complex128)
        counts = np.zeros((n_ant, n_chan), dtype=int)
        np.add.at(ref_data, ant_map, data[sel])
        np.add.at(counts, ant_map, flags[sel] == 0)
        np.divide(ref_data, counts[:, :, None], where=counts[:, :, None] != 0, out=ref_data)

        fft_data = np.abs(np.fft.fft(ref_data, n=n_chan*pad_factor, axis=1))
        fft_data = np.fft.fftshift(fft_data, axes=1)

        delta_freq = chan_freq[1] - chan_freq[0]
        fft_freq = np.fft.fftfreq(n_chan*pad_factor, delta_freq)
        fft_freq = np.fft.fftshift(fft_freq)

        delay_est_ind_00 = np.argmax(fft_data[..., 0], axis=1)
        delay_est_00 = fft_freq[delay_est_ind_00]

        if n_corr > 1:
            delay_est_ind_11 = np.argmax(fft_data[..., -1], axis=1)
            delay_est_11 = fft_freq[delay_est_ind_11]

        for t, p, q in zip(t_map[sel], a1[sel], a2[sel]):
            if p == refant:
                param[t, 0, q, 0, 1] = -delay_est_00[q]
                if n_corr > 1:
                    param[t, 0, q, 0, 3] = -delay_est_11[q]
            else:
                param[t, 0, p, 0, 1] = delay_est_00[p]
                if n_corr > 1:
                    param[t, 0, p, 0, 3] = delay_est_11[p]

    coeffs00 = param[..., 1]*kwargs["chan_freqs"][None, :,  None, None]
    gain[..., 0] = np.exp(2j*np.pi*coeffs00)

    if n_corr > 1:
        coeffs11 = param[..., 3]*kwargs["chan_freqs"][None, :, None, None]
        gain[..., -1] = np.exp(2j*np.pi*coeffs11)
