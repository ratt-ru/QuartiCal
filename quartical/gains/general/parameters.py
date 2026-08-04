# -*- coding: utf-8 -*-
"""Parameter-space helpers shared by the parameterised gain solvers."""
import numpy as np

import quartical.gains.general.factories as factories
from quartical.gains.general.flagging import (
    apply_gain_flags_to_gains,
    apply_param_flags_to_params
)


def get_n_param(corr_mode, params_per_corr):
    """Return the length of a term's flat parameter vector.

    This one number is both the length of the parameter vector and the
    dimension of the (n_param, n_param) jhj element, so the identity vector
    below and the accumulator layout in ``accumulator.py`` derive from the same
    source and cannot disagree.

    Args:
        corr_mode: A numba literal holding the number of correlations.
        params_per_corr: The number of parameters the term solves per diagonal
            correlation, or None for a term whose single parameter set acts on
            the full 2x2. Those terms - crosshand phase, rotation and rotation
            measure - each solve one parameter and support four correlations
            only. They are exactly the terms which pass
            ``params_per_corr=None`` to ``build_param_solver_impl``, because
            collapsing jhj/jhr to a scalar solve is possible if and only if
            there is a parameter set per correlation to collapse.

    Returns:
        The number of parameters, which agrees with
        ``len(term_class.make_param_names(correlations))``.

    Raises:
        ValueError: If corr_mode is unsupported for this parameterisation.
    """

    if params_per_corr is None:
        if corr_mode.literal_value != 4:
            raise ValueError("Unsupported number of correlations.")
        return 1

    if corr_mode.literal_value in (2, 4):
        return 2*params_per_corr
    elif corr_mode.literal_value == 1:
        return params_per_corr
    else:
        raise ValueError("Unsupported number of correlations.")


def get_identity_params(corr_mode, params_per_corr, *, fill=0.0):
    """Return the parameter vector which produces an identity gain.

    Args:
        corr_mode: A numba literal holding the number of correlations.
        params_per_corr: The number of parameters the term solves per diagonal
            correlation. See :func:`get_n_param`.
        fill: The parameter value which yields an identity gain - zero for
            phase-like parameters, one for amplitudes.

    Returns:
        A flat float64 array of identity parameters.

    Raises:
        ValueError: If corr_mode is unsupported for this parameterisation.
    """

    n_param = get_n_param(corr_mode, params_per_corr)

    return np.full((n_param,), fill, dtype=np.float64)


def reference_params_factory(params_to_gains):
    """Produce the referencing hook for a frequency-dependent term.

    A per-antenna solution is only determined up to a value common to every
    antenna, so referencing subtracts the reference antenna's parameters to fix
    that freedom. Flagged parameters are skipped, since they hold an identity
    value rather than a solution. Rebuilding the gains from the referenced
    parameters writes every gain, flagged or not, so both sets of flags are
    reapplied afterwards to restore the identity wherever they are raised.

    Args:
        params_to_gains: The term's routine for rebuilding gains from
            parameters, taking ``(params, gains, chan_freq, min_freq, max_freq,
            param_freq_map, rescaled)``. Terms whose gains vary across the band
            share this signature; ``phase`` does not, and so states its own
            referencing hook.

    Returns:
        An inlineable ``reference_params(ms_inputs, mapping_inputs,
        chain_inputs, meta_inputs)`` suitable for
        ``build_param_solver_impl``.
    """

    def reference_params(ms_inputs, mapping_inputs, chain_inputs, meta_inputs):

        active_term = meta_inputs.active_term
        ref_ant = meta_inputs.reference_antenna

        gains = chain_inputs.gains[active_term]
        gain_flags = chain_inputs.gain_flags[active_term]
        params = chain_inputs.params[active_term]
        param_flags = chain_inputs.param_flags[active_term]

        param_freq_map = mapping_inputs.param_freq_maps[active_term]

        n_ti, n_fi, n_ant, n_dir, _ = params.shape

        # Copied, because the reference antenna's own parameters are needed
        # after the loop has begun subtracting them in place.
        ref_params = params[:, :, ref_ant: ref_ant + 1, :, :].copy()

        for t in range(n_ti):
            for f in range(n_fi):
                for a in range(n_ant):
                    for d in range(n_dir):

                        p = params[t, f, a, d]
                        rp = ref_params[t, f, 0, d]

                        if param_flags[t, f, a, d] == 1:
                            continue
                        else:
                            p -= rp

        params_to_gains(
            params,
            gains,
            ms_inputs.CHAN_FREQ,
            ms_inputs.MIN_FREQ,
            ms_inputs.MAX_FREQ,
            param_freq_map,
            rescaled=True
        )

        # Referencing may move flagged gains/params from identity.
        apply_param_flags_to_params(param_flags, params, 0)
        apply_gain_flags_to_gains(gain_flags, gains)

    return factories.qcjit(reference_params)
