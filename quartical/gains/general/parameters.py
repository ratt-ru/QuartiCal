# -*- coding: utf-8 -*-
"""Parameter-space helpers shared by the parameterised gain solvers."""
import numpy as np


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
