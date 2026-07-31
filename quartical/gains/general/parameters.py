# -*- coding: utf-8 -*-
"""Parameter-space helpers shared by the parameterised gain solvers."""
import numpy as np


def get_identity_params(corr_mode, n_param, *, per_correlation, fill=0.0):
    """Return the parameter vector which produces an identity gain.

    Args:
        corr_mode: A numba literal holding the number of correlations.
        n_param: The number of parameters the term solves for - per diagonal
            correlation when ``per_correlation`` is True, in total otherwise.
        per_correlation: Whether the parameters are duplicated per diagonal
            correlation. Terms whose single parameter set acts on the full 2x2
            (crosshand phase, rotation, rotation measure) pass False and
            support four correlations only.
        fill: The parameter value which yields an identity gain - zero for
            phase-like parameters, one for amplitudes.

    Returns:
        A flat float64 array of identity parameters.

    Raises:
        ValueError: If corr_mode is unsupported for this parameterisation.
    """

    if per_correlation:
        if corr_mode.literal_value in (2, 4):
            n_element = 2 * n_param
        elif corr_mode.literal_value == 1:
            n_element = n_param
        else:
            raise ValueError("Unsupported number of correlations.")
    else:
        if corr_mode.literal_value != 4:
            raise ValueError("Unsupported number of correlations.")
        n_element = n_param

    return np.full((n_element,), fill, dtype=np.float64)
