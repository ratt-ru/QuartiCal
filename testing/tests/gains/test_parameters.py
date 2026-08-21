# -*- coding: utf-8 -*-
"""Check the shared parameter-count helper against the gain classes.

``get_n_param`` derives a term's flat parameter-vector length from the
``PARAMS_PER_CORR`` its kernel declares, and the accumulator layout in
``general/accumulator.py`` is sized from the same number. The independent
statement of that length is each gain class's ``make_param_names``, which is
what actually shapes the params array and the output zarr dataset, so the two
have to agree for every term and correlation mode.
"""
import importlib

import numpy as np
import pytest
from numba import types

from quartical.gains import TERM_TYPES
from quartical.gains.general.parameters import (
    get_identity_params,
    get_n_param,
)

# Term type -> the kernel module declaring its PARAMS_PER_CORR.
PARAMETERISED_TERMS = {
    "amplitude": "quartical.gains.amplitude.kernel",
    "phase": "quartical.gains.phase.kernel",
    "delay": "quartical.gains.delay.kernel",
    "delay_and_offset": "quartical.gains.delay_and_offset.kernel",
    "delay_and_tec": "quartical.gains.delay_and_tec.kernel",
    "tec_and_offset": "quartical.gains.tec_and_offset.kernel",
    "delay_tec_and_offset": "quartical.gains.delay_tec_and_offset.kernel",
    "crosshand_phase": "quartical.gains.crosshand_phase.kernel",
    "crosshand_phase_null_v": "quartical.gains.crosshand_phase.null_v_kernel",
    "rotation": "quartical.gains.rotation.kernel",
    "rotation_measure": "quartical.gains.rotation_measure.kernel",
}

# The correlation labels a Measurement Set presents at each corr mode, in both
# linear and circular feed bases - crosshand_phase parameterises XX/RR only, so
# the basis matters.
CORRELATIONS = {
    ("linear", 4): ["XX", "XY", "YX", "YY"],
    ("linear", 2): ["XX", "YY"],
    ("linear", 1): ["XX"],
    ("circular", 4): ["RR", "RL", "LR", "LL"],
    ("circular", 2): ["RR", "LL"],
    ("circular", 1): ["RR"],
}


def params_per_corr(term_type):
    """Return the PARAMS_PER_CORR its kernel module declares."""

    module = importlib.import_module(PARAMETERISED_TERMS[term_type])

    return module.PARAMS_PER_CORR


@pytest.mark.parametrize("term_type", PARAMETERISED_TERMS)
@pytest.mark.parametrize("basis, n_corr", CORRELATIONS)
def test_n_param_matches_param_names(term_type, basis, n_corr):
    """get_n_param must agree with the class's own parameter-name list."""

    correlations = CORRELATIONS[basis, n_corr]
    term_class = TERM_TYPES[term_type]
    declared = params_per_corr(term_type)

    if declared is None and n_corr != 4:
        # A term whose single parameter set acts on the full 2x2 supports four
        # correlations only, and says so rather than sizing anything.
        with pytest.raises(ValueError):
            get_n_param(types.literal(n_corr), declared)
        return

    n_param = get_n_param(types.literal(n_corr), declared)

    assert n_param == len(term_class.make_param_names(correlations))


@pytest.mark.parametrize("term_type", PARAMETERISED_TERMS)
@pytest.mark.parametrize("n_corr", [1, 2, 4])
def test_identity_params_length_matches_n_param(term_type, n_corr):
    """The identity vector is sized by the same rule as the accumulator."""

    declared = params_per_corr(term_type)
    corr_mode = types.literal(n_corr)

    if declared is None and n_corr != 4:
        # A term whose single parameter set acts on the full 2x2 sizes nothing
        # outside four correlations.
        with pytest.raises(ValueError):
            get_identity_params(corr_mode, declared)
        return

    identity_params = get_identity_params(corr_mode, declared)

    assert identity_params.shape == (get_n_param(corr_mode, declared),)
    assert identity_params.dtype == np.float64


def test_amplitude_identity_params_are_unity():
    """Amplitude is the one term whose identity parameters are not zero.

    The fill comes from the kernel's own ``IDENTITY_FILL``, which is also what
    ``init_term`` writes into flagged parameters, so a term declaring the wrong
    identity fails here rather than silently writing it to the gain dataset.
    """

    module = importlib.import_module(PARAMETERISED_TERMS["amplitude"])
    declared = params_per_corr("amplitude")

    identity_params = get_identity_params(
        types.literal(4), declared, fill=module.IDENTITY_FILL
    )

    assert np.all(identity_params == 1.0)


@pytest.mark.parametrize("n_corr", [0, 3, 5])
def test_unsupported_corr_mode_raises(n_corr):
    """An unsupported correlation count is rejected, not silently sized."""

    with pytest.raises(ValueError, match="Unsupported number of correlations"):
        get_n_param(types.literal(n_corr), 1)
