# -*- coding: utf-8 -*-
"""Check what init_term does with flags.

A parameter interval with no unflagged data backing it is never solved for, so
init_term fills it with the parameter value which produces an identity gain.
That value is term-specific - zero for every phase-like parameter, one for an
amplitude - and the solvers obtain it from ``get_identity_params``, to which
each kernel supplies its own fill. init_term has to agree: a flagged interval
holding anything else is written to the output gain dataset and read back by
``load_from`` interpolation.

init_term is also where the flags accompanying loaded solutions enter the
solve. They describe intervals which the interpolation could not fill - most
importantly antennas which were absent from the loaded solutions altogether -
and are merged with the flags derived from the data.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from quartical.calibration.constructor import term_spec_tup
from quartical.gains import TERM_TYPES

# Term type -> the parameter value which produces an identity gain. Excludes
# parallactic_angle, whose init_term overwrites params with angles computed via
# casacore measures and so cannot be driven from synthetic inputs.
IDENTITY_PARAM = {
    "amplitude": 1.0,
    "phase": 0.0,
    "delay": 0.0,
    "delay_and_offset": 0.0,
    "delay_and_tec": 0.0,
    "tec_and_offset": 0.0,
    "delay_tec_and_offset": 0.0,
    "crosshand_phase": 0.0,
    "rotation": 0.0,
    "rotation_measure": 0.0,
}

# Terms whose init_term branches on initial_estimate, filling flags on either
# side of the branch. Both sides need checking - they are separate call sites.
ESTIMATING_TERMS = {
    "delay",
    "delay_and_offset",
    "delay_and_tec",
    "tec_and_offset",
    "delay_tec_and_offset",
}

# Four correlations, which is the only mode every term above supports.
CORRELATIONS = ["XX", "XY", "YX", "YY"]

N_TIME = 2
N_CHAN = 4
N_ANT = 3
N_DIR = 1
N_CORR = len(CORRELATIONS)

# Every row at this time index carries a raised flag, leaving the interval
# unbacked by data and so flagged for every antenna.
FLAGGED_TIME = 1


def term_options(term_type, initial_estimate, load_from=None):
    """Build the minimal options object the Gain constructor reads."""

    return SimpleNamespace(
        type=term_type,
        solve_per="antenna",
        scalar=False,
        direction_dependent=False,
        pinned_directions=[],
        time_interval=1,
        freq_interval=1,
        respect_scan_boundaries=False,
        initial_estimate=initial_estimate,
        load_from=load_from,
        interp_mode="reim",
        interp_method="2dlinear",
    )


def synthetic_inputs(n_param):
    """Build the term_spec and the ms/term kwargs which init_term indexes.

    Rows cover every baseline at every time, and every row at
    ``FLAGGED_TIME`` is flagged, so exactly one of the two time intervals
    survives.
    """

    baselines = [(a, b) for a in range(N_ANT) for b in range(a + 1, N_ANT)]
    rows = [(t, bl) for t in range(N_TIME) for bl in baselines]

    time_map = np.empty(len(rows), dtype=np.int32)
    ant1 = np.empty(len(rows), dtype=np.int32)
    ant2 = np.empty(len(rows), dtype=np.int32)
    flag = np.zeros((len(rows), N_CHAN), dtype=np.int8)

    for row, (t, (a, b)) in enumerate(rows):
        time_map[row], ant1[row], ant2[row] = t, a, b
        if t == FLAGGED_TIME:
            flag[row] = 1

    freq_map = np.arange(N_CHAN, dtype=np.int32)
    chan_freq = np.linspace(1e9, 2e9, N_CHAN)

    # The parameter frequency grid is one bin spanning the band, as it is for a
    # real delay solve. The initial-estimate paths need several channels per
    # parameter interval to derive a resolution from.
    param_freq_map = np.zeros(N_CHAN, dtype=np.int32)

    spec = term_spec_tup(
        "G",
        None,
        (N_TIME, N_CHAN, N_ANT, N_DIR, N_CORR),
        (N_TIME, 1, N_ANT, N_DIR, n_param),
    )
    # Unit visibilities: the initial-estimate paths read DATA, and what they
    # estimate from it is irrelevant here - only that they then fill flagged
    # parameters with the identity.
    data = np.ones((len(rows), N_CHAN, N_CORR), dtype=np.complex128)

    ms_kwargs = {
        "DATA": data,
        "FLAG": flag,
        "ANTENNA1": ant1,
        "ANTENNA2": ant2,
        "ROW_MAP": None,
        "CHAN_FREQ": chan_freq,
        "MIN_FREQ": chan_freq.min(),
        "MAX_FREQ": chan_freq.max(),
    }
    term_kwargs = {
        "G_time_map": time_map,
        "G_freq_map": freq_map,
        "G_param_time_map": time_map,
        "G_param_freq_map": param_freq_map,
    }

    return spec, ms_kwargs, term_kwargs


# One case per distinct flag-filling call site: both sides of the branch for
# the estimating terms, the single path for the rest.
CASES = [
    (term_type, initial_estimate)
    for term_type in IDENTITY_PARAM
    for initial_estimate in (
        (False, True) if term_type in ESTIMATING_TERMS else (False,)
    )
]


@pytest.fixture(
    params=CASES,
    ids=lambda case: f"{case[0]}-estimate{case[1]:d}",
    scope="module",
)
def case(request):
    return request.param


@pytest.fixture(scope="module")
def term_type(case):
    return case[0]


@pytest.fixture(scope="module")
def init_term_output(case):
    """Drive a term's init_term over a grid with one fully-flagged interval."""

    term_type, initial_estimate = case
    term_class = TERM_TYPES[term_type]
    n_param = len(term_class.make_param_names(CORRELATIONS))

    spec, ms_kwargs, term_kwargs = synthetic_inputs(n_param)
    term = term_class("G", term_options(term_type, initial_estimate))

    return term.init_term(spec, 0, ms_kwargs, term_kwargs)


@pytest.fixture(scope="module")
def identity_params(term_type):
    """The identity parameter vector this term is expected to fill with."""

    n_param = len(TERM_TYPES[term_type].make_param_names(CORRELATIONS))

    return np.full(n_param, IDENTITY_PARAM[term_type])


def test_flagged_interval_exists(init_term_output):
    """The inputs must actually produce flagged parameters to check."""

    *_, param_flags = init_term_output

    assert param_flags[FLAGGED_TIME].all()
    assert not param_flags[1 - FLAGGED_TIME].any()


def test_flagged_params_hold_identity(init_term_output, identity_params):
    """Flagged parameters must equal the term's identity parameter vector."""

    *_, params, param_flags = init_term_output

    flagged = params[param_flags.astype(bool)]

    assert flagged.size
    np.testing.assert_array_equal(
        flagged, np.broadcast_to(identity_params, flagged.shape)
    )


# ---------------------------------loaded flags--------------------------------

# The antenna which is missing from the loaded solutions and is consequently
# flagged for every interval.
MISSING_ANTENNA = 2

# Any path will do - init_term only checks that load_from is set.
LOAD_PATH = "loads.qc/G"


def loaded_flags(shape):
    """Flags in which a single antenna is flagged for every interval."""

    flags = np.zeros(shape[:-1], dtype=np.int8)
    flags[:, :, MISSING_ANTENNA] = 1

    return flags


@pytest.fixture(params=["complex", "delay"], scope="module")
def loaded_term_type(request):
    return request.param


@pytest.fixture(scope="module")
def loaded_init_term_output(loaded_term_type):
    """Drive init_term with loaded solutions containing flags of their own."""

    term_class = TERM_TYPES[loaded_term_type]
    term = term_class("G", term_options(loaded_term_type, False, LOAD_PATH))

    if term.is_parameterized:
        n_param = len(term_class.make_param_names(CORRELATIONS))
    else:
        n_param = 1  # Unused - unparameterised terms have no parameters.

    spec, ms_kwargs, term_kwargs = synthetic_inputs(n_param)

    term_kwargs["G_initial_gain"] = np.ones(spec.shape, dtype=np.complex128)
    term_kwargs["G_initial_gain_flags"] = loaded_flags(spec.shape)
    term_kwargs["G_initial_params"] = np.zeros(spec.pshape, dtype=np.float64)
    term_kwargs["G_initial_param_flags"] = loaded_flags(spec.pshape)

    return term.init_term(spec, 0, ms_kwargs, term_kwargs)


def test_loaded_gain_flags_merged(loaded_init_term_output):
    """An antenna missing from the loaded solutions is fully flagged."""

    _, gain_flags, *_ = loaded_init_term_output

    assert gain_flags[:, :, MISSING_ANTENNA].all()


def test_loaded_gain_flags_are_additional(loaded_init_term_output):
    """The loaded flags do not flag intervals which the data supports."""

    _, gain_flags, *_ = loaded_init_term_output

    remaining = np.delete(gain_flags, MISSING_ANTENNA, axis=2)

    assert not remaining[1 - FLAGGED_TIME].any()


def test_loaded_param_flags_merged(loaded_init_term_output):
    """A missing antenna is flagged on the parameter grid too."""

    if len(loaded_init_term_output) == 2:
        pytest.skip("Term is not parameterised.")

    *_, param_flags = loaded_init_term_output

    assert param_flags[:, :, MISSING_ANTENNA].all()
