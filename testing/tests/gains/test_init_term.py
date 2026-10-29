# -*- coding: utf-8 -*-
"""Check what init_term does with flags.

A parameter interval with no unflagged data backing it is never solved for, so
init_term fills it with the parameter value which produces an identity gain.
That value is term-specific - zero for every phase-like parameter, one for an
amplitude - and the solvers obtain it from ``get_identity_params``, to which
each kernel supplies its own fill. init_term has to agree: a flagged interval
holding anything else is written to the output gain dataset and read back by
``load_from`` interpolation.

init_term starts from the values and flags carried by the term's gain dataset
scaffold. These hold the identity and no raised flags unless the term was loaded
from disk, in which case they hold the interpolated solutions and flag the
antennas which the interpolation could not fill. The scaffold flags are merged
with the flags derived from the data.
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
        referenced=True,
    )


def synthetic_inputs(n_param, param_fill=0.0, per_channel_params=False):
    """Build the term_spec and the ms/term kwargs which init_term indexes.

    Rows cover every baseline at every time, and every row at
    ``FLAGGED_TIME`` is flagged, so exactly one of the two time intervals
    survives. The scaffold inputs hold identity gains, parameters equal to
    param_fill and no raised flags. The parameter frequency grid is one bin
    spanning the band unless per_channel_params is set, in which case it
    matches the gain frequency grid.
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

    # A single parameter frequency bin is what a real delay solve uses, and the
    # initial-estimate paths need several channels per parameter interval to
    # derive a resolution from. Phase-like terms instead index their
    # parameters on the gain grid, so converting them to gains needs one
    # parameter per channel.
    if per_channel_params:
        param_freq_map = np.arange(N_CHAN, dtype=np.int32)
    else:
        param_freq_map = np.zeros(N_CHAN, dtype=np.int32)

    n_param_freq = param_freq_map.max() + 1

    spec = term_spec_tup(
        "G",
        None,
        (N_TIME, N_CHAN, N_ANT, N_DIR, N_CORR),
        (N_TIME, n_param_freq, N_ANT, N_DIR, n_param),
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
    identity_gains = np.zeros(spec.shape, dtype=np.complex128)
    identity_gains[..., (0, 3)] = 1

    term_kwargs = {
        "G_time_map": time_map,
        "G_freq_map": freq_map,
        "G_param_time_map": time_map,
        "G_param_freq_map": param_freq_map,
        "G_initial_gain": identity_gains,
        "G_initial_gain_flags": np.zeros(spec.shape[:-1], dtype=np.int8),
        "G_initial_params": np.full(spec.pshape, param_fill),
        "G_initial_param_flags": np.zeros(spec.pshape[:-1], dtype=np.int8),
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

    spec, ms_kwargs, term_kwargs = synthetic_inputs(
        n_param, IDENTITY_PARAM[term_type]
    )
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


# --------------------------------scaffold flags-------------------------------

# The antenna which the scaffold flags for every interval, as it does for an
# antenna missing from the loaded solutions.
FLAGGED_ANTENNA = 2

# Any path will do - init_term only checks that load_from is set.
LOAD_PATH = "loads.qc/G"


def antenna_flags(shape):
    """Flags in which a single antenna is flagged for every interval."""

    flags = np.zeros(shape[:-1], dtype=np.int8)
    flags[:, :, FLAGGED_ANTENNA] = 1

    return flags


@pytest.fixture(
    params=[
        ("complex", None),
        ("complex", LOAD_PATH),
        ("delay", None),
        ("delay", LOAD_PATH),
    ],
    ids=lambda case: f"{case[0]}-loaded{case[1] is not None:d}",
    scope="module",
)
def flagged_scaffold_case(request):
    return request.param


@pytest.fixture(scope="module")
def flagged_scaffold_init_term_output(flagged_scaffold_case):
    """Drive init_term with scaffold flags raised for a single antenna."""

    term_type, load_from = flagged_scaffold_case
    term_class = TERM_TYPES[term_type]
    term = term_class("G", term_options(term_type, False, load_from))

    if term.is_parameterized:
        n_param = len(term_class.make_param_names(CORRELATIONS))
    else:
        n_param = 1  # Unused - unparameterised terms have no parameters.

    spec, ms_kwargs, term_kwargs = synthetic_inputs(n_param)

    term_kwargs["G_initial_gain_flags"] = antenna_flags(spec.shape)
    term_kwargs["G_initial_param_flags"] = antenna_flags(spec.pshape)

    return term.init_term(spec, 0, ms_kwargs, term_kwargs)


def test_scaffold_gain_flags_merged(flagged_scaffold_init_term_output):
    """An antenna flagged by the scaffold is flagged for every interval."""

    _, gain_flags, *_ = flagged_scaffold_init_term_output

    assert gain_flags[:, :, FLAGGED_ANTENNA].all()


def test_scaffold_gain_flags_are_additional(
    flagged_scaffold_init_term_output
):
    """The scaffold flags do not flag intervals which the data supports."""

    _, gain_flags, *_ = flagged_scaffold_init_term_output

    remaining = np.delete(gain_flags, FLAGGED_ANTENNA, axis=2)

    assert not remaining[1 - FLAGGED_TIME].any()


def test_scaffold_param_flags_merged(flagged_scaffold_init_term_output):
    """An antenna flagged by the scaffold is flagged on the parameter grid."""

    if len(flagged_scaffold_init_term_output) == 2:
        pytest.skip("Term is not parameterised.")

    *_, param_flags = flagged_scaffold_init_term_output

    assert param_flags[:, :, FLAGGED_ANTENNA].all()


# -------------------------------scaffold values-------------------------------

# Term type -> a scaffold value which differs from the identity. The terms are
# ones without an initial estimate, whose starting point is the scaffold as is.
SCAFFOLD_VALUE = {
    "complex": 2.0,
    "amplitude": 0.5,
    "phase": 0.25,
}


@pytest.fixture(params=SCAFFOLD_VALUE.keys(), scope="module")
def scaffold_value_term_type(request):
    return request.param


@pytest.fixture(scope="module")
def scaffold_value_init_term_output(scaffold_value_term_type):
    """Drive an unloaded term's init_term from non-identity scaffold values."""

    term_type = scaffold_value_term_type
    term_class = TERM_TYPES[term_type]
    term = term_class("G", term_options(term_type, False))

    if term.is_parameterized:
        n_param = len(term_class.make_param_names(CORRELATIONS))
    else:
        n_param = 1  # Unused - unparameterised terms have no parameters.

    value = SCAFFOLD_VALUE[term_type]

    spec, ms_kwargs, term_kwargs = synthetic_inputs(
        n_param, value, per_channel_params=True
    )

    term_kwargs["G_initial_gain"] *= value

    return term.init_term(spec, 0, ms_kwargs, term_kwargs)


def test_init_term_starts_from_scaffold(
    scaffold_value_init_term_output,
    scaffold_value_term_type
):
    """An unloaded term starts from its scaffold values where unflagged."""

    value = SCAFFOLD_VALUE[scaffold_value_term_type]

    if len(scaffold_value_init_term_output) == 2:
        values, flags = scaffold_value_init_term_output
        expected = value * np.array([1, 0, 0, 1])
    else:
        _, _, values, flags = scaffold_value_init_term_output
        expected = value

    unflagged = values[~flags.astype(bool)]

    assert unflagged.size
    np.testing.assert_array_equal(
        unflagged, np.broadcast_to(expected, unflagged.shape)
    )


# ---------------------------parameter identity fill---------------------------

# Every parameterised term except parallactic_angle, whose init_term overwrites
# the parameters with angles computed via casacore measures.
PARAMETERISED_TERM_TYPES = [
    term_type for term_type, term_class in TERM_TYPES.items()
    if term_class.is_parameterized and term_type != "parallactic_angle"
]


@pytest.mark.parametrize("term_type", PARAMETERISED_TERM_TYPES)
def test_param_identity_fill_yields_identity_gains(term_type):
    """A term's declared parameter identity produces identity gains."""

    term_class = TERM_TYPES[term_type]
    n_param = len(term_class.make_param_names(CORRELATIONS))

    spec, ms_kwargs, term_kwargs = synthetic_inputs(
        n_param, term_class.param_identity_fill, per_channel_params=True
    )
    term = term_class("G", term_options(term_type, False))

    gains, *_ = term.init_term(spec, 0, ms_kwargs, term_kwargs)

    np.testing.assert_allclose(
        gains, np.broadcast_to([1, 0, 0, 1], gains.shape), atol=1e-12
    )
