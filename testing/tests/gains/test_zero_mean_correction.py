"""The zero-mean correction bridges the stored and solver parameter bases.

QuartiCal stores the parameters of the TEC-like terms in a plain basis, in
which a term's phase is its offset plus its frequency-dependent parts:

    phase(nu) = offset + 2 pi TEC / nu + 2 pi delay nu.

The solvers instead work in a basis where each frequency-dependent part has
zero mean across the band, which lifts the near-degeneracy between the offset
and the rest. ``params_to_gains`` implements that basis in both of its
``rescaled`` modes, so a stored offset has to be shifted on the way into a
solve and shifted back on the way out. ``apply_zero_mean_correction`` is that
shift.

Its coefficients - ``mid_freq`` for a delay and
``log(nu_min/nu_max)/bandwidth`` for a TEC - are defined on parameters in
native units, which fixes where it sits relative to the solver's parameter
rescaling: after the unscaling in post_solve, and before the rescaling in
pre_solve.

These tests drive each kernel's own ``pre_solve`` and ``post_solve`` hooks, so
they constrain the order of the statements inside them and not merely the
arithmetic of the correction.
"""
from collections import namedtuple

import numpy as np
import pytest

from quartical.gains.delay_tec_and_offset.kernel import (
    delay_tec_and_offset_params_to_gains,
    post_solve as delay_tec_and_offset_post_solve,
    pre_solve as delay_tec_and_offset_pre_solve
)
from quartical.gains.tec_and_offset.kernel import (
    post_solve as tec_and_offset_post_solve,
    pre_solve as tec_and_offset_pre_solve,
    tec_and_offset_params_to_gains
)

N_ANT = 3
N_CHAN = 8
N_CORR = 4

CHAN_FREQ = np.linspace(8.5e8, 1.7e9, N_CHAN)
MIN_FREQ, MAX_FREQ = CHAN_FREQ[0], CHAN_FREQ[-1]

# A TEC and a delay which each contribute of order a radian of phase across the
# band. The correction applied to a rescaled rather than a native parameter is
# wrong by a factor of the bandwidth (TEC) or the band midpoint (delay), so
# realistic magnitudes make the difference unmistakable.
TEC = 1e9 / (2 * np.pi)
DELAY = 1e-9 / (2 * np.pi)
OFFSET = 0.5

# The hooks reach their inputs through the standardised solver-loop arguments.
# Only the fields below are touched, so stand-in namedtuples carrying just
# those are enough to call them; numba resolves the attribute access by name.
MsInputs = namedtuple("MsInputs", ("MIN_FREQ", "MAX_FREQ"))
ChainInputs = namedtuple("ChainInputs", ("params", "param_flags"))
MetaInputs = namedtuple("MetaInputs", ("active_term",))
NativeImdry = namedtuple("NativeImdry", ("jhj",))


def tec_and_offset_phase(p, chan_freq):
    """Stored-basis phase of an (offset, TEC) parameter pair."""

    return p[0] + 2 * np.pi * p[1] / chan_freq


def delay_tec_and_offset_phase(p, chan_freq):
    """Stored-basis phase of an (offset, TEC, delay) parameter triple."""

    return (
        p[0]
        + 2 * np.pi * p[1] / chan_freq
        + 2 * np.pi * p[2] * chan_freq
    )


# Per term: the native parameter values of one correlation, the phase they
# imply, the term's pre/post-solve hooks, and its parameters-to-gains routine.
TERMS = {
    "tec_and_offset": (
        (OFFSET, TEC),
        tec_and_offset_phase,
        tec_and_offset_pre_solve,
        tec_and_offset_post_solve,
        tec_and_offset_params_to_gains
    ),
    "delay_tec_and_offset": (
        (OFFSET, TEC, DELAY),
        delay_tec_and_offset_phase,
        delay_tec_and_offset_pre_solve,
        delay_tec_and_offset_post_solve,
        delay_tec_and_offset_params_to_gains
    )
}


def make_params(native_corr):
    """Return native parameters which differ per antenna and per correlation.

    The second correlation is negated so that a correction applied to the wrong
    parameter slot cannot pass by symmetry.
    """

    params_per_corr = len(native_corr)
    params = np.zeros((1, 1, N_ANT, 1, 2 * params_per_corr))

    for a in range(N_ANT):
        first = [(a + 1) * v for v in native_corr]
        params[0, 0, a, 0, :params_per_corr] = first
        params[0, 0, a, 0, params_per_corr:] = [-v for v in first]

    return params


def make_hook_inputs(params):
    """Return the standardised hook arguments wrapping a parameter array."""

    n_param = params.shape[-1]
    param_flags = np.zeros(params.shape[:-1], dtype=np.int8)
    jhj = np.zeros(params.shape + (n_param,))

    # params/param_flags are indexed by the active term, whereas jhj has
    # already been narrowed to it by the solver loop.
    return (
        MsInputs(MIN_FREQ, MAX_FREQ),
        ChainInputs((params,), (param_flags,)),
        MetaInputs(0),
        NativeImdry(jhj)
    )


def make_truth_gains(params, params_per_corr, phase):
    """Return the gains the native parameters imply in the stored basis."""

    gains = np.zeros((1, N_CHAN, N_ANT, 1, N_CORR), dtype=np.complex128)

    for a in range(N_ANT):
        for corr, start in ((0, 0), (-1, params_per_corr)):
            p = params[0, 0, a, 0, start:start + params_per_corr]
            gains[0, :, a, 0, corr] = np.exp(1j * phase(p, CHAN_FREQ))

    return gains


@pytest.fixture(params=list(TERMS), ids=list(TERMS))
def term(request):
    return request.param


def test_pre_solve_preserves_gains(term):
    """pre_solve must not change the gains a solution describes.

    This is the contract which fixes the correction's position: the parameters
    handed to the solver have to describe the same gains as the stored
    parameters they came from, read through params_to_gains in the solver's
    rescaled basis. It only holds if the correction sees native units.
    """

    native_corr, phase, pre_solve, _, params_to_gains = TERMS[term]
    params_per_corr = len(native_corr)

    params = make_params(native_corr)
    truth_gains = make_truth_gains(params, params_per_corr, phase)

    ms_inputs, chain_inputs, meta_inputs, _ = make_hook_inputs(params)
    pre_solve(ms_inputs, chain_inputs, meta_inputs)

    solver_gains = np.zeros_like(truth_gains)
    params_to_gains(
        params,
        solver_gains,
        CHAN_FREQ,
        MIN_FREQ,
        MAX_FREQ,
        np.zeros(N_CHAN, dtype=np.int64),
        rescaled=True
    )

    assert np.any(truth_gains[..., (0, -1)]), "All gains are zero!"
    np.testing.assert_allclose(
        solver_gains[..., (0, -1)], truth_gains[..., (0, -1)], rtol=1e-10
    )


def test_pre_solve_post_solve_round_trip(term):
    """The two hooks must compose to the identity on the parameters.

    A solve which does no iterations runs pre_solve and post_solve and nothing
    between them, so it has to return the parameters it was given.
    """

    native_corr, _, pre_solve, post_solve, _ = TERMS[term]

    params = make_params(native_corr)
    native_params = params.copy()

    ms_inputs, chain_inputs, meta_inputs, native_imdry = \
        make_hook_inputs(params)

    pre_solve(ms_inputs, chain_inputs, meta_inputs)
    post_solve(ms_inputs, chain_inputs, meta_inputs, native_imdry)

    np.testing.assert_allclose(params, native_params, rtol=1e-12, atol=1e-12)
