"""The stored and solver parameter bases must describe the same gains.

The delay/tec family solves in a rescaled parameter basis: a delay is scaled by
the band midpoint and a TEC by the bandwidth, which keeps the parameters and
their derivatives at comparable magnitudes. ``pre_solve`` enters that basis and
``post_solve`` leaves it, so the parameters written to disk are always in
native units.

That rescaling is a change of *units* and nothing else. In particular it does
not move the reference point of a term's frequency-dependent coefficients -
subtracting the band's mean frequency from a delay's coefficient, or the band's
mean of ``1/nu`` from a TEC's, is what decorrelates those parameters from the
offset, and it belongs to the term's model. ``params_to_gains`` therefore
carries it in both of its ``rescaled`` modes, and the two modes describe
identical gains.

These tests drive each kernel's own ``pre_solve`` and ``post_solve`` hooks from
a jitted caller, as the solver loop does, so they constrain the statements
inside them and not merely the arithmetic of the rescaling. Both oracles are
``params_to_gains``, so a term whose native and rescaled maps drift apart fails
here rather than passing against a hand-written formula that encodes one of the
two.
"""
from collections import namedtuple

import numpy as np
import pytest
from numba import njit

from quartical.gains.delay.kernel import (
    delay_params_to_gains,
    post_solve as delay_post_solve,
    pre_solve as delay_pre_solve
)
from quartical.gains.delay_and_offset.kernel import (
    delay_and_offset_params_to_gains,
    post_solve as delay_and_offset_post_solve,
    pre_solve as delay_and_offset_pre_solve
)
from quartical.gains.delay_and_tec.kernel import (
    delay_and_tec_params_to_gains,
    post_solve as delay_and_tec_post_solve,
    pre_solve as delay_and_tec_pre_solve
)
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
# band, and an offset of the same order. A rescaling applied to the wrong
# parameter slot, or in the wrong direction, is wrong by a factor of the
# bandwidth (TEC) or the band midpoint (delay), so realistic magnitudes make
# the difference unmistakable.
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

# Per term: the native parameter values of one correlation, the term's
# pre/post-solve hooks, and its parameters-to-gains routine. The values follow
# each term's own slot order, fixed by its make_param_names: an offset first
# where the term has one, then a TEC, then a delay.
TERMS = {
    "delay": (
        (DELAY,),
        delay_pre_solve,
        delay_post_solve,
        delay_params_to_gains
    ),
    "delay_and_offset": (
        (OFFSET, DELAY),
        delay_and_offset_pre_solve,
        delay_and_offset_post_solve,
        delay_and_offset_params_to_gains
    ),
    "delay_and_tec": (
        (TEC, DELAY),
        delay_and_tec_pre_solve,
        delay_and_tec_post_solve,
        delay_and_tec_params_to_gains
    ),
    "tec_and_offset": (
        (OFFSET, TEC),
        tec_and_offset_pre_solve,
        tec_and_offset_post_solve,
        tec_and_offset_params_to_gains
    ),
    "delay_tec_and_offset": (
        (OFFSET, TEC, DELAY),
        delay_tec_and_offset_pre_solve,
        delay_tec_and_offset_post_solve,
        delay_tec_and_offset_params_to_gains
    )
}


def make_hook_drivers(pre_solve, post_solve):
    """Return jitted drivers for a term's pre-solve and post-solve hooks.

    The hooks are cached, so calling one from Python makes numba pickle the
    argument types - and with them the module declaring the namedtuples above -
    into the hook's on-disk cache index. Nothing on ``sys.path`` guarantees a
    test module, and the index unpickles as a whole, so a failure to import it
    makes every signature of that hook unloadable.

    Binding the hooks as freevars of an uncached jitted caller is how the
    solver loop reaches them: ``qcjit``'s ``inline="always"`` means they are
    never lowered as their own cache unit, and no cache index records anything
    about this module.
    """

    @njit
    def drive_pre_solve(ms_inputs, chain_inputs, meta_inputs):
        pre_solve(ms_inputs, chain_inputs, meta_inputs)

    @njit
    def drive_round_trip(ms_inputs, chain_inputs, meta_inputs, native_imdry):
        pre_solve(ms_inputs, chain_inputs, meta_inputs)
        post_solve(ms_inputs, chain_inputs, meta_inputs, native_imdry)

    return drive_pre_solve, drive_round_trip


# Built at import time, which costs nothing - numba compiles a driver on its
# first call - so each term's hooks are compiled once for the whole module.
DRIVERS = {
    name: make_hook_drivers(pre_solve, post_solve)
    for name, (_, pre_solve, post_solve, _) in TERMS.items()
}


def make_params(native_corr):
    """Return native parameters which differ per antenna and per correlation.

    The second correlation is negated so that a rescaling applied to the wrong
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


def make_gains(params_to_gains, params, rescaled):
    """Return the gains a parameter array implies in the requested basis."""

    gains = np.zeros((1, N_CHAN, N_ANT, 1, N_CORR), dtype=np.complex128)

    params_to_gains(
        params,
        gains,
        CHAN_FREQ,
        MIN_FREQ,
        MAX_FREQ,
        np.zeros(N_CHAN, dtype=np.int64),
        rescaled=rescaled
    )

    return gains


@pytest.fixture(params=list(TERMS), ids=list(TERMS))
def term(request):
    return request.param


def test_pre_solve_preserves_gains(term):
    """pre_solve must not change the gains a solution describes.

    This is the contract which makes the rescaling a change of units: the
    parameters handed to the solver, read through params_to_gains in the
    rescaled basis, have to describe the same gains as the native parameters
    they came from read in the native basis.
    """

    native_corr, _, _, params_to_gains = TERMS[term]
    drive_pre_solve, _ = DRIVERS[term]

    params = make_params(native_corr)
    native_gains = make_gains(params_to_gains, params.copy(), False)

    ms_inputs, chain_inputs, meta_inputs, _ = make_hook_inputs(params)
    drive_pre_solve(ms_inputs, chain_inputs, meta_inputs)

    solver_gains = make_gains(params_to_gains, params, True)

    assert np.any(native_gains[..., (0, -1)]), "All gains are zero!"
    np.testing.assert_allclose(
        solver_gains[..., (0, -1)], native_gains[..., (0, -1)], rtol=1e-10
    )


def test_pre_solve_post_solve_round_trip(term):
    """The two hooks must compose to the identity on the parameters.

    A solve which does no iterations runs pre_solve and post_solve and nothing
    between them, so it has to return the parameters it was given.
    """

    native_corr, *_ = TERMS[term]
    _, drive_round_trip = DRIVERS[term]

    params = make_params(native_corr)
    native_params = params.copy()

    ms_inputs, chain_inputs, meta_inputs, native_imdry = \
        make_hook_inputs(params)

    drive_round_trip(ms_inputs, chain_inputs, meta_inputs, native_imdry)

    np.testing.assert_allclose(params, native_params, rtol=1e-12, atol=1e-12)
