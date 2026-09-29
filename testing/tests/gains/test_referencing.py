# -*- coding: utf-8 -*-
"""Check that the per-term ``referenced`` option gates the referencing stage.

Referencing right-multiplies every antenna's gain by a constant unitary built
from the reference antenna, which drives the reference antenna's diagonal
phases (equivalently, its parameters) to zero. Turning the option off leaves
that gauge wherever the solve happened to land, which for real data is not
zero. The two cases are therefore distinguishable by looking at the reference
antenna alone, with no reference to the truth.
"""
from copy import deepcopy

import numpy as np
import pytest

from quartical.config import Gain as GainConfig
from quartical.gains import TERM_TYPES

# The reference antenna the solve is told to use. Antenna zero is the schema
# default, but the test states it so the assertions have something to index.
REFERENCE_ANTENNA = 0

# Terms whose referencing acts on parameters rather than on gains. The rest of
# the terms exercised here are referenced in gain space.
PARAM_REFERENCED_TERMS = {"phase"}

# Below this, the reference antenna's gauge is pinned; above it, it is free.
# Solved phases on real data are O(1) radians, so the gap is enormous and the
# exact threshold is not delicate.
PINNED_ATOL = 1e-6
FREE_GAUGE_FLOOR = 1e-2


@pytest.fixture(params=["complex", "diag_complex", "phase"], scope="module")
def term_type(request):
    return request.param


@pytest.fixture(params=[True, False], scope="module")
def referenced(request):
    return request.param


@pytest.fixture(scope="module")
def opts(base_opts, term_type, referenced):

    # Don't overwrite base config - instead create a copy and update.

    _opts = deepcopy(base_opts)

    _opts.solver.terms = ['G']
    _opts.solver.iter_recipe = [25]
    _opts.solver.propagate_flags = False
    _opts.solver.reference_antenna = REFERENCE_ANTENNA
    _opts.solver.threads = 2
    _opts.G.type = term_type
    _opts.G.referenced = referenced

    return _opts


@pytest.fixture(scope="module")
def raw_xds_list(read_xds_list_output):
    # Only use the first xds. This overloads the global fixture.
    return read_xds_list_output[0][:1]


def reference_antenna_gauge(gain_xds, term_type):
    """Return the quantity which referencing drives to zero.

    Args:
        gain_xds: A solved gain xarray.Dataset.
        term_type: The name of the solved term.

    Returns:
        An array of the reference antenna's referenceable degrees of freedom -
        its parameters for a parameterised term, otherwise the phases of its
        diagonal gain elements.
    """

    if term_type in PARAM_REFERENCED_TERMS:
        return np.asarray(gain_xds.params.data)[:, :, REFERENCE_ANTENNA]

    gains = np.asarray(gain_xds.gains.data)

    # A four-correlation gain stores its diagonal at 0 and 3; a one or two
    # correlation gain is diagonal throughout.
    diagonal = slice(None, None, 3) if gains.shape[-1] == 4 else slice(None)

    return np.angle(gains[:, :, REFERENCE_ANTENNA, :, diagonal])


# -----------------------------------------------------------------------------

@pytest.mark.parametrize("referenced_opt", [True, False])
@pytest.mark.parametrize("registered_term_type", sorted(TERM_TYPES))
def test_every_term_accepts_referenced(registered_term_type, referenced_opt):
    # A term with no referencing stage resolves the hook to a build-time no-op,
    # so it must take the option without complaint rather than reject it.
    term_opts = GainConfig(
        type=registered_term_type,
        referenced=referenced_opt,
        # Crosshand phase is only ever solved over the whole array, and says so
        # from its own post-init rather than tolerating the default.
        solve_per=(
            "array" if registered_term_type == "crosshand_phase" else "antenna"
        ),
    )

    term = TERM_TYPES[registered_term_type]("G", term_opts)

    assert term.referenced is referenced_opt


def test_referenced_reaches_the_gain_object(chain, referenced):
    # The option has to survive the trip from config to instantiated term.
    (term,) = chain
    assert term.referenced is referenced


def test_referenced_gates_reference_antenna_pinning(
    cmp_gain_xds_lod, term_type, referenced
):
    # With referencing on, the reference antenna's gauge is pinned to zero;
    # with it off, the solve leaves it wherever it landed.
    for gain_dict in cmp_gain_xds_lod:

        gauge = reference_antenna_gauge(gain_dict["G"], term_type)

        if referenced:
            np.testing.assert_allclose(gauge, 0, atol=PINNED_ATOL)
        else:
            assert np.max(np.abs(gauge)) > FREE_GAUGE_FLOOR
