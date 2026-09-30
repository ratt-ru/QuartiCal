# -*- coding: utf-8 -*-
"""Check the gain dataset scaffolds which make_gain_xds_lod produces.

Every scaffold carries identity values and unraised flags. These are the
starting point of every solve, and the arrays which interpolation updates when
a term is loaded from disk, so every scaffold has the same variables whether or
not its term is loaded.
"""
from copy import deepcopy

import numpy as np
import pytest
from dask.core import get_dependencies

from quartical.config.internal import gains_to_chain
from quartical.gains.datasets import make_gain_xds_lod

# Term type -> the parameter value which produces an identity gain, or None for
# a term without parameters.
SCAFFOLD_PARAM_IDENTITY = {
    "complex": None,
    "diag_complex": None,
    "amplitude": 1.0,
    "phase": 0.0,
    "delay": 0.0,
    "rotation": 0.0,
}

# The test MS has four correlations, for which the identity is a 2x2 unit
# matrix.
IDENTITY_GAIN = np.array([1, 0, 0, 1], dtype=np.complex128)


@pytest.fixture(scope="module")
def opts(base_opts):

    _opts = deepcopy(base_opts)

    # A short time chunk gives each scaffold several chunks to compare.
    _opts.input_ms.time_chunk = 10

    return _opts


@pytest.fixture(
    scope="module",
    params=SCAFFOLD_PARAM_IDENTITY.keys()
)
def term_type(request):
    return request.param


@pytest.fixture(scope="module")
def scaffold_xds_list(predicted_xds_list, opts, term_type):

    _opts = deepcopy(opts)
    _opts.solver.terms = ["G"]
    _opts.G.type = term_type

    gain_xds_lod = make_gain_xds_lod(predicted_xds_list, gains_to_chain(_opts))

    return [gain_xds_dict["G"] for gain_xds_dict in gain_xds_lod]


def scaffold_fields(xds):
    """The (data variable, spec attribute) pairs a scaffold carries."""

    fields = [("gains", "GAIN_SPEC"), ("gain_flags", "GAIN_SPEC")]

    if hasattr(xds, "PARAM_SPEC"):
        fields += [("params", "PARAM_SPEC"), ("param_flags", "PARAM_SPEC")]

    return fields


def test_scaffold_gains_hold_identity(scaffold_xds_list):
    """Every scaffold gain is the identity."""

    for xds in scaffold_xds_list:
        gains = xds.gains.values
        np.testing.assert_array_equal(
            gains, np.broadcast_to(IDENTITY_GAIN, gains.shape)
        )


def test_scaffold_params_hold_identity(scaffold_xds_list, term_type):
    """Every scaffold parameter produces an identity gain."""

    if SCAFFOLD_PARAM_IDENTITY[term_type] is None:
        pytest.skip("Term is not parameterised.")

    for xds in scaffold_xds_list:
        np.testing.assert_array_equal(
            xds.params.values, SCAFFOLD_PARAM_IDENTITY[term_type]
        )


def test_scaffold_flags_unraised(scaffold_xds_list):
    """No scaffold flag is raised - flags come from the data or the loads."""

    for xds in scaffold_xds_list:
        for field, _ in scaffold_fields(xds):
            if field.endswith("flags"):
                assert not xds[field].values.any()


def test_scaffold_chunks_match_specs(scaffold_xds_list):
    """Scaffold chunks follow the specs which the solver is blocked on."""

    for xds in scaffold_xds_list:
        for field, spec_name in scaffold_fields(xds):
            spec = tuple(getattr(xds, spec_name))
            n_dims = xds[field].ndim
            assert xds[field].data.chunks == spec[:n_dims]


@pytest.fixture(scope="module")
def chain_scaffold_xds_list(predicted_xds_list, opts):
    """Scaffolds for the default two-term chain, whose terms share a shape."""

    gain_xds_lod = make_gain_xds_lod(predicted_xds_list, gains_to_chain(opts))

    return [xds for gain_xds_dict in gain_xds_lod
            for xds in gain_xds_dict.values()]


def test_scaffold_chunks_share_no_tasks(chain_scaffold_xds_list):
    """No two scaffold chunks share a task, within or across scaffolds.

    The scheduler plugin pins each chunk's subtree to a worker by grouping the
    tasks which share root ancestors. A root shared by several chunks - of one
    scaffold, of several terms or of several datasets - would merge their
    otherwise independent subtrees into one group.
    """

    seen = set()

    for xds in chain_scaffold_xds_list:
        for field, _ in scaffold_fields(xds):
            array = xds[field].data
            graph = dict(array.__dask_graph__())

            for key in flatten_keys(array.__dask_keys__()):
                ancestors = ancestors_of(graph, key)
                assert not ancestors & seen
                seen |= ancestors


def flatten_keys(keys):
    """Flatten the nested lists returned by __dask_keys__."""

    if isinstance(keys, list):
        return [k for sub in keys for k in flatten_keys(sub)]

    return [keys]


def ancestors_of(graph, key):
    """The set containing key and every task on which it depends."""

    ancestors = set()
    stack = [key]

    while stack:
        current = stack.pop()
        if current in ancestors:
            continue
        ancestors.add(current)
        stack.extend(get_dependencies(graph, current))

    return ancestors
