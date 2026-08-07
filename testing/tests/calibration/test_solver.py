from copy import deepcopy
import numpy as np
import pytest


@pytest.fixture(scope="module", params=["parallactic_angle", "feed_flip"])
def unsolvable_type(request):
    # The term types with no solver. For these, solver_wrapper allocates jhj
    # itself rather than receiving it from a solver.
    return request.param


@pytest.fixture(scope="module")
def opts(base_opts, unsolvable_type):

    # Don't overwrite base config - instead create a copy and update.

    _opts = deepcopy(base_opts)

    _opts.input_model.apply_p_jones = False
    _opts.output.apply_p_jones_inv = False
    _opts.solver.terms = ['G', 'B']
    _opts.solver.iter_recipe = [0, 0]
    _opts.solver.threads = 2

    # Placing the unsolvable term ahead of another term is what makes the
    # active term differ from the last term in the chain.
    _opts.G.type = unsolvable_type
    _opts.B.type = "complex"

    return _opts


@pytest.fixture(scope="module")
def raw_xds_list(read_xds_list_output):
    # Only use the first xds. This overloads the global fixture.
    return read_xds_list_output[0][:1]


# -----------------------------------------------------------------------------

@pytest.mark.parametrize("term_name", ["G", "B"])
def test_jhj_matches_term_spec(cmp_gain_xds_lod, term_name):
    """Each term's jhj is sized from that term's own spec.

    Sizing it from the wrong term's spec goes unnoticed for the last term in
    the chain and for single-term chains, hence the two-term chain whose first
    term is the unsolvable one.
    """

    for term_dict in cmp_gain_xds_lod:
        term_xds = term_dict[term_name]
        reference = term_xds.params if "params" in term_xds else term_xds.gains

        assert term_xds.jhj.shape == reference.shape


@pytest.mark.parametrize("term_name", ["G", "B"])
def test_jhj_dtype_matches_term_spec(cmp_gain_xds_lod, term_name):
    """Each term's jhj carries the dtype declared for it.

    A parameterised term's jhj is real and an unparameterised term's is
    complex, mirroring the two output declarations in
    calibration/constructor.py.
    """

    for term_dict in cmp_gain_xds_lod:
        term_xds = term_dict[term_name]
        expected = np.float64 if "params" in term_xds else np.complex128

        assert term_xds.jhj.dtype == expected
