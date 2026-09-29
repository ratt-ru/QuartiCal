from copy import deepcopy
import numpy as np
import pytest

from quartical.calibration.constructor import term_spec_tup
from quartical.calibration.solver import get_collapsed_inputs
from quartical.gains.gain import Gain


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


# -----------------------------------------------------------------------------

def collapse_inputs(n_term, active_term, n_model_dir, direction_dependent):
    """Drive get_collapsed_inputs over a fabricated chain of complex terms.

    Args:
        n_term: Number of terms in the chain.
        active_term: Index of the term being solved.
        n_model_dir: Number of directions in the model.
        direction_dependent: Whether every term is direction dependent.

    Returns:
        The mapping_inputs and chain_inputs namedtuples for the collapsed
        chain.
    """

    n_row, n_ant, n_corr = 10, 3, 2
    n_time, n_freq = 5, 4
    n_gain_dir = n_model_dir if direction_dependent else 1

    gain_shape = (n_time, n_freq, n_ant, n_gain_dir, n_corr)
    term_spec_list = [
        term_spec_tup(f"T{i}", "complex", gain_shape, ())
        for i in range(n_term)
    ]

    time_bins = np.arange(n_time, dtype=np.int32)
    time_map = np.zeros(n_row, dtype=np.int32)
    freq_map = np.arange(n_freq, dtype=np.int32)
    dir_map = Gain._make_dir_map(n_model_dir, direction_dependent)

    mapping_kwargs = {
        "time_bins": (time_bins,) * n_term,
        "time_maps": (time_map,) * n_term,
        "freq_maps": (freq_map,) * n_term,
        "dir_maps": (dir_map,) * n_term,
        "param_time_bins": (time_bins,) * n_term,
        "param_time_maps": (time_map,) * n_term,
        "param_freq_maps": (freq_map,) * n_term,
    }

    gain = np.ones(gain_shape, dtype=np.complex128)
    flag = np.zeros(gain_shape[:-1], dtype=np.int8)

    chain_kwargs = {
        "gains": (gain,) * n_term,
        "gain_flags": (flag,) * n_term,
        "params": (np.zeros(gain_shape[:-1] + (1,)),) * n_term,
        "param_flags": (flag,) * n_term,
    }

    ms_kwargs = {"TIME": np.linspace(0, 1, n_row)}

    # get_collapsed_inputs reads nothing from the chain elements but the two
    # namedtuple types, which are class attributes on Gain.
    chain = [Gain] * n_term

    mapping_inputs, chain_inputs, _ = get_collapsed_inputs(
        ms_kwargs,
        mapping_kwargs,
        chain_kwargs,
        term_spec_list,
        chain,
        active_term
    )

    return mapping_inputs, chain_inputs


@pytest.mark.parametrize(
    "n_term, active_term", [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]
)
@pytest.mark.parametrize("n_model_dir", [1, 2, 3])
@pytest.mark.parametrize("direction_dependent", [False, True])
def test_collapsed_dir_maps_index_gains(
    n_term, active_term, n_model_dir, direction_dependent
):
    """Every collapsed direction map spans the model and indexes its gain.

    The solver kernels loop over the model's directions and use each map to
    look up the direction of the corresponding gain. A map sized from the gains
    rather than the model is shorter than that loop whenever no term is
    direction dependent, and the kernels then index off its end.
    """

    mapping_inputs, chain_inputs = collapse_inputs(
        n_term, active_term, n_model_dir, direction_dependent
    )

    for dir_map, gain in zip(mapping_inputs.dir_maps, chain_inputs.gains):
        assert dir_map.size == n_model_dir
        assert dir_map.max() < gain.shape[3]
