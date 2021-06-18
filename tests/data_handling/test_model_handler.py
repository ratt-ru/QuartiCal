import pytest
from quartical.data_handling.ms_handler import read_xds_list
from quartical.config.preprocess import interpret_model
from quartical.data_handling.model_handler import add_model_graph
from copy import deepcopy


recipes = {"{col1}:{sky_model}@dE": 9,
           "{sky_model}:{sky_model}@dE": 9,
           "{col1}~{col2}": 1,
           "{col1}+{col2}": 1,
           "{sky_model}@dE~{col1}:{col2}": 9,
           "{col1}+{sky_model}@dE:{col2}": 9}


@pytest.fixture(params=recipes.items(),
                scope="module",
                ids=recipes.keys())
def model_expectations(request, lsm_name):

    recipe = request.param[0].format(sky_model=lsm_name,
                                     col1="DATA",
                                     col2="MODEL_DATA")

    expected_ndir = request.param[1]

    return recipe, expected_ndir


@pytest.fixture(scope="module")
def model_recipe(model_expectations):
    return model_expectations[0]


@pytest.fixture(scope="module")
def expected_ndir(model_expectations):
    return model_expectations[1]


@pytest.fixture(scope="module")
def opts(base_opts, freq_chunk, time_chunk, model_recipe):

    _opts = deepcopy(base_opts)

    _opts.input_ms.freq_chunk = freq_chunk
    _opts.input_ms.time_chunk = time_chunk
    _opts.input_model.recipe = model_recipe

    interpret_model(_opts)

    return _opts


@pytest.fixture(scope="module")
def _add_model_graph(opts):

    ms_xds_list, _ = read_xds_list(opts)

    return add_model_graph(ms_xds_list, opts)


# ------------------------------add_model_graph--------------------------------


@pytest.mark.model_handler
def test_assigned_model(_add_model_graph):

    model_xds_list = _add_model_graph

    assert all(hasattr(xds, "MODEL_DATA") for xds in model_xds_list)


@pytest.mark.model_handler
def test_model_shape(_add_model_graph, expected_ndir):

    model_xds_list = _add_model_graph

    assert all(xds.sizes["dir"] == expected_ndir for xds in model_xds_list)

# -----------------------------------------------------------------------------
