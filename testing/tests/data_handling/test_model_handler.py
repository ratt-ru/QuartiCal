import pytest
from quartical.data_handling.ms_handler import read_xds_list
from quartical.config.preprocess import transcribe_recipe
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
def model_opts(base_opts, model_recipe):
    model_opts = deepcopy(base_opts.input_model)

    model_opts.recipe = model_recipe

    return model_opts


@pytest.fixture(scope="module")
def ms_opts(base_opts, freq_chunk, time_chunk):

    ms_opts = deepcopy(base_opts.input_ms)

    ms_opts.freq_chunk = freq_chunk
    ms_opts.time_chunk = time_chunk

    return ms_opts


@pytest.fixture(scope="module")
def recipe(model_opts):
    return transcribe_recipe(model_opts.recipe)


@pytest.fixture(scope="module")
def xds_list(recipe, ms_opts):
    xds_list, _ = read_xds_list(recipe.ingredients.model_columns, ms_opts)
    return xds_list


@pytest.fixture(scope="module")
def predicted_xds_list(xds_list, recipe, ms_name, model_opts):
    return add_model_graph(xds_list, recipe, ms_name, model_opts)

# ------------------------------add_model_graph--------------------------------


@pytest.mark.model_handler
def test_assigned_model(predicted_xds_list):
    assert all(hasattr(xds, "MODEL_DATA") for xds in predicted_xds_list)


@pytest.mark.model_handler
def test_model_shape(predicted_xds_list, expected_ndir):
    assert all(xds.sizes["dir"] == expected_ndir for xds in predicted_xds_list)

# -----------------------------------------------------------------------------
