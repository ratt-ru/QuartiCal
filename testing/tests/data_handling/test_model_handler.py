import pytest
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
def raw_model_recipe(model_expectations):
    return model_expectations[0]


@pytest.fixture(scope="module")
def expected_ndir(model_expectations):
    return model_expectations[1]


@pytest.fixture(scope="module")
def opts(base_opts, freq_chunk, time_chunk, raw_model_recipe):

    # Don't overwrite base config - instead create a copy and update.

    _opts = deepcopy(base_opts)

    _opts.input_ms.freq_chunk = freq_chunk
    _opts.input_ms.time_chunk = time_chunk
    _opts.input_model.recipe = raw_model_recipe

    return _opts


# ------------------------------add_model_graph--------------------------------


@pytest.mark.model_handler
def test_assigned_model(predicted_xds_list):
    assert all(hasattr(xds, "MODEL_DATA") for xds in predicted_xds_list)


@pytest.mark.model_handler
def test_model_shape(predicted_xds_list, expected_ndir):
    assert all(xds.sizes["dir"] == expected_ndir for xds in predicted_xds_list)

# -----------------------------------------------------------------------------
