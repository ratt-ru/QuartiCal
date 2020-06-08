import pytest
from cubicalv2.data_handling.ms_handler import read_ms
from cubicalv2.parser.preprocess import interpret_model
from cubicalv2.data_handling.model_handler import add_model_graph
from argparse import Namespace


recipes = {"{col1}:{sky_model}@dE": 9,
           "{sky_model}:{sky_model}@dE": 9,
           "{col1}~{col2}": 1,
           "{col1}+{col2}": 1,
           "{sky_model}@dE~{col1}:{col2}": 9,
           "{col1}+{sky_model}@dE:{col2}": 9}


@pytest.fixture(params=zip(recipes.keys(), recipes.values()), scope="module",
                ids=list(recipes.keys()))
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

    # Don't overwrite base config - instead create a new Namespace and update.

    options = Namespace(**vars(base_opts))

    options.input_ms_freq_chunk = freq_chunk
    options.input_ms_time_chunk = time_chunk
    options.input_ms_correlation_mode = "full"
    options.input_model_recipe = model_recipe

    interpret_model(options)

    return options


@pytest.fixture(scope="module")
def _add_model_graph(opts):

    ms_xds_list, col_kwrds = read_ms(opts)

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
