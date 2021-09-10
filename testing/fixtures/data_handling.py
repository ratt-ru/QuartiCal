import pytest
from quartical.data_handling.ms_handler import read_xds_list
from quartical.data_handling.model_handler import add_model_graph


# EXTERNAL FIXTURES:
#   ms_opts
#   recipe
#   ms_name
#   model_opts

@pytest.fixture(scope="module")
def read_xds_list_output(ms_opts, recipe):
    return read_xds_list(recipe.ingredients.model_columns, ms_opts)


@pytest.fixture(scope="module")
def data_xds_list(read_xds_list_output):
    return read_xds_list_output[0]


@pytest.fixture(scope="module")
def ref_xds_list(read_xds_list_output):
    return read_xds_list_output[1]


@pytest.fixture(scope="module")
def predicted_xds_list(data_xds_list, recipe, ms_name, model_opts):
    return add_model_graph(data_xds_list, recipe, ms_name, model_opts)


@pytest.fixture(scope="module")
def data_xds_list_w_model_col(ms_opts):
    return read_xds_list(["MODEL_DATA"], ms_opts)[0]
