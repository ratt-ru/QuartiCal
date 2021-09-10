import pytest
from quartical.data_handling.ms_handler import read_xds_list


@pytest.fixture(scope="module")
def read_xds_list_output(ms_opts, recipe):
    return read_xds_list(recipe.ingredients.model_columns, ms_opts)


@pytest.fixture(scope="module")
def data_xds_list(read_xds_list_output):
    return read_xds_list_output[0]


@pytest.fixture(scope="module")
def ref_xds_list(read_xds_list_output):
    return read_xds_list_output[1]
