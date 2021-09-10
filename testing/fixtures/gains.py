import pytest
import dask


@pytest.fixture(scope="module")
def cmp_calibration_graph_outputs(add_calibration_graph_outputs):
    return dask.compute(*add_calibration_graph_outputs)


@pytest.fixture(scope="module")
def cmp_gain_xds_lod(cmp_calibration_graph_outputs):
    return cmp_calibration_graph_outputs[0]


@pytest.fixture(scope="module")
def cmp_net_xds_list(cmp_calibration_graph_outputs):
    return cmp_calibration_graph_outputs[1]


@pytest.fixture(scope="module")
def cmp_post_solve_data_xds_list(cmp_calibration_graph_outputs):
    return cmp_calibration_graph_outputs[2]


@pytest.fixture(scope="module")
def cmp_residuals(post_solve_data_xds_list):
    return [xds._RESIDUALS.data for xds in post_solve_data_xds_list]
