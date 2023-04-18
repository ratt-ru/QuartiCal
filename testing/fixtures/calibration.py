import pytest
from quartical.calibration.calibrate import add_calibration_graph
from quartical.calibration.mapping import make_mapping_datasets


@pytest.fixture(scope="module")
def mapping_xds_list(predicted_xds_list, chain):
    return make_mapping_datasets(predicted_xds_list, chain)


@pytest.fixture(scope="module")
def add_calibration_graph_outputs(predicted_xds_list, stats_xds_list,
                                  solver_opts, chain, output_opts):
    return add_calibration_graph(predicted_xds_list, stats_xds_list,
                                 solver_opts, chain, output_opts)


@pytest.fixture(scope="module")
def gain_xds_lod(add_calibration_graph_outputs):
    return add_calibration_graph_outputs[0]


@pytest.fixture(scope="module")
def net_gain_xds_list(add_calibration_graph_outputs):
    return add_calibration_graph_outputs[1]


@pytest.fixture(scope="module")
def post_cal_data_xds_list(add_calibration_graph_outputs):
    return add_calibration_graph_outputs[2]


@pytest.fixture(scope="module")
def term_xds_dict(gain_xds_lod):
    return gain_xds_lod[0]
