import pytest
from quartical.gains.datasets import (compute_interval_chunking,
                                      compute_dataset_coords)
from quartical.calibration.calibrate import add_calibration_graph
from quartical.calibration.mapping import (make_t_maps,
                                           make_f_maps,
                                           make_d_maps)


@pytest.fixture(scope="module")
def add_calibration_graph_outputs(predicted_xds_list, solver_opts, chain_opts,
                                  output_opts):
    return add_calibration_graph(predicted_xds_list, solver_opts, chain_opts,
                                 output_opts)


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
def tbin_list_tmap_list(predicted_xds_list, chain_opts):
    return make_t_maps(predicted_xds_list, chain_opts)


@pytest.fixture(scope="module")
def t_bin_list(tbin_list_tmap_list):
    return tbin_list_tmap_list[0]


@pytest.fixture(scope="module")
def t_map_list(tbin_list_tmap_list):
    return tbin_list_tmap_list[1]


@pytest.fixture(scope="module")
def f_map_list(predicted_xds_list, chain_opts):
    return make_f_maps(predicted_xds_list, chain_opts)


@pytest.fixture(scope="module")
def d_map_list(predicted_xds_list, chain_opts):
    return make_d_maps(predicted_xds_list, chain_opts)


@pytest.fixture(scope="module")
def compute_interval_chunking_outputs(predicted_xds_list,
                                      t_map_list,
                                      f_map_list):
    return compute_interval_chunking(predicted_xds_list,
                                     t_map_list,
                                     f_map_list)


@pytest.fixture(scope="module")
def tipc_list(compute_interval_chunking_outputs):
    return compute_interval_chunking_outputs[0]


@pytest.fixture(scope="module")
def fipc_list(compute_interval_chunking_outputs):
    return compute_interval_chunking_outputs[1]


@pytest.fixture(scope="module")
def coords_per_xds(predicted_xds_list,
                   t_bin_list,
                   f_map_list,
                   tipc_list,
                   fipc_list,
                   solver_opts):
    return compute_dataset_coords(
        predicted_xds_list,
        t_bin_list,
        f_map_list,
        tipc_list,
        fipc_list,
        solver_opts.terms
    )


@pytest.fixture(scope="module")
def term_xds_dict(gain_xds_lod):
    return gain_xds_lod[0]
