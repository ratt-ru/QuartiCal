import pytest
from cubicalv2.data_handling.ms_handler import read_ms
from cubicalv2.parser.preprocess import interpret_model
from cubicalv2.data_handling.predict import (predict,
                                             parse_sky_models,
                                             daskify_sky_model_dict)
from argparse import Namespace


@pytest.fixture(scope="module")
def opts(base_opts, freq_chunk, time_chunk, correlation_mode, model_recipe):

    # Don't overwrite base config - instead create a new Namespace and update.

    options = Namespace(**vars(base_opts))

    options.input_ms_freq_chunk = freq_chunk
    options.input_ms_time_chunk = time_chunk
    options.input_ms_correlation_mode = correlation_mode
    options.input_model_recipe = model_recipe

    interpret_model(options)

    return options


@pytest.fixture(scope="module")
def _predict(opts):

    ms_xds_list, col_kwrds = read_ms(opts)

    return predict(ms_xds_list, opts)


@pytest.fixture(scope="module")
def _di_sky_dict(lsm_name):

    options = Namespace(**{"input_model_recipe": lsm_name,
                           "input_model_source_chunks": 10})

    interpret_model(options)

    return parse_sky_models(options)


@pytest.fixture(scope="module")
def _dd_sky_dict(lsm_name):

    options = Namespace(**{"input_model_recipe": lsm_name + "@dE",
                           "input_model_source_chunks": 10})

    interpret_model(options)

    return parse_sky_models(options)


@pytest.fixture(scope="module")
def _dask_di_sky_dict(_di_sky_dict):

    return daskify_sky_model_dict(_di_sky_dict)


@pytest.fixture(scope="module")
def _dask_dd_sky_dict(_dd_sky_dict):

    return daskify_sky_model_dict(_dd_sky_dict)


# -----------------------------parse_sky_models--------------------------------


@pytest.mark.parametrize("field", ["radec", "stokes", "spi", "ref_freq"])
def test_expected_fields_points_di(_di_sky_dict, field):

    # Check that we have all the fields we expect.

    model = list(_di_sky_dict.keys())[0]

    assert field in _di_sky_dict[model]["DIE"]["point"]


@pytest.mark.parametrize("field", ["radec", "stokes", "spi", "ref_freq",
                                   "shape"])
def test_expected_fields_gauss_di(_di_sky_dict, field):

    # Check that we have all the fields we expect.

    model = list(_di_sky_dict.keys())[0]

    assert field in _di_sky_dict[model]["DIE"]["gauss"]


def test_npoint_di(_di_sky_dict):

    # Check for the expected number of point sources.

    model = list(_di_sky_dict.keys())[0]

    point = _di_sky_dict[model]["DIE"]["point"]

    expected_n_point = 30

    assert all([len(f) == expected_n_point for f in point.values()])


def test_ngauss_di(_di_sky_dict):

    # Check for the expected number of gaussian sources.

    model = list(_di_sky_dict.keys())[0]

    gauss = _di_sky_dict[model]["DIE"]["gauss"]

    expected_n_gauss = 30

    assert all([len(f) == expected_n_gauss for f in gauss.values()])


@pytest.mark.parametrize("field", ["radec", "stokes", "spi", "ref_freq"])
def test_expected_fields_points_dd(_dd_sky_dict, field):

    # Check that we have all the fields we expect.

    check = all([field in _dd_sky_dict[m][c]["point"]
                 for m in _dd_sky_dict.keys()
                 for c in _dd_sky_dict[m].keys()
                 if "point" in _dd_sky_dict[m][c]])

    assert check is True


@pytest.mark.parametrize("field", ["radec", "stokes", "spi", "ref_freq",
                                   "shape"])
def test_expected_fields_gauss_dd(_dd_sky_dict, field):

    # Check that we have all the fields we expect.

    check = all([field in _dd_sky_dict[m][c]["gauss"]
                 for m in _dd_sky_dict.keys()
                 for c in _dd_sky_dict[m].keys()
                 if "gauss" in _dd_sky_dict[m][c]])

    assert check is True


expected_clusters = {"DIE": {"n_point": 24, "n_gauss": 24},
                     "B290": {"n_point": 2, "n_gauss": 1},
                     "C242": {"n_point": 0, "n_gauss": 1},
                     "G195": {"n_point": 0, "n_gauss": 1},
                     "H194": {"n_point": 0, "n_gauss": 2},
                     "I215": {"n_point": 0, "n_gauss": 1},
                     "K285": {"n_point": 1, "n_gauss": 0},
                     "O265": {"n_point": 1, "n_gauss": 0},
                     "R283": {"n_point": 1, "n_gauss": 0},
                     "W317": {"n_point": 1, "n_gauss": 0}}


def test_npoint_dd(_dd_sky_dict):

    # Check for the expected number of point sources.

    check = True

    for sky_model_name, sky_model in _dd_sky_dict.items():
        for cluster_name, cluster in sky_model.items():
            if "point" in cluster:
                n_point = len(cluster["point"]["stokes"])
                expected_n_point = expected_clusters[cluster_name]["n_point"]
                check &= n_point == expected_n_point

    assert check is True


def test_ngauss_dd(_dd_sky_dict):

    # Check for the expected number of gaussian sources.

    check = True

    for sky_model_name, sky_model in _dd_sky_dict.items():
        for cluster_name, cluster in sky_model.items():
            if "gauss" in cluster:
                n_gauss = len(cluster["gauss"]["stokes"])
                expected_n_gauss = expected_clusters[cluster_name]["n_gauss"]
                check &= n_gauss == expected_n_gauss

    assert check is True


# -----------------------daskify_sky_model_dict--------------------------------
