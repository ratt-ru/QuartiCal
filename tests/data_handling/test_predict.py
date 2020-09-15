import pytest
from quartical.data_handling.ms_handler import read_xds_list
from quartical.parser.preprocess import interpret_model
from quartical.data_handling.predict import (predict,
                                             parse_sky_models,
                                             daskify_sky_model_dict,
                                             get_support_tables)
from argparse import Namespace
import dask.array as da
import numpy as np
from numpy.testing import assert_array_almost_equal


expected_clusters = {"DIE": {"n_point": 22, "n_gauss": 24},
                     "B290": {"n_point": 1, "n_gauss": 2},
                     "C242": {"n_point": 0, "n_gauss": 1},
                     "G195": {"n_point": 0, "n_gauss": 1},
                     "H194": {"n_point": 0, "n_gauss": 2},
                     "I215": {"n_point": 0, "n_gauss": 1},
                     #  "K285": {"n_point": 1, "n_gauss": 0},  #noqa
                     #  "O265": {"n_point": 1, "n_gauss": 0},  #noqa
                     "R283": {"n_point": 1, "n_gauss": 0},
                     #  "W317": {"n_point": 1, "n_gauss": 0}}  #noqa
                     "V317": {"n_point": 0, "n_gauss": 1}}


@pytest.fixture(params=["", "@dE"], ids=["di", "dd"],
                scope="module")
def model_recipe(request, lsm_name):
    return lsm_name + request.param


@pytest.fixture(scope="module")
def opts(base_opts, freq_chunk, time_chunk, model_recipe, beam_name):

    # Don't overwrite base config - instead create a new Namespace and update.

    options = Namespace(**vars(base_opts))

    options.input_ms_freq_chunk = freq_chunk
    options.input_ms_time_chunk = time_chunk
    options.input_ms_correlation_mode = "full"
    options.input_model_recipe = model_recipe
    options.input_model_beam = \
        beam_name + "/JVLA-L-centred-$(corr)_$(reim).fits"
    options.input_model_beam_l_axis = "-X"
    options.input_model_beam_m_axis = "Y"

    interpret_model(options)

    return options


@pytest.fixture(scope="module")
def _predict(opts):

    # Forcefully add this to ensure that the comparison data is read.
    opts._model_columns = ["MODEL_DATA"]

    ms_xds_list, _, _ = read_xds_list(opts)

    return predict(ms_xds_list, opts), ms_xds_list


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

    options = Namespace(**{"input_model_source_chunks": 10})

    return daskify_sky_model_dict(_di_sky_dict, options)


@pytest.fixture(scope="module")
def _dask_dd_sky_dict(_dd_sky_dict):

    options = Namespace(**{"input_model_source_chunks": 10})

    return daskify_sky_model_dict(_dd_sky_dict, options)


@pytest.fixture(scope="module")
def _support_tables(base_opts):

    return get_support_tables(base_opts)


# -----------------------------parse_sky_models--------------------------------


@pytest.mark.predict
@pytest.mark.parametrize("field", ["radec", "stokes", "spi", "ref_freq"])
def test_expected_fields_points_di(_di_sky_dict, field):

    # Check that we have all the fields we expect.

    model = list(_di_sky_dict.keys())[0]

    assert field in _di_sky_dict[model]["DIE"]["point"]


@pytest.mark.predict
@pytest.mark.parametrize("field", ["radec", "stokes", "spi", "ref_freq",
                                   "shape"])
def test_expected_fields_gauss_di(_di_sky_dict, field):

    # Check that we have all the fields we expect.

    model = list(_di_sky_dict.keys())[0]

    assert field in _di_sky_dict[model]["DIE"]["gauss"]


@pytest.mark.predict
def test_npoint_di(_di_sky_dict):

    # Check for the expected number of point sources.

    model = list(_di_sky_dict.keys())[0]

    point = _di_sky_dict[model]["DIE"]["point"]

    expected_n_point = sum([s["n_point"] for s in expected_clusters.values()])

    assert all([len(f) == expected_n_point for f in point.values()])


@pytest.mark.predict
def test_ngauss_di(_di_sky_dict):

    # Check for the expected number of gaussian sources.

    model = list(_di_sky_dict.keys())[0]

    gauss = _di_sky_dict[model]["DIE"]["gauss"]

    expected_n_gauss = sum(s["n_gauss"] for s in expected_clusters.values())

    assert all([len(f) == expected_n_gauss for f in gauss.values()])


@pytest.mark.predict
@pytest.mark.parametrize("field", ["radec", "stokes", "spi", "ref_freq"])
def test_expected_fields_points_dd(_dd_sky_dict, field):

    # Check that we have all the fields we expect.

    check = all([field in _dd_sky_dict[m][c]["point"]
                 for m in _dd_sky_dict.keys()
                 for c in _dd_sky_dict[m].keys()
                 if "point" in _dd_sky_dict[m][c]])

    assert check is True


@pytest.mark.predict
@pytest.mark.parametrize("field", ["radec", "stokes", "spi", "ref_freq",
                                   "shape"])
def test_expected_fields_gauss_dd(_dd_sky_dict, field):

    # Check that we have all the fields we expect.

    check = all([field in _dd_sky_dict[m][c]["gauss"]
                 for m in _dd_sky_dict.keys()
                 for c in _dd_sky_dict[m].keys()
                 if "gauss" in _dd_sky_dict[m][c]])

    assert check is True


@pytest.mark.predict
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


@pytest.mark.predict
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

# -------------------------daskify_sky_model_dict------------------------------


@pytest.mark.predict
def test_chunking_di(_dask_di_sky_dict):

    # Check for consistent chunking.

    check = True

    for sky_model_name, sky_model in _dask_di_sky_dict.items():
        for cluster_name, cluster in sky_model.items():
            for source_type, sources in cluster.items():
                for arr in sources:
                    check &= all([c <= 10 for c in arr.chunks[0]])

    assert check is True


@pytest.mark.predict
def test_chunking_dd(_dask_dd_sky_dict):

    # Check for consistent chunking.

    check = True

    for sky_model_name, sky_model in _dask_dd_sky_dict.items():
        for cluster_name, cluster in sky_model.items():
            for source_type, sources in cluster.items():
                for arr in sources:
                    check &= all([c <= 10 for c in arr.chunks[0]])

    assert check is True

# ----------------------------get_support_tables-------------------------------


@pytest.mark.predict
@pytest.mark.parametrize("table", ["ANTENNA", "DATA_DESCRIPTION", "FIELD",
                                   "SPECTRAL_WINDOW", "POLARIZATION"])
def test_support_fields(_support_tables, table):

    # Check that we have all expected support tables.

    assert table in _support_tables


@pytest.mark.predict
@pytest.mark.parametrize("table", ["ANTENNA"])
def test_lazy_tables(_support_tables, table):

    # Check that the antenna table is lazily evaluated.

    assert all([isinstance(dvar.data, da.Array)
                for dvar in _support_tables[table][0].data_vars.values()])


@pytest.mark.predict
@pytest.mark.parametrize("table", ["DATA_DESCRIPTION", "FIELD",
                                   "SPECTRAL_WINDOW", "POLARIZATION"])
def test_nonlazy_tables(_support_tables, table):

    # Check that the expected tables are not lazily evaluated.

    assert all([isinstance(dvar.data, np.ndarray)
                for dvar in _support_tables[table][0].data_vars.values()])


# ---------------------------------predict-------------------------------------

# NOTE: No coverage attempt is made for the predict internals copied from
# https://github.com/ska-sa/codex-africanus. This is because the majority
# of this functionality should be tested by codex-africanus. We do check that
# both the direction-independent predict and direction-dependent predict work
# for a number of different input values.

@pytest.mark.predict
def test_predict(_predict):

    # Check that the predicted visibilities are consistent with the MeqTrees
    # visibilities stored in MODEL_DATA.

    predict_per_xds, ms_xds_list = _predict

    for xds_ind in range(len(predict_per_xds)):
        for predict_list in predict_per_xds[xds_ind].values():
            predicted_vis = sum(predict_list)
            expected_vis = ms_xds_list[xds_ind].MODEL_DATA.data
            assert_array_almost_equal(predicted_vis, expected_vis)

# -----------------------------------------------------------------------------
