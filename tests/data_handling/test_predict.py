from copy import deepcopy
import pytest
from quartical.data_handling.ms_handler import read_xds_list
from quartical.parser.preprocess import interpret_model
from quartical.data_handling.predict import (predict,
                                             parse_sky_models,
                                             daskify_sky_model_dict,
                                             get_support_tables)
import dask.array as da
import numpy as np
from numpy.testing import assert_array_almost_equal


expected_clusters = {"DIE": {"point": 22, "gauss": 24},
                     "B290": {"point": 1, "gauss": 2},
                     "C242": {"point": 0, "gauss": 1},
                     "G195": {"point": 0, "gauss": 1},
                     "H194": {"point": 0, "gauss": 2},
                     "I215": {"point": 0, "gauss": 1},
                     #  "K285": {"point": 1, "n_gauss": 0},  #noqa
                     #  "O265": {"point": 1, "n_gauss": 0},  #noqa
                     "R283": {"point": 1, "gauss": 0},
                     #  "W317": {"point": 1, "n_gauss": 0}}  #noqa
                     "V317": {"point": 0, "gauss": 1}}


@pytest.fixture(params=["", "@dE"], ids=["di", "dd"],
                scope="module")
def model_recipe(request, lsm_name):
    return lsm_name + request.param


@pytest.fixture(scope="module")
def opts(base_opts, freq_chunk, time_chunk, model_recipe, beam_name):

    _opts = deepcopy(base_opts)

    _opts.input_ms.freq_chunk = freq_chunk
    _opts.input_ms.time_chunk = time_chunk
    _opts.input_model.recipe = model_recipe
    _opts.input_model.beam = beam_name + "/JVLA-L-centred-$(corr)_$(reim).fits"
    _opts.input_model.beam_l_axis = "-X"
    _opts.input_model.beam_m_axis = "Y"

    interpret_model(_opts)

    return _opts


@pytest.fixture(scope="module")
def _predict(opts):

    # Forcefully add this to ensure that the comparison data is read.
    opts._model_columns = ["MODEL_DATA"]

    ms_xds_list, _ = read_xds_list(opts)

    return predict(ms_xds_list, opts), ms_xds_list


@pytest.fixture(scope="function")
def _sky_dict(base_opts, model_recipe):

    base_opts.input_model.recipe = model_recipe
    interpret_model(base_opts)

    return parse_sky_models(base_opts)


@pytest.fixture(scope="function")
def _dask_sky_dict(_sky_dict, opts):

    return daskify_sky_model_dict(_sky_dict, opts)


@pytest.fixture(scope="module")
def _support_tables(base_opts):

    return get_support_tables(base_opts)


# -----------------------------parse_sky_models--------------------------------


@pytest.mark.predict
@pytest.mark.parametrize("source_fields", [
    ("point", ["radec", "stokes", "spi", "ref_freq"]),
    ("gauss", ["radec", "stokes", "spi", "ref_freq", "shape"])
])
def test_expected_fields(_sky_dict, source_fields):

    # Check that we have all the fields we expect.

    source_type, fields = source_fields

    check = True

    for clusters in _sky_dict.values():
        for cluster in clusters.values():
            for field in fields:
                if source_type in cluster:
                    check &= field in cluster[source_type]

    assert check


@pytest.mark.predict
@pytest.mark.parametrize("source_fields", [
    ("point", ["radec", "stokes", "spi", "ref_freq"]),
    ("gauss", ["radec", "stokes", "spi", "ref_freq", "shape"])
])
def test_nsource(_sky_dict, source_fields):

    # Check for the expected number of point sources.

    source_type, fields = source_fields

    expected_n_source = [s[source_type] for s in expected_clusters.values()]

    for field in fields:
        n_source = [len(cluster.get(source_type, {field: []})[field])
                    for clusters in _sky_dict.values()
                    for cluster in clusters.values()]

        if len(n_source) == 1:
            expected_n_source = [sum(expected_n_source)]

        assert n_source == expected_n_source


# -------------------------daskify_sky_model_dict------------------------------


@pytest.mark.predict
def test_chunking(_dask_sky_dict):

    # Check for consistent chunking.

    check = True

    for sky_model_name, sky_model in _dask_sky_dict.items():
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
