from copy import deepcopy
import pytest
from quartical.data_handling.predict import (parse_sky_models,
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
                     "R283": {"point": 1, "gauss": 0},
                     "V317": {"point": 0, "gauss": 1}}


@pytest.fixture(params=["", "@dE"], ids=["di", "dd"],
                scope="module")
def raw_model_recipe(request, lsm_name):
    return lsm_name + request.param


@pytest.fixture(scope="module")
def model_opts(base_opts, raw_model_recipe, beam_name):

    model_opts = deepcopy(base_opts.input_model)

    model_opts.recipe = raw_model_recipe
    model_opts.beam = beam_name + "/JVLA-L-centred-$(corr)_$(reim).fits"
    model_opts.beam_l_axis = "-X"
    model_opts.beam_m_axis = "Y"

    return model_opts


@pytest.fixture(scope="module")
def ms_opts(base_opts, freq_chunk, time_chunk):

    ms_opts = deepcopy(base_opts.input_ms)

    ms_opts.freq_chunk = freq_chunk
    ms_opts.time_chunk = time_chunk

    return ms_opts


@pytest.fixture(scope="function")
def sky_model_dict(recipe):
    return parse_sky_models(recipe.ingredients.sky_models)


@pytest.fixture(scope="function")
def dask_sky_dict(sky_model_dict):

    return daskify_sky_model_dict(sky_model_dict, 10)


@pytest.fixture(scope="module")
def support_tables(ms_name):
    return get_support_tables(ms_name)


# -----------------------------parse_sky_models--------------------------------


@pytest.mark.predict
@pytest.mark.parametrize("source_fields", [
    ("point", ["radec", "stokes", "spi", "ref_freq"]),
    ("gauss", ["radec", "stokes", "spi", "ref_freq", "shape"])
])
def test_expected_fields(sky_model_dict, source_fields):

    # Check that we have all the fields we expect.

    source_type, fields = source_fields

    check = True

    for clusters in sky_model_dict.values():
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
def test_nsource(sky_model_dict, source_fields):

    # Check for the expected number of point sources.

    source_type, fields = source_fields

    expected_n_source = [s[source_type] for s in expected_clusters.values()]

    for field in fields:
        n_source = [len(cluster.get(source_type, {field: []})[field])
                    for clusters in sky_model_dict.values()
                    for cluster in clusters.values()]

        if len(n_source) == 1:
            expected_n_source = [sum(expected_n_source)]

        assert n_source == expected_n_source


# -------------------------daskify_sky_model_dict------------------------------


@pytest.mark.predict
def test_chunking(dask_sky_dict):

    # Check for consistent chunking.

    check = True

    for sky_model_name, sky_model in dask_sky_dict.items():
        for cluster_name, cluster in sky_model.items():
            for source_type, sources in cluster.items():
                for arr in sources.values():
                    check &= all([c <= 10 for c in arr.chunks[0]])

    assert check is True


# ----------------------------get_support_tables-------------------------------


@pytest.mark.predict
@pytest.mark.parametrize("table", ["ANTENNA", "DATA_DESCRIPTION", "FIELD",
                                   "SPECTRAL_WINDOW", "POLARIZATION"])
def test_support_fields(support_tables, table):
    # Check that we have all expected support tables.
    assert table in support_tables


@pytest.mark.predict
@pytest.mark.parametrize("table", ["ANTENNA"])
def test_lazy_tables(support_tables, table):
    # Check that the antenna table is lazily evaluated.
    assert all([isinstance(dvar.data, da.Array)
                for dvar in support_tables[table][0].data_vars.values()])


@pytest.mark.predict
@pytest.mark.parametrize("table", ["DATA_DESCRIPTION", "FIELD",
                                   "SPECTRAL_WINDOW", "POLARIZATION"])
def test_nonlazy_tables(support_tables, table):
    # Check that the expected tables are not lazily evaluated.
    assert all([isinstance(dvar.data, np.ndarray)
                for dvar in support_tables[table][0].data_vars.values()])


# ---------------------------------predict-------------------------------------

# NOTE: No coverage attempt is made for the predict internals copied from
# https://github.com/ska-sa/codex-africanus. This is because the majority
# of this functionality should be tested by codex-africanus. We do check that
# both the direction-independent predict and direction-dependent predict work
# for a number of different input values.

@pytest.mark.predict
def test_predict(predicted_xds_list, data_xds_list_w_model_col):

    # Check that the predicted visibilities are consistent with the MeqTrees
    # visibilities stored in MODEL_DATA.

    for xds_ind in range(len(predicted_xds_list)):
        predicted_vis = predicted_xds_list[xds_ind].MODEL_DATA.data
        predicted_vis = predicted_vis.sum(axis=2)  # Sum over directions.
        expected_vis = data_xds_list_w_model_col[xds_ind].MODEL_DATA.data
        assert_array_almost_equal(predicted_vis, expected_vis)

# -----------------------------------------------------------------------------
