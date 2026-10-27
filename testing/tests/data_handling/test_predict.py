from copy import deepcopy
import pytest
from quartical.data_handling.predict import (parse_sky_models,
                                             daskify_sky_model_dict,
                                             get_support_tables,
                                             build_rime_spec,
                                             PointedBeamCubeDDE)
import dask.array as da
import numpy as np
import xarray
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


@pytest.fixture
def feed_receptor_angles():
    # One pair of receptor angles per FEED row.
    return np.arange(6, dtype=np.float64).reshape(3, 2)


@pytest.fixture
def row_partitioned_feed_xds(monkeypatch, feed_receptor_angles):
    # The FEED dataset get_support_tables returns for a table stored with one
    # row per partition, as dask-ms convert writes it to zarr. See
    # ratt-ru/QuartiCal#352.
    feed_partitions = [
        xarray.Dataset(
            {
                "RECEPTOR_ANGLE": (
                    ("row", "receptors"),
                    da.from_array(row_angles[None], chunks=-1)
                )
            }
        )
        for row_angles in feed_receptor_angles
    ]

    def fake_xds_from_storage_table(path, columns=None, group_cols=None):
        if path.endswith("::FEED"):
            return [xds[columns] if columns else xds for xds in feed_partitions]
        return [xarray.Dataset()]

    monkeypatch.setattr(
        "quartical.data_handling.predict.xds_from_storage_table",
        fake_xds_from_storage_table
    )

    [feed_xds] = get_support_tables("fake.ms")["FEED"]

    return feed_xds


@pytest.mark.predict
def test_support_tables_join_feed_partitions(
    row_partitioned_feed_xds, feed_receptor_angles
):
    # The partitions are joined into the single dataset predict uses.
    np.testing.assert_array_equal(
        row_partitioned_feed_xds.RECEPTOR_ANGLE.values, feed_receptor_angles
    )


@pytest.mark.predict
def test_support_tables_hold_feed_in_one_chunk(row_partitioned_feed_xds):
    # predict clones the receptor angles for every dataset, which would clone
    # a read per FEED row if the joined table kept one chunk per partition.
    assert row_partitioned_feed_xds.RECEPTOR_ANGLE.data.numblocks == (1, 1)


# ----------------------------build_rime_spec----------------------------------


@pytest.mark.predict
def test_beam_term_samples_beam_lm(model_opts):

    # The beam term must consume its own lm coordinates so that the beam and
    # the phase term can be centred on different directions.

    spec = build_rime_spec(["I", "Q", "U", "V"],
                           ["XX", "XY", "YX", "YY"],
                           "point",
                           model_opts)

    beam_terms = [t for t in spec.terms
                  if isinstance(t, PointedBeamCubeDDE)]

    assert len(beam_terms) == 2  # One on either side of the onion.
    assert all("beam_lm" in t.ALL_ARGS for t in beam_terms)


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
