import dask.array as da
import numpy as np
import pytest
import xarray

from quartical.data_handling import angles
from quartical.data_handling.angles import (
    assert_parangle_supported,
    assign_parangle_data,
)

N_ANT = 3
N_RECEPTOR = 2


def _make_xds(multi_field=None):
    """Build a minimal dataset carrying only the attrs under test."""
    attrs = {} if multi_field is None else {"MULTI_FIELD": multi_field}
    return xarray.Dataset(attrs=attrs)


def test_assert_parangle_supported_single_field():
    """Datasets partitioned by FIELD_ID carry no MULTI_FIELD flag and pass."""
    data_xds_list = [_make_xds(), _make_xds()]
    assert_parangle_supported(data_xds_list)  # Should not raise.


def test_assert_parangle_supported_multi_field():
    """A flagged multi-field dataset must raise rather than silently proceed."""
    data_xds_list = [_make_xds(), _make_xds(multi_field=True)]
    with pytest.raises(ValueError, match="not partitioned by FIELD_ID"):
        assert_parangle_supported(data_xds_list)


def _antenna_table():
    """A minimal ANTENNA subtable carrying only POSITION."""
    return [
        xarray.Dataset(
            {"POSITION": (("ant", "xyz"), da.zeros((N_ANT, 3), chunks=-1))}
        )
    ]


def _field_table(n_field=2):
    """A minimal FIELD subtable with one PHASE_DIR per field."""
    # PHASE_DIR has shape (field, poly, radec); a single polynomial coefficient.
    phase_dirs = np.arange(n_field * 2, dtype=float).reshape(n_field, 1, 2)
    return [
        xarray.Dataset({"PHASE_DIR": (("field", "poly", "radec"), phase_dirs)})
    ]


def _feed_group(spw_id, receptor_angle):
    """A minimal FEED group for a single spectral window.

    daskms groups the FEED table by SPECTRAL_WINDOW_ID, exposing the group key
    as a scalar coordinate on each returned dataset.
    """
    angle = da.full((N_ANT, N_RECEPTOR), receptor_angle, chunks=-1)
    return xarray.Dataset(
        {
            "POLARIZATION_TYPE": (
                ("ant", "receptors"),
                np.array([["X", "Y"]] * N_ANT),
            ),
            "RECEPTOR_ANGLE": (("ant", "receptors"), angle),
        },
        coords={"SPECTRAL_WINDOW_ID": spw_id},
    )


def _patch_tables(monkeypatch, feed_groups, n_field=2):
    """Patch xds_from_storage_table to serve in-memory subtables."""

    def fake_xds_from_storage_table(path, columns=None, group_cols=None):
        if path.endswith("::ANTENNA"):
            return _antenna_table()
        if path.endswith("::FEED"):
            return feed_groups
        if path.endswith("::FIELD"):
            return _field_table(n_field)
        raise ValueError(f"Unexpected table request: {path}")

    monkeypatch.setattr(
        angles, "xds_from_storage_table", fake_xds_from_storage_table
    )


def _data_xds(spw_id, field_id=None):
    """A data xds carrying only the attrs assign_parangle_data consumes."""
    attrs = {"SPECTRAL_WINDOW_ID": spw_id}
    if field_id is not None:
        attrs["FIELD_ID"] = field_id
    return xarray.Dataset(attrs=attrs)


def test_assign_parangle_data_selects_per_spw(monkeypatch):
    """Receptor angles are selected from the matching SPECTRAL_WINDOW_ID."""
    feed_groups = [_feed_group(0, 0.5), _feed_group(1, 1.5)]
    _patch_tables(monkeypatch, feed_groups)

    [xds] = assign_parangle_data("fake.ms", [_data_xds(spw_id=1, field_id=0)])

    # The angles must come from the SPW-1 FEED group, not the SPW-0 group.
    np.testing.assert_array_equal(
        xds.RECEPTOR_ANGLE.data.compute(),
        np.full((N_ANT, N_RECEPTOR), 1.5),
    )
    assert xds.attrs["FEED_TYPE"] == "linear"
    assert "MULTI_FIELD" not in xds.attrs


def test_assign_parangle_data_global_feeds(monkeypatch):
    """A SPECTRAL_WINDOW_ID == -1 FEED group applies to every spectral window."""
    feed_groups = [_feed_group(-1, 2.0)]
    _patch_tables(monkeypatch, feed_groups)

    # The dataset's SPW (5) has no matching FEED group; the global group is used.
    [xds] = assign_parangle_data("fake.ms", [_data_xds(spw_id=5, field_id=1)])

    np.testing.assert_array_equal(
        xds.RECEPTOR_ANGLE.data.compute(),
        np.full((N_ANT, N_RECEPTOR), 2.0),
    )


def test_assign_parangle_data_sparse_unordered_spw(monkeypatch):
    """Selection is keyed by SPECTRAL_WINDOW_ID, not group position.

    The FEED groups are deliberately sparse (no SPWs 0-4) and out of order, so
    positional indexing (the previous behaviour) would either select the wrong
    group or raise IndexError. Keyed lookup must still return the SPW-5 group.
    """
    feed_groups = [_feed_group(5, 5.5), _feed_group(2, 2.5)]
    _patch_tables(monkeypatch, feed_groups)

    [xds] = assign_parangle_data("fake.ms", [_data_xds(spw_id=5, field_id=0)])

    np.testing.assert_array_equal(
        xds.RECEPTOR_ANGLE.data.compute(),
        np.full((N_ANT, N_RECEPTOR), 5.5),
    )


def test_assign_parangle_data_missing_spw_raises(monkeypatch):
    """A SPW with no matching FEED group and no global (-1) entry must raise."""
    feed_groups = [_feed_group(0, 0.0)]
    _patch_tables(monkeypatch, feed_groups)

    with pytest.raises(ValueError, match="No FEED table entry"):
        assign_parangle_data("fake.ms", [_data_xds(spw_id=3, field_id=0)])


def test_assign_parangle_data_flags_multi_field(monkeypatch):
    """Data not partitioned by FIELD_ID is flagged and given placeholders."""
    feed_groups = [_feed_group(0, 0.0)]
    _patch_tables(monkeypatch, feed_groups)

    [xds] = assign_parangle_data("fake.ms", [_data_xds(spw_id=0)])  # No FIELD_ID.

    assert xds.attrs["MULTI_FIELD"] is True
    assert xds.attrs["FIELD_ID"] == 0
    # The placeholder field centre is taken from the first field's PHASE_DIR.
    assert xds.attrs["FIELD_CENTRE"] == tuple(np.arange(2, dtype=float))
