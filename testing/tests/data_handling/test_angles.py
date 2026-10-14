from types import SimpleNamespace

import dask.array as da
import numpy as np
import pytest
import xarray
from africanus.rime import parallactic_angles

from quartical.calibration.constructor import term_spec_tup
from quartical.data_handling import angles
from quartical.data_handling.angles import (
    assert_parangle_supported,
    assign_parangle_data,
    make_parangle_xds_list,
)
from quartical.gains import TERM_TYPES

N_ANT = 3
N_RECEPTOR = 2

# ITRF antenna positions in metres, near MeerKAT, so that the measures behind
# the parallactic angles see a real site.
ANT_POSITIONS = np.array([5109224.0, 2006790.0, -3239100.0]) + np.array(
    [[0.0, 0.0, 0.0], [100.0, -50.0, 30.0], [-80.0, 120.0, 60.0]]
)

# Three hours of unique times, in UTC seconds since MJD 0. Every baseline is
# present at each, and the rows split into chunks along time boundaries.
UTIMES = 5.0e9 + 1800.0 * np.arange(6)
UTIME_CHUNKS = (3, 3)
BASELINES = [(0, 1), (0, 2), (1, 2)]

# A rephased field: the phase centre has moved one degree away from the
# pointing, which REFERENCE_DIR keeps.
POINTING = (np.deg2rad(80.0), np.deg2rad(-45.0))
PHASE_CENTRE = (np.deg2rad(80.0), np.deg2rad(-44.0))

# Radians. A casacore measures server caches precession and nutation between
# calls, so angles for the same time differ by a few nanoradians depending on
# which times the server saw before. The rephase above moves them by ~1e-2.
ANGLE_TOLERANCE = 1e-7


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
    """A minimal ANTENNA subtable carrying only NAME and POSITION."""
    names = np.array([f"m{a:03d}" for a in range(N_ANT)])
    return [
        xarray.Dataset(
            {
                "NAME": (("ant",), da.from_array(names)),
                "POSITION": (("ant", "xyz"), da.from_array(ANT_POSITIONS)),
            }
        )
    ]


def _field_table(n_field=2, reference_dirs=None, phase_dirs=None):
    """A minimal FIELD subtable with one direction per field.

    The direction columns have shape (field, poly, radec); a single polynomial
    coefficient. PHASE_DIR defaults to distinct placeholder values per field.
    REFERENCE_DIR is only present when supplied, mimicking the writers which
    do not populate it.
    """
    if phase_dirs is None:
        phase_dirs = np.arange(n_field * 2, dtype=float)
    phase_dirs = np.array(phase_dirs, dtype=float).reshape(n_field, 1, 2)
    data_vars = {"PHASE_DIR": (("field", "poly", "radec"), phase_dirs)}

    if reference_dirs is not None:
        reference_dirs = np.array(reference_dirs, dtype=float)
        data_vars["REFERENCE_DIR"] = (
            ("field", "poly", "radec"), reference_dirs.reshape(n_field, 1, 2)
        )

    return [xarray.Dataset(data_vars)]


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


def _patch_tables(
    monkeypatch,
    feed_groups,
    n_field=2,
    reference_dirs=None,
    phase_dirs=None,
):
    """Patch xds_from_storage_table to serve in-memory subtables."""

    def fake_xds_from_storage_table(path, columns=None, group_cols=None):
        if path.endswith("::ANTENNA"):
            return _antenna_table()
        if path.endswith("::FEED"):
            return feed_groups
        if path.endswith("::FIELD"):
            return _field_table(n_field, reference_dirs, phase_dirs)
        raise ValueError(f"Unexpected table request: {path}")

    monkeypatch.setattr(
        angles, "xds_from_storage_table", fake_xds_from_storage_table
    )


def _data_xds(spw_id, field_id=None):
    """A data xds carrying only what the parallactic angle code reads."""
    attrs = {"SPECTRAL_WINDOW_ID": spw_id, "UTIME_CHUNKS": UTIME_CHUNKS}
    if field_id is not None:
        attrs["FIELD_ID"] = field_id

    ant1, ant2 = np.array(BASELINES).T
    row_chunks = (tuple(len(BASELINES) * n for n in UTIME_CHUNKS),)

    def row_column(values):
        return (("row",), da.from_array(values, chunks=row_chunks))

    return xarray.Dataset(
        {
            "TIME": row_column(np.repeat(UTIMES, len(BASELINES))),
            "ANTENNA1": row_column(np.tile(ant1, UTIMES.size)),
            "ANTENNA2": row_column(np.tile(ant2, UTIMES.size)),
        },
        coords={"ant": np.arange(N_ANT)},
        attrs=attrs,
    )


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


def test_assign_parangle_data_uses_pointing(monkeypatch):
    """The field centre is the pointing, which a rephase leaves in place.

    The parallactic angles describe antenna orientation, so they must follow
    REFERENCE_DIR rather than the freely-shiftable PHASE_DIR.
    """
    feed_groups = [_feed_group(0, 0.0)]
    reference_dirs = [(0.5, 0.6), (0.7, 0.8)]
    _patch_tables(monkeypatch, feed_groups, reference_dirs=reference_dirs)

    [xds] = assign_parangle_data("fake.ms", [_data_xds(spw_id=0, field_id=1)])

    assert xds.attrs["FIELD_CENTRE"] == (0.7, 0.8)


def test_assign_parangle_data_pointing_falls_back(monkeypatch):
    """With no REFERENCE_DIR, PHASE_DIR remains the field centre."""
    feed_groups = [_feed_group(0, 0.0)]
    _patch_tables(monkeypatch, feed_groups)

    [xds] = assign_parangle_data("fake.ms", [_data_xds(spw_id=0, field_id=1)])

    assert xds.attrs["FIELD_CENTRE"] == tuple(np.arange(2, 4, dtype=float))


def test_assign_parangle_data_flags_multi_field(monkeypatch):
    """Data not partitioned by FIELD_ID is flagged and given placeholders."""
    feed_groups = [_feed_group(0, 0.0)]
    _patch_tables(monkeypatch, feed_groups)

    [xds] = assign_parangle_data("fake.ms", [_data_xds(spw_id=0)])  # No FIELD_ID.

    assert xds.attrs["MULTI_FIELD"] is True
    assert xds.attrs["FIELD_ID"] == 0
    # The placeholder field centre is taken from the first field's PHASE_DIR.
    assert xds.attrs["FIELD_CENTRE"] == tuple(np.arange(2, dtype=float))


@pytest.fixture
def rephased_xds(monkeypatch):
    """A data xds for a rephased field, after assign_parangle_data."""
    # Zero receptor angles leave the bare parallactic angles.
    _patch_tables(
        monkeypatch,
        [_feed_group(0, 0.0)],
        n_field=1,
        reference_dirs=[POINTING],
        phase_dirs=[PHASE_CENTRE],
    )
    [xds] = assign_parangle_data("fake.ms", [_data_xds(spw_id=0, field_id=0)])
    return xds


@pytest.fixture
def rephased_parangles(rephased_xds):
    """PARALLACTIC_ANGLES for the rephased field, as (utime, ant, receptor)."""
    [parangle_xds] = make_parangle_xds_list("fake.ms", [rephased_xds])
    return parangle_xds.PARALLACTIC_ANGLES.values


def test_make_parangle_xds_list_uses_pointing(rephased_parangles):
    """PARALLACTIC_ANGLES are computed about the pointing, not the phase centre.

    The expected angles come from africanus, which computes the P-Jones the
    predict applies to sky model components.
    """
    about_pointing = parallactic_angles(
        UTIMES, ANT_POSITIONS, np.array(POINTING)
    )
    about_phase_centre = parallactic_angles(
        UTIMES, ANT_POSITIONS, np.array(PHASE_CENTRE)
    )

    # The rephase must move the angles, or this test cannot tell the two
    # directions apart.
    assert not np.allclose(
        about_phase_centre, about_pointing, rtol=0, atol=ANGLE_TOLERANCE
    )

    for receptor in range(N_RECEPTOR):
        np.testing.assert_allclose(
            rephased_parangles[..., receptor],
            about_pointing,
            rtol=0,
            atol=ANGLE_TOLERANCE,
        )


def _parallactic_angle_term():
    """A parallactic_angle term built from the options its constructor reads."""
    options = SimpleNamespace(
        type="parallactic_angle",
        solve_per="antenna",
        scalar=False,
        direction_dependent=False,
        pinned_directions=[],
        time_interval=1,
        freq_interval=0,
        respect_scan_boundaries=False,
        initial_estimate=False,
        load_from=None,
        interp_mode="reim",
        interp_method="2dlinear",
        referenced=False,
    )
    return TERM_TYPES["parallactic_angle"]("P", options)


def test_parallactic_angle_term_matches_parangle_array(
    rephased_xds, rephased_parangles
):
    """The parallactic_angle term uses the angles in PARALLACTIC_ANGLES.

    The term and input_model.apply_p_jones describe the same rotation, so they
    must agree on a rephased field.
    """
    n_utime, n_row = UTIMES.size, rephased_xds.sizes["row"]
    time_map = np.unique(rephased_xds.TIME.values, return_inverse=True)[1]
    freq_map = np.zeros(1, dtype=np.int32)

    spec = term_spec_tup(
        "P",
        "parallactic_angle",
        (n_utime, 1, N_ANT, 1, 4),
        (n_utime, 1, N_ANT, 1, 1),
    )
    ms_kwargs = {
        "TIME": rephased_xds.TIME.values,
        "FLAG": np.zeros((n_row, 1), dtype=np.int8),
        "ANTENNA1": rephased_xds.ANTENNA1.values,
        "ANTENNA2": rephased_xds.ANTENNA2.values,
        "ROW_MAP": None,
        "POSITION": rephased_xds.POSITION.values,
        "RECEPTOR_ANGLE": rephased_xds.RECEPTOR_ANGLE.values,
    }
    term = _parallactic_angle_term()

    # init_term starts from the scaffold, which holds identity values and no
    # raised flags.
    identity_gains = np.zeros(spec.shape, dtype=np.complex128)
    identity_gains[..., (0, 3)] = 1
    term_kwargs = {
        "P_time_map": time_map,
        "P_freq_map": freq_map,
        "P_param_time_map": time_map,
        "P_param_freq_map": freq_map,
        "P_initial_gain": identity_gains,
        "P_initial_gain_flags": np.zeros(spec.shape[:-1], dtype=np.int8),
        "P_initial_params": np.full(spec.pshape, term.param_identity_fill),
        "P_initial_param_flags": np.zeros(spec.pshape[:-1], dtype=np.int8),
    }

    _, _, params, _ = term.init_term(
        spec, 0, ms_kwargs, term_kwargs, meta=dict(rephased_xds.attrs)
    )

    # The term keeps a single angle per antenna, from the first receptor.
    np.testing.assert_allclose(
        params[:, 0, :, 0, 0],
        rephased_parangles[..., 0],
        rtol=0,
        atol=ANGLE_TOLERANCE,
    )
