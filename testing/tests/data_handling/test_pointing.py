import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from xarray import Dataset

from quartical.data_handling.pointing import get_pointing_dir


def make_field_xds(n_field=1, **directions):
    """Makes a FIELD-like dataset from the given (ra, dec) directions.

    Each direction may be a single (ra, dec) pair or one pair per field.
    """

    return Dataset({
        column: (("row", "field-poly", "field-dir"),
                 np.array(value, dtype=np.float64).reshape(n_field, 1, 2))
        for column, value in directions.items()
    })


# ------------------------------get_pointing_dir-------------------------------


def test_pointing_dir_prefers_reference_dir():

    # REFERENCE_DIR survives a rephase and is consequently preferred.

    field_xds = make_field_xds(PHASE_DIR=(0.1, 0.2),
                               REFERENCE_DIR=(0.3, 0.4),
                               DELAY_DIR=(0.5, 0.6))

    column, direction = get_pointing_dir(field_xds)

    assert column == "REFERENCE_DIR"
    assert_array_almost_equal(direction, (0.3, 0.4))


def test_pointing_dir_skips_unpopulated_columns():

    field_xds = make_field_xds(PHASE_DIR=(0.1, 0.2),
                               REFERENCE_DIR=(np.nan, np.nan),
                               DELAY_DIR=(0.5, 0.6))

    column, direction = get_pointing_dir(field_xds)

    assert column == "DELAY_DIR"
    assert_array_almost_equal(direction, (0.5, 0.6))


def test_pointing_dir_accepts_zero_direction():

    # (0, 0) is a valid sky position rather than an unpopulated column.

    field_xds = make_field_xds(PHASE_DIR=(0.1, 0.2),
                               REFERENCE_DIR=(0.0, 0.0))

    column, direction = get_pointing_dir(field_xds)

    assert column == "REFERENCE_DIR"
    assert_array_almost_equal(direction, (0.0, 0.0))


def test_pointing_dir_falls_back_to_phase_dir():

    field_xds = make_field_xds(PHASE_DIR=(0.1, 0.2))

    column, direction = get_pointing_dir(field_xds)

    assert column == "PHASE_DIR"
    assert_array_almost_equal(direction, (0.1, 0.2))


def test_pointing_dir_selects_by_field_id():

    field_xds = make_field_xds(n_field=2,
                               PHASE_DIR=[(0.1, 0.2), (0.3, 0.4)],
                               REFERENCE_DIR=[(0.5, 0.6), (0.7, 0.8)])

    column, direction = get_pointing_dir(field_xds, field_id=1)

    assert column == "REFERENCE_DIR"
    assert_array_almost_equal(direction, (0.7, 0.8))


def test_pointing_dir_without_usable_direction():

    field_xds = make_field_xds(PHASE_DIR=(np.nan, np.nan))

    with pytest.raises(ValueError):
        get_pointing_dir(field_xds)
