import pytest
import xarray

from quartical.data_handling.angles import assert_parangle_supported


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
