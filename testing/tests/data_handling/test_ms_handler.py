from copy import deepcopy
import pytest
from quartical.data_handling.ms_handler import write_xds_list
import numpy as np

# EXTERNAL FIXTURES:
#   base_opts
#   weight_column
#   freq_chunk
#   time_chunk
#   select_corr
#   data_xds_list
#   ms_opts
#   output_opts
#   ref_xds_list
#   ms_name


@pytest.fixture(scope="module")
def opts(base_opts, weight_column, freq_chunk, time_chunk, select_corr):

    # Don't overwrite base config - instead create a copy and update.

    _opts = deepcopy(base_opts)

    _opts.input_ms.weight_column = weight_column
    _opts.input_ms.freq_chunk = freq_chunk
    _opts.input_ms.time_chunk = time_chunk
    _opts.input_ms.select_corr = select_corr

    return _opts


# -------------------------------read_xds_list---------------------------------

@pytest.mark.data_handling
def test_read_ms_nxds(data_xds_list):

    # Check that we produce one xds per scan.
    assert len(data_xds_list) == 2


@pytest.mark.data_handling
def test_read_ms_cols(data_xds_list):

    expected_col_names = ["TIME",
                          "ANTENNA1",
                          "ANTENNA2",
                          "DATA",
                          "FLAG",
                          "FLAG_ROW",
                          "UVW",
                          "MODEL_DATA"]

    # Check that all requested columns are present on each xds.
    assert np.all([hasattr(xds, col_name)
                   for xds in data_xds_list
                   for col_name in expected_col_names])


@pytest.mark.data_handling
def test_read_ms_time_chunks(data_xds_list, ms_opts):

    # Check that the time axis is correctly chunked.
    expected_t_dim = ms_opts.time_chunk or np.inf  # or handles 0.

    assert np.all([chunk <= expected_t_dim
                   for xds in data_xds_list
                   for chunk in xds.UTIME_CHUNKS])


@pytest.mark.data_handling
def test_read_ms_freq_chunks(data_xds_list, ms_opts):

    # Check that the frequency axis is correctly chunked.
    expected_f_dim = ms_opts.freq_chunk or np.inf  # or handles 0.

    assert np.all([chunk <= expected_f_dim
                   for xds in data_xds_list
                   for chunk in xds.chunks["chan"]])


# -------------------------------write_xds_list--------------------------------

@pytest.fixture(scope="module")
def written_xds_list(data_xds_list, ref_xds_list, ms_name, output_opts):

    data_xds_list = [xds.assign({"_RESIDUAL": xds.DATA,
                                 "_CORRECTED_DATA": xds.DATA,
                                 "_CORRECTED_RESIDUAL": xds.DATA})
                     for xds in data_xds_list]

    data_xds_list = [xds.assign_attrs({"WRITE_COLS": ("DATA",)})
                     for xds in data_xds_list]

    return write_xds_list(data_xds_list, ref_xds_list, ms_name, output_opts)


@pytest.mark.data_handling
def test_write_columns_present(written_xds_list):
    # Check that the column to be written is on the writable_xds.
    assert np.all([hasattr(xds, "DATA") for xds in written_xds_list])

# -----------------------------------------------------------------------------
