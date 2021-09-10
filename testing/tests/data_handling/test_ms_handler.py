from copy import deepcopy
import pytest
from quartical.data_handling.ms_handler import read_xds_list, write_xds_list
import numpy as np


@pytest.fixture(scope="module")
def ms_opts(base_opts, weight_column, freq_chunk, time_chunk, select_corr):

    # Don't overwrite base config - instead create a new Namespace and update.

    ms_opts = deepcopy(base_opts.input_ms)

    ms_opts.weight_column = weight_column
    ms_opts.freq_chunk = freq_chunk
    ms_opts.time_chunk = time_chunk
    ms_opts.select_corr = select_corr

    return ms_opts


@pytest.fixture(scope="module")
def output_opts(base_opts):
    return base_opts.output


@pytest.fixture(scope="module")
def _read_xds_list(ms_opts):
    return read_xds_list(["MODEL_DATA"], ms_opts)


@pytest.fixture(scope="module")
def xds_list(_read_xds_list):
    return _read_xds_list[0]


@pytest.fixture(scope="module")
def ref_xds_list(_read_xds_list):
    return _read_xds_list[1]


@pytest.mark.data_handling
def test_read_ms_nxds(xds_list):

    # Check that we produce one xds per scan.
    assert len(xds_list) == 2


@pytest.mark.data_handling
def test_read_ms_cols(xds_list):

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
                   for xds in xds_list
                   for col_name in expected_col_names])


@pytest.mark.data_handling
def test_read_ms_time_chunks(xds_list, ms_opts):

    # Check that the time axis is correctly chunked.
    expected_t_dim = ms_opts.time_chunk or np.inf  # or handles 0.

    assert np.all([chunk <= expected_t_dim
                   for xds in xds_list
                   for chunk in xds.UTIME_CHUNKS])


@pytest.mark.data_handling
def test_read_ms_freq_chunks(xds_list, ms_opts):

    # Check that the frequency axis is correctly chunked.
    expected_f_dim = ms_opts.freq_chunk or np.inf  # or handles 0.

    assert np.all([chunk <= expected_f_dim
                   for xds in xds_list
                   for chunk in xds.chunks["chan"]])


@pytest.fixture(scope="module")
def written_xds_list(xds_list, ref_xds_list, ms_name, output_opts):

    xds_list = [xds.assign({"_RESIDUAL": xds.DATA,
                            "_CORRECTED_DATA": xds.DATA,
                            "_CORRECTED_RESIDUAL": xds.DATA})
                for xds in xds_list]

    xds_list = [xds.assign_attrs({"WRITE_COLS": ("DATA",)})
                for xds in xds_list]

    return write_xds_list(xds_list, ref_xds_list, ms_name, output_opts)


@pytest.mark.data_handling
def test_write_columns_present(written_xds_list):
    # Check that the column to be written is on the writable_xds.
    assert np.all([hasattr(xds, "DATA") for xds in written_xds_list])
