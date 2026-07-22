import shutil
from copy import deepcopy
from pathlib import Path
import pytest
from quartical.data_handling.ms_handler import write_xds_list
import numpy as np


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
def test_read_ms_nxds(raw_xds_list):

    # Check that we produce one xds per scan.
    assert len(raw_xds_list) == 2


@pytest.mark.data_handling
def test_read_ms_cols(raw_xds_list):

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
                   for xds in raw_xds_list
                   for col_name in expected_col_names])


@pytest.mark.data_handling
def test_read_ms_time_chunks(raw_xds_list, ms_opts):

    # Check that the time axis is correctly chunked.
    expected_t_dim = ms_opts.time_chunk or np.inf  # or handles 0.

    assert np.all([chunk <= expected_t_dim
                   for xds in raw_xds_list
                   for chunk in xds.UTIME_CHUNKS])


@pytest.mark.data_handling
def test_read_ms_freq_chunks(raw_xds_list, ms_opts):

    # Check that the frequency axis is correctly chunked.
    expected_f_dim = ms_opts.freq_chunk or np.inf  # or handles 0.

    assert np.all([chunk <= expected_f_dim
                   for xds in raw_xds_list
                   for chunk in xds.chunks["chan"]])


# -------------------------------write_xds_list--------------------------------

@pytest.fixture(scope="module")
def writable_ms(ms_name, tmp_path_factory):
    # write_xds_list creates its output columns on disk at graph-build time,
    # which mutates the target table's column schema. Under pytest-xdist the
    # other workers hold the shared MS open for reading and would then fail with
    # "Table::lock cannot sync ... another process changed the number of
    # columns". Redirect the write onto a private copy of the MS instead. Under
    # xdist tmp_path_factory is rooted in a per-worker temporary directory, so
    # each worker gets its own copy; module scope means we copy once per worker
    # rather than once per parameter combination (writable_ms depends on no
    # parametrised fixtures, so pytest caches it for the whole module).
    ms_copy = tmp_path_factory.mktemp("writable_ms") / Path(ms_name).name
    shutil.copytree(ms_name, ms_copy)

    return str(ms_copy)


@pytest.fixture(scope="module")
def written_xds_list(raw_xds_list, ref_xds_list, writable_ms, output_opts):

    raw_xds_list = [xds.assign({"_RESIDUAL": xds.DATA,
                                "_CORRECTED_DATA": xds.DATA,
                                "_CORRECTED_RESIDUAL": xds.DATA})
                    for xds in raw_xds_list]

    return write_xds_list(raw_xds_list, ref_xds_list, writable_ms, output_opts)


@pytest.mark.data_handling
def test_write_columns_present(written_xds_list):
    # Check that the column to be written is on the writable_xds.
    assert np.all([hasattr(xds, "TEST_RESIDUALS") for xds in written_xds_list])

# -----------------------------------------------------------------------------
