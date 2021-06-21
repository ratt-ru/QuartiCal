from copy import deepcopy
import pytest
from quartical.data_handling.ms_handler import read_xds_list, write_xds_list
from quartical.config.preprocess import transcribe_recipe
import numpy as np


@pytest.fixture(scope="module")
def opts(base_opts, weight_column, freq_chunk, time_chunk, select_corr):

    # Don't overwrite base config - instead create a new Namespace and update.

    _opts = deepcopy(base_opts)

    _opts.input_ms.weight_column = weight_column
    _opts.input_ms.freq_chunk = freq_chunk
    _opts.input_ms.time_chunk = time_chunk
    _opts.input_ms.select_corr = select_corr
    _opts.input_model.recipe = "MODEL_DATA"

    return _opts


@pytest.fixture(scope="module")
def _read_xds_list(opts):

    recipe = transcribe_recipe(opts)

    return read_xds_list(recipe.ingredients.model_columns, opts)


@pytest.mark.data_handling
def test_read_ms_nxds(_read_xds_list):

    ms_xds_list, _ = _read_xds_list

    # Check that we produce one xds per scan.
    assert len(ms_xds_list) == 2


@pytest.mark.data_handling
def test_read_ms_cols(_read_xds_list):

    ms_xds_list, _ = _read_xds_list

    expected_col_names = ["TIME",
                          "ANTENNA1",
                          "ANTENNA2",
                          "DATA",
                          "FLAG",
                          "FLAG_ROW",
                          "UVW",
                          "MODEL_DATA"]

    # Check that all requested columns are present on each xds.
    assert np.all([hasattr(xds, cn)
                   for xds in ms_xds_list
                   for cn in expected_col_names])


@pytest.mark.data_handling
def test_read_ms_time_chunks(_read_xds_list, opts):

    ms_xds_list, _ = _read_xds_list

    # Check that the time axis is correctly chunked.
    expected_t_dim = opts.input_ms.time_chunk or np.inf  # or handles 0.

    assert np.all([c <= expected_t_dim
                   for xds in ms_xds_list
                   for c in xds.UTIME_CHUNKS])


@pytest.mark.data_handling
def test_read_ms_freq_chunks(_read_xds_list, opts):

    ms_xds_list, _ = _read_xds_list

    # Check that the frequency axis is correctly chunked.
    expected_f_dim = opts.input_ms.freq_chunk or np.inf  # or handles 0.

    assert np.all([c <= expected_f_dim
                   for xds in ms_xds_list
                   for c in xds.DATA.data.chunks[1]])


@pytest.fixture(scope="module")
def _write_xds_list(_read_xds_list, opts):

    ms_xds_list, ref_xds_list = _read_xds_list

    ms_xds_list = [xds.assign({"_RESIDUAL": xds.DATA,
                               "_CORRECTED_DATA": xds.DATA,
                               "_CORRECTED_RESIDUAL": xds.DATA})
                   for xds in ms_xds_list]

    ms_xds_list = [xds.assign_attrs({"WRITE_COLS": ("DATA",)})
                   for xds in ms_xds_list]

    return write_xds_list(ms_xds_list, ref_xds_list, opts)


@pytest.mark.data_handling
def test_write_columns_present(_write_xds_list):

    written_xds_list = _write_xds_list

    # Check that the column to be written is on the writable_xds.
    assert np.all([hasattr(xds, "DATA") for xds in written_xds_list])
