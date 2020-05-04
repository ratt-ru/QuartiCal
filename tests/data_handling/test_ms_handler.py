import pytest
from cubicalv2.data_handling.ms_handler import read_ms, write_columns
from argparse import Namespace
import numpy as np


@pytest.fixture(scope="module")
def opts(base_opts, weight_column, freq_chunk, time_chunk, correlation_mode):

    # Don't overwrite base config - instead create a new Namespace and update.

    options = Namespace(**vars(base_opts))

    options.input_ms_weight_column = weight_column
    options.input_ms_freq_chunk = freq_chunk
    options.input_ms_time_chunk = time_chunk
    options.input_ms_correlation_mode = correlation_mode
    options._model_columns = ["MODEL_DATA"]

    return options


@pytest.mark.slow
@pytest.mark.data_handling
@pytest.mark.usefixtures("requires_data")
def test_read_ms(opts):

    col_names = ["TIME",
                 "ANTENNA1",
                 "ANTENNA2",
                 "DATA",
                 "FLAG",
                 "FLAG_ROW",
                 "UVW",
                 "BITFLAG",
                 "BITFLAG_ROW",
                 *opts._model_columns]

    ms_xds_list, col_kwrds = read_ms(opts)

    # Check that we produce one xds per scan.
    assert len(ms_xds_list) == 3

    # Check that all requested columns are present on each xds.
    assert np.all([hasattr(xds, cn)
                   for xds in ms_xds_list
                   for cn in col_names])

    # Check that the time axis is correctly chunked.
    expected_t_dim = opts.input_ms_time_chunk or np.inf  # or handles 0.

    assert np.all([c <= expected_t_dim
                   for xds in ms_xds_list
                   for c in xds.UTIME_CHUNKS])

    # Check that the frequency axis is correctly chunked.
    expected_f_dim = opts.input_ms_freq_chunk or np.inf  # or handles 0.

    assert np.all([c <= expected_f_dim
                   for xds in ms_xds_list
                   for c in xds.DATA.data.chunks[1]])


@pytest.mark.slow
@pytest.mark.data_handling
@pytest.mark.usefixtures("requires_data")
def test_write_columns(opts):

    ms_xds_list, col_kwrds = read_ms(opts)

    ms_xds_list = [xds.assign_attrs({"WRITE_COLS": ["DATA"]})
                   for xds in ms_xds_list]

    write_xds_list = write_columns(ms_xds_list, col_kwrds, opts)

    # Check that the column to be written is on the writable_xds.
    assert np.all([hasattr(xds, "DATA") for xds in write_xds_list])

    # Check that the chunks on the input agree with chunks on the output.
    assert np.all([ms_xds.DATA.data.npartitions == w_xds.DATA.data.npartitions
                  for ms_xds, w_xds in zip(ms_xds_list, write_xds_list)])
