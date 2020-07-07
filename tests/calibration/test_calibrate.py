import pytest
from quartical.parser import preprocess
from quartical.data_handling.ms_handler import (read_xds_list,
                                                preprocess_xds_list)
from quartical.data_handling.model_handler import add_model_graph
from quartical.calibration.calibrate import (make_t_mappings,
                                             make_f_mappings,
                                             make_d_mappings,
                                             make_gain_xds_list,
                                             add_calibration_graph)
from argparse import Namespace
import dask.array as da
import numpy as np
from numpy.testing import assert_array_equal


@pytest.fixture(scope="module")
def mapping_opts(base_opts):

    # Don't overwrite base config - instead create a new Namespace and update.

    options = Namespace(**vars(base_opts))

    options._model_columns = ["MODEL_DATA"]

    return options


@pytest.fixture(scope="module")
def xds_opts(base_opts, time_chunk, freq_chunk, time_int, freq_int):

    # Don't overwrite base config - instead create a new Namespace and update.

    options = Namespace(**vars(base_opts))

    options._model_columns = ["MODEL_DATA"]
    options.input_ms_time_chunk = time_chunk
    options.input_ms_freq_chunk = freq_chunk
    options.G_time_interval = time_int
    options.B_time_interval = 2*time_int
    options.G_freq_interval = freq_int
    options.B_freq_interval = 2*freq_int
    options.flags_mad_enable = True

    return options


@pytest.fixture(scope="module")
def _read_xds_list(xds_opts):

    preprocess.interpret_model(xds_opts)

    return read_xds_list(xds_opts)


@pytest.fixture(scope="module")
def data_xds_list(_read_xds_list, xds_opts):

    ms_xds_list, col_kwrds = _read_xds_list

    preprocessed_xds_list = \
        preprocess_xds_list(ms_xds_list, col_kwrds, xds_opts)

    data_xds_list = add_model_graph(preprocessed_xds_list, xds_opts)

    return data_xds_list


@pytest.fixture(scope="module")
def col_kwrds(_read_xds_list):

    return _read_xds_list[1]


@pytest.fixture(scope="module")
def data_xds(data_xds_list):

    return data_xds_list[0]  # We only need to test on one.


@pytest.fixture(scope="module")
def _add_calibration_graph(data_xds_list, col_kwrds, xds_opts):

    return add_calibration_graph(data_xds_list, col_kwrds, xds_opts)


@pytest.fixture(scope="module")
def post_cal_gain_xds_dict(_add_calibration_graph):

    return _add_calibration_graph[0]


@pytest.fixture(scope="module")
def post_cal_data_xds_list(_add_calibration_graph):

    return _add_calibration_graph[1]

# ------------------------------make_t_mappings--------------------------------


@pytest.mark.parametrize("time_chunk", [33, 60])
def test_t_mappings(time_int, time_chunk, mapping_opts):
    """Test construction of time mappings for different chunks/intervals."""

    opts = mapping_opts
    opts.G_time_interval = time_int  # Setting time interval on first gain.
    opts.B_time_interval = time_int*2  # Setting time interval on second gain.

    n_time = 100  # Total number of unique times to consider.
    n_bl = 351  # 27 antenna array - VLA-like.

    time_stamps = da.arange(100, chunks=time_chunk)

    time_col = da.map_blocks(
        np.repeat,
        time_stamps,
        n_bl,
        chunks=(tuple(c*n_bl for c in time_stamps.chunks[0]),))

    utime_ind = da.map_blocks(
        lambda a, **kwargs: np.unique(a, **kwargs)[1],
        time_col,
        return_inverse=True)

    da_t_maps = make_t_mappings(utime_ind, opts)

    # Set up and compute numpy values to test against.

    t_ints = [getattr(opts, t + "_time_interval") or n_time
              for t in opts.solver_gain_terms]

    np_t_maps = np.array(list(map(lambda ti: utime_ind//ti, t_ints))).T

    assert_array_equal(da_t_maps, np_t_maps)


# ------------------------------make_f_mappings--------------------------------


@pytest.mark.parametrize("freq_chunk", [30, 64])
def test_f_mappings(freq_int, freq_chunk, mapping_opts):
    """Test construction of freq mappings for different chunks/intervals."""

    opts = mapping_opts
    opts.G_time_interval = freq_int  # Setting time interval on first gain.
    opts.B_time_interval = freq_int*2  # Setting time interval on second gain.

    n_freq = 64  # Total number of channels to consider.

    chan_freqs = da.arange(n_freq, chunks=freq_chunk)

    da_f_maps = make_f_mappings(chan_freqs, opts)

    # Set up and compute numpy values to test against.

    f_ints = [getattr(opts, t + "_freq_interval") or n_freq
              for t in opts.solver_gain_terms]

    np_f_maps = [np.array(list(map(lambda fi: np.arange(nf)//fi, f_ints))).T
                 for nf in chan_freqs.chunks[0]]
    np_f_maps = np.concatenate(np_f_maps, axis=0)

    assert_array_equal(da_f_maps, np_f_maps)


# ------------------------------make_d_mappings--------------------------------


@pytest.mark.parametrize("n_dir", [2, 10])
@pytest.mark.parametrize("has_dd_term", [False, True])
def test_d_mappings(n_dir, has_dd_term, mapping_opts):
    """Test construction of direction mappings for different n_dir."""

    opts = mapping_opts
    opts.B_direction_dependent = has_dd_term

    d_maps = make_d_mappings(n_dir, opts)  # Not a dask array.

    # Set up and compute numpy values to test against.

    dd_terms = [getattr(opts, t + "_direction_dependent")
                for t in opts.solver_gain_terms]

    np_d_maps = np.array(list(map(lambda dd: np.arange(n_dir)*dd, dd_terms)))

    assert_array_equal(d_maps, np_d_maps)


# ----------------------------make_gain_xds_list-------------------------------


def test_nterm(data_xds, xds_opts):
    """Each gain term should produce a gain xds."""

    gain_xds_list = make_gain_xds_list(data_xds, xds_opts)

    assert len(xds_opts.solver_gain_terms) == len(gain_xds_list)


def test_data_coords(data_xds, xds_opts):
    """Check that dimensions shared between the gains and data are the same."""

    gain_xds_list = make_gain_xds_list(data_xds, xds_opts)

    data_coords = ["ant", "dir", "corr"]

    assert all(data_xds.dims[d] == gxds.dims[d]
               for gxds in gain_xds_list
               for d in data_coords)


def test_t_chunking(data_xds, xds_opts):
    """Check that time chunking of the gain xds list is correct."""

    gain_xds_list = make_gain_xds_list(data_xds, xds_opts)

    assert all(len(data_xds.UTIME_CHUNKS) == gxds.dims["t_chunk"]
               for gxds in gain_xds_list)


def test_f_chunking(data_xds, xds_opts):
    """Check that frequency chunking of the gain xds list is correct."""

    gain_xds_list = make_gain_xds_list(data_xds, xds_opts)

    assert all(len(data_xds.chunks["chan"]) == gxds.dims["f_chunk"]
               for gxds in gain_xds_list)


def test_t_ints(data_xds, xds_opts):
    """Check that the time intervals are correct."""

    gain_xds_list = make_gain_xds_list(data_xds, xds_opts)

    n_row = data_xds.dims["row"]
    t_ints = [getattr(xds_opts, term + "_time_interval") or n_row
              for term in xds_opts.solver_gain_terms]

    expected_t_ints = [sum([np.ceil(tc/ti) for tc in data_xds.UTIME_CHUNKS])
                       for ti in t_ints]

    assert all(int(eti) == gxds.dims["time_int"]
               for eti, gxds in zip(expected_t_ints, gain_xds_list))


def test_f_ints(data_xds, xds_opts):
    """Check that the frequency intervals are correct."""

    gain_xds_list = make_gain_xds_list(data_xds, xds_opts)

    n_chan = data_xds.dims["chan"]
    f_ints = [getattr(xds_opts, term + "_freq_interval") or n_chan
              for term in xds_opts.solver_gain_terms]

    expected_f_ints = [sum([np.ceil(fc/fi) for fc in data_xds.chunks["chan"]])
                       for fi in f_ints]

    assert all(int(efi) == gxds.dims["freq_int"]
               for efi, gxds in zip(expected_f_ints, gain_xds_list))


def test_attribues(data_xds, xds_opts):
    """Check that the attributes of the gains are the same as the data."""

    gain_xds_list = make_gain_xds_list(data_xds, xds_opts)

    data_attributes = ["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"]

    assert all(data_xds.attrs[a] == gxds.attrs[a]
               for gxds in gain_xds_list
               for a in data_attributes)


def test_chunk_spec(data_xds, xds_opts):
    """Check that the chunking specs are correct."""

    gain_xds_list = make_gain_xds_list(data_xds, xds_opts)

    n_row, n_chan, n_ant, n_dir, n_corr = \
        [data_xds.dims[d] for d in ["row", "chan", "ant", "dir", "corr"]]

    t_ints = [getattr(xds_opts, term + "_time_interval") or n_row
              for term in xds_opts.solver_gain_terms]

    expected_t_ints = [[np.int(np.ceil(tc/ti))
                        for tc in data_xds.UTIME_CHUNKS]
                       for ti in t_ints]

    f_ints = [getattr(xds_opts, term + "_freq_interval") or n_chan
              for term in xds_opts.solver_gain_terms]

    expected_f_ints = [[np.int(np.ceil(fc/fi))
                        for fc in data_xds.chunks["chan"]]
                       for fi in f_ints]

    specs = [tuple([tuple(tic), tuple(fic), (n_ant,), (n_dir,), (n_corr,)])
             for tic, fic in zip(expected_t_ints, expected_f_ints)]

    assert all(spec == gxds.attrs["CHUNK_SPEC"]
               for spec, gxds in zip(specs, gain_xds_list))

# ---------------------------add_calibration_graph-----------------------------


def test_ngains(post_cal_gain_xds_dict, xds_opts):
    """Check that calibration produces one xds per gain per input xds."""

    assert len(post_cal_gain_xds_dict) == len(xds_opts.solver_gain_terms)


def test_has_gain_field(post_cal_gain_xds_dict):
    """Check that calibration assigns the gains to the relevant xds."""

    assert all([hasattr(gxds, "gains")
                for gxds_list in post_cal_gain_xds_dict.values()
                for gxds in gxds_list])


def test_has_output_field(post_cal_data_xds_list, xds_opts):
    """Check that calibration assigns the output to the data xds."""

    assert all([hasattr(xds, xds_opts.output_column)
                for xds in post_cal_data_xds_list])


def test_has_bitflag_field(post_cal_data_xds_list):
    """Check that calibration assigns the bitflags to the data xds."""

    assert all([hasattr(xds, "CUBI_BITFLAG")
                for xds in post_cal_data_xds_list])


def test_write_columns(post_cal_data_xds_list, xds_opts):
    """Check that the output column name is added to WRTIE_COLS."""

    assert all([xds_opts.output_column in xds.attrs["WRITE_COLS"]
                for xds in post_cal_data_xds_list])

# -----------------------------------------------------------------------------
