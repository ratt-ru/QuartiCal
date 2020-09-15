import pytest
from quartical.parser import preprocess
from quartical.data_handling.ms_handler import (read_xds_list,
                                                preprocess_xds_list)
from quartical.data_handling.model_handler import add_model_graph
from quartical.calibration.calibrate import make_gain_xds_list
from quartical.calibration.constructor import (construct_solver,
                                               expand_specs)
from argparse import Namespace
import dask.array as da
import numpy as np


@pytest.fixture(scope="module")
def opts(base_opts, time_chunk, freq_chunk):

    # Don't overwrite base config - instead create a new Namespace and update.

    options = Namespace(**vars(base_opts))

    options._model_columns = ["MODEL_DATA"]
    options.input_ms_time_chunk = time_chunk
    options.input_ms_freq_chunk = freq_chunk

    return options


@pytest.fixture(scope="module")
def _read_xds_list(opts):

    preprocess.interpret_model(opts)

    return read_xds_list(opts)


@pytest.fixture(scope="module")
def data_xds(_read_xds_list, opts):

    ms_xds_list, _, col_kwrds = _read_xds_list

    # We only need to test on one.
    ms_xds_list = ms_xds_list[:1]

    preprocessed_xds_list = preprocess_xds_list(ms_xds_list, col_kwrds, opts)

    data_xds_list = add_model_graph(preprocessed_xds_list, opts)

    return data_xds_list[0]


@pytest.fixture(scope="module")
def _construct_solver(data_xds, opts):

    # Grab the relevant columns.
    model_col = data_xds.MODEL_DATA.data
    data_col = data_xds.DATA.data

    # Make some fake mappings - this test doesn't need the values.
    n_row, n_chan, n_dir, _ = model_col.shape
    n_term = len(opts.solver_gain_terms)
    t_map_arr = da.empty((n_row, n_term), chunks=(data_col.chunks[0], n_term))
    f_map_arr = da.empty((n_chan, n_term), chunks=(data_col.chunks[1], n_term))
    d_map_arr = np.empty((n_dir, n_term))

    # Make a gain xds list for this data xds. The results will be assigned to
    # this xds.
    gain_xds_list = make_gain_xds_list(data_xds, opts)

    # Call the construct solver function with the relevant inputs.
    gain_xds_list = construct_solver(data_xds,
                                     t_map_arr,
                                     f_map_arr,
                                     d_map_arr,
                                     opts.input_ms_correlation_mode,
                                     gain_xds_list,
                                     opts)

    return gain_xds_list


@pytest.fixture(scope="module")
def _expand_specs(_construct_solver):

    gain_xds_list = _construct_solver

    return expand_specs(gain_xds_list)

# -----------------------------construct_solver--------------------------------


def test_fields(_construct_solver, opts):
    """Check that the expected output fields have been added to the xds."""

    gain_xds_list = _construct_solver

    fields = ["gains", "conv_perc", "conv_iter"]

    assert all([hasattr(gxds, field)
               for gxds in gain_xds_list
               for field in fields])


def test_t_chunks(_construct_solver, data_xds):
    """Check that the time chunking on the gain xdss is what we expect."""

    gain_xds_list = _construct_solver

    expected_t_chunks = data_xds.DATA.data.numblocks[0]

    assert all([len(gxds.chunks["t_chunk"]) == expected_t_chunks
               for gxds in gain_xds_list])


def test_f_chunks(_construct_solver, data_xds):
    """Check that the freq chunking on the gain xdss is what we expect."""

    gain_xds_list = _construct_solver

    expected_f_chunks = data_xds.DATA.data.numblocks[1]

    assert all([len(gxds.chunks["f_chunk"]) == expected_f_chunks
               for gxds in gain_xds_list])

# -------------------------------expand_specs----------------------------------


def test_nchunk(_expand_specs, data_xds):
    """Test that the expanded CHUNK_SPEC has the correct number of chunks."""

    spec_list = _expand_specs

    expected_nchunk = data_xds.DATA.data.npartitions

    assert len(spec_list) * len(spec_list[0]) == expected_nchunk


def test_shapes(_expand_specs, _construct_solver):
    """Test that the expanded CHUNK_SPEC has the correct shapes."""

    gain_xds_list = _construct_solver
    spec_list = _expand_specs

    # Flattens out the nested spec list to make comparison easier.
    expanded_shapes = \
        [spec.shape for tc in spec_list for fc in tc for spec in fc]

    ref_shapes = []

    for gxds in gain_xds_list:

        chunk_spec = gxds.CHUNK_SPEC
        ac = chunk_spec.achunk[0]
        dc = chunk_spec.dchunk[0]
        cc = chunk_spec.cchunk[0]

        local_shape = []

        for tc in chunk_spec.tchunk:
            for fc in chunk_spec.fchunk:

                local_shape.append((tc, fc, ac, dc, cc))

        ref_shapes.append(local_shape)

    ref_shapes = [val for pair in zip(*ref_shapes) for val in pair]

    assert ref_shapes == expanded_shapes

# -----------------------------------------------------------------------------
