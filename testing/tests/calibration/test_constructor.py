from copy import deepcopy
import pickle
import pytest
from quartical.calibration.constructor import (construct_solver,
                                               expand_specs)


@pytest.fixture(scope="module")
def opts(base_opts, time_chunk, freq_chunk):

    # Don't overwrite base config - instead create a copy and update.

    _opts = deepcopy(base_opts)

    _opts.input_ms.time_chunk = time_chunk
    _opts.input_ms.freq_chunk = freq_chunk

    return _opts


@pytest.fixture(scope="module")
def raw_xds_list(read_xds_list_output):
    # Only use the first xds. This overloads the global fixture.
    return read_xds_list_output[0][:1]


@pytest.fixture(scope="module")
def construct_solver_output(
    predicted_xds_list,
    mapping_xds_list,
    stats_xds_list,
    gain_xds_lod,
    solver_opts,
    chain
):

    # Call the construct solver function with the relevant inputs.
    output = construct_solver(
        predicted_xds_list,
        mapping_xds_list,
        stats_xds_list,
        gain_xds_lod,
        solver_opts,
        chain
    )

    return output


@pytest.fixture(scope="module")
def solver_xds_list(construct_solver_output):
    return construct_solver_output[0]


@pytest.fixture(scope="module")
def solver_data_xds_list(construct_solver_output):
    return construct_solver_output[1]


@pytest.fixture(scope="module")
def expanded_specs(solver_xds_list):

    term_xds_list = solver_xds_list[0]

    return expand_specs(term_xds_list)


# ---------------------------------pickling------------------------------------


@pytest.mark.xfail(reason="Dynamic classes cannot be pickled (easily).")
def test_pickling(solver_xds_list):
    # NOTE: This fails due to the dynamic construction of chain. It does
    # not seem to break the distributed scheduler, so marking as xfail for now.
    assert pickle.loads(pickle.dumps(solver_xds_list)) == solver_xds_list


# -----------------------------construct_solver--------------------------------


def test_fields(solver_xds_list):
    """Check that the expected output fields have been added to the xds."""

    term_xds_dict = solver_xds_list[0]

    fields = ["gains", "conv_perc", "conv_iter"]

    assert all([hasattr(gxds, field)
               for gxds in term_xds_dict.values()
               for field in fields])


def test_t_chunks(solver_xds_list, predicted_xds_list):
    """Check that the time chunking on the gain xdss is what we expect."""

    term_xds_dict = solver_xds_list[0]

    expected_t_chunks = predicted_xds_list[0].DATA.data.numblocks[0]

    assert all([len(gxds.chunks["t_chunk"]) == expected_t_chunks
               for gxds in term_xds_dict.values()])


def test_f_chunks(solver_xds_list, predicted_xds_list):
    """Check that the freq chunking on the gain xdss is what we expect."""

    term_xds_dict = solver_xds_list[0]

    expected_f_chunks = predicted_xds_list[0].DATA.data.numblocks[1]

    assert all([len(gxds.chunks["f_chunk"]) == expected_f_chunks
               for gxds in term_xds_dict.values()])

# -------------------------------expand_specs----------------------------------


def test_nchunk(expanded_specs, predicted_xds_list):
    """Test that the expanded GAIN_SPEC has the correct number of chunks."""

    spec_list = expanded_specs

    expected_nchunk = predicted_xds_list[0].DATA.data.npartitions

    assert len(spec_list) * len(spec_list[0]) == expected_nchunk


def test_shapes(expanded_specs, solver_xds_list):
    """Test that the expanded GAIN_SPEC has the correct shapes."""

    term_xds_dict = solver_xds_list[0]
    spec_list = expanded_specs

    # Flattens out the nested spec list to make comparison easier.
    expanded_shapes = \
        [spec.shape for tc in spec_list for fc in tc for spec in fc]

    ref_shapes = []

    for gxds in term_xds_dict.values():

        chunk_spec = gxds.GAIN_SPEC
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
