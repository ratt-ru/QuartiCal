from copy import deepcopy
import pickle

import pytest
from quartical.config.preprocess import transcribe_recipe
from quartical.data_handling.ms_handler import (read_xds_list,
                                                preprocess_xds_list)
from quartical.data_handling.model_handler import add_model_graph
from quartical.calibration.calibrate import make_gain_xds_list
from quartical.calibration.constructor import (construct_solver,
                                               expand_specs)
from quartical.calibration.mapping import (make_t_maps,
                                           make_f_maps,
                                           make_d_maps)


@pytest.fixture(scope="module")
def opts(base_opts, time_chunk, freq_chunk):

    # Don't overwrite base config - instead create a new Namespace and update.

    _opts = deepcopy(base_opts)

    _opts.input_model.recipe = "MODEL_DATA"
    _opts.input_ms.time_chunk = time_chunk
    _opts.input_ms.freq_chunk = freq_chunk

    return _opts


@pytest.fixture(scope="module")
def recipe(opts):
    return transcribe_recipe(opts.input_model.recipe)


@pytest.fixture(scope="module")
def xds_list(recipe, opts):
    model_columns = recipe.ingredients.model_columns
    # We only need to test on one for these tests.
    return read_xds_list(model_columns, opts.input_ms)[0][:1]


@pytest.fixture(scope="module")
def data_xds_list(xds_list, recipe, ms_name, opts):

    weight_col_name = opts.input_ms.weight_column

    preprocessed_xds_list = preprocess_xds_list(xds_list, weight_col_name)

    data_xds_list = add_model_graph(preprocessed_xds_list,
                                    recipe,
                                    ms_name,
                                    opts)

    return data_xds_list


@pytest.fixture(scope="module")
def t_bin_list(data_xds_list, opts):
    return make_t_maps(data_xds_list, opts)[0]


@pytest.fixture(scope="module")
def t_map_list(data_xds_list, opts):
    return make_t_maps(data_xds_list, opts)[1]


@pytest.fixture(scope="module")
def f_map_list(data_xds_list, opts):
    return make_f_maps(data_xds_list, opts)


@pytest.fixture(scope="module")
def d_map_list(data_xds_list, opts):
    return make_d_maps(data_xds_list, opts)


@pytest.fixture(scope="module")
def gain_xds_list(data_xds_list, t_map_list, t_bin_list, f_map_list, opts):
    return make_gain_xds_list(data_xds_list, t_map_list, t_bin_list,
                              f_map_list, opts)


@pytest.fixture(scope="module")
def solver_xds_list(data_xds_list, gain_xds_list, t_bin_list, t_map_list,
                    f_map_list, d_map_list, opts):

    # Call the construct solver function with the relevant inputs.
    solver_xds_list = construct_solver(data_xds_list,
                                       gain_xds_list,
                                       t_bin_list,
                                       t_map_list,
                                       f_map_list,
                                       d_map_list,
                                       opts)

    return solver_xds_list


@pytest.fixture(scope="module")
def expanded_specs(solver_xds_list):

    term_xds_list = solver_xds_list[0]

    return expand_specs(term_xds_list)


# ---------------------------------pickling------------------------------------


def test_pickling(solver_xds_list):
    assert pickle.loads(pickle.dumps(solver_xds_list)) == solver_xds_list


# -----------------------------construct_solver--------------------------------


def test_fields(solver_xds_list):
    """Check that the expected output fields have been added to the xds."""

    term_xds_list = solver_xds_list[0]

    fields = ["gains", "conv_perc", "conv_iter"]

    assert all([hasattr(gxds, field)
               for gxds in term_xds_list
               for field in fields])


def test_t_chunks(solver_xds_list, data_xds_list):
    """Check that the time chunking on the gain xdss is what we expect."""

    term_xds_list = solver_xds_list[0]

    expected_t_chunks = data_xds_list[0].DATA.data.numblocks[0]

    assert all([len(gxds.chunks["t_chunk"]) == expected_t_chunks
               for gxds in term_xds_list])


def test_f_chunks(solver_xds_list, data_xds_list):
    """Check that the freq chunking on the gain xdss is what we expect."""

    term_xds_list = solver_xds_list[0]

    expected_f_chunks = data_xds_list[0].DATA.data.numblocks[1]

    assert all([len(gxds.chunks["f_chunk"]) == expected_f_chunks
               for gxds in term_xds_list])

# -------------------------------expand_specs----------------------------------


def test_nchunk(expanded_specs, data_xds_list):
    """Test that the expanded GAIN_SPEC has the correct number of chunks."""

    spec_list = expanded_specs

    expected_nchunk = data_xds_list[0].DATA.data.npartitions

    assert len(spec_list) * len(spec_list[0]) == expected_nchunk


def test_shapes(expanded_specs, solver_xds_list):
    """Test that the expanded GAIN_SPEC has the correct shapes."""

    term_xds_list = solver_xds_list[0]
    spec_list = expanded_specs

    # Flattens out the nested spec list to make comparison easier.
    expanded_shapes = \
        [spec.shape for tc in spec_list for fc in tc for spec in fc]

    ref_shapes = []

    for gxds in term_xds_list:

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
