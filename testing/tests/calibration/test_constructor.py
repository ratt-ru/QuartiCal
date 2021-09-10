from copy import deepcopy
import pickle
import pytest
from quartical.config.preprocess import transcribe_recipe
from quartical.config.internal import gains_to_chain
from quartical.data_handling.ms_handler import (read_xds_list,
                                                preprocess_xds_list)
from quartical.data_handling.model_handler import add_model_graph
from quartical.gains.datasets import (make_gain_xds_lod,
                                      compute_interval_chunking,
                                      compute_dataset_coords)
from quartical.calibration.constructor import (construct_solver,
                                               expand_specs)
from quartical.calibration.mapping import (make_t_maps,
                                           make_f_maps,
                                           make_d_maps)


@pytest.fixture(scope="module")
def opts(base_opts, time_chunk, freq_chunk):

    # Don't overwrite base config - instead create a copy and update.

    _opts = deepcopy(base_opts)

    _opts.input_model.recipe = "MODEL_DATA"
    _opts.input_ms.time_chunk = time_chunk
    _opts.input_ms.freq_chunk = freq_chunk

    return _opts


@pytest.fixture(scope="module")
def model_opts(opts):
    return opts.input_model


@pytest.fixture(scope="module")
def ms_opts(opts):
    return opts.input_ms


@pytest.fixture(scope="module")
def solver_opts(opts):
    return opts.solver


@pytest.fixture(scope="module")
def chain_opts(opts):
    return gains_to_chain(opts)


@pytest.fixture(scope="module")
def recipe(model_opts):
    return transcribe_recipe(model_opts.recipe)


@pytest.fixture(scope="module")
def xds_list(recipe, ms_opts):
    model_columns = recipe.ingredients.model_columns
    # We only need to test on one for these tests.
    return read_xds_list(model_columns, ms_opts)[0][:1]


@pytest.fixture(scope="module")
def preprocessed_xds_list(xds_list, ms_opts):
    return preprocess_xds_list(xds_list, ms_opts)


@pytest.fixture(scope="module")
def data_xds_list(preprocessed_xds_list, recipe, ms_name, model_opts):
    return add_model_graph(preprocessed_xds_list, recipe, ms_name, model_opts)


@pytest.fixture(scope="module")
def t_bin_list(data_xds_list, chain_opts):
    return make_t_maps(data_xds_list, chain_opts)[0]


@pytest.fixture(scope="module")
def t_map_list(data_xds_list, chain_opts):
    return make_t_maps(data_xds_list, chain_opts)[1]


@pytest.fixture(scope="module")
def f_map_list(data_xds_list, chain_opts):
    return make_f_maps(data_xds_list, chain_opts)


@pytest.fixture(scope="module")
def d_map_list(data_xds_list, chain_opts):
    return make_d_maps(data_xds_list, chain_opts)


@pytest.fixture(scope="module")
def _compute_interval_chunking(data_xds_list, t_map_list, f_map_list):
    return compute_interval_chunking(data_xds_list, t_map_list, f_map_list)


@pytest.fixture(scope="module")
def tipc_list(_compute_interval_chunking):
    return _compute_interval_chunking[0]


@pytest.fixture(scope="module")
def fipc_list(_compute_interval_chunking):
    return _compute_interval_chunking[1]


@pytest.fixture(scope="module")
def coords_per_xds(data_xds_list,
                   t_bin_list,
                   f_map_list,
                   tipc_list,
                   fipc_list,
                   solver_opts):
    return compute_dataset_coords(
        data_xds_list,
        t_bin_list,
        f_map_list,
        tipc_list,
        fipc_list,
        solver_opts.terms
    )


@pytest.fixture(scope="module")
def gain_xds_lod(data_xds_list, tipc_list, fipc_list, coords_per_xds,
                 chain_opts):
    return make_gain_xds_lod(data_xds_list, tipc_list, fipc_list,
                             coords_per_xds, chain_opts)


@pytest.fixture(scope="module")
def solver_xds_list(data_xds_list, gain_xds_lod, t_bin_list, t_map_list,
                    f_map_list, d_map_list, solver_opts, chain_opts):

    # Call the construct solver function with the relevant inputs.
    solver_xds_list = construct_solver(data_xds_list,
                                       gain_xds_lod,
                                       t_bin_list,
                                       t_map_list,
                                       f_map_list,
                                       d_map_list,
                                       solver_opts,
                                       chain_opts)

    return solver_xds_list


@pytest.fixture(scope="module")
def expanded_specs(solver_xds_list):

    term_xds_list = solver_xds_list[0]

    return expand_specs(term_xds_list)


# ---------------------------------pickling------------------------------------


@pytest.mark.xfail(reason="Dynamic classes cannot be pickled (easily).")
def test_pickling(solver_xds_list):
    # NOTE: This fails due to the dynamic construction of chain_opts. It does
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


def test_t_chunks(solver_xds_list, data_xds_list):
    """Check that the time chunking on the gain xdss is what we expect."""

    term_xds_dict = solver_xds_list[0]

    expected_t_chunks = data_xds_list[0].DATA.data.numblocks[0]

    assert all([len(gxds.chunks["t_chunk"]) == expected_t_chunks
               for gxds in term_xds_dict.values()])


def test_f_chunks(solver_xds_list, data_xds_list):
    """Check that the freq chunking on the gain xdss is what we expect."""

    term_xds_dict = solver_xds_list[0]

    expected_f_chunks = data_xds_list[0].DATA.data.numblocks[1]

    assert all([len(gxds.chunks["f_chunk"]) == expected_f_chunks
               for gxds in term_xds_dict.values()])

# -------------------------------expand_specs----------------------------------


def test_nchunk(expanded_specs, data_xds_list):
    """Test that the expanded GAIN_SPEC has the correct number of chunks."""

    spec_list = expanded_specs

    expected_nchunk = data_xds_list[0].DATA.data.npartitions

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
