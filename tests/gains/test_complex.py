from copy import deepcopy
import pytest
import numpy as np
import dask
import dask.array as da
from quartical.data_handling.ms_handler import (read_xds_list,
                                                preprocess_xds_list)
from quartical.data_handling.model_handler import add_model_graph
from quartical.calibration.calibrate import add_calibration_graph
from tests.testing_utils.gains import apply_gains


@pytest.fixture(scope="module")
def opts(base_opts):

    # Don't overwrite base config - instead create a new Namespace and update.

    _opts = deepcopy(base_opts)

    _opts.input_model.recipe = "MODEL_DATA"
    _opts.solver.terms = ['G']
    _opts.solver.iter_recipe = [25]
    _opts.solver.converging_criteria = 0
    _opts.G.type = "complex"

    return _opts


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
def true_gain_list(data_xds_list):

    gain_list = []

    for xds in data_xds_list:

        n_ant = xds.dims["ant"]
        utime_chunks = xds.UTIME_CHUNKS
        n_time = sum(utime_chunks)
        chan_chunks = xds.chunks["chan"]
        n_chan = xds.dims["chan"]
        n_dir = xds.dims["dir"]
        n_corr = xds.dims["corr"]

        chunking = (utime_chunks, chan_chunks, n_ant, n_dir, n_corr)

        da.random.seed(0)
        amp = da.random.normal(size=(n_time, n_chan, n_ant, n_dir, n_corr),
                               loc=1,
                               scale=0.05,
                               chunks=chunking)
        phase = da.random.normal(size=(n_time, n_chan, n_ant, n_dir, n_corr),
                                 loc=0,
                                 scale=0.25,
                                 chunks=chunking)

        if n_corr == 4:
            amp *= da.array([1, 0.1, 0.1, 1])

        gains = amp*da.exp(1j*phase)

        gain_list.append(gains)

    return gain_list


@pytest.fixture(scope="module")
def corrupted_data_xds_list(data_xds_list, true_gain_list):

    corrupted_data_xds_list = []

    for xds, gains in zip(data_xds_list, true_gain_list):

        n_corr = xds.dims["corr"]

        ant1 = xds.ANTENNA1.data
        ant2 = xds.ANTENNA2.data
        time = xds.TIME.data

        row_inds = \
            time.map_blocks(lambda x: np.unique(x, return_inverse=True)[1])

        model = da.ones(xds.MODEL_DATA.data.shape, dtype=np.complex128)

        if n_corr == 4:
            model *= da.array([1, 0, 0, 1])

        data = da.blockwise(apply_gains, ("rfc"),
                            model, ("rfdc"),
                            gains, ("rfadc"),
                            ant1, ("r"),
                            ant2, ("r"),
                            row_inds, ("r"),
                            n_corr, None,
                            align_arrays=False,
                            concatenate=True,
                            dtype=model.dtype)

        corrupted_xds = xds.assign({
            "DATA": ((xds.DATA.dims), data),
            "MODEL_DATA": ((xds.MODEL_DATA.dims), model),
            "FLAG": ((xds.FLAG.dims), da.zeros_like(xds.FLAG.data)),
            "WEIGHT": ((xds.WEIGHT.dims), da.ones_like(xds.WEIGHT.data))
            }
        )

        corrupted_data_xds_list.append(corrupted_xds)

    return corrupted_data_xds_list


@pytest.fixture(scope="module")
def _add_calibration_graph(corrupted_data_xds_list, solver_opts, chain_opts):
    gain_xds_lod, net_xds_list, cal_data_xds_list = \
        add_calibration_graph(corrupted_data_xds_list, solver_opts, chain_opts)

    return dask.compute(gain_xds_lod, cal_data_xds_list[0]._RESIDUAL.data)


@pytest.fixture(scope="module")
def gain_xds_lod(_add_calibration_graph):
    return _add_calibration_graph[0]


@pytest.fixture(scope="module")
def residuals(_add_calibration_graph):
    return _add_calibration_graph[1]


# -----------------------------------------------------------------------------

def test_residual_magnitude(residuals):
    # Magnitude of the residuals should tend to zero.
    np.testing.assert_array_less(np.abs(residuals), 1e-14)


def test_gains(gain_xds_lod, true_gain_list):

    for solved_gain_dict, true_gain in zip(gain_xds_lod, true_gain_list):
        solved_gain = solved_gain_dict["G"].gains.values
        true_gain = true_gain.compute()  # TODO: This could be done elsewhere.

        solved_gain = solved_gain.reshape(solved_gain.shape[:-1] + (2, 2))
        true_gain = true_gain.reshape(true_gain.shape[:-1] + (2, 2))

        # Indices for the transpose.
        inds = (0, 1, 2, 3, 5, 4)

        # We want to rotate all the gains to a ref antenna (0) for comparison.
        solved_gain = \
            solved_gain @ solved_gain.transpose(inds)[:, :, :1].conj()
        true_gain = \
            true_gain @ true_gain.transpose(inds)[:, :, :1].conj()

        # If we have missing antennas, we zero them for comparison.
        true_gain[solved_gain == 0] = 0

        # To ensure the missing antenna handling doesn't reder this test
        # useless, check that we have non-zero entries first.
        assert np.any(solved_gain), "All gains are zero!"
        np.testing.assert_array_almost_equal(true_gain, solved_gain)

# -----------------------------------------------------------------------------
