from copy import deepcopy
import pytest
import numpy as np
import dask.array as da
from quartical.calibration.calibrate import add_calibration_graph
from testing.utils.gains import apply_gains, reference_gains


@pytest.fixture(params=["complex", "approx_complex"], scope="module")
def solver_type(request):
    return request.param


@pytest.fixture(scope="module")
def opts(base_opts, solver_type, select_corr, solve_per):

    # Don't overwrite base config - instead create a copy and update.

    _opts = deepcopy(base_opts)

    _opts.input_ms.select_corr = select_corr
    _opts.solver.terms = ['G']
    _opts.solver.iter_recipe = [30]
    _opts.solver.propagate_flags = False
    _opts.solver.convergence_criteria = 1e-8
    _opts.G.type = solver_type
    _opts.G.solve_per = solve_per

    return _opts


@pytest.fixture(scope="module")
def raw_xds_list(read_xds_list_output):
    # Only use the first xds. This overloads the global fixture.
    return read_xds_list_output[0][:1]


@pytest.fixture(scope="module")
def true_gain_list(predicted_xds_list, solve_per):

    gain_list = []

    for xds in predicted_xds_list:

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
        phase = da.random.uniform(size=(n_time, n_chan, n_ant, n_dir, n_corr),
                                  high=np.pi/2,
                                  low=-np.pi/2,
                                  chunks=chunking)

        if n_corr == 4:  # Reduce amplitude of leakage components.
            amp *= da.array([1, 0.1, 0.1, 1])

        gains = amp*da.exp(1j*phase)

        if solve_per == "array":
            gains = da.broadcast_to(gains[:, :, :1], gains.shape)

        gain_list.append(gains)

    return gain_list


@pytest.fixture(scope="module")
def corrupted_data_xds_list(predicted_xds_list, true_gain_list):

    corrupted_data_xds_list = []

    for xds, gains in zip(predicted_xds_list, true_gain_list):

        n_corr = xds.dims["corr"]

        ant1 = xds.ANTENNA1.data
        ant2 = xds.ANTENNA2.data
        time = xds.TIME.data

        row_inds = \
            time.map_blocks(lambda x: np.unique(x, return_inverse=True)[1])

        model = da.ones(xds.MODEL_DATA.data.shape, dtype=np.complex128)

        # TODO: Commenting out the below breaks tests. Why?

        if n_corr == 4:  # Decrease magnitude of off-diagonal elements.
            model *= da.array([1, 0.1, 0.1, 1])

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
def add_calibration_graph_outputs(corrupted_data_xds_list,
                                  solver_opts, chain_opts, output_opts):
    # Overload this fixture as we need to use the corrupted xdss.
    return add_calibration_graph(corrupted_data_xds_list,
                                 solver_opts, chain_opts, output_opts)


# -----------------------------------------------------------------------------

def test_residual_magnitude(cmp_post_solve_data_xds_list):
    # Magnitude of the residuals should tend to zero.
    for xds in cmp_post_solve_data_xds_list:
        np.testing.assert_array_almost_equal(np.abs(xds._RESIDUAL.data), 0)


def test_solver_flags(cmp_post_solve_data_xds_list):
    # The solver should not add addiitonal flags to the test data.
    for xds in cmp_post_solve_data_xds_list:
        np.testing.assert_array_equal(xds._FLAG.data, xds.FLAG.data)


def test_gains(gain_xds_lod, true_gain_list):

    for solved_gain_dict, true_gain in zip(gain_xds_lod, true_gain_list):
        solved_gain_xds = solved_gain_dict["G"]
        solved_gain, solved_flags = da.compute(solved_gain_xds.gains.data,
                                               solved_gain_xds.gain_flags.data)
        true_gain = true_gain.compute()  # TODO: This could be done elsewhere.

        n_corr = true_gain.shape[-1]

        solved_gain = reference_gains(solved_gain, n_corr)
        true_gain = reference_gains(true_gain, n_corr)

        true_gain[np.where(solved_flags)] = 0
        solved_gain[np.where(solved_flags)] = 0

        # To ensure the gain flags don't render this test useless, check that
        # we have non-zero entries first.
        assert np.any(solved_gain), "All gains are zero!"
        np.testing.assert_array_almost_equal(true_gain, solved_gain)


def test_gain_flags(gain_xds_lod):

    for solved_gain_dict in gain_xds_lod:
        solved_gain_xds = solved_gain_dict["G"]
        solved_flags = solved_gain_xds.gain_flags.values

        frows, fchans, fants, fdir = np.where(solved_flags)

        # We know that these antennas are missing in the test data. No other
        # antennas should have flags.
        assert set(np.unique(fants)) == {18, 20}

# -----------------------------------------------------------------------------
