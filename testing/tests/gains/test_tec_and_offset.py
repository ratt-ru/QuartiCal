from copy import deepcopy
import pytest
import numpy as np
import dask.array as da
import xarray as xr
from quartical.calibration.calibrate import add_calibration_graph
from testing.utils.gains import apply_gains


@pytest.fixture(scope="module")
def opts(base_opts, select_corr, scalar_mode):

    # Don't overwrite base config - instead create a copy and update.

    _opts = deepcopy(base_opts)

    _opts.input_ms.select_corr = select_corr
    _opts.solver.terms = ['G']
    _opts.solver.iter_recipe = [50]
    _opts.solver.propagate_flags = False
    _opts.solver.convergence_fraction = 1.0
    _opts.solver.convergence_criteria = 1e-7
    _opts.solver.threads = 2
    _opts.G.type = "tec_and_offset"
    _opts.G.freq_interval = 0
    _opts.G.initial_estimate = True
    _opts.G.scalar = scalar_mode

    return _opts


@pytest.fixture(scope="module")
def raw_xds_list(read_xds_list_output):
    # Only use the first xds. This overloads the global fixture.
    return read_xds_list_output[0][:1]


@pytest.fixture(scope="module")
def true_gain_list(predicted_xds_list, scalar_mode):

    gain_xds_list = []

    for xds in predicted_xds_list:

        n_ant = xds.sizes["ant"]
        utime_chunks = xds.UTIME_CHUNKS
        n_time = sum(utime_chunks)
        chan_chunks = xds.chunks["chan"]
        n_chan = xds.sizes["chan"]
        n_dir = xds.sizes["dir"]
        n_corr = xds.sizes["corr"]

        chan_freq = xds.CHAN_FREQ.data

        min_freq = chan_freq.min()
        max_freq = chan_freq.max()

        # Set the maximum delay and tec based on the number of times they
        # may wrap across the bandwidth.
        max_tec_wraps = 5
        max_tec = max_tec_wraps * (min_freq * max_freq) / (max_freq - min_freq)

        chunking = (utime_chunks, chan_chunks, n_ant, n_dir, n_corr)
        param_chunking = (utime_chunks, 1, n_ant, n_dir, n_corr)

        da.random.seed(0)
        tec = da.random.uniform(
            size=(n_time, 1, n_ant, n_dir, n_corr),
            low=-max_tec,
            high=max_tec,
            chunks=param_chunking
        )
        tec[:, :, 0, :, :] = 0  # Zero the reference antenna for safety.

        # Using the full 2pi range makes some tests fail - this may be due to
        # the fact that we only have 8 channels/degeneracy between parameters.
        offsets = da.random.uniform(
            size=(n_time, 1, n_ant, n_dir, n_corr),
            low=-np.pi,
            high=np.pi,
            chunks=param_chunking
        )
        offsets[:, :, 0, :, :] = 0  # Zero the reference antenna for safety.

        amp = da.ones(
            (n_time, n_chan, n_ant, n_dir, n_corr),
            chunks=chunking
        )

        if n_corr == 4:
            amp[..., 1] = 0
            amp[..., 2] = 0

        origin_chan_freq = chan_freq # - band_centre
        origin_chan_freq = origin_chan_freq[None, :, None, None, None]
        phase = (
            2 * np.pi * tec * (1 / origin_chan_freq) +
            offsets
        )
        gains = amp*da.exp(1j*phase)

        if n_corr == 1:
            n_param = 2
        else:
            n_param = 4

        params = da.zeros(
            (n_time, 1, n_ant, n_dir, n_param),
            chunks=param_chunking[:-1] + (n_param,)
        )
        params[..., 0] = offsets[..., 0]
        params[..., 1] = tec[..., 0]
        if n_corr > 1:
            params[..., 2] = offsets[..., -1]
            params[..., 3] = tec[..., -1]

        if scalar_mode and n_corr > 1:
            gains[..., -1] = gains[..., 0]
            params[..., 2:] = params[..., :2]

        gxds = xr.Dataset(
            data_vars={
                "GAINS": (("time", "freq", "ant", "dir", "corr"), gains),
                "PARAMS":  (("ti", "fi", "ant", "dir", "param"), params)
            }
        )

        gain_xds_list.append(gxds)

    return gain_xds_list


@pytest.fixture(scope="module")
def corrupted_data_xds_list(predicted_xds_list, true_gain_list):

    corrupted_data_xds_list = []

    for xds, gxds in zip(predicted_xds_list, true_gain_list):

        gains = gxds.GAINS.data

        n_corr = xds.sizes["corr"]

        ant1 = xds.ANTENNA1.data
        ant2 = xds.ANTENNA2.data
        time = xds.TIME.data

        row_inds = \
            time.map_blocks(lambda x: np.unique(x, return_inverse=True)[1])

        model = da.ones(xds.MODEL_DATA.data.shape, dtype=np.complex128)

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
def add_calibration_graph_outputs(corrupted_data_xds_list, stats_xds_list,
                                  solver_opts, chain, output_opts):
    # Overload this fixture as we need to use the corrupted xdss.
    return add_calibration_graph(corrupted_data_xds_list, stats_xds_list,
                                 solver_opts, chain, output_opts)


# -----------------------------------------------------------------------------

def test_residual_magnitude(cmp_post_solve_data_xds_list):
    # Magnitude of the residuals should tend to zero.
    for xds in cmp_post_solve_data_xds_list:
        residual = xds._RESIDUAL.data
        if residual.shape[-1] == 4:
            residual = residual[..., (0, 3)]  # Only check on-diagonal terms.
        np.testing.assert_array_almost_equal(np.abs(residual), 0)


def test_solver_flags(cmp_post_solve_data_xds_list):
    # The solver should not add addiitonal flags to the test data.
    for xds in cmp_post_solve_data_xds_list:
        np.testing.assert_array_equal(xds._FLAG.data, xds.FLAG.data)


def test_gains(cmp_gain_xds_lod, true_gain_list):

    for solved_gain_dict, true_gain_xds in zip(cmp_gain_xds_lod, true_gain_list):
        solved_gain_xds = solved_gain_dict["G"]
        solved_gain, solved_flags = da.compute(solved_gain_xds.gains.data,
                                               solved_gain_xds.gain_flags.data)
        true_gain = true_gain_xds.GAINS.values

        true_gain[np.where(solved_flags)] = 0
        solved_gain[np.where(solved_flags)] = 0

        # To ensure the missing antenna handling doesn't render this test
        # useless, check that we have non-zero entries first.
        assert np.any(solved_gain), "All gains are zero!"
        np.testing.assert_array_almost_equal(true_gain, solved_gain)


def test_params(cmp_gain_xds_lod, true_gain_list):

    for solved_gain_dict, true_gain_xds in zip(cmp_gain_xds_lod, true_gain_list):
        solved_gain_xds = solved_gain_dict["G"]
        solved_params, solved_flags = da.compute(solved_gain_xds.params.data,
                                                 solved_gain_xds.param_flags.data)
        true_params = true_gain_xds.PARAMS.values

        solved_params[...,::2] = np.angle(np.exp(1j*solved_params[..., ::2]))

        true_params[np.where(solved_flags)] = 0
        solved_params[np.where(solved_flags)] = 0

        # To ensure the missing antenna handling doesn't render this test
        # useless, check that we have non-zero entries first.
        assert np.any(solved_params), "All params are zero!"
        np.testing.assert_allclose(true_params, solved_params, rtol=1e-5)


def test_gain_flags(cmp_gain_xds_lod):

    for solved_gain_dict in cmp_gain_xds_lod:
        solved_gain_xds = solved_gain_dict["G"]
        solved_flags = solved_gain_xds.gain_flags.values

        frows, fchans, fants, fdir = np.where(solved_flags)

        # We know that these antennas are missing in the test data. No other
        # antennas should have flags.
        assert set(np.unique(fants)) == {18, 20}


# -----------------------------------------------------------------------------
