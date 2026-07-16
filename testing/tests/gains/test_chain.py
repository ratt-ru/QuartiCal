from pathlib import Path
import pytest
import numpy as np
import dask.array as da
from quartical.config.parser import parse_inputs
from quartical.calibration.calibrate import add_calibration_graph
from testing.utils.gains import apply_gains, reference_gains

# This module verifies the mechanics of a multi-term chain (per-term
# time/freq/direction mappings and the left/right operator products inside
# the solver kernels) rather than any individual term. Individual terms are
# not compared against the truth as effects can bleed between terms (e.g. a
# sufficiently resolved complex term can absorb a delay); instead the net
# product of the chain, which is what the data actually constrains, is
# compared against the product of the true Jones terms.
#
# The module is marked as slow: kernel compilation depends on the length and
# composition of the chain, so these tests cannot reuse the kernels compiled
# by the single-term tests. To keep the cost to a single set of chain
# compilations, both variants (DI and DD) share one three-term chain
# signature (complex, delay, diag_complex) and a single correlation mode.

pytestmark = pytest.mark.slow


@pytest.fixture(params=[False, True], ids=["di", "dd"], scope="module")
def direction_dependent(request):
    return request.param


@pytest.fixture(scope="module")
def opts(test_data_location, ms_name, tmp_path_factory, direction_dependent):

    # The set of terms on the config object is fixed at parse time and the
    # base config only knows about G and B. Parse an overlay config which
    # introduces the K term rather than deep-copying base_opts.

    base_conf_path = Path(test_data_location, "test_config.yaml")
    chain_conf_path = tmp_path_factory.mktemp("chain_conf") / "chain.yaml"
    # One-term-at-a-time chain updates converge linearly when terms are
    # strongly coupled, so the chain needs many cycles over the terms (the
    # DD variant in particular, where the directions couple through the sum
    # over the model). Converged terms exit almost immediately, so the large
    # iteration budget is cheap.
    chain_conf_path.write_text(
        "solver:\n"
        "  terms: [G, K, B]\n"
        f"  iter_recipe: [{', '.join(['25'] * 60)}]\n"
    )

    _opts, _ = parse_inputs(
        bypass_sysargv=[
            'goquartical', str(base_conf_path), str(chain_conf_path)
        ]
    )
    _opts.input_ms.path = ms_name  # Ensure the ms path is correct.

    # Deliberately a single correlation mode - every additional mode would
    # compile the entire chain from scratch.
    _opts.input_ms.select_corr = [0, 1, 2, 3]
    _opts.solver.propagate_flags = False
    _opts.solver.convergence_criteria = 1e-7
    _opts.solver.convergence_fraction = 1
    _opts.solver.threads = 2
    # The net gain is the product of the chain at full data resolution.
    _opts.output.net_gains = [["G", "K", "B"]]

    _opts.G.type = "complex"
    _opts.G.time_interval = 1
    _opts.G.freq_interval = 0
    _opts.K.type = "delay"
    _opts.K.time_interval = 0
    _opts.K.freq_interval = 0
    _opts.K.initial_estimate = True
    _opts.B.type = "diag_complex"
    _opts.B.time_interval = 0
    _opts.B.direction_dependent = direction_dependent
    # In the DD case B must be frequency-constant: with per-channel freedom
    # in every direction only the sum over directions is constrained by the
    # data and the net gains would not be unique.
    _opts.B.freq_interval = 0 if direction_dependent else 1

    return _opts


@pytest.fixture(scope="module")
def raw_xds_list(read_xds_list_output):
    # Only use the first xds. This overloads the global fixture.
    return read_xds_list_output[0][:1]


def compose_2x2(op1, op2):
    """Compose two (..., 4)-correlation Jones arrays via 2x2 matmul."""

    shape = np.broadcast_shapes(op1.shape, op2.shape)

    op1 = np.broadcast_to(op1, shape).reshape(shape[:-1] + (2, 2))
    op2 = np.broadcast_to(op2, shape).reshape(shape[:-1] + (2, 2))

    return (op1 @ op2).reshape(shape)


@pytest.fixture(scope="module")
def true_net_gain_list(predicted_xds_list, direction_dependent):

    # The true gains are constructed in numpy (they are small at solver
    # resolution) as forming the net gain requires composing the terms with
    # per-element 2x2 matrix products.

    net_gain_list = []

    for xds in predicted_xds_list:

        n_ant = xds.sizes["ant"]
        n_time = sum(xds.UTIME_CHUNKS)
        n_chan = xds.sizes["chan"]
        n_corr = xds.sizes["corr"]
        n_dir = 2 if direction_dependent else 1

        diag = np.array([1, 0, 0, 1])

        rng = np.random.default_rng(0)

        # G: time-variable, frequency-constant, diagonal. The truth is kept
        # diagonal throughout (and the model unpolarised - see the corrupted
        # data fixture) because the diagonal-type terms in the chain (K, B)
        # are only constrained by the parallel-hand visibilities: with
        # per-channel structure in the truth, the relative (crosshand) phase
        # between the two polarisation chains would be unconstrained and the
        # solve ill-posed. This mirrors test_delay.py.
        g_shape = (n_time, 1, n_ant, 1, n_corr)
        g_amp = rng.normal(loc=1, scale=0.05, size=g_shape) * diag
        g_phase = rng.uniform(low=-np.pi/2, high=np.pi/2, size=g_shape)
        true_g = g_amp * np.exp(1j * g_phase)

        # K: diagonal delay, time-constant, phase referenced to the band
        # centre (mirrors the construction in test_delay.py).
        chan_freq = xds.CHAN_FREQ.values
        chan_width = chan_freq[1] - chan_freq[0]
        band_centre = (chan_freq[0] + chan_freq[-1]) / 2

        k_shape = (1, 1, n_ant, 1, n_corr)
        delays = rng.uniform(
            low=-1/(2*chan_width), high=1/(2*chan_width), size=k_shape
        )
        delays[:, :, 0] = 0  # Zero the reference antenna for safety.
        delays *= diag
        origin_chan_freq = chan_freq - band_centre
        phase = 2*np.pi*delays*origin_chan_freq[None, :, None, None, None]
        true_k = np.exp(1j*phase) * diag

        # B: diagonal; per-channel in the DI case, per-direction (and
        # frequency-constant - see the opts fixture) in the DD case.
        if direction_dependent:
            b_shape = (1, 1, n_ant, n_dir, n_corr)
        else:
            b_shape = (1, n_chan, n_ant, 1, n_corr)
        b_amp = rng.normal(loc=1, scale=0.05, size=b_shape) * diag
        b_phase = rng.uniform(low=-np.pi/2, high=np.pi/2, size=b_shape)
        true_b = b_amp * np.exp(1j * b_phase)

        # The net gain multiplies in chain order, i.e. G@K@B, per direction.
        # Broadcasting the per-term shapes yields the full data resolution.
        true_net = compose_2x2(compose_2x2(true_g, true_k), true_b)

        net_gain_list.append(true_net)

    return net_gain_list


@pytest.fixture(scope="module")
def corrupted_data_xds_list(
    predicted_xds_list, true_net_gain_list, direction_dependent
):

    corrupted_data_xds_list = []

    for xds, net_gain in zip(predicted_xds_list, true_net_gain_list):

        n_row = xds.sizes["row"]
        n_chan = xds.sizes["chan"]
        n_corr = xds.sizes["corr"]
        n_dir = net_gain.shape[3]

        ant1 = xds.ANTENNA1.data
        ant2 = xds.ANTENNA2.data
        time = xds.TIME.data

        row_inds = \
            time.map_blocks(lambda x: np.unique(x, return_inverse=True)[1])

        # The model is rebuilt from scratch as the DD case needs a second
        # direction which the recipe (a single column) does not provide.
        row_chunks, chan_chunks = xds.MODEL_DATA.data.chunks[:2]
        model = da.ones(
            (n_row, n_chan, n_dir, n_corr),
            chunks=(row_chunks, chan_chunks, n_dir, n_corr),
            dtype=np.complex128
        )
        # The model is unpolarised (zero cross-hands): combined with the
        # diagonal truth this keeps the cross-hand visibilities identically
        # zero, so the crosshand phase (which parallel-hand data cannot
        # constrain) never enters the problem.
        model *= np.array([1, 0, 0, 1])

        if direction_dependent:
            # Give the second direction a distinct amplitude and a phase
            # slope in frequency so that the directions are separable.
            rotation = 0.5 * np.exp(2j * np.pi * 0.2 * np.arange(n_chan))
            dir_factor = np.stack([np.ones(n_chan), rotation], axis=-1)
            model *= dir_factor[None, :, :, None]

        gains = da.from_array(
            net_gain,
            chunks=(xds.UTIME_CHUNKS, chan_chunks, -1, -1, -1)
        )

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
            "MODEL_DATA": (("row", "chan", "dir", "corr"), model),
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
    # Magnitude of the residuals should tend to zero if the chain converged.
    for xds in cmp_post_solve_data_xds_list:
        residual = xds._RESIDUAL.data
        if residual.shape[-1] == 4:
            residual = residual[..., (0, 3)]  # Only check on-diagonal terms.
        np.testing.assert_array_almost_equal(np.abs(residual), 0)


def test_solver_flags(cmp_post_solve_data_xds_list):
    # The solver should not add additional flags to the test data.
    for xds in cmp_post_solve_data_xds_list:
        np.testing.assert_array_equal(xds._FLAG.data, xds.FLAG.data)


def test_net_gains(cmp_net_xds_list, true_net_gain_list):
    # The individual solutions are not unique (structure can bleed between
    # terms) but the net product of the chain is (up to the usual reference
    # ambiguity). This is the assertion which probes the chain mechanics.
    for net_dict, true_net in zip(cmp_net_xds_list, true_net_gain_list):

        net_xds = net_dict["GKB-net"]
        solved_net = net_xds.gains.values
        net_flags = net_xds.gain_flags.values

        n_corr = true_net.shape[-1]

        solved_net = reference_gains(solved_net, n_corr)
        true_net = reference_gains(true_net, n_corr)

        true_net[np.where(net_flags)] = 0
        solved_net[np.where(net_flags)] = 0

        # To ensure the missing antenna handling doesn't render this test
        # useless, check that we have non-zero entries first.
        assert np.any(solved_net), "All gains are zero!"
        np.testing.assert_array_almost_equal(true_net, solved_net)


def test_gain_flags(cmp_gain_xds_lod):

    for solved_gain_dict in cmp_gain_xds_lod:
        for term_name, solved_gain_xds in solved_gain_dict.items():
            solved_flags = solved_gain_xds.gain_flags.values

            frows, fchans, fants, fdir = np.where(solved_flags)

            # We know that these antennas are missing in the test data. No
            # other antennas should have flags in any term.
            assert set(np.unique(fants)) == {18, 20}, \
                f"Unexpected gain flags on term {term_name}."


# -----------------------------------------------------------------------------
