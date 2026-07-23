# -*- coding: utf-8 -*-
"""Kernel-level tests for the null-V crosshand-phase solver.

These tests drive ``null_v_crosshand_phase_solver`` directly with synthetic,
in-memory inputs (no Measurement Set), mirroring the production call path
after ``get_collapsed_inputs`` has collapsed the chain. They pin down the
convergence properties that the integration tests cannot see:

* **Speed**: the solver must converge in a handful of iterations, not tens.
  The update is the exact minimiser of the projected Stokes-V objective (see
  the module docstring in ``null_v_kernel.py``), so noise-free solves land on
  the solution in a single step and are declared converged as soon as the
  convergence bookkeeping allows (iteration 3).
* **Noise robustness**: the per-visibility crosshand SNR must not attenuate
  the step. The legacy Gauss-Newton update built its JHJ from the (noisy)
  data, which inflated the curvature by the noise power and shrank every
  step by U^2/(U^2 + sigma^2) - tens to hundreds of iterations at realistic
  cross-hand SNR. The coherent-product update is unbiased in expectation.
* **Stall-free geometry**: starting a quarter turn (pi/2) from the truth must
  not stall. The legacy iteration had a repelling fixed point there.
* **Chain transport**: with flanking terms in the (collapsed) chain, the
  inverse-chain forging must still recover the crosshand phase.

The recovered phase is compared branch-agnostically (modulo pi): V-nulling
data cannot distinguish phi from phi + pi (the documented sign ambiguity).
"""

import numpy as np
import pytest

from quartical.calibration.solver import meta_args_nt
from quartical.gains import TERM_TYPES

# Sizes are kept small - the runtime is dominated by numba compilation.
N_ANT = 8
N_TIME = 64
SEED = 42

# Per-channel true crosshand phases, chosen to probe the objective's
# geometry: a benign small offset, a negative offset, a value 0.02 rad from
# the pi/2 stationary point of the legacy iteration, and a value on the
# sign-flipped branch (recovered as psi - pi).
TRUE_PSI = np.array([0.3, -1.2, 1.55, 3.0])

# A linearly polarised, unpolarised-V source: XX = I + Q, XY = YX = U.
STOKES_I, STOKES_Q, STOKES_U = 1.0, 0.02, 0.1


def make_synthetic_inputs(noise=0.0, with_chain_term=False):
    """Build solver inputs mimicking the collapsed production call path.

    Constructs a single-(time-)interval, per-channel (freq_interval=1),
    solve_per="array" problem: the configuration in which this term is
    typically run. Data are corrupted by a per-channel crosshand phase and,
    optionally, by a dense near-identity full-Jones flanking term standing in
    for the combined (amplitude/delay/leakage) left of the active term, as
    produced by ``get_collapsed_inputs``.

    Args:
        noise: Standard deviation of the complex noise added to the data.
        with_chain_term: If True, include the flanking term in the chain.

    Returns:
        Tuple of (ms_inputs, mapping_inputs, chain_inputs, term_class) plus
        the true per-channel phases.
    """
    rng = np.random.default_rng(SEED)

    n_chan = TRUE_PSI.size
    n_bl = N_ANT * (N_ANT - 1) // 2
    n_row = N_TIME * n_bl

    # Time-ordered rows with all baselines per unique time, as get_extents
    # assumes (contiguous solution intervals along the row axis).
    p, q = np.triu_indices(N_ANT, k=1)
    antenna1 = np.tile(p.astype(np.int32), N_TIME)
    antenna2 = np.tile(q.astype(np.int32), N_TIME)
    utime_idx = np.repeat(np.arange(N_TIME, dtype=np.int32), n_bl)
    time_col = utime_idx * 8.0

    # The sky coherency: [I+Q, U+iV, U-iV, I-Q] with V = 0.
    b_mat = np.array(
        [
            [STOKES_I + STOKES_Q, STOKES_U],
            [STOKES_U, STOKES_I - STOKES_Q],
        ],
        dtype=np.complex128,
    )

    # Corrupt with the crosshand phase: D = G_p B G_q^H, G = diag(e^{i psi}, 1).
    # G is antenna-independent (solve_per="array" truth), so the cross-hands
    # counter-rotate: XY -> e^{i psi} XY, YX -> e^{-i psi} YX.
    data = np.empty((n_row, n_chan, 4), dtype=np.complex128)
    data[..., 0] = b_mat[0, 0]
    data[..., 1] = b_mat[0, 1] * np.exp(1j * TRUE_PSI)[None, :]
    data[..., 2] = b_mat[1, 0] * np.exp(-1j * TRUE_PSI)[None, :]
    data[..., 3] = b_mat[1, 1]

    if with_chain_term:
        # A dense near-identity full-Jones flanking gain on the (utime, chan)
        # grid, exactly as get_collapsed_inputs produces for the combined
        # terms left of the active one. Data becomes L_p D L_q^H.
        l_shape = (N_TIME, n_chan, N_ANT, 1, 4)
        l_gain = np.zeros(l_shape, dtype=np.complex128)
        l_gain[..., (0, 3)] = 1 + 0.05 * (
            rng.standard_normal(l_shape[:-1] + (2,))
            + 1j * rng.standard_normal(l_shape[:-1] + (2,))
        )
        l_gain[..., (1, 2)] = 0.05 * (
            rng.standard_normal(l_shape[:-1] + (2,))
            + 1j * rng.standard_normal(l_shape[:-1] + (2,))
        )

        l_p = l_gain[utime_idx, :, antenna1, 0].reshape(n_row, n_chan, 2, 2)
        l_q = l_gain[utime_idx, :, antenna2, 0].reshape(n_row, n_chan, 2, 2)
        d_2x2 = data.reshape(n_row, n_chan, 2, 2)
        data = (l_p @ d_2x2 @ l_q.conj().swapaxes(-1, -2)).reshape(
            n_row, n_chan, 4
        )

    if noise:
        data = data + noise * (
            rng.standard_normal(data.shape)
            + 1j * rng.standard_normal(data.shape)
        )

    term_class = TERM_TYPES["crosshand_phase_null_v"]

    ms_inputs = term_class.ms_inputs(
        MODEL_DATA=np.zeros((n_row, n_chan, 1, 4), dtype=np.complex64),
        DATA=data.astype(np.complex64),
        ANTENNA1=antenna1,
        ANTENNA2=antenna2,
        WEIGHT=np.ones((n_row, n_chan, 4), dtype=np.float32),
        FLAG=np.zeros((n_row, n_chan), dtype=np.int8),
        ROW_MAP=None,
        ROW_WEIGHTS=None,
        TIME=time_col,
    )

    # Active-term maps: one time interval over everything, per-channel freq.
    time_bins = np.zeros(N_TIME, dtype=np.int32)
    time_map = np.zeros(n_row, dtype=np.int32)
    freq_map = np.arange(n_chan, dtype=np.int32)
    dir_map = np.zeros(1, dtype=np.int32)

    # Identity starting gains and zeroed parameters for the active term.
    gain_shape = (1, n_chan, N_ANT, 1, 4)
    gains = np.zeros(gain_shape, dtype=np.complex128)
    gains[..., (0, 3)] = 1
    gain_flags = np.zeros(gain_shape[:-1], dtype=np.int8)
    params = np.zeros((1, n_chan, N_ANT, 1, 1), dtype=np.float64)
    param_flags = np.zeros(params.shape[:-1], dtype=np.int8)

    if with_chain_term:
        # Flanking-term maps are dense (per utime/channel), as production's
        # net maps are. The flanking params entry is the active params array,
        # mirroring get_collapsed_inputs.
        net_t_bins = np.arange(N_TIME, dtype=np.int32)
        net_t_map = utime_idx.astype(np.int32)
        net_f_map = np.arange(n_chan, dtype=np.int32)
        l_flags = np.zeros(l_shape[:-1], dtype=np.int8)

        mapping_inputs = term_class.mapping_inputs(
            time_bins=(net_t_bins, time_bins),
            time_maps=(net_t_map, time_map),
            freq_maps=(net_f_map, freq_map),
            dir_maps=(dir_map, dir_map),
            param_time_bins=(net_t_bins, time_bins),
            param_time_maps=(net_t_map, time_map),
            param_freq_maps=(net_f_map, freq_map),
        )
        chain_inputs = term_class.chain_inputs(
            gains=(l_gain, gains),
            gain_flags=(l_flags, gain_flags),
            params=(params, params),
            param_flags=(l_flags, param_flags),
        )
        active_term = 1
    else:
        mapping_inputs = term_class.mapping_inputs(
            time_bins=(time_bins,),
            time_maps=(time_map,),
            freq_maps=(freq_map,),
            dir_maps=(dir_map,),
            param_time_bins=(time_bins,),
            param_time_maps=(time_map,),
            param_freq_maps=(freq_map,),
        )
        chain_inputs = term_class.chain_inputs(
            gains=(gains,),
            gain_flags=(gain_flags,),
            params=(params,),
            param_flags=(param_flags,),
        )
        active_term = 0

    meta_inputs = meta_args_nt(
        iters=15,
        active_term=active_term,
        stop_frac=0.99,
        stop_crit=1e-6,
        threads=1,
        robust=False,
        reference_antenna=0,
        scalar=False,
        dd_term=False,
        pinned_directions=(0,),
        solve_per="array",
    )

    return ms_inputs, mapping_inputs, chain_inputs, meta_inputs, term_class


def run_solver(noise=0.0, with_chain_term=False):
    """Run the null-V solver on synthetic inputs; return results.

    Returns:
        Tuple of (recovered per-channel phases, conv_iter, conv_perc).
    """
    ms_in, map_in, chain_in, meta_in, term_class = make_synthetic_inputs(
        noise=noise, with_chain_term=with_chain_term
    )

    _, conv_iter, conv_perc = term_class.solver(
        ms_in, map_in, chain_in, meta_in, 4
    )

    # solve_per="array" - every antenna holds the same solution; read the
    # per-channel phase from antenna 0.
    params = chain_in.params[meta_in.active_term]
    recovered = params[0, :, 0, 0, 0]

    return recovered, int(conv_iter), float(conv_perc)


def assert_phases_match_mod_pi(recovered, expected, atol):
    """Branch-agnostic phase comparison: phi and phi + pi are equivalent."""
    # The doubled-phase unit vectors coincide for both branches.
    mismatch = np.angle(np.exp(2j * (recovered - expected))) / 2
    np.testing.assert_allclose(mismatch, 0, atol=atol)


# -----------------------------------------------------------------------------


def test_noise_free_recovery():
    # Includes a channel 0.02 rad from the legacy pi/2 stall point and one on
    # the sign-flipped branch - both must converge promptly.
    recovered, conv_iter, conv_perc = run_solver()

    assert_phases_match_mod_pi(recovered, TRUE_PSI, atol=1e-4)
    assert conv_perc == 1.0, "Solver did not report full convergence."
    assert conv_iter <= 5, (
        f"Expected convergence in <=5 iterations, took {conv_iter}."
    )


def test_noisy_recovery():
    # Per-visibility crosshand SNR of 0.5 - realistic for a few-percent
    # polarised calibrator at single-channel resolution. The step must not be
    # attenuated by the noise (the legacy solver's data-built JHJ shrank the
    # step by U^2/(U^2 + sigma^2) ~ 0.2 here and could not converge in 15
    # iterations).
    noise = 2 * STOKES_U  # Per-component sigma; SNR = U/sigma = 0.5.
    recovered, conv_iter, conv_perc = run_solver(noise=noise)

    # Statistical tolerance: ~3 sigma of the coherent estimator for this
    # visibility count and SNR.
    assert_phases_match_mod_pi(recovered, TRUE_PSI, atol=0.15)
    assert conv_perc == 1.0, "Solver did not report full convergence."
    assert conv_iter <= 6, (
        f"Expected convergence in <=6 iterations, took {conv_iter}."
    )


def test_chain_recovery():
    # A dense near-identity full-Jones flanking term (the collapsed
    # amplitude/delay/leakage product) must be transported exactly by the
    # inverse-chain forging: with the flanking term known exactly, the
    # crosshand phase is recovered without bias.
    recovered, conv_iter, conv_perc = run_solver(with_chain_term=True)

    assert_phases_match_mod_pi(recovered, TRUE_PSI, atol=1e-4)
    assert conv_perc == 1.0, "Solver did not report full convergence."
    assert conv_iter <= 5, (
        f"Expected convergence in <=5 iterations, took {conv_iter}."
    )
