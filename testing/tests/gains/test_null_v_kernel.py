# -*- coding: utf-8 -*-
"""Kernel-level tests for the forward-model null-V crosshand-phase solver.

These tests drive ``null_v_crosshand_phase_solver`` directly with synthetic,
in-memory inputs (no Measurement Set), mirroring the production call path
after ``get_collapsed_inputs`` has collapsed the chain. They pin down the
properties the integration tests cannot see:

* **Predictability over the full circle**: the recovered phase must match the
  true phase (modulo the irreducible pi/sign ambiguity) for truths anywhere
  in (-pi, pi], including exactly +/-pi/2 - the watershed/repeller points of
  earlier update rules - and pi itself.
* **Forward-model consistency**: the solver corrupts a model, it never
  corrects the data. The supplied model's cross-hands are treated as unknown
  (the V = 0 nuisance completion absorbs all linear polarisation), so a
  wildly wrong model cross-hand must not change the solution, while the
  model's parallel hands (total intensity) must be used to predict any
  leakage mixing.
* **Noise robustness**: the update is a linear least-squares estimate per
  iteration, so low per-visibility cross-hand SNR must reduce precision, not
  convergence speed.
* **Sky-side leakage**: with a full-Jones (leakage-bearing) term between the
  crosshand term and the model - the configuration in which the data-side
  null-V estimator converges confidently to a biased answer - the forward
  solver must recover the truth, because the model's total intensity
  predicts the leakage mixing in the residual.

The recovered phase is compared branch-agnostically (modulo pi) unless a
test explicitly targets the branch. The sign of the calibrator's Stokes U
selects the branch: the solver's nuisance amplitude is non-negative by
construction, so a negative true U lands exactly pi away.
"""

import numpy as np
import pytest

from quartical.calibration.solver import meta_args_nt
from quartical.gains import TERM_TYPES

# Sizes are kept small - the runtime is dominated by numba compilation.
N_ANT = 8
N_TIME = 64
SEED = 42

# Per-channel true crosshand phases covering the full circle, including
# exactly +/-pi/2 (the watershed of update rules that estimate the nuisance
# amplitude and the phase separately, and the repeller of the legacy
# Gauss-Newton iteration), 0, and pi (the branch edge).
TRUE_PSI = np.linspace(-np.pi, np.pi, 17)[1:]

# A linearly polarised source with no circular polarisation:
# XX = I + Q, XY = YX = U, V = 0.
STOKES_I, STOKES_Q, STOKES_U = 1.0, 0.02, 0.1


def wrap(angle):
    """Wrap angles into (-pi, pi]."""
    return np.angle(np.exp(1j * angle))


def make_synthetic_inputs(
    noise=0.0,
    with_chain_terms=False,
    stokes_u=STOKES_U,
    model_crosshand=None,
):
    """Build solver inputs mimicking the collapsed production call path.

    Constructs a single-(time-)interval, per-channel (freq_interval=1),
    solve_per="array" problem: the configuration in which this term is
    typically run. Data are corrupted by a per-channel crosshand phase and,
    optionally, by a diagonal data-side flanking term plus a full-Jones
    (leakage-bearing) sky-side flanking term - i.e. the physical ordering
    [G, X, B] in which the data-side null-V estimator is known to fail.

    Args:
        noise: Standard deviation per real/imaginary component of the
            complex noise added to the data.
        with_chain_terms: If True, flank the active term with a diagonal
            term on the data side and a full-Jones term on the sky side.
        stokes_u: True Stokes U of the synthetic source (may be negative).
        model_crosshand: Value planted in the supplied model's cross-hand
            slots. Defaults to the true Stokes U; tests may plant garbage to
            verify the model cross-hands are ignored.

    Returns:
        Tuple of (ms_inputs, mapping_inputs, chain_inputs, meta_inputs,
        term_class).
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

    # The true sky coherency: [I+Q, U; U, I-Q] with V = 0.
    b_mat = np.array(
        [
            [STOKES_I + STOKES_Q, stokes_u],
            [stokes_u, STOKES_I - STOKES_Q],
        ],
        dtype=np.complex128,
    )

    # The per-antenna Jones chain on the (utime, chan, ant) grid. The active
    # crosshand term is antenna-independent (solve_per="array" truth).
    x_jones = np.zeros((1, n_chan, 1, 2, 2), dtype=np.complex128)
    x_jones[..., 0, 0] = np.exp(1j * TRUE_PSI)[None, :, None]
    x_jones[..., 1, 1] = 1

    if with_chain_terms:
        # Data-side diagonal term (amplitude + phase) on the dense
        # (utime, chan) grid, as production's collapsed maps produce.
        l_shape = (N_TIME, n_chan, N_ANT, 1, 4)
        l_gain = np.zeros(l_shape, dtype=np.complex128)
        diag_amp = 1 + 0.1 * rng.standard_normal(l_shape[:-1] + (2,))
        diag_phase = 0.2 * rng.standard_normal(l_shape[:-1] + (2,))
        l_gain[..., (0, 3)] = diag_amp * np.exp(1j * diag_phase)

        # Sky-side full-Jones term (a bandpass carrying few-percent leakage,
        # constant in time). Off-diagonal scale ~3% against a fractional
        # polarisation of ~10%: d/p is O(1), the regime in which the
        # data-side estimator's contaminating harmonics dominate.
        b_shape = (1, n_chan, N_ANT, 1, 4)
        b_gain = np.zeros(b_shape, dtype=np.complex128)
        b_gain[..., (0, 3)] = 1 + 0.05 * (
            rng.standard_normal(b_shape[:-1] + (2,))
            + 1j * rng.standard_normal(b_shape[:-1] + (2,))
        )
        b_gain[..., (1, 2)] = 0.03 * (
            rng.standard_normal(b_shape[:-1] + (2,))
            + 1j * rng.standard_normal(b_shape[:-1] + (2,))
        )

        l_2x2 = l_gain.reshape(N_TIME, n_chan, N_ANT, 2, 2)
        b_2x2 = b_gain.reshape(1, n_chan, N_ANT, 2, 2)

        # Full chain J = L X B per (utime, chan, ant).
        j_2x2 = l_2x2 @ x_jones @ b_2x2
    else:
        j_2x2 = np.broadcast_to(x_jones, (N_TIME, n_chan, N_ANT, 2, 2))

    j_p = j_2x2[utime_idx, :, antenna1]
    j_q = j_2x2[utime_idx, :, antenna2]

    data = (j_p @ b_mat @ j_q.conj().swapaxes(-1, -2)).reshape(
        n_row, n_chan, 4
    )

    if noise:
        data = data + noise * (
            rng.standard_normal(data.shape)
            + 1j * rng.standard_normal(data.shape)
        )

    # The supplied model: correct parallel hands (I, Q known), cross-hands
    # planted with model_crosshand (ignored by the solver - the V = 0
    # nuisance completion replaces them).
    model_crosshand = stokes_u if model_crosshand is None else model_crosshand
    model = np.empty((n_row, n_chan, 1, 4), dtype=np.complex64)
    model[..., 0] = b_mat[0, 0]
    model[..., 1] = model_crosshand
    model[..., 2] = model_crosshand
    model[..., 3] = b_mat[1, 1]

    term_class = TERM_TYPES["crosshand_phase_null_v"]

    ms_inputs = term_class.ms_inputs(
        MODEL_DATA=model,
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

    if with_chain_terms:
        # Flanking-term maps: the data-side term is dense in time, the
        # sky-side term is time-constant. The flanking params entries reuse
        # the active params array, mirroring get_collapsed_inputs.
        dense_t_bins = np.arange(N_TIME, dtype=np.int32)
        dense_t_map = utime_idx.astype(np.int32)
        l_flags = np.zeros(l_gain.shape[:-1], dtype=np.int8)
        b_flags = np.zeros(b_gain.shape[:-1], dtype=np.int8)

        mapping_inputs = term_class.mapping_inputs(
            time_bins=(dense_t_bins, time_bins, time_bins),
            time_maps=(dense_t_map, time_map, time_map),
            freq_maps=(freq_map, freq_map, freq_map),
            dir_maps=(dir_map, dir_map, dir_map),
            param_time_bins=(dense_t_bins, time_bins, time_bins),
            param_time_maps=(dense_t_map, time_map, time_map),
            param_freq_maps=(freq_map, freq_map, freq_map),
        )
        chain_inputs = term_class.chain_inputs(
            gains=(l_gain, gains, b_gain),
            gain_flags=(l_flags, gain_flags, b_flags),
            params=(params, params, params),
            param_flags=(l_flags, param_flags, b_flags),
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


def run_solver(**kwargs):
    """Run the null-V solver on synthetic inputs; return results.

    Returns:
        Tuple of (recovered per-channel phases, conv_iter, conv_perc).
    """
    ms_in, map_in, chain_in, meta_in, term_class = make_synthetic_inputs(
        **kwargs
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


def test_noise_free_recovery_over_full_circle():
    # Truths span (-pi, pi] including exactly +/-pi/2 and pi - every channel
    # must be recovered; no watershed, stall, or repeller anywhere.
    recovered, conv_iter, conv_perc = run_solver()

    assert conv_perc == 1.0
    assert_phases_match_mod_pi(recovered, TRUE_PSI, atol=1e-5)


def test_noise_free_convergence_is_prompt():
    # The per-iteration update is an exact conditional minimiser: iteration 1
    # lands on the solution and the bookkeeping needs two confirming
    # iterations, so convergence must be reported at iteration 3.
    _, conv_iter, conv_perc = run_solver()

    assert conv_perc == 1.0
    assert conv_iter <= 4


def test_positive_stokes_u_recovers_exact_branch():
    # With U > 0 the non-negative nuisance amplitude selects the true branch:
    # the recovered phase must equal psi exactly, not just modulo pi.
    recovered, _, _ = run_solver()

    np.testing.assert_allclose(wrap(recovered - TRUE_PSI), 0, atol=1e-5)


def test_negative_stokes_u_lands_on_flipped_branch():
    # With U < 0 the (U, phi) -> (-U, phi + pi) gauge places the solution
    # exactly pi from the truth - the documented, predictable sign flip.
    recovered, _, _ = run_solver(stokes_u=-STOKES_U)

    np.testing.assert_allclose(
        np.abs(wrap(recovered - TRUE_PSI)), np.pi, atol=1e-5
    )


def test_model_crosshands_are_ignored():
    # The V = 0 completion treats the source's linear polarisation as
    # unknown: planting garbage in the model cross-hands must not move the
    # solution.
    recovered, _, conv_perc = run_solver(model_crosshand=0.7)

    assert conv_perc == 1.0
    assert_phases_match_mod_pi(recovered, TRUE_PSI, atol=1e-5)


def test_noisy_recovery():
    # noise=0.2 gives per-visibility cross-hand SNR rho = U/sigma ~ 0.35
    # (sigma^2 = 2*noise^2 per complex sample). The per-iteration linear
    # least-squares estimate predicts sigma_phi ~ 1/(2*rho*sqrt(N)) ~ 0.034
    # for N = 1792 visibilities per channel; allow ~4.5 sigma. Low SNR must
    # not stall convergence.
    recovered, conv_iter, conv_perc = run_solver(noise=0.2)

    assert conv_perc == 1.0
    assert conv_iter <= 6
    assert_phases_match_mod_pi(recovered, TRUE_PSI, atol=0.15)


def test_sky_side_leakage_chain_recovery():
    # The physical ordering [G, X, B]: a diagonal term data-side of the
    # active term and a full-Jones leakage-bearing term sky-side of it, all
    # held at their exact values. The data-side null-V estimator is O(1)
    # biased here; the forward solver must recover the truth because the
    # model's parallel hands predict the leakage mixing exactly.
    recovered, conv_iter, conv_perc = run_solver(with_chain_terms=True)

    assert conv_perc == 1.0
    assert_phases_match_mod_pi(recovered, TRUE_PSI, atol=1e-4)
