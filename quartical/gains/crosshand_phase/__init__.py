import numpy as np
from quartical.gains.conversion import trig_to_angle
from quartical.gains.parameterized_gain import ParameterizedGain
from quartical.gains.crosshand_phase.kernel import (
    crosshand_phase_solver,
    crosshand_params_to_gains
)
from quartical.gains.crosshand_phase.null_v_kernel import (
    null_v_crosshand_phase_solver
)
from quartical.gains.general.flagging import (
    apply_gain_flags_to_gains,
    apply_param_flags_to_params
)


class CrosshandPhase(ParameterizedGain):

    solver = staticmethod(crosshand_phase_solver)

    native_to_converted = (
        (0, (np.cos,)),
        (1, (np.sin,))
    )
    converted_to_native = (
        (2, trig_to_angle),
    )
    converted_dtype = np.float64
    native_dtype = np.float64

    def __init__(self, term_name, term_opts):

        super().__init__(term_name, term_opts)

    @classmethod
    def make_param_names(cls, correlations):

        # TODO: This is not dasky, unlike the other functions. Delayed?
        parameterisable = ["XX", "RR"]

        param_corr = [c for c in correlations if c in parameterisable]

        return [f"crosshand_phase_{c}" for c in param_corr]

    def init_term(self, term_spec, ref_ant, ms_kwargs, term_kwargs, meta=None):
        """Initialise the gains (and parameters)."""

        gains, gain_flags, params, param_flags = super().init_term(
            term_spec, ref_ant, ms_kwargs, term_kwargs
        )

        # Convert the parameters into gains.
        crosshand_params_to_gains(params, gains)

        # Apply flags to gains and parameters.
        apply_param_flags_to_params(param_flags, params, 0)
        apply_gain_flags_to_gains(gain_flags, gains)

        return gains, gain_flags, params, param_flags


class CrosshandPhaseNullV(CrosshandPhase):

    solver = staticmethod(null_v_crosshand_phase_solver)

    def init_term(self, term_spec, ref_ant, ms_kwargs, term_kwargs, meta=None):
        """Initialise the gains (and parameters).

        When ``initial_estimate`` is enabled, seeds the crosshand phase with a
        closed-form estimate derived from the cross-hand visibilities. This
        places the first iterate inside the quadratic convergence basin of the
        correct minimum, avoiding the ``+/-pi/2`` repeller (an unstable
        stationary point of the V-nulling objective) that otherwise stalls the
        solver when it must walk from identity to the true phase.

        The estimate uses the coherent-product form (see the null-V solver
        notes, section 5): for each cross-hand baseline the product
        ``D_XY * conj(D_YX) = U**2 * exp(2j*psi)`` carries twice the
        instrumental crosshand phase ``psi`` while the per-baseline geometric
        phase cancels between the two cross-hands (they share a uv point), so
        the products add coherently even for resolved sources. The seed is a
        single array-wide value per solution interval,
        ``0.5 * arg(sum_baselines D_XY * conj(D_YX))``, applied to all antennas
        (the ``solve_per="array"`` regime for which the closed form is exact).

        This is a deliberately hacky seed for evaluating convergence behaviour:

        * It is computed from the raw ``DATA`` cross-hands, i.e. it is NOT
          corrected through the inverse of the rest of the Jones chain. A real
          rotation (parallactic angle / feed rotation) preserves Stokes V and
          so leaves the estimate unbiased, but a leakage term on the sky side
          biases it at ``O(leakage * frac_pol)``. As a seed it only needs to be
          good, not exact.
        * It does NOT resolve the irreducible pi sign-ambiguity of the V-nulling
          objective (the two minima ``psi`` and ``psi + pi`` produce identical
          V = 0). Selecting the physical branch requires the sign of the
          calibrator's model Stokes U, which is not used here.
        """
        gains, gain_flags, params, param_flags = super(
            CrosshandPhase, self
        ).init_term(term_spec, ref_ant, ms_kwargs, term_kwargs)

        if self.load_from or not self.initial_estimate:
            crosshand_params_to_gains(params, gains)
            apply_param_flags_to_params(param_flags, params, 0)
            apply_gain_flags_to_gains(gain_flags, gains)
            return gains, gain_flags, params, param_flags

        data = ms_kwargs["DATA"]  # (row, chan, corr)
        flags = ms_kwargs["FLAG"]  # (row, chan)
        a1 = ms_kwargs["ANTENNA1"]
        a2 = ms_kwargs["ANTENNA2"]
        t_map = term_kwargs[f"{self.name}_time_map"]  # (row,)
        f_map = term_kwargs[f"{self.name}_param_freq_map"]  # (chan,)

        # The crosshand phase in solve_per="array" is a single phase shared by
        # every antenna, so the seed is a single array-wide value per solution
        # interval, written to ALL antennas. This differs from e.g. the delay
        # term, whose per-antenna estimate is referenced against ref_ant: here
        # seeding only the non-reference antennas (and leaving ref_ant at
        # identity) produces an inconsistent starting point that the array-mode
        # solver cannot reconcile. ref_ant is therefore intentionally unused.
        cross = a1 != a2  # Autocorrelations carry no crosshand phase.
        t_map = t_map[cross]
        data = data[cross].copy()  # Copy so we can zero out flagged samples.
        flags = flags[cross]

        # Zero flagged samples so they contribute nothing to the coherent sum.
        data[flags == 1] = 0

        # Cross-hand coherent product per visibility (corr 1 = XY, 2 = YX).
        # D_XY * conj(D_YX) = U**2 * exp(2j*psi) carries twice the crosshand
        # phase with the per-baseline geometric phase cancelled.
        xh_product = data[..., 1] * np.conj(data[..., 2])  # (row, chan)

        utint = np.unique(t_map)
        ufint = np.unique(f_map)

        for ut in utint:
            trows = np.where(t_map == ut)[0]
            for uf in ufint:
                chans = np.where(f_map == uf)[0]
                accumulated = xh_product[np.ix_(trows, chans)].sum()
                if accumulated == 0:  # Fully flagged - leave at identity.
                    continue
                params[ut, uf, :, :, 0] = 0.5 * np.angle(accumulated)

        crosshand_params_to_gains(params, gains)

        apply_param_flags_to_params(param_flags, params, 0)
        apply_gain_flags_to_gains(gain_flags, gains)

        return gains, gain_flags, params, param_flags