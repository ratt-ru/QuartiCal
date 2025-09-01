import numpy as np
from quartical.gains.conversion import trig_to_angle
from quartical.gains.parameterized_gain import ParameterizedGain
from quartical.gains.crosshand_phase.kernel import (
    crosshand_phase_solver,
    crosshand_params_to_gains,
    get_mean_xy_phase
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

        if self.load_from or not self.initial_estimate:

            apply_param_flags_to_params(param_flags, params, 0)
            apply_gain_flags_to_gains(gain_flags, gains)

            return gains, gain_flags, params, param_flags

        data = ms_kwargs["DATA"]  # (row, chan, corr)
        flags = ms_kwargs["FLAG"]  # (row, chan)
        a1 = ms_kwargs["ANTENNA1"]
        a2 = ms_kwargs["ANTENNA2"]
        t_map = term_kwargs[f"{term_spec.name}_time_map"]
        f_map = term_kwargs[f"{term_spec.name}_param_freq_map"]

        # We only need the autocorrelations.
        sel = np.where(a1 == a2)
        a1 = a1[sel]
        a2 = a2[sel]
        t_map = t_map[sel]
        data = data[sel]
        flags = flags[sel]

        params[...] = np.angle(get_mean_xy_phase(data, t_map, f_map, a1))

        # NOTE(JSKenyon): This is a hack for now - ideally, we would be able to
        # trust the flags on the autos when doing this.
        gain_flags[...] = 0
        param_flags[...] = 0

        # Convert the parameters into gains.
        crosshand_params_to_gains(params, gains)

        # Apply flags to gains and parameters.
        apply_param_flags_to_params(param_flags, params, 0)
        apply_gain_flags_to_gains(gain_flags, gains)

        return gains, gain_flags, params, param_flags


class CrosshandPhaseNullV(CrosshandPhase):

    solver = staticmethod(null_v_crosshand_phase_solver)
