import numpy as np
from quartical.gains.conversion import trig_to_angle
from quartical.gains.parameterized_gain import ParameterizedGain
from quartical.gains.crosshand_phase.kernel import (
    crosshand_phase_solver,
    crosshand_params_to_gains
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

    def init_term(self, term_spec, ref_ant, ms_kwargs, term_kwargs):
        """Initialise the gains (and parameters)."""

        gains, params = super().init_term(
            term_spec, ref_ant, ms_kwargs, term_kwargs
        )

        # Convert the parameters into gains.
        crosshand_params_to_gains(params, gains)

        return gains, params
