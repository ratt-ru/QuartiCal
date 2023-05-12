import numpy as np
from quartical.gains.conversion import no_op
from quartical.gains.gain import ParameterizedGain
from quartical.gains.amplitude.kernel import amplitude_solver, amplitude_args


class Amplitude(ParameterizedGain):

    solver = staticmethod(amplitude_solver)
    term_args = amplitude_args

    native_to_converted = (
        (1, (no_op,)),
    )
    converted_to_native = (
        (1, no_op),
    )
    converted_dtype = np.float64
    native_dtype = np.float64

    def __init__(self, term_name, term_opts):

        super().__init__(term_name, term_opts)

    @classmethod
    def make_param_names(cls, correlations):

        # TODO: This is not dasky, unlike the other functions. Delayed?
        parameterisable = ["XX", "YY", "RR", "LL"]

        param_corr = [c for c in correlations if c in parameterisable]

        return [f"amplitude_{c}" for c in param_corr]

    def init_term(self, term_spec, ref_ant, ms_kwargs, term_kwargs):
        """Initialise the gains (and parameters)."""

        gains, params = super().init_term(
            term_spec, ref_ant, ms_kwargs, term_kwargs
        )

        if not self.load_from:
            params[...] = 1  # Amplitudes start at unity unless loaded.

        return gains, params
