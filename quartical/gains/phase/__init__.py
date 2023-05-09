import numpy as np
from quartical.gains.conversion import trig_to_phase
from quartical.gains.gain import ParameterizedGain
from quartical.gains.phase.kernel import phase_solver, phase_args


class Phase(ParameterizedGain):

    solver = staticmethod(phase_solver)
    term_args = phase_args

    native_to_converted = (
        (0, (np.cos,)),
        (1, (np.sin,))
    )
    converted_to_native = (
        (2, trig_to_phase),
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

        return [f"phase_{c}" for c in param_corr]
