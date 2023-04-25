from quartical.gains.gain import ParameterizedGain
from quartical.gains.crosshand_phase.kernel import (crosshand_phase_solver,
                                                    crosshand_phase_args)


class CrosshandPhase(ParameterizedGain):

    solver = staticmethod(crosshand_phase_solver)
    term_args = crosshand_phase_args

    def __init__(self, term_name, term_opts):

        super().__init__(term_name, term_opts)

    @classmethod
    def make_param_names(cls, correlations):

        # TODO: This is not dasky, unlike the other functions. Delayed?
        parameterisable = ["XX", "RR"]

        param_corr = [c for c in correlations if c in parameterisable]

        return [f"crosshand_phase_{c}" for c in param_corr]
