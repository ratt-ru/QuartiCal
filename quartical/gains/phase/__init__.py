from quartical.gains.gain import ParameterizedGain
from quartical.gains.phase.kernel import phase_solver, phase_args


class Phase(ParameterizedGain):

    solver = staticmethod(phase_solver)
    term_args = phase_args

    def __init__(self, term_name, term_opts):

        super().__init__(term_name, term_opts)

        self.gain_axes = (
            "gain_time",
            "gain_freq",
            "antenna",
            "direction",
            "correlation"
        )
        self.param_axes = (
            "param_time",
            "param_freq",
            "antenna",
            "direction",
            "param_name"
        )

    @classmethod
    def make_param_names(cls, correlations):

        # TODO: This is not dasky, unlike the other functions. Delayed?
        parameterisable = ["XX", "YY", "RR", "LL"]

        param_corr = [c for c in correlations if c in parameterisable]

        return [f"phase_{c}" for c in param_corr]
