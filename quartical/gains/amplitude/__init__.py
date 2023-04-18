from quartical.gains.gain import ParameterizedGain
from quartical.gains.amplitude.kernel import amplitude_solver, amplitude_args


class Amplitude(ParameterizedGain):

    solver = staticmethod(amplitude_solver)
    term_args = amplitude_args

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

        return [f"amplitude_{c}" for c in param_corr]

    @staticmethod
    def init_term(
        gain, param, term_ind, term_spec, term_opts, ref_ant, **kwargs
    ):
        """Initialise the gains (and parameters)."""

        super(Amplitude, Amplitude).init_term(
            gain, param, term_ind, term_spec, term_opts, ref_ant, **kwargs
        )

        param[:] = 1  # Amplitudes start at unity. TODO: Estimate?
