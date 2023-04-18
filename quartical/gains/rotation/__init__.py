from quartical.gains.gain import ParameterizedGain
from quartical.gains.rotation.kernel import rotation_solver, rotation_args


class Rotation(ParameterizedGain):

    solver = staticmethod(rotation_solver)
    term_args = rotation_args

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

        return ["rotation_angle"]
