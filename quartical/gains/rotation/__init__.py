import numpy as np
from quartical.gains.conversion import trig_to_angle
from quartical.gains.gain import ParameterizedGain
from quartical.gains.rotation.kernel import rotation_solver, rotation_args


class Rotation(ParameterizedGain):

    solver = staticmethod(rotation_solver)
    term_args = rotation_args

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

        return ["rotation_angle"]
