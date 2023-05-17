import numpy as np
from quartical.gains.conversion import trig_to_angle
from quartical.gains.parameterized_gain import ParameterizedGain
from quartical.gains.rotation.kernel import (
    rotation_solver,
    rotation_params_to_gains
)


class Rotation(ParameterizedGain):

    solver = staticmethod(rotation_solver)

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

    def init_term(self, term_spec, ref_ant, ms_kwargs, term_kwargs):
        """Initialise the gains (and parameters)."""

        gains, params = super().init_term(
            term_spec, ref_ant, ms_kwargs, term_kwargs
        )

        # Convert the parameters into gains.
        rotation_params_to_gains(params, gains)

        return gains, params
