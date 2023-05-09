import numpy as np
from quartical.gains.converter import amp_trig_to_complex
from quartical.gains.gain import Gain
from quartical.gains.leakage.kernel import leakage_solver, leakage_args


class Leakage(Gain):

    solver = staticmethod(leakage_solver)
    term_args = leakage_args

    native_to_converted = (
        (0, (np.abs,)),
        (0, (np.angle, np.cos)),
        (1, (np.angle, np.sin))
    )
    converted_to_native = (
        (3, amp_trig_to_complex),
    )
    converted_dtype = np.float64
    native_dtype = np.complex128

    def __init__(self, term_name, term_opts):

        super().__init__(term_name, term_opts)
