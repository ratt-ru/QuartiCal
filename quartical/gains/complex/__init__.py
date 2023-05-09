import numpy as np
from quartical.gains.gain import Gain
from quartical.gains.converter import amp_trig_to_complex
from quartical.gains.complex.kernel import complex_solver, complex_args
from quartical.gains.complex.diag_kernel import diag_complex_solver


class Complex(Gain):

    solver = staticmethod(complex_solver)
    term_args = complex_args
    # Conversion functions required for interpolation NOTE: Non-parameterised
    # gains will always be reinterpreted and parameterised in amplitude and
    # phase for the sake of simplicity.
    conversion_functions = (
        (0, (np.abs,)),
        (0, (np.angle, np.cos)),
        (1, (np.angle, np.sin))
    )
    reversion_functions = (
        (3, amp_trig_to_complex),
    )
    converted_dtype = np.float64
    native_dtype = np.complex128

    def __init__(self, term_name, term_opts):

        super().__init__(term_name, term_opts)


class DiagComplex(Complex):

    solver = staticmethod(diag_complex_solver)

    def __init__(self, term_name, term_opts):

        super().__init__(term_name, term_opts)
