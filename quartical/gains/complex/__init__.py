from quartical.gains.gain import Gain
from quartical.gains.complex.kernel import complex_solver, complex_args
from quartical.gains.complex.diag_kernel import diag_complex_solver


class Complex(Gain):

    solver = staticmethod(complex_solver)
    term_args = complex_args

    def __init__(self, term_name, term_opts):

        super().__init__(term_name, term_opts)


class DiagComplex(Complex):

    solver = staticmethod(diag_complex_solver)

    def __init__(self, term_name, term_opts):

        super().__init__(term_name, term_opts)
