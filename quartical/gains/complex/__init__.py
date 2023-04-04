from quartical.gains.gain import Gain, gain_spec_tup
from quartical.gains.complex.kernel import complex_solver, complex_args
from quartical.gains.complex.diag_kernel import diag_complex_solver


class Complex(Gain):

    solver = complex_solver
    term_args = complex_args

    def __init__(self, term_name, term_opts):

        Gain.__init__(self, term_name, term_opts)

        self.n_ppa = 0
        self.gain_axes = (
            "gain_time",
            "gain_freq",
            "antenna",
            "direction",
            "correlation"
        )


class DiagComplex(Complex):

    solver = diag_complex_solver

    def __init__(self, term_name, term_opts):

        Complex.__init__(self, term_name, term_opts)
