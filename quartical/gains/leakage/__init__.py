from quartical.gains.gain import Gain
from quartical.gains.leakage.kernel import leakage_solver, leakage_args


class Leakage(Gain):

    solver = staticmethod(leakage_solver)
    term_args = leakage_args

    def __init__(self, term_name, term_opts):

        super().__init__(term_name, term_opts)
