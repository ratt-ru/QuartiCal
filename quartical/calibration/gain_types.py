from collections import namedtuple
from quartical.kernels.complex import complex_solver
from quartical.kernels.phase import phase_solver


term_types = {"complex": namedtuple("cmplx", ("gains", "flags")),
              "phase": namedtuple("phase", ("gains", "flags", "params"))}

term_solvers = {"complex": complex_solver,
                "phase": phase_solver}
