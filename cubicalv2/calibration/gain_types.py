from collections import namedtuple
from cubicalv2.kernels.complex import complex_solver
from cubicalv2.kernels.phase import phase_solver


term_types = {"complex": namedtuple("cmplx", ("gains", "flags", "parms")),
              "phase": namedtuple("phase", ("gains", "flags", "parms"))}

term_solvers = {"cmplx": complex_solver,
                "phase": phase_solver}
