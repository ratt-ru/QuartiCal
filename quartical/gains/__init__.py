from quartical.gains.complex import Complex
from quartical.gains.phase import Phase
from quartical.gains.delay import Delay


term_types = {"complex": Complex,
              "phase": Phase,
              "delay": Delay}

term_solvers = {"complex": Complex.solver,
                "phase": Phase.solver,
                "delay": Delay.solver}
