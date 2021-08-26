from quartical.gains.complex import Complex, SlowComplex
from quartical.gains.phase import Phase
from quartical.gains.delay import Delay
from quartical.gains.tec import TEC


TERM_TYPES = {"complex": Complex,
              "phase": Phase,
              "delay": Delay,
              "slow_complex": SlowComplex,
              "tec": TEC}
