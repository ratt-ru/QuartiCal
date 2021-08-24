from quartical.gains.complex import Complex, SlowComplex
from quartical.gains.phase import Phase, SlowPhase
from quartical.gains.delay import Delay, SlowDelay
from quartical.gains.tec import TEC


TERM_TYPES = {"complex": Complex,
              "phase": Phase,
              "delay": Delay,
              "slow_complex": SlowComplex,
              "slow_phase": SlowPhase,
              "slow_delay": SlowDelay,
              "tec": TEC}
