from quartical.gains.complex import Complex, SlowComplex
from quartical.gains.amplitude import Amplitude
from quartical.gains.phase import Phase
from quartical.gains.delay import Delay
from quartical.gains.tec import TEC
from quartical.gains.rotation_measure import RotationMeasure


TERM_TYPES = {"complex": Complex,
              "amplitude": Amplitude,
              "phase": Phase,
              "delay": Delay,
              "slow_complex": SlowComplex,
              "tec": TEC,
              "rotation_measure": RotationMeasure}
