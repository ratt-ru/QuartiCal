from quartical.gains.complex import Complex, SlowComplex, DiagComplex
from quartical.gains.amplitude import Amplitude
from quartical.gains.phase import Phase
from quartical.gains.delay import Delay
from quartical.gains.tec import TEC
from quartical.gains.rotation_measure import RotationMeasure


TERM_TYPES = {"complex": Complex,
              "slow_complex": SlowComplex,
              "diag_complex": DiagComplex,
              "amplitude": Amplitude,
              "phase": Phase,
              "delay": Delay,
              "tec": TEC,
              "rotation_measure": RotationMeasure}
