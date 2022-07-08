from quartical.gains.complex import Complex, DiagComplex
from quartical.gains.amplitude import Amplitude
from quartical.gains.phase import Phase
from quartical.gains.delay import Delay
from quartical.gains.tec import TEC
from quartical.gains.rotation_measure import RotationMeasure
from quartical.gains.crosshand_phase import CrosshandPhase


TERM_TYPES = {"complex": Complex,
              "diag_complex": DiagComplex,
              "amplitude": Amplitude,
              "phase": Phase,
              "delay": Delay,
              "tec": TEC,
              "rotation_measure": RotationMeasure,
              "crosshand_phase": CrosshandPhase}
