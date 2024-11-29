from quartical.gains.complex import Complex, DiagComplex
from quartical.gains.amplitude import Amplitude
from quartical.gains.phase import Phase
from quartical.gains.delay import Delay
from quartical.gains.delay_and_offset import DelayAndOffset
from quartical.gains.tec_and_offset import TecAndOffset
from quartical.gains.rotation import Rotation
from quartical.gains.rotation_measure import RotationMeasure
from quartical.gains.crosshand_phase import CrosshandPhase, CrosshandPhaseNullV
from quartical.gains.leakage import Leakage
from quartical.gains.delay_and_tec import DelayAndTec
from quartical.gains.parallactic_angle import ParallacticAngle


TERM_TYPES = {
    "complex": Complex,
    "diag_complex": DiagComplex,
    "amplitude": Amplitude,
    "phase": Phase,
    "delay": Delay,
    "delay_and_offset": DelayAndOffset,
    "tec_and_offset": TecAndOffset,
    "rotation": Rotation,
    "rotation_measure": RotationMeasure,
    "crosshand_phase": CrosshandPhase,
    "crosshand_phase_null_v": CrosshandPhaseNullV,
    "leakage": Leakage,
    "delay_and_tec": DelayAndTec,
    "parallactic_angle": ParallacticAngle
}
