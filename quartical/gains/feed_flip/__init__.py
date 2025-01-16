import numpy as np
from quartical.gains.gain import Gain
from quartical.gains.conversion import amp_trig_to_complex


class FeedFlip(Gain):

    solver = None # This is not a solvable term.
    # Conversion functions required for interpolation NOTE: Non-parameterised
    # gains will always be reinterpreted and parameterised in amplitude and
    # phase for the sake of simplicity.
    # TODO: Make this a real valued term - took simple approach for now.
    native_to_converted = (
        (0, (np.abs,)),
        (0, (np.angle, np.cos)),
        (1, (np.angle, np.sin))
    )
    converted_to_native = (
        (3, amp_trig_to_complex),
    )
    converted_dtype = np.float64
    native_dtype = np.complex128

    def __init__(self, term_name, term_opts):

        super().__init__(term_name, term_opts)

        self.time_interval = 0
        self.freq_interval = 0

    def init_term(self, term_spec, ref_ant, ms_kwargs, term_kwargs, meta=None):
        """Initialise the gains (and parameters)."""

        (_, _, gain_shape, _) = term_spec

        gains = np.ones(gain_shape, dtype=np.complex128)

        if gain_shape[-1] == 4:
            gains[..., (0, 3)] = 0  # 2-by-2 antidiagonal.
        else:
            raise ValueError(
                "Feed flip unsupported for less than four correlations"
            )

        gain_flags = np.zeros(gains.shape[:-1], dtype=np.int8)

        return gains, gain_flags
