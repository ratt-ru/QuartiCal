import numpy as np
from quartical.gains.conversion import no_op, trig_to_angle
from quartical.gains.gain import ParameterizedGain
from quartical.gains.tec.kernel import (
    tec_solver,
    tec_args,
    tec_params_to_gains
)


class TEC(ParameterizedGain):

    solver = staticmethod(tec_solver)
    term_args = tec_args

    native_to_converted = (
        (0, (np.cos,)),
        (1, (np.sin,)),
        (1, (no_op,))
    )
    converted_to_native = (
        (2, trig_to_angle),
        (1, no_op)
    )
    converted_dtype = np.float64
    native_dtype = np.float64

    def __init__(self, term_name, term_opts):

        super().__init__(term_name, term_opts)

    @classmethod
    def _make_freq_map(cls, chan_freqs, chan_widths, freq_interval):
        # Overload gain mapping construction - we evaluate it in every channel.
        return np.arange(chan_freqs.size, dtype=np.int32)

    @classmethod
    def make_param_names(cls, correlations):

        # TODO: This is not dasky, unlike the other functions. Delayed?
        parameterisable = ["XX", "YY", "RR", "LL"]

        param_corr = [c for c in correlations if c in parameterisable]

        template = ("phase_offset_{}", "TEC_{}")

        return [n.format(c) for c in param_corr for n in template]

    def init_term(self, term_spec, ref_ant, ms_kwargs, term_kwargs):
        """Initialise the gains (and parameters)."""

        gains, params = super().init_term(
            term_spec, ref_ant, ms_kwargs, term_kwargs
        )

        # Convert the parameters into gains.
        tec_params_to_gains(
            params,
            gains,
            ms_kwargs["CHAN_FREQ"],
            term_kwargs[f"{self.name}_param_freq_map"],
        )

        return gains, params
