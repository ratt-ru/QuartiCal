import numpy as np
from collections import namedtuple
from quartical.gains.conversion import no_op, trig_to_angle
from quartical.gains.parameterized_gain import ParameterizedGain
from quartical.gains.tec_and_offset.kernel import (
    tec_and_offset_solver,
    tec_and_offset_params_to_gains
)
from quartical.gains.general.flagging import (
    apply_gain_flags_to_gains,
    apply_param_flags_to_params
)
from quartical.gains.general.estimates import estimate_delay_and_tec


# Overload the default measurement set inputs to include the frequencies.
ms_inputs = namedtuple(
    'ms_inputs', ParameterizedGain.ms_inputs._fields + (
        'CHAN_FREQ',
        'MIN_FREQ',
        'MAX_FREQ'
    )
)


class TecAndOffset(ParameterizedGain):

    solver = staticmethod(tec_and_offset_solver)
    ms_inputs = ms_inputs

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

    def init_term(self, term_spec, ref_ant, ms_kwargs, term_kwargs, meta=None):
        """Initialise the gains (and parameters)."""

        gains, gain_flags, params, param_flags = super().init_term(
            term_spec, ref_ant, ms_kwargs, term_kwargs
        )

        # Convert the parameters into gains.
        tec_and_offset_params_to_gains(
            params,
            gains,
            ms_kwargs["CHAN_FREQ"],
            ms_kwargs["MIN_FREQ"],
            ms_kwargs["MAX_FREQ"],
            term_kwargs[f"{self.name}_param_freq_map"],
        )

        if self.load_from or not self.initial_estimate:

            apply_param_flags_to_params(param_flags, params, 0)
            apply_gain_flags_to_gains(gain_flags, gains)

            return gains, gain_flags, params, param_flags

        data = ms_kwargs["DATA"]  # (row, chan, corr)
        flags = ms_kwargs["FLAG"]  # (row, chan)
        a1 = ms_kwargs["ANTENNA1"]
        a2 = ms_kwargs["ANTENNA2"]
        chan_freq = ms_kwargs["CHAN_FREQ"]
        t_map = term_kwargs[f"{term_spec.name}_time_map"]
        f_map = term_kwargs[f"{term_spec.name}_param_freq_map"]

        # Rescale the channel frequencies.
        scale_factor = (chan_freq.min() + chan_freq.max()) / 2
        scaled_chan_freq = chan_freq / scale_factor

        estimates = estimate_delay_and_tec(
            data,
            flags,
            a1,
            a2,
            t_map,
            f_map,
            scaled_chan_freq,
            gains.shape,
            ref_ant=ref_ant
        )

        # Pack the estimates into the parameter array and undo rescaling.
        params[:, :, :, 0, 1::2] = estimates[..., 1] * scale_factor

        # Convert the parameters into gains.
        tec_and_offset_params_to_gains(
            params,
            gains,
            ms_kwargs["CHAN_FREQ"],
            ms_kwargs["MIN_FREQ"],
            ms_kwargs["MAX_FREQ"],
            term_kwargs[f"{self.name}_param_freq_map"],
        )

        # Apply flags to gains and parameters.
        apply_param_flags_to_params(param_flags, params, 1)
        apply_gain_flags_to_gains(gain_flags, gains)

        return gains, gain_flags, params, param_flags
