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
import finufft

# Overload the default measurement set inputs to include the frequencies.
ms_inputs = namedtuple(
    'ms_inputs', ParameterizedGain.ms_inputs._fields + ('CHAN_FREQ',)
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

    def init_term(self, term_spec, ref_ant, ms_kwargs, term_kwargs):
        """Initialise the gains (and parameters)."""

        gains, gain_flags, params, param_flags = super().init_term(
            term_spec, ref_ant, ms_kwargs, term_kwargs
        )


        # Convert the parameters into gains.
        tec_and_offset_params_to_gains(
            params,
            gains,
            ms_kwargs["CHAN_FREQ"],
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
        _, n_chan, n_ant, n_dir, n_corr = gains.shape

        # We only need the baselines which include the ref_ant.
        sel = np.where((a1 == ref_ant) | (a2 == ref_ant))
        a1 = a1[sel]
        a2 = a2[sel]
        t_map = t_map[sel]
        data = data[sel]
        flags = flags[sel]

        data[flags == 1] = 0  # Ignore UV-cut, otherwise there may be no est.

        utint = np.unique(t_map)
        ufint = np.unique(f_map)

        for ut in utint:
            sel = np.where((t_map == ut) & (a1 != a2))
            ant_map_pq = np.where(a1[sel] == ref_ant, a2[sel], 0)
            ant_map_qp = np.where(a2[sel] == ref_ant, a1[sel], 0)
            ant_map = ant_map_pq + ant_map_qp

            ref_data = np.zeros((n_ant, n_chan, n_corr), dtype=np.complex128)
            counts = np.zeros((n_ant, n_chan), dtype=int)
            np.add.at(
                ref_data,
                ant_map,
                data[sel]
            )
            np.add.at(
                counts,
                ant_map,
                flags[sel] == 0
            )
            np.divide(
                ref_data,
                counts[:, :, None],
                where=counts[:, :, None] != 0,
                out=ref_data
            )

            for uf in ufint:

                fsel = np.where(f_map == uf)[0]
                sel_n_chan = fsel.size

                ###fsel_data = visibilities corresponding to the fsel frequencies
                fsel_data = ref_data[:, fsel]
                valid_ant = fsel_data.any(axis=(1, 2))
                ##in inverse frequency domain
                invfreq = 1./chan_freq

                ##factor for rescaling frequency
                ffactor = 1 #1e8
                invfreq *= ffactor

                #delta_freq is the smallest difference between the frequency values
                delta_freq = invfreq[-2] - invfreq[-1]
                max_tec = 2 * np.pi/ (delta_freq)
                super_res = 10
                nyq_freq = 1./(2*(invfreq.max()-invfreq.min()))
                lim0 = 0.5 * max_tec

                ##choosing resolution
                n = int(super_res * max_tec/ nyq_freq)
                #domain to pick tec_est from
                fft_freq = np.linspace(-lim0, lim0, n)

                if n_corr == 1:
                    n_param_tec = 1
                elif n_corr in (2, 4):
                    n_param_tec = 2
                else:
                    raise ValueError("Unsupported number of correlations.")

                tec_est = np.zeros((n_ant, n_param_tec), dtype=np.float64)

                for p in range(n_ant):
                    for k in range(n_param_tec):
                        vis_finufft = finufft.nufft1d3(invfreq,
                                                       fsel_data[p, :, k],
                                                       fft_freq,
                                                       eps=1e-11, isign=-1)
                        fft_data_pk = np.abs(vis_finufft)
                        tec_est[p, k] = fft_freq[np.argmax(fft_data_pk)]


                tec_est[~valid_ant, :] = 0

                for t, p, q in zip(t_map[sel], a1[sel], a2[sel]):
                    if p == ref_ant:
                        params[t, uf, q, 0, 1] = -tec_est[q, 0]
                        if n_corr > 1:
                            params[t, uf, q, 0, 3] = -tec_est[q, 1]
                    elif q == ref_ant:
                        params[t, uf, p, 0, 1] = tec_est[p, 0]
                        if n_corr > 1:
                            params[t, uf, p, 0, 3] = tec_est[p, 1]

        # Convert the parameters into gains.
        tec_and_offset_params_to_gains(
            params,
            gains,
            ms_kwargs["CHAN_FREQ"],
            term_kwargs[f"{self.name}_param_freq_map"],
        )

        # Apply flags to gains and parameters.
        apply_param_flags_to_params(param_flags, params, 1)
        apply_gain_flags_to_gains(gain_flags, gains)

        return gains, gain_flags, params, param_flags
