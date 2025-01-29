import numpy as np
import finufft
from collections import namedtuple
from scipy.constants import c as lightspeed
from quartical.gains.conversion import no_op
from quartical.gains.parameterized_gain import ParameterizedGain
from quartical.gains.rotation_measure.kernel import (
    rm_solver,
    rm_params_to_gains
)
from quartical.gains.general.flagging import (
    apply_gain_flags_to_gains,
    apply_param_flags_to_params
)

# Overload the default measurement set inputs to include the frequencies.
ms_inputs = namedtuple(
    'ms_inputs', ParameterizedGain.ms_inputs._fields + ('CHAN_FREQ',)
)


class RotationMeasure(ParameterizedGain):

    solver = staticmethod(rm_solver)
    ms_inputs = ms_inputs

    native_to_converted = (
        (1, (no_op,)),
    )
    converted_to_native = (
        (1, no_op),
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

        return ["rotation_measure"]

    @staticmethod
    def make_f_maps(chan_freqs, chan_widths, f_int):
        """Internals of the frequency interval mapper."""

        n_chan = chan_freqs.size

        # The leading dimension corresponds (gain, param). For unparameterised
        # gains, the parameter mapping is irrelevant.
        f_map_arr = np.zeros((2, n_chan,), dtype=np.int32)

        if isinstance(f_int, float):
            net_ivl = 0
            bin_num = 0
            for i, ivl in enumerate(chan_widths):
                f_map_arr[1, i] = bin_num
                net_ivl += ivl
                if net_ivl >= f_int:
                    net_ivl = 0
                    bin_num += 1
        else:
            f_map_arr[1, :] = np.arange(n_chan)//f_int

        f_map_arr[0, :] = np.arange(n_chan)

        return f_map_arr

    def init_term(self, term_spec, ref_ant, ms_kwargs, term_kwargs, meta=None):
        """Initialise the gains (and parameters)."""

        gains, gain_flags, params, param_flags = super().init_term(
            term_spec, ref_ant, ms_kwargs, term_kwargs
        )

        chan_freq = ms_kwargs["CHAN_FREQ"]
        lambda_sq = (299792458 / chan_freq) ** 2

        # Convert the parameters into gains.
        rm_params_to_gains(
            params,
            gains,
            lambda_sq,
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

                # fsel_data = visibilities corresponding to the fsel
                # frequencies
                fsel_data = ref_data[:, fsel]
                valid_ant = fsel_data.any(axis=(1, 2))
                ##in inverse frequency domain
                invfreq2 = 1 / (chan_freq ** 2)

                # delta_freq is the smallest difference between the frequency
                # values
                delta_freq = invfreq2[-2] - invfreq2[-1]
                max_rm = 2 * np.pi / delta_freq
                super_res = 1000 #100
                nyq_freq = 1./(2*(invfreq2.max() - invfreq2.min()))
                lim0 = 0.5 * max_rm

                # choosing resolution
                n = int(super_res * max_rm / nyq_freq)
                # domain to pick tec_est from
                fft_freq = np.linspace(-lim0, lim0, n)

                if n_corr != 4:
                    raise ValueError(
                        f"Rotation measure term does not support {n_corr} "
                        f"correlation data."
                    )

                rm_est = np.zeros((n_ant, 1), dtype=np.float64)
                fft_arr = np.zeros(
                    (n_ant, n, 1), dtype=fsel_data.dtype
                )

                vis_finufft = finufft.nufft1d3(
                    invfreq2,
                    fsel_data[:, :, 0] - 1j * fsel_data[:, :, 1],
                    fft_freq,
                    eps=1e-6,
                    isign=-1
                )
                fft_arr[:, :, 0] = vis_finufft
                fft_data_pk = np.abs(vis_finufft)
                rm_est[:, 0] = fft_freq[np.argmax(fft_data_pk, axis=1)]

                rm_est[~valid_ant, :] = 0
                # NOTE: Correct RM values as we assumed (c**2/nu**2) * RM i.e.
                # the peak will be at c**2 * RM.
                rm_est /= lightspeed ** 2

                for t, p, q in zip(t_map[sel], a1[sel], a2[sel]):
                    if p == ref_ant:
                        params[t, uf, q, 0, 0] = -rm_est[q, 0]
                    elif q == ref_ant:
                        params[t, uf, p, 0, 0] = rm_est[p, 0]

        # Convert the parameters into gains.
        rm_params_to_gains(
            params,
            gains,
            lambda_sq,
            term_kwargs[f"{self.name}_param_freq_map"],
        )

        # Apply flags to gains and parameters.
        apply_param_flags_to_params(param_flags, params, 1)
        apply_gain_flags_to_gains(gain_flags, gains)

        return gains, gain_flags, params, param_flags
