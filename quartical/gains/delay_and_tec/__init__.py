import numpy as np
import finufft
from collections import namedtuple
from scipy.integrate import cumulative_trapezoid
from quartical.gains.conversion import no_op, trig_to_angle
from quartical.gains.parameterized_gain import ParameterizedGain
from quartical.gains.delay_and_tec.kernel import (
    delay_and_tec_solver,
    delay_and_tec_params_to_gains
)
from quartical.gains.general.flagging import (
    apply_gain_flags_to_gains,
    apply_param_flags_to_params
)

# Overload the default measurement set inputs to include the frequencies.
ms_inputs = namedtuple(
    'ms_inputs', ParameterizedGain.ms_inputs._fields + ('CHAN_FREQ',)
)


class DelayAndTec(ParameterizedGain):

    solver = staticmethod(delay_and_tec_solver)
    ms_inputs = ms_inputs

    native_to_converted = (
        (1, (no_op,)),
        (1, (no_op,))
    )
    converted_to_native = (
        (1, no_op),
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

        template = ("tec_{}", "delay_{}")

        return [n.format(c) for c in param_corr for n in template]

    def init_term(self, term_spec, ref_ant, ms_kwargs, term_kwargs, meta=None):
        """Initialise the gains (and parameters)."""

        gains, gain_flags, params, param_flags = super().init_term(
            term_spec, ref_ant, ms_kwargs, term_kwargs
        )

        # Convert the parameters into gains.
        delay_and_tec_params_to_gains(
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
        n_tint = utint.size
        n_fint = ufint.size
        # NOTE: This determines the number of subintervals which are used to
        # estimate the delay and tec values. More subintervals will typically
        # yield better estimates at the cost of SNR. TODO: Thus doesn't factor
        # in the flagging behaviour which may make some estimates worse than
        # others. Should we instead only consider unflagged regions or weight
        # the mean calaculation?
        n_subint = max(int(np.ceil(n_chan / 512)), 2)

        if n_corr == 1:
            n_paramt = 1 #number of parameters in TEC
            n_paramk = 1 #number of parameters in delay
        elif n_corr in (2, 4):
            n_paramt = 2
            n_paramk = 2
        else:
            raise ValueError("Unsupported number of correlations.")

        n_param = params.shape[-1]
        assert n_param == n_paramk + n_paramt

        ctz_delay = np.empty((n_tint, n_fint, n_subint, n_ant, n_paramk))
        gradients = np.empty((n_tint, n_fint, n_subint))

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
                fsel_nchan = fsel.size
                fsel_chan = chan_freq[fsel]

                fsel_data = ref_data[:, fsel]
                valid_ant = fsel_data.any(axis=(1, 2))

                subint_stride = int(np.ceil(fsel_nchan / n_subint))

                for i, si in enumerate(range(0, fsel_nchan, subint_stride)):

                    si_sel = slice(si, si + subint_stride)

                    subint_data = fsel_data[:, si_sel]

                    # NOTE: Collapse correlation axis when term is scalar.
                    if self.scalar:
                        subint_data[..., :] = subint_data.sum(
                            axis=-1, keepdims=True
                        )

                    subint_freq = fsel_chan[si_sel]
                    subint_ifreq = 1/subint_freq

                    gradients[ut, uf, i], _ = np.polyfit(
                        subint_freq, subint_ifreq, deg=1
                    )

                    #Initialise array to contain delay and tec estimates
                    delay_est = np.zeros((n_ant, n_paramk), dtype=np.float64)
                    delay_est, fft_arrk, fft_freqk = self.initial_estimates(
                        subint_data, delay_est, subint_freq, valid_ant, type="k"
                    )

                    psk = (fft_arrk * fft_arrk.conj()).real
                    for ai in range(n_ant):
                        ctz = cumulative_trapezoid(psk[ai], fft_freqk, axis=0)
                        for p in range(n_paramk):
                            half_max = 0.5 * ctz[:, p].max()
                            median_i = np.argwhere(ctz[:, p] >= half_max)[0]
                            ctz_delay[ut, uf, i, ai, p] = fft_freqk[median_i]

                    # TODO: Investigate whether it is better to use the delay
                    # estimates with a higher number of subintervals over the
                    # median of the power spectrum.
                    ctz_delay[ut, uf, i] = delay_est

                # Zero the reference antenna/antennas without data.
                ctz_delay[ut, uf, :, ~valid_ant] = 0

        gradients = gradients[..., None, None]  # Add antenna and param axes.
        tec_numerator = np.diff(ctz_delay, axis=2)
        tec_denominator = np.diff(gradients, axis=2)
        tec_est = (tec_numerator / tec_denominator)
        delay_est = ctz_delay[:, :, :-1] - gradients[:, :, :-1] * tec_est

        tec_est = tec_est.mean(axis=2)
        delay_est = delay_est.mean(axis=2)

        # Flip the estimates on antennas > reference as they correspond to G^H.
        tec_est[:, :, ref_ant:] = -tec_est[:, :, ref_ant:]
        delay_est[:, :, ref_ant:] = -delay_est[:, :, ref_ant:]

        params[:, :, :, 0, 0::2] = tec_est
        params[:, :, :, 0, 1::2] = delay_est

        delay_and_tec_params_to_gains(
            params,
            gains,
            ms_kwargs["CHAN_FREQ"],
            term_kwargs[f"{self.name}_param_freq_map"],
        )

        apply_param_flags_to_params(param_flags, params, 0)
        apply_gain_flags_to_gains(gain_flags, gains)

        delay_and_tec_params_to_gains(
            params,
            gains,
            ms_kwargs["CHAN_FREQ"],
            term_kwargs[f"{self.name}_param_freq_map"],
        )

        return gains, gain_flags, params, param_flags


    def initial_estimates(self, fsel_data, est_arr, freq, valid_ant, type="k"):
        """
        This function return the set of initial estimates for each param in params.
        type is either k (delay) or t (tec).

        """

        n_ant, n_param = est_arr.shape

        dfreq = np.abs(freq[-2] - freq[-1])
        #Maximum reconstructable delta
        max_delta = 1/ dfreq
        nyq_rate = 1./ (2*(freq.max() - freq.min()))
        nbins = int(max_delta/ nyq_rate)

        if type == "k":
            nbins = max(2 * freq.size, 4096)  # Need adequate samples.
            fft_freq = np.fft.fftfreq(nbins, dfreq)
            fft_freq = np.fft.fftshift(fft_freq)
            #when not using finufft
            # fft_arr = np.abs(
            #     np.fft.fft(fsel_data, n=nbins, axis=1)
            # )
            # fft_arr = np.fft.fftshift(fft_arr, axes=1)
        elif type == "t":
            nbins = max(2 * freq.size, 4096)
            ##factor for rescaling frequency
            ffactor = 1 #1e8
            freq *= ffactor
            fft_freq = np.linspace(0.5*-max_delta, 0.5*max_delta, nbins)
        else:
            raise TypeError("Unsupported parameter type.")

        fft_arr = np.zeros((n_ant, nbins, n_param), dtype=fsel_data.dtype)

        for i in range(n_param):
            if i == 0:
                datak = fsel_data[:, :, 0]
            elif i == 1:
                datak = fsel_data[:, :, -1]
            else:
                raise ValueError("Unsupported number of parameters for delay.")

            # TODO: Why does the normal FFT disagree with the nufft? Ideally
            # we want to use numpy as it reduces our dependencies.
            # vis_fft = np.fft.fftshift(np.fft.fft(datak.copy(), axis=-1, n=nbins))

            vis_finufft = finufft.nufft1d3(
                2 * np.pi * freq,
                datak.copy(),
                fft_freq,
                eps=1e-6,
                isign=-1
            )

            # fft_ps = (vis_fft * vis_fft.conj()).real
            # nufft_ps = (vis_finufft * vis_finufft.conj()).real

            # plt.plot(fft_freq, fft_ps[1])
            # plt.plot(fft_freq, nufft_ps[1], c="r")
            # plt.show()

            fft_arr[:, :, i] = vis_finufft
            est_arr[:, i] = fft_freq[np.argmax(np.abs(vis_finufft), axis=1)]

        est_arr[~valid_ant] = 0

        return est_arr, fft_arr, fft_freq
