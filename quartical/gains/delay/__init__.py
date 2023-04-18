from quartical.gains.gain import ParameterizedGain
from quartical.gains.delay.kernel import delay_solver, delay_args
from quartical.gains.delay.pure_kernel import pure_delay_solver
import numpy as np


class Delay(ParameterizedGain):

    solver = staticmethod(delay_solver)
    term_args = delay_args

    def __init__(self, term_name, term_opts):

        super().__init__(term_name, term_opts)

        self.gain_axes = (
            "gain_time",
            "gain_freq",
            "antenna",
            "direction",
            "correlation"
        )
        self.param_axes = (
            "param_time",
            "param_freq",
            "antenna",
            "direction",
            "param_name"
        )

    @classmethod
    def _make_freq_map(cls, chan_freqs, chan_widths, freq_interval):
        # Overload gain mapping construction - we evaluate it in every channel.
        return np.arange(chan_freqs.size, dtype=np.int32)

    @classmethod
    def make_param_names(cls, correlations):

        # TODO: This is not dasky, unlike the other functions. Delayed?
        parameterisable = ["XX", "YY", "RR", "LL"]

        param_corr = [c for c in correlations if c in parameterisable]

        template = ("phase_offset_{}", "delay_{}")

        return [n.format(c) for c in param_corr for n in template]

    @staticmethod
    def init_term(
        gain, param, term_ind, term_spec, term_opts, ref_ant, **kwargs
    ):
        """Initialise the gains (and parameters)."""

        loaded = super(Delay, Delay).init_term(
            gain, param, term_ind, term_spec, term_opts, ref_ant, **kwargs
        )

        if loaded or not term_opts.initial_estimate:
            return

        data = kwargs["DATA"]  # (row, chan, corr)
        flags = kwargs["FLAG"]  # (row, chan)
        a1 = kwargs["ANTENNA1"]
        a2 = kwargs["ANTENNA2"]
        chan_freq = kwargs["CHAN_FREQ"]
        # TODO: This whole process is a bit dodgy - improve with new changes.
        t_map = kwargs[f"{term_spec.name}-time-map"]
        f_map = kwargs[f"{term_spec.name}-param-freq-map"]
        _, n_chan, n_ant, n_dir, n_corr = gain.shape

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
                n = int(np.ceil(2 ** 15 / sel_n_chan)) * sel_n_chan

                fsel_data = ref_data[:, fsel]
                valid_ant = fsel_data.any(axis=(1, 2))

                fft_data = np.abs(
                    np.fft.fft(fsel_data, n=n, axis=1)
                )
                fft_data = np.fft.fftshift(fft_data, axes=1)

                delta_freq = chan_freq[1] - chan_freq[0]
                fft_freq = np.fft.fftfreq(n, delta_freq)
                fft_freq = np.fft.fftshift(fft_freq)

                delay_est_ind_00 = np.argmax(fft_data[..., 0], axis=1)
                delay_est_00 = fft_freq[delay_est_ind_00]
                delay_est_00[~valid_ant] = 0

                if n_corr > 1:
                    delay_est_ind_11 = np.argmax(fft_data[..., -1], axis=1)
                    delay_est_11 = fft_freq[delay_est_ind_11]
                    delay_est_11[~valid_ant] = 0

                for t, p, q in zip(t_map[sel], a1[sel], a2[sel]):
                    if p == ref_ant:
                        param[t, uf, q, 0, 1] = -delay_est_00[q]
                        if n_corr > 1:
                            param[t, uf, q, 0, 3] = -delay_est_11[q]
                    else:
                        param[t, uf, p, 0, 1] = delay_est_00[p]
                        if n_corr > 1:
                            param[t, uf, p, 0, 3] = delay_est_11[p]

        for ut in utint:
            for f in range(n_chan):
                fm = f_map[f]
                cf = 2j * np.pi * chan_freq[f]

                gain[ut, f, :, :, 0] = np.exp(cf * param[ut, fm, :, :, 1])

                if n_corr > 1:
                    gain[ut, f, :, :, -1] = np.exp(cf * param[ut, fm, :, :, 3])


class PureDelay(Delay):

    solver = staticmethod(pure_delay_solver)

    def __init__(self, term_name, term_opts):

        super().__init__(term_name, term_opts)
