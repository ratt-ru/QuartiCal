from quartical.gains.gain import Gain, gain_spec_tup, param_spec_tup
from quartical.gains.tec.kernel import tec_solver, tec_args
import numpy as np
import finufft

class TEC(Gain):

    solver = tec_solver
    term_args = tec_args

    def __init__(self, term_name, term_opts, data_xds, coords, tipc, fipc):

        Gain.__init__(self, term_name, term_opts, data_xds, coords, tipc, fipc)

        parameterisable = ["XX", "YY", "RR", "LL"]

        self.parameterised_corr = \
            [ct for ct in self.corr_types if ct in parameterisable]
        self.n_param = 2 * len(self.parameterised_corr)

        self.gain_chunk_spec = gain_spec_tup(self.n_tipc_g,
                                             self.n_fipc_g,
                                             (self.n_ant,),
                                             (self.n_dir,),
                                             (self.n_corr,))
        self.param_chunk_spec = param_spec_tup(self.n_tipc_g,
                                               self.n_fipc_p,
                                               (self.n_ant,),
                                               (self.n_dir,),
                                               (self.n_param,))

        self.gain_axes = ("gain_t", "gain_f", "ant", "dir", "corr")
        self.param_axes = ("param_t", "param_f", "ant", "dir", "param")

    def make_xds(self):

        xds = Gain.make_xds(self)

        param_template = ["phase_offset_{}", "TEC_{}"]

        param_labels = [pt.format(ct) for ct in self.parameterised_corr
                        for pt in param_template]

        xds = xds.assign_coords({"param": np.array(param_labels),
                                 "param_t": self.gain_times,
                                 "param_f": self.param_freqs})
        xds = xds.assign_attrs({"GAIN_SPEC": self.gain_chunk_spec,
                                "PARAM_SPEC": self.param_chunk_spec,
                                "GAIN_AXES": self.gain_axes,
                                "PARAM_AXES": self.param_axes})

        return xds

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


    @staticmethod 
    def init_term(gain, param, term_ind, term_spec, term_opts, ref_ant, **kwargs ):
        """Initialise the gains (and parameters)."""
 
        loaded = Gain.init_term(
            gain, param, term_ind, term_spec, term_opts, ref_ant, **kwargs
        )

        if loaded or not term_opts.initial_estimate: 
            return

        # import ipdb; ipdb.set_trace()

        data = kwargs["data"] # (row, chan, corr)
        flags = kwargs["flags"] # (row, chan)
        a1 = kwargs["a1"]
        a2 = kwargs["a2"]
        chan_freq = kwargs["chan_freqs"]
        t_map = kwargs["t_map_arr"][0, :, term_ind] # time -> solint
        f_map = kwargs["f_map_arr"][1, :, term_ind] # freq -> solint 
        _, n_chan, n_ant, n_dir, n_corr = gain.shape
        n_param = param.shape[4]

        # We only need the baselines which include the ref_ant.
        sel = np.where((a1 == ref_ant) | (a2 == ref_ant))
        a1 = a1[sel]
        a2 = a2[sel]
        t_map = t_map[sel]
        data = data[sel]
        flags = flags[sel]
        data[flags == 1] = 0 # Ignore UV-cut, otherwise there may be no est. 
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
                ffactor = 1
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
                fft_arr = np.zeros((n_ant, n, n_param_tec), dtype=fsel_data.dtype)

                for p in range(n_ant): 
                    for k in range(n_param_tec):
                        vis_finufft = finufft.nufft1d3(invfreq, fsel_data[p, :, k], fft_freq, eps=1e-11, isign=-1)
                        fft_arr[p, :, k] = vis_finufft
                        fft_data_pk = np.abs(vis_finufft)
                        tec_est[p, k] = fft_freq[np.argmax(fft_data_pk)]


                tec_est[~valid_ant, :] = 0

                for t, p, q in zip(t_map[sel], a1[sel], a2[sel]):
                    if p == ref_ant:
                        param[t, uf, q, 0, 1] = -tec_est[q, 0]
                        if n_corr > 1:
                            param[t, uf, q, 0, 3] = -tec_est[q, 1]
                    elif q == ref_ant:
                        param[t, uf, p, 0, 1] = tec_est[p, 0]
                        if n_corr > 1:
                            param[t, uf, p, 0, 3] = tec_est[p, 1]

                np.save("/home/russeeawon/testing/sim_tecest_expt1/tecest.npy", tec_est)
                np.save("/home/russeeawon/testing/sim_tecest_expt1/fftarr.npy", fft_arr)
                np.save("/home/russeeawon/testing/sim_tecest_expt1/fft_freq.npy", fft_freq)
        
        # import ipdb; ipdb.set_trace()
        for ut in utint:
            for f in range(n_chan):
                fm = f_map[f]
                cf = 1j/ chan_freq[f]
                gain[ut, f, :, :, 0] = np.exp(cf * param[ut, fm, :, :, 1])

                if n_corr > 1:
                    gain[ut, f, :, :, -1] = np.exp(cf * param[ut, fm, :, :, 3])