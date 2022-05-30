from quartical.gains.gain import Gain, gain_spec_tup, param_spec_tup
from quartical.gains.crosshand_phase.kernel import (crosshand_phase_solver,
                                                    crosshand_phase_args)
import numpy as np


class CrosshandPhase(Gain):

    solver = crosshand_phase_solver
    term_args = crosshand_phase_args

    def __init__(self, term_name, term_opts, data_xds, coords, tipc, fipc):

        Gain.__init__(self, term_name, term_opts, data_xds, coords, tipc, fipc)

        parameterisable = ["XX", "RR"]

        self.parameterised_corr = \
            [ct for ct in self.corr_types if ct in parameterisable]
        self.n_param = len(self.parameterised_corr)

        self.gain_chunk_spec = gain_spec_tup(self.n_tipc_g,
                                             self.n_fipc_g,
                                             (self.n_ant,),
                                             (self.n_dir,),
                                             (self.n_corr,))
        self.param_chunk_spec = param_spec_tup(self.n_tipc_g,
                                               self.n_fipc_g,
                                               (self.n_ant,),
                                               (self.n_dir,),
                                               (self.n_param,))
        self.gain_axes = ("gain_t", "gain_f", "ant", "dir", "corr")
        self.param_axes = ("param_t", "param_f", "ant", "dir", "param")

    def make_xds(self):

        xds = Gain.make_xds(self)

        param_template = ["crosshand_phase_{}"]

        param_labels = [pt.format(ct) for ct in self.parameterised_corr
                        for pt in param_template]

        xds = xds.assign_coords({"param": np.array(param_labels),
                                 "param_t": self.param_times,
                                 "param_f": self.param_freqs})
        xds = xds.assign_attrs({"GAIN_SPEC": self.gain_chunk_spec,
                                "PARAM_SPEC": self.param_chunk_spec,
                                "GAIN_AXES": self.gain_axes,
                                "PARAM_AXES": self.param_axes})

        return xds

    @staticmethod
    def init_term(gain, param, term_ind, term_spec, ref_ant, **kwargs):
        """Initialise the gains (and parameters)."""

        loaded = Gain.init_term(
            gain, param, term_ind, term_spec, ref_ant, **kwargs
        )

        if loaded:
            return

        data = kwargs["data"]  # (row, chan, corr)
        model = kwargs["model"]  # (row, chan, corr)
        flags = kwargs["flags"]  # (row, chan)
        t_map = kwargs["t_map_arr"][0, :, term_ind]  # time -> solint
        f_map = kwargs["f_map_arr"][0, :, term_ind]  # freq -> solint

        offdiag_model = model[..., (1, 2)].sum(axis=2)
        offdiag_data = data[..., (1, 2)]

        model_inv = np.divide(
            np.conj(offdiag_model),
            np.abs(offdiag_model),
            where=flags[..., None] == 0
        )

        est_data = (offdiag_data * model_inv).sum(0)
        offdiag_data_abs = np.abs(offdiag_data).sum(0)
        est_data = est_data[..., 0] + np.conj(est_data[..., 1])
        offdiag_data_abs = \
            offdiag_data_abs[..., 0] + np.conj(offdiag_data_abs[..., 1])

        crosshand_phase = np.angle(
            np.divide(est_data, offdiag_data_abs, where=offdiag_data_abs != 0)
        )

        param[0, :, :, 0, 0] = crosshand_phase[:, None]
        gain[0, :, :, 0, 0] = np.exp(1j*crosshand_phase[:, None])
