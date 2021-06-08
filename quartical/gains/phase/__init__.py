from quartical.gains.gain import Gain, gain_spec_tup, param_spec_tup
from quartical.gains.phase.kernel import phase_solver
import numpy as np


class Phase(Gain):

    solver = phase_solver

    def __init__(self, term_name, data_xds, coords, tipc, fipc, opts):

        Gain.__init__(self, term_name, data_xds, coords, tipc, fipc, opts)

        self.n_ppa = 1
        self.gain_chunk_spec = gain_spec_tup(self.n_tipc_g,
                                             self.n_fipc_g,
                                             (self.n_ant,),
                                             (self.n_dir,),
                                             (self.n_corr,))
        self.param_chunk_spec = param_spec_tup(self.n_tipc_g,
                                               self.n_fipc_g,
                                               (self.n_ant,),
                                               (self.n_dir,),
                                               (self.n_ppa,),
                                               (self.n_corr,))
        self.gain_axes = ("gain_t", "gain_f", "ant", "dir", "corr")
        self.param_axes = ("param_t", "param_f", "ant", "dir", "param", "corr")

    def make_xds(self):

        xds = Gain.make_xds(self)

        xds = xds.assign_coords({"param": np.array(["phase"]),
                                 "param_t": self.param_times,
                                 "param_f": self.param_freqs})
        xds = xds.assign_attrs({"GAIN_SPEC": self.gain_chunk_spec,
                                "PARAM_SPEC": self.param_chunk_spec,
                                "GAIN_AXES": self.gain_axes,
                                "PARAM_AXES": self.param_axes})

        return xds