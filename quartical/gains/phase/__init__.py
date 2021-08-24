from quartical.gains.gain import Gain, gain_spec_tup, param_spec_tup
from quartical.gains.phase.kernel import phase_solver, phase_args
from quartical.gains.phase.slow_kernel import phase_solver as slow_phase
import numpy as np


class Phase(Gain):

    solver = phase_solver
    term_args = phase_args

    def __init__(self, term_name, term_opts, data_xds, coords, tipc, fipc):

        Gain.__init__(self, term_name, term_opts, data_xds, coords, tipc, fipc)

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


class SlowPhase(Phase):

    solver = slow_phase

    def __init__(self, term_name, term_opts, data_xds, coords, tipc, fipc):

        Phase.__init__(self, term_name, term_opts, data_xds, coords, tipc,
                       fipc)
