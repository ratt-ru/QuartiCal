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
