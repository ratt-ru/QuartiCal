from quartical.gains.gain import Gain, gain_spec_tup
from quartical.gains.leakage.kernel import leakage_solver, leakage_args


class Leakage(Gain):

    solver = leakage_solver
    term_args = leakage_args

    def __init__(self, term_name, term_opts, data_xds, coords, tipc, fipc):

        Gain.__init__(self, term_name, term_opts, data_xds, coords, tipc, fipc)

        if self.n_corr != 4:
            raise ValueError(
                "Leakage terms are only supported for 4 correlation data."
            )

        self.n_ppa = 0
        self.gain_chunk_spec = gain_spec_tup(self.n_tipc_g,
                                             self.n_fipc_g,
                                             (self.n_ant,),
                                             (self.n_dir,),
                                             (self.n_corr,))
        self.gain_axes = ("gain_t", "gain_f", "ant", "dir", "corr")

    def make_xds(self):

        xds = Gain.make_xds(self)

        xds = xds.assign_attrs({"GAIN_SPEC": self.gain_chunk_spec,
                                "GAIN_AXES": self.gain_axes})

        return xds
