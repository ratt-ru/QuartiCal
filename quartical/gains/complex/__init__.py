from quartical.gains.gain import Gain, gain_spec_tup
from quartical.gains.complex.kernel2 import complex_solver, complex_args


class Complex(Gain):

    solver = complex_solver
    term_args = complex_args

    def __init__(self, term_name, term_opts, data_xds, coords, tipc, fipc):

        Gain.__init__(self, term_name, term_opts, data_xds, coords, tipc, fipc)

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
