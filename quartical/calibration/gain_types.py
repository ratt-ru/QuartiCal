from collections import namedtuple
from quartical.kernels.complex import complex_solver
from quartical.kernels.phase import phase_solver
from quartical.kernels.delay import delay_solver
from quartical.kernels.kalman import kalman_solver
import numpy as np
import xarray


term_solvers = {"complex": complex_solver,
                "phase": phase_solver,
                "delay": delay_solver,
                "kalman": kalman_solver}

gain_spec_tup = namedtuple("gain_spec_tup",
                           "tchunk fchunk achunk dchunk cchunk")
param_spec_tup = namedtuple("param_spec_tup",
                            "tchunk fchunk achunk dchunk pchunk cchunk")


class Gain:

    def __init__(self, term_name, data_xds, tipc, fipc, opts):

        self.name = term_name
        self.index = opts.solver_gain_terms.index(self.name)
        self.dd_term = getattr(opts, f"{self.name}_direction_dependent")
        self.type = getattr(opts, f"{self.name}_type")
        self.n_chan = data_xds.dims["chan"]
        self.n_ant = data_xds.dims["ant"]
        self.n_dir = data_xds.dims["dir"]
        self.n_corr = data_xds.dims["corr"]
        self.id_fields = {f: data_xds.attrs[f]
                          for f in ["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"]}
        self.utime_chunks = list(map(int, data_xds.UTIME_CHUNKS))
        self.freq_chunks = list(map(int, data_xds.chunks["chan"]))
        self.n_t_chunk = len(self.utime_chunks)
        self.n_f_chunk = len(self.freq_chunks)

        self.n_tipc = tuple(map(int, tipc[:, self.index]))
        self.n_tint = np.sum(self.n_tipc)

        self.n_fipc = tuple(map(int, fipc[:, self.index]))
        self.n_fint = np.sum(self.n_fipc)

    def make_xds(self):

        # Set up an xarray.Dataset describing the gain term.
        xds = xarray.Dataset(
            coords={"time_int": ("time_int",
                                 np.arange(self.n_tint, dtype=np.int32)),
                    "freq_int": ("freq_int",
                                 np.arange(self.n_fint, dtype=np.int32)),
                    "ant": ("ant", np.arange(self.n_ant, dtype=np.int32)),
                    "dir": ("dir", np.arange(self.n_dir if self.dd_term else 1,
                                             dtype=np.int32)),
                    "corr": ("corr", np.arange(self.n_corr, dtype=np.int32)),
                    "t_chunk": ("t_chunk",
                                np.arange(self.n_t_chunk, dtype=np.int32)),
                    "f_chunk": ("f_chunk",
                                np.arange(self.n_f_chunk, dtype=np.int32))},
            attrs={"NAME": self.name,
                   "TYPE": self.type,
                   **self.id_fields})

        return xds


class Complex(Gain):

    def __init__(self, term_name, data_xds, tipc, fipc, opts):

        Gain.__init__(self, term_name, data_xds, tipc, fipc, opts)

        self.n_ppa = 0
        self.gain_chunk_spec = gain_spec_tup(self.n_tipc,
                                             self.n_fipc,
                                             (self.n_ant,),
                                             (self.n_dir,),
                                             (self.n_corr,))

    def make_xds(self):

        xds = Gain.make_xds(self)

        xds = xds.assign_attrs({"GAIN_SPEC": self.gain_chunk_spec})

        return xds


class Phase(Gain):

    def __init__(self, term_name, data_xds, tipc, fipc, opts):

        Gain.__init__(self, term_name, data_xds, tipc, fipc, opts)

        self.n_ppa = 1
        self.gain_chunk_spec = gain_spec_tup(self.n_tipc,
                                             self.n_fipc,
                                             (self.n_ant,),
                                             (self.n_dir,),
                                             (self.n_corr,))
        self.param_chunk_spec = param_spec_tup(self.n_tipc,
                                               self.n_fipc,
                                               (self.n_ant,),
                                               (self.n_dir,),
                                               (self.n_ppa,),
                                               (self.n_corr,))

    def make_xds(self):

        xds = Gain.make_xds(self)

        xds = xds.assign_coords({"param": np.arange(self.n_ppa)})
        xds = xds.assign_attrs({"GAIN_SPEC": self.gain_chunk_spec,
                                "PARAM_SPEC": self.param_chunk_spec})

        return xds


class Delay(Gain):

    def __init__(self, term_name, data_xds, tipc, fipc, opts):

        Gain.__init__(self, term_name, data_xds, tipc, fipc, opts)

        self.n_ppa = 2
        self.gain_chunk_spec = gain_spec_tup(self.utime_chunks,
                                             self.freq_chunks,
                                             (self.n_ant,),
                                             (self.n_dir,),
                                             (self.n_corr,))
        self.param_chunk_spec = param_spec_tup(self.n_tipc,
                                               self.n_fipc,
                                               (self.n_ant,),
                                               (self.n_dir,),
                                               (self.n_ppa,),
                                               (self.n_corr,))

    def make_xds(self):

        xds = Gain.make_xds(self)

        xds = xds.assign_coords({"param": np.arange(self.n_ppa),
                                 "time": np.arange(sum(self.utime_chunks)),
                                 "freq": np.arange(sum(self.freq_chunks))})
        xds = xds.assign_attrs({"GAIN_SPEC": self.gain_chunk_spec,
                                "PARAM_SPEC": self.param_chunk_spec})

        return xds


term_types = {"complex": Complex,
              "phase": Phase,
              "kalman": Complex,
              "delay": Delay}
