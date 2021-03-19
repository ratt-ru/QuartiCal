from collections import namedtuple
from daskms.reads import PARTITION_KEY
from quartical.kernels.complex import complex_solver
from quartical.kernels.phase import phase_solver
from quartical.kernels.delay import delay_solver
from quartical.kernels.kalman import kalman_solver
from quartical.scheduling import dataset_partition
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

    def __init__(self, term_name, data_xds, coords, tipc, fipc, opts):

        self.name = term_name
        self.dd_term = getattr(opts, f"{self.name}_direction_dependent")
        self.type = getattr(opts, f"{self.name}_type")
        self.n_chan = data_xds.dims["chan"]
        self.n_ant = data_xds.dims["ant"]
        self.n_dir = data_xds.dims["dir"]
        self.n_corr = data_xds.dims["corr"]
        self.id_fields = {f: data_xds.attrs[f]
                          for f in opts.input_ms_group_by}
        self.utime_chunks = list(map(int, data_xds.UTIME_CHUNKS))
        self.freq_chunks = list(map(int, data_xds.chunks["chan"]))
        self.n_t_chunk = len(self.utime_chunks)
        self.n_f_chunk = len(self.freq_chunks)

        self.n_tipc = tuple(map(int, tipc))
        self.n_tint = np.sum(self.n_tipc)

        self.n_fipc = tuple(map(int, fipc))
        self.n_fint = np.sum(self.n_fipc)

        self.unique_times = coords["time"]
        self.unique_freqs = coords["freq"]

        self.interval_times = coords[f"{self.name}_mean_time"]
        self.interval_freqs = coords[f"{self.name}_mean_freq"]

        self.additional_args = []
        self.partition = {**dict(dataset_partition(data_xds)),
                          PARTITION_KEY: getattr(data_xds, PARTITION_KEY)}

    def make_xds(self):

        # Set up an xarray.Dataset describing the gain term.
        xds = xarray.Dataset(
            data_vars={},
            coords={"ant": ("ant", np.arange(self.n_ant, dtype=np.int32)),
                    "dir": ("dir", np.arange(self.n_dir if self.dd_term else 1,
                                             dtype=np.int32)),
                    "corr": ("corr", np.arange(self.n_corr, dtype=np.int32)),
                    "t_chunk": ("t_chunk",
                                np.arange(self.n_t_chunk, dtype=np.int32)),
                    "f_chunk": ("f_chunk",
                                np.arange(self.n_f_chunk, dtype=np.int32)),
                    "t_int": ("t_int", self.interval_times),
                    "f_int": ("f_int", self.interval_freqs)},
            attrs={"NAME": self.name,
                   "TYPE": self.type,
                   "ARGS": self.additional_args,
                   **self.id_fields,
                   **self.partition})

        return xds


class Complex(Gain):

    def __init__(self, term_name, data_xds, coords, tipc, fipc, opts):

        Gain.__init__(self, term_name, data_xds, coords, tipc, fipc, opts)

        self.n_ppa = 0
        self.gain_chunk_spec = gain_spec_tup(self.n_tipc,
                                             self.n_fipc,
                                             (self.n_ant,),
                                             (self.n_dir,),
                                             (self.n_corr,))
        self.gain_axes = ("t_int", "f_int", "ant", "dir", "corr")

    def make_xds(self):

        xds = Gain.make_xds(self)

        xds = xds.assign_attrs({"GAIN_SPEC": self.gain_chunk_spec,
                                "GAIN_AXES": self.gain_axes})

        return xds


class Phase(Gain):

    def __init__(self, term_name, data_xds, coords, tipc, fipc, opts):

        Gain.__init__(self, term_name, data_xds, coords, tipc, fipc, opts)

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
        self.gain_axes = ("t_int", "f_int", "ant", "dir", "corr")
        self.param_axes = \
            ("t_int", "f_int", "ant", "dir", "param", "corr")

    def make_xds(self):

        xds = Gain.make_xds(self)

        xds = xds.assign_coords({"param": np.array(["phase"])})
        xds = xds.assign_attrs({"GAIN_SPEC": self.gain_chunk_spec,
                                "PARAM_SPEC": self.param_chunk_spec,
                                "GAIN_AXES": self.gain_axes,
                                "PARAM_AXES": self.param_axes})

        return xds


class Delay(Gain):

    def __init__(self, term_name, data_xds, coords, tipc, fipc, opts):

        Gain.__init__(self, term_name, data_xds, coords, tipc, fipc, opts)

        self.n_ppa = 2
        self.additional_args = ["chan_freqs", "t_bin_arr"]
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
        self.gain_axes = ("time", "freq", "ant", "dir", "corr")
        self.param_axes = ("t_int", "f_int", "ant", "dir", "param", "corr")

    def make_xds(self):

        xds = Gain.make_xds(self)

        xds = xds.assign_coords({"param": np.array(["phase_offset", "delay"]),
                                 "time": ("time", self.unique_times),
                                 "freq": ("freq", self.unique_freqs)})
        xds = xds.assign_attrs({"GAIN_SPEC": self.gain_chunk_spec,
                                "PARAM_SPEC": self.param_chunk_spec,
                                "GAIN_AXES": self.gain_axes,
                                "PARAM_AXES": self.param_axes})

        return xds


term_types = {"complex": Complex,
              "phase": Phase,
              "kalman": Complex,
              "delay": Delay}
