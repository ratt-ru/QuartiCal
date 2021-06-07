from collections import namedtuple
from quartical.kernels.complex import complex_solver
from quartical.kernels.phase import phase_solver
from quartical.kernels.delay import delay_solver
import numpy as np
import xarray


term_solvers = {"complex": complex_solver,
                "phase": phase_solver,
                "delay": delay_solver}

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
        self.n_dir = data_xds.dims["dir"] if self.dd_term else 1
        self.n_corr = data_xds.dims["corr"]
        self.id_fields = {f: data_xds.attrs[f]
                          for f in opts.input_ms_group_by}
        self.utime_chunks = list(map(int, data_xds.UTIME_CHUNKS))
        self.freq_chunks = list(map(int, data_xds.chunks["chan"]))
        self.n_t_chunk = len(self.utime_chunks)
        self.n_f_chunk = len(self.freq_chunks)

        self.n_tipc = tuple(map(int, tipc))
        self.n_tint = np.sum(self.n_tipc)

        self.n_fipc_g = tuple(map(int, fipc[0]))
        self.n_fint_g = np.sum(self.n_fipc_g)
        self.n_fipc_p = tuple(map(int, fipc[1]))
        self.n_fint_p = np.sum(self.n_fipc_p)

        self.unique_times = coords["time"]
        self.unique_freqs = coords["freq"]

        self.interval_times = coords[f"{self.name}_mean_time"]
        self.gain_freqs = coords[f"{self.name}_mean_gfreqs"]
        self.param_freqs = coords[f"{self.name}_mean_pfreqs"]

        self.additional_args = []

    def make_xds(self):

        # Set up an xarray.Dataset describing the gain term.
        xds = xarray.Dataset(
            data_vars={},
            coords={"ant": ("ant", np.arange(self.n_ant, dtype=np.int32)),
                    "dir": ("dir", np.arange(self.n_dir, dtype=np.int32)),
                    "corr": ("corr", np.arange(self.n_corr, dtype=np.int32)),
                    "t_chunk": ("t_chunk",
                                np.arange(self.n_t_chunk, dtype=np.int32)),
                    "f_chunk": ("f_chunk",
                                np.arange(self.n_f_chunk, dtype=np.int32)),
                    "gain_t": ("gain_t", self.interval_times),
                    "gain_f": ("gain_f", self.gain_freqs)},
            attrs={"NAME": self.name,
                   "TYPE": self.type,
                   "ARGS": self.additional_args,
                   **self.id_fields})

        return xds

    @staticmethod
    def make_f_maps(chan_freqs, chan_widths, f_int):
        """Internals of the frequency interval mapper."""

        n_chan = chan_freqs.size

        # The leading dimension corresponds (gain, param). For unparameterised
        # gains, the parameter mapping is irrelevant.
        f_map_arr = np.empty((2, n_chan,), dtype=np.int32)

        if isinstance(f_int, float):
            net_ivl = 0
            bin_num = 0
            for i, ivl in enumerate(chan_widths):
                f_map_arr[:, i] = bin_num
                net_ivl += ivl
                if net_ivl >= f_int:
                    net_ivl = 0
                    bin_num += 1
        else:
            f_map_arr[:, :] = np.arange(n_chan)//f_int

        return f_map_arr


class Complex(Gain):

    def __init__(self, term_name, data_xds, coords, tipc, fipc, opts):

        Gain.__init__(self, term_name, data_xds, coords, tipc, fipc, opts)

        self.n_ppa = 0
        self.gain_chunk_spec = gain_spec_tup(self.n_tipc,
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


class Phase(Gain):

    def __init__(self, term_name, data_xds, coords, tipc, fipc, opts):

        Gain.__init__(self, term_name, data_xds, coords, tipc, fipc, opts)

        self.n_ppa = 1
        self.gain_chunk_spec = gain_spec_tup(self.n_tipc,
                                             self.n_fipc_g,
                                             (self.n_ant,),
                                             (self.n_dir,),
                                             (self.n_corr,))
        self.param_chunk_spec = param_spec_tup(self.n_tipc,
                                               self.n_fipc_g,
                                               (self.n_ant,),
                                               (self.n_dir,),
                                               (self.n_ppa,),
                                               (self.n_corr,))
        self.gain_axes = ("gain_t", "gain_f", "ant", "dir", "corr")
        self.param_axes = \
            ("param_t", "param_f", "ant", "dir", "param", "corr")

    def make_xds(self):

        xds = Gain.make_xds(self)

        xds = xds.assign_coords({"param": np.array(["phase"]),
                                 "param_t": self.interval_times,
                                 "param_f": self.param_freqs})
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
        self.gain_chunk_spec = gain_spec_tup(self.n_tipc,
                                             self.n_fipc_g,
                                             (self.n_ant,),
                                             (self.n_dir,),
                                             (self.n_corr,))
        self.param_chunk_spec = param_spec_tup(self.n_tipc,
                                               self.n_fipc_p,
                                               (self.n_ant,),
                                               (self.n_dir,),
                                               (self.n_ppa,),
                                               (self.n_corr,))

        self.gain_axes = ("gain_t", "gain_f", "ant", "dir", "corr")
        self.param_axes = ("param_t", "param_f", "ant", "dir", "param", "corr")

    def make_xds(self):

        xds = Gain.make_xds(self)

        xds = xds.assign_coords({"param": np.array(["phase_offset", "delay"]),
                                 "param_t": self.interval_times,
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


term_types = {"complex": Complex,
              "phase": Phase,
              "delay": Delay}
