from collections import namedtuple
import numpy as np
import xarray


gain_spec_tup = namedtuple(
    "gain_spec_tup",
    (
        "tchunk",
        "fchunk",
        "achunk",
        "dchunk",
        "cchunk"
    )
)

param_spec_tup = namedtuple(
    "param_spec_tup",
    (
        "tchunk",
        "fchunk",
        "achunk",
        "dchunk",
        "pchunk"
    )
)

base_args = namedtuple(
    "base_args",
    (
        "model",
        "data",
        "a1",
        "a2",
        "weights",
        "flags",
        "t_map_arr",
        "f_map_arr",
        "d_map_arr",
        "inverse_gains",
        "gains",
        "gain_flags",
        "row_map",
        "row_weights"
    )
)


class Gain:

    base_args = base_args

    def __init__(self, term_name, term_opts, data_xds, coords, tipc, fipc):

        self.name = term_name
        self.dd_term = term_opts.direction_dependent
        self.type = term_opts.type
        self.n_chan = data_xds.dims["chan"]
        self.n_ant = data_xds.dims["ant"]
        self.n_dir = data_xds.dims["dir"] if self.dd_term else 1
        self.n_corr = data_xds.dims["corr"]
        partition_schema = data_xds.__daskms_partition_schema__
        self.id_fields = {f: data_xds.attrs[f] for f, _ in partition_schema}
        self.field_name = data_xds.FIELD_NAME
        self.utime_chunks = list(map(int, data_xds.UTIME_CHUNKS))
        self.freq_chunks = list(map(int, data_xds.chunks["chan"]))
        self.n_t_chunk = len(self.utime_chunks)
        self.n_f_chunk = len(self.freq_chunks)

        self.ant_names = data_xds.ant.values
        self.corr_types = data_xds.corr.values

        self.n_tipc_g = tuple(map(int, tipc[0]))
        self.n_tint_g = np.sum(self.n_tipc_g)
        self.n_tipc_p = tuple(map(int, tipc[1]))
        self.n_tint_p = np.sum(self.n_tipc_p)

        self.n_fipc_g = tuple(map(int, fipc[0]))
        self.n_fint_g = np.sum(self.n_fipc_g)
        self.n_fipc_p = tuple(map(int, fipc[1]))
        self.n_fint_p = np.sum(self.n_fipc_p)

        self.unique_times = coords["time"]
        self.unique_freqs = coords["freq"]

        self.gain_times = coords.get(f"{self.name}_mean_gtime",
                                     coords["time"])
        self.param_times = coords.get(f"{self.name}_mean_ptime",
                                      coords["time"])
        self.gain_freqs = coords.get(f"{self.name}_mean_gfreq",
                                     coords["freq"])
        self.param_freqs = coords.get(f"{self.name}_mean_pfreq",
                                      coords["freq"])

    def make_xds(self):

        # Set up an xarray.Dataset describing the gain term.
        xds = xarray.Dataset(
            coords={"ant": ("ant", self.ant_names),
                    "dir": ("dir", np.arange(self.n_dir, dtype=np.int32)),
                    "corr": ("corr", self.corr_types),
                    "t_chunk": ("t_chunk",
                                np.arange(self.n_t_chunk, dtype=np.int32)),
                    "f_chunk": ("f_chunk",
                                np.arange(self.n_f_chunk, dtype=np.int32)),
                    "gain_t": ("gain_t", self.gain_times),
                    "gain_f": ("gain_f", self.gain_freqs)},
            attrs={"NAME": self.name,
                   "TYPE": self.type,
                   "FIELD_NAME": self.field_name,
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

    @staticmethod
    def make_t_bins(n_utime, utime_intervals, t_int):
        """Internals of the time binner."""

        tbin_arr = np.empty((2, utime_intervals.size), dtype=np.int32)

        if isinstance(t_int, float):
            net_ivl = 0
            bin_num = 0
            for i, ivl in enumerate(utime_intervals):
                tbin_arr[:, i] = bin_num
                net_ivl += ivl
                if net_ivl >= t_int:
                    net_ivl = 0
                    bin_num += 1
        else:
            tbin_arr[:, :] = np.floor_divide(np.arange(n_utime),
                                             t_int,
                                             dtype=np.int32)

        return tbin_arr
