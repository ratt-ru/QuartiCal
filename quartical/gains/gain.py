from collections import namedtuple
import numpy as np
import dask.array as da
from uuid import uuid4


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
        "MODEL_DATA",
        "DATA",
        "ANTENNA1",
        "ANTENNA2",
        "WEIGHT",
        "FLAG",
        "gains",
        "gain_flags",
        "time_bins",
        "time_maps",
        "freq_maps",
        "dir_maps",
        "row_map",
        "row_weights"
    )
)


class Gain:

    base_args = base_args

    def __init__(self, term_name, term_opts):

        self.name = term_name
        self.type = term_opts.type
        self.solve_per = term_opts.solve_per
        self.direction_dependent = term_opts.direction_dependent
        self.time_interval = term_opts.time_interval
        self.freq_interval = term_opts.freq_interval
        self.respect_scan_boundaries = term_opts.respect_scan_boundaries
        self.initial_estimate = term_opts.initial_estimate

    @classmethod
    def make_time_bins(
        cls,
        time_col,
        interval_col,
        scan_col,
        time_interval,
        respect_scan_boundaries
    ):

        time_bins = da.map_blocks(
            cls._make_time_bins,
            time_col,
            interval_col,
            scan_col,
            time_interval,
            respect_scan_boundaries,
            dtype=np.int64
        )

        return time_bins

    @classmethod
    def _make_time_bins(
        cls,
        time_col,
        interval_col,
        scan_col,
        time_interval,
        respect_scan_boundaries
    ):
        """Internals of the time binner.

        Args:
            n_utime: An interger number of unique times.
            utime_intervals: The intervals associated with the unique times.
            utime_scan_numbers: The scan numbers assosciated with the unique
                times.
            t_int: A int or float describing the time intervals. Floats
                correspond to value based solution intervals (e.g. 10s)
                and ints correspond to an interger number of timeslots.
            rsb: A boolean indicating whether respect_scan_boundaries is
                enabled for the current term.
        """

        utime, utime_ind = np.unique(time_col, return_index=True)

        utime_intervals = interval_col[utime_ind]
        utime_scans = scan_col[utime_ind]

        time_bins = np.empty((utime_intervals.size,), dtype=np.int32)

        if respect_scan_boundaries:
            _, scan_boundaries = np.unique(utime_scans, return_index=True)
            scan_boundaries = list(scan_boundaries - 1)  # Offset.
            scan_boundaries.pop(0)  # The first boundary will be zero.
            scan_boundaries.append(utime.size)  # Add a final boundary.
        else:
            scan_boundaries = [-1]

        scan_id = 0
        bin_size = 0
        bin_num = 0
        break_interval = False
        for i, ivl in enumerate(utime_intervals):
            time_bins[i] = bin_num
            bin_size += ivl if isinstance(time_interval, float) else 1
            if i == scan_boundaries[scan_id]:
                scan_id += 1
                break_interval = True
            if bin_size >= time_interval or break_interval:
                bin_size = 0
                bin_num += 1
                break_interval = False

        return time_bins

    @classmethod
    def make_time_map(cls, time_col, time_bins):

        time_map = da.map_blocks(
            cls._make_time_map,
            time_col,
            time_bins,
            dtype=np.int64
        )

        return time_map

    @classmethod
    def _make_time_map(cls, time_col, time_bins):

        _, utime_inv = np.unique(time_col, return_inverse=True)

        return time_bins[utime_inv]

    @classmethod
    def make_time_chunks(cls, time_bins):

        time_chunks = da.map_blocks(
            cls._make_time_chunks,
            time_bins,
            chunks=(1,),
            dtype=np.int64
        )

        return time_chunks

    @classmethod
    def _make_time_chunks(cls, time_bins):
        return time_bins.max() + 1

    @classmethod
    def make_time_coords(cls, time_col, time_bins):

        time_coords = da.map_blocks(
            cls._make_time_coords,
            time_col,
            time_bins,
            dtype=time_col.dtype
        )

        return time_coords

    @classmethod
    def _make_time_coords(cls, time_col, time_bins):

        unique_values = np.unique(time_col)

        sums = np.zeros(time_bins.max() + 1, dtype=np.float64)
        counts = np.zeros_like(sums, dtype=np.int64)

        np.add.at(sums, time_bins, unique_values)
        np.add.at(counts, time_bins, 1)

        return sums / counts

    @classmethod
    def make_freq_map(cls, chan_freqs, chan_widths, freq_interval):

        freq_map = da.map_blocks(
            cls._make_freq_map,
            chan_freqs,
            chan_widths,
            freq_interval,
            dtype=np.int64
        )

        return freq_map

    @classmethod
    def _make_freq_map(cls, chan_freqs, chan_widths, freq_interval):
        """Internals of the frequency interval mapper."""

        n_chan = chan_freqs.size

        freq_map = np.empty((n_chan,), dtype=np.int32)

        if isinstance(freq_interval, float):
            net_ivl = 0
            bin_num = 0
            for i, ivl in enumerate(chan_widths):
                freq_map[i] = bin_num
                net_ivl += ivl
                if net_ivl >= freq_interval:
                    net_ivl = 0
                    bin_num += 1
        else:
            freq_map[:] = np.arange(n_chan)//freq_interval

        return freq_map

    @classmethod
    def make_freq_chunks(cls, freq_map):

        freq_chunks = da.map_blocks(
            cls._make_freq_chunks,
            freq_map,
            chunks=(1,),
            dtype=np.int64
        )

        return freq_chunks

    @classmethod
    def _make_freq_chunks(cls, freq_map):
        return freq_map.max() + 1

    @classmethod
    def make_freq_coords(cls, chan_freq, freq_map):

        freq_coords = da.map_blocks(
            cls._make_freq_coords,
            chan_freq,
            freq_map,
            dtype=chan_freq.dtype
        )

        return freq_coords

    @classmethod
    def _make_freq_coords(cls, chan_freq, freq_map):

        unique_values = np.unique(chan_freq)

        sums = np.zeros(freq_map.max() + 1, dtype=np.float64)
        counts = np.zeros_like(sums, dtype=np.int64)

        np.add.at(sums, freq_map, unique_values)
        np.add.at(counts, freq_map, 1)

        return sums / counts

    @staticmethod
    def make_dir_map(n_dir, direction_dependent):

        # NOTE: Does not call the numpy implementation.
        # TODO: arange doesn't accept a name parameter - should we clone?
        if direction_dependent:
            dir_map = da.arange(
                n_dir,
                dtype=np.int64
            )
        else:
            dir_map = da.zeros(
                n_dir,
                name="dirmap-" + uuid4().hex,
                dtype=np.int64
            )

        return dir_map

    @staticmethod
    def _make_dir_map(n_dir, direction_dependent):

        if direction_dependent:
            dir_map = np.arange(
                n_dir,
                dtype=np.int64
            )
        else:
            dir_map = np.zeros(
                n_dir,
                dtype=np.int64
            )

        return dir_map

    @staticmethod
    def init_term(
        gain, param, term_ind, term_spec, term_opts, ref_ant, **kwargs
    ):
        """Initialise the gains (and parameters)."""

        (term_name, term_type, term_shape, term_pshape) = term_spec

        # TODO: This needs to be more sophisticated on parameterised terms.
        if f"{term_name}_initial_gain" in kwargs:
            gain[:] = kwargs[f"{term_name}_initial_gain"]
            loaded = True
        else:
            gain[..., (0, -1)] = 1  # Set first and last correlations to 1.
            loaded = False

        return loaded
