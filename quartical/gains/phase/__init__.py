from quartical.gains.gain import Gain
from quartical.gains.phase.kernel import phase_solver, phase_args
import numpy as np
import dask.array as da


class Phase(Gain):

    solver = phase_solver
    term_args = phase_args

    def __init__(self, term_name, term_opts):

        Gain.__init__(self, term_name, term_opts)

        self.gain_axes = (
            "gain_time",
            "gain_freq",
            "antenna",
            "direction",
            "correlation"
        )
        self.param_axes = (
            "param_time",
            "param_freq",
            "antenna",
            "direction",
            "param_name"
        )

    @classmethod
    def make_param_time_bins(
        cls,
        time_col,
        interval_col,
        scan_col,
        time_interval,
        respect_scan_boundaries
    ):

        time_bins = da.map_blocks(
            cls._make_param_time_bins,
            time_col,
            interval_col,
            scan_col,
            time_interval,
            respect_scan_boundaries,
            dtype=np.int64
        )

        return time_bins

    @classmethod
    def _make_param_time_bins(
        cls,
        time_col,
        interval_col,
        scan_col,
        time_interval,
        respect_scan_boundaries
    ):
        return cls._make_time_bins(
            time_col,
            interval_col,
            scan_col,
            time_interval,
            respect_scan_boundaries
        )

    @classmethod
    def make_param_time_map(cls, time_col, param_time_bins):

        param_time_map = da.map_blocks(
            cls._make_param_time_map,
            time_col,
            param_time_bins,
            dtype=np.int64
        )

        return param_time_map

    @classmethod
    def _make_param_time_map(cls, time_col, param_time_bins):
        return cls._make_time_map(time_col, param_time_bins)

    @classmethod
    def make_param_time_chunks(cls, param_time_bins):

        param_time_chunks = da.map_blocks(
            cls._make_param_time_chunks,
            param_time_bins,
            chunks=(1,),
            dtype=np.int64
        )

        return param_time_chunks

    @classmethod
    def _make_param_time_chunks(cls, param_time_bins):
        return cls._make_time_chunks(param_time_bins)

    @classmethod
    def make_param_freq_map(cls, chan_freqs, chan_widths, freq_interval):

        param_freq_map = da.map_blocks(
            cls._make_param_freq_map,
            chan_freqs,
            chan_widths,
            freq_interval,
            dtype=np.int64
        )

        return param_freq_map

    @classmethod
    def _make_param_freq_map(cls, chan_freqs, chan_widths, freq_interval):
        return cls._make_freq_map(chan_freqs, chan_widths, freq_interval)

    @classmethod
    def make_param_freq_chunks(cls, param_freq_map):

        param_freq_chunks = da.map_blocks(
            cls._make_param_freq_chunks,
            param_freq_map,
            chunks=(1,),
            dtype=np.int64
        )

        return param_freq_chunks

    @classmethod
    def _make_param_freq_chunks(cls, param_freq_map):
        return cls._make_freq_chunks(param_freq_map)

    @classmethod
    def make_param_names(cls, correlations):

        # TODO: This is not dasky, unlike the other functions. Delayed?
        parameterisable = ["XX", "YY", "RR", "LL"]

        param_corr = [c for c in correlations if c in parameterisable]

        return [f"phase_{c}" for c in param_corr]
