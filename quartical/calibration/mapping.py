import dask.array as da
import numpy as np
import xarray
from uuid import uuid4


def make_mapping_datasets(data_xds_list, chain):

    mapping_xds_list = []

    for data_xds in data_xds_list:

        mappings = {}

        for gain_obj in chain:

            # Check whether we are dealing with BDA data.
            if hasattr(data_xds, "UPSAMPLED_TIME"):
                time_col = data_xds.UPSAMPLED_TIME.data
                interval_col = data_xds.UPSAMPLED_INTERVAL.data
            else:
                time_col = data_xds.TIME.data
                interval_col = data_xds.INTERVAL.data

            # If SCAN_NUMBER was a partitioning column it will not be present
            # on the dataset - we reintroduce it for cases where we need to
            # ensure solution intervals don't span scan boundaries.
            # TODO: This could actually be moved to the underlying code by
            # passing in/handling None.
            if "SCAN_NUMBER" in data_xds.data_vars.keys():
                scan_col = data_xds.SCAN_NUMBER.data
            else:
                scan_col = da.zeros_like(
                    time_col,
                    dtype=np.int32,
                    name="scan_number-" + uuid4().hex
                )

            time_interval = gain_obj.time_interval
            respect_scan_boundaries = gain_obj.respect_scan_boundaries

            time_bins = gain_obj.make_time_bins(
                time_col,
                interval_col,
                scan_col,
                time_interval,
                respect_scan_boundaries,
                chunks=(data_xds.UTIME_CHUNKS,)
            )

            time_map = gain_obj.make_time_map(
                time_col,
                time_bins
            )

            chan_freqs = data_xds.CHAN_FREQ.data
            chan_widths = data_xds.CHAN_WIDTH.data
            freq_interval = gain_obj.freq_interval

            freq_map = gain_obj.make_freq_map(
                chan_freqs,
                chan_widths,
                freq_interval
            )

            n_dir = data_xds.sizes["dir"]

            dir_map = gain_obj.make_dir_map(
                n_dir,
                gain_obj.direction_dependent
            )

            mappings[f"{gain_obj.name}_time_bins"] = (("time",), time_bins)
            mappings[f"{gain_obj.name}_time_map"] = (("row",), time_map)
            mappings[f"{gain_obj.name}_freq_map"] = (("chan",), freq_map)
            mappings[f"{gain_obj.name}_dir_map"] = (("dir",), dir_map)

            if hasattr(gain_obj, "param_axes"):
                param_time_bins = gain_obj.make_param_time_bins(
                    time_col,
                    interval_col,
                    scan_col,
                    time_interval,
                    respect_scan_boundaries,
                    chunks=(data_xds.UTIME_CHUNKS,)
                )

                param_time_map = gain_obj.make_time_map(
                    time_col,
                    param_time_bins
                )

                param_freq_map = gain_obj.make_param_freq_map(
                    chan_freqs,
                    chan_widths,
                    freq_interval
                )

                mappings[f"{gain_obj.name}_param_time_bins"] = (
                    (("time",), param_time_bins)
                )
                mappings[f"{gain_obj.name}_param_time_map"] = (
                    (("row",), param_time_map)
                )
                mappings[f"{gain_obj.name}_param_freq_map"] = (
                    (("chan",), param_freq_map)
                )

        mapping_xds_list.append(xarray.Dataset(mappings))

    return mapping_xds_list
