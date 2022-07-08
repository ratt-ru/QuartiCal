import dask.array as da
import numpy as np
import xarray
from quartical.statistics.stat_kernels import (compute_mean_presolve_chisq,
                                               compute_mean_postsolve_chisq)


def make_stats_xds_list(data_xds_list):
    """Make a list of xarray.Datasets to hold statistical information."""

    stats_xds_list = []

    attr_fields = ["SCAN_NUMBER", "DATA_DESC_ID", "FIELD_ID", "FIELD_NAME"]

    for xds in data_xds_list:

        attrs = {k: v for k, v in xds.attrs.items() if k in attr_fields}

        coords = {
            "t_chunk": (("t_chunk",), np.arange(len(xds.chunks["row"]))),
            "f_chunk": (("f_chunk",), np.arange(len(xds.chunks["chan"])))
        }

        stats_xds = xarray.Dataset(coords=coords, attrs=attrs)

        stats_xds_list.append(stats_xds)

    return stats_xds_list


def assign_presolve_chisq(data_xds_list, stats_xds_list):
    """Assigns pre-solve chi-squared values to the appropriate dataset."""

    chisq_per_xds = []

    for xds, sxds in zip(data_xds_list, stats_xds_list):

        data = xds.DATA.data
        model = xds.MODEL_DATA.data
        weight = xds.WEIGHT.data
        flags = xds.FLAG.data

        chisq = da.blockwise(
            compute_mean_presolve_chisq, "rf",
            data, "rfc",
            model, "rfdc",
            weight, "rfc",
            flags, "rf",
            adjust_chunks={"r": 1, "f": 1},
            align_arrays=False,
            concatenate=True,
            dtype=np.float64,
        )

        chisq_per_xds.append(
            sxds.assign(
                {"PRESOLVE_CHISQ": (("t_chunk", "f_chunk"), chisq)}
            )
        )

    return chisq_per_xds


def assign_postsolve_chisq(data_xds_list, stats_xds_list):
    """Assigns post-solve chi-squared values to the appropriate dataset."""

    chisq_per_xds = []

    for xds, sxds in zip(data_xds_list, stats_xds_list):

        residual = xds._RESIDUAL.data
        weight = xds._WEIGHT.data
        flags = xds.FLAG.data

        chisq = da.blockwise(
            compute_mean_postsolve_chisq, "rf",
            residual, "rfc",
            weight, "rfc",
            flags, "rf",
            adjust_chunks={"r": 1, "f": 1},
            align_arrays=False,
            concatenate=True,
            dtype=np.float64,
        )

        chisq_per_xds.append(
            sxds.assign(
                {"POSTSOLVE_CHISQ": (("t_chunk", "f_chunk"), chisq)}
            )
        )

    return chisq_per_xds
