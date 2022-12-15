import numpy as np
import xarray


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
