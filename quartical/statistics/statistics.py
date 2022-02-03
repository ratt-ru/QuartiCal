import dask.array as da
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


def assign_presolve_chisq(data_xds_list, stats_xds_list):
    """Assigns pre-solve chi-squared values to the appropriate dataset."""

    chisq_per_xds = []

    for xds, sxds in zip(data_xds_list, stats_xds_list):

        data = xds.DATA.data
        model = xds.MODEL_DATA.data
        weights = xds.WEIGHT.data
        inv_flags = da.where(xds.FLAG.data == 0, 1, 0)[:, :, None]

        residual = data - model.sum(axis=2)

        chisq = da.map_blocks(
            compute_chisq,
            residual,
            weights,
            inv_flags,
            chunks=(1, 1),
            drop_axis=-1,
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

        weights = xds._WEIGHT.data
        residual = xds._RESIDUAL.data
        inv_flags = da.where(xds.FLAG.data == 0, 1, 0)[:, :, None]

        chisq = da.map_blocks(
            compute_chisq,
            residual,
            weights,
            inv_flags,
            chunks=(1, 1),
            drop_axis=-1,
        )

        chisq_per_xds.append(
            sxds.assign(
                {"POSTSOLVE_CHISQ": (("t_chunk", "f_chunk"), chisq)}
            )
        )

    return chisq_per_xds


def compute_chisq(residual, weights, inv_flags):
    """Compute the mean chi-squared for a given chunk.

    Args:
        residual: A (row, chan, corr) numpy.ndarray containing residual values.
        weights: A (row, chan, corr) numpy.ndarray containing weight values.
        inv_flags: A (row, chan, 1) numpy.ndarray containing inverse flags i.e.
            set where data is valid.

    Returns:
        mean_chisq: A (1, 1) numpy.ndarray containing the mean chi-squared.
    """

    eff_weights = weights * inv_flags

    chisq = (residual * eff_weights * residual.conj()).real
    chisq = chisq.sum(keepdims=True)

    counts = inv_flags.sum(keepdims=True) * residual.shape[-1]

    if counts:
        mean_chisq = (chisq/counts)[..., -1]
    else:
        mean_chisq = np.array([[np.nan]])

    return mean_chisq
