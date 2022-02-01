import dask.array as da
import numpy as np
import xarray
from loguru import logger


def make_stats_xds_list(data_xds_list):

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

    eff_weights = weights * inv_flags

    chisq = (residual * eff_weights * residual.conj()).real
    chisq = chisq.sum(keepdims=True)

    counts = inv_flags.sum(keepdims=True) * residual.shape[-1]

    if counts:
        mean_chisq = (chisq/counts)[..., -1]
    else:
        mean_chisq = np.array([[np.nan]])

    return mean_chisq


def _log_chisq(pre, post, attrs, block_info=None):

    t_chunk, f_chunk = block_info[0]['chunk-location']

    ddid = attrs.get("DATA_DESC_ID", "?")
    scan = attrs.get("SCAN_NUMBER", "?")
    field = attrs.get("FIELD_ID", "?")

    msg = "\n    "

    msg += f"FIELD: {field} DDID: {ddid} SCAN: {scan} "
    msg += f"T_CHUNK: {t_chunk} "
    msg += f"F_CHUNK: {f_chunk} "

    if pre.item() > post.item():
        msg += f"<green>CHISQ: {pre.item():.2f} -> {post.item():.2f}</green>"
    elif pre.item() <= post.item():
        msg += f"<yellow>CHISQ: {pre.item():.2f} -> {post.item():.2f}</yellow>"
    else:
        msg += f"<red>CHISQ: {pre.item():.2f} -> {post.item():.2f}</red>"

    logger.opt(colors=True).info(msg)

    return np.array([[True]])


def log_stats(stats_xds_list):

    log_per_xds = []

    for sxds in stats_xds_list:

        pre = sxds.PRESOLVE_CHISQ.data
        post = sxds.POSTSOLVE_CHISQ.data

        message = da.map_blocks(_log_chisq, pre, post, sxds.attrs, dtype=bool)

        log_per_xds.append(message)

    return log_per_xds
