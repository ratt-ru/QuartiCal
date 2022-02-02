import dask.array as da
import numpy as np
import xarray
from loguru import logger
import os
import re


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

    msg += f"FLD: {field} SPW: {ddid} SCN: {scan} "
    msg += f"T_CHUNK: {t_chunk} "
    msg += f"F_CHUNK: {f_chunk} "

    if pre.item() > post.item():
        msg += f"<green>CHISQ: {pre.item():.2f} -> {post.item():.2f}</green>"
    elif pre.item() <= post.item():
        msg += f"<yellow>CHISQ: {pre.item():.2f} -> {post.item():.2f}</yellow>"
    else:
        msg += f"<red>CHISQ: {pre.item():.2f} -> {post.item():.2f}</red>"

    logger.opt(colors=True).info(msg)

    return post


def embed_stats_logging(stats_xds_list):

    stats_log_xds_list = []

    for sxds in stats_xds_list:

        pre = sxds.PRESOLVE_CHISQ.data
        post = sxds.POSTSOLVE_CHISQ.data

        # This is dirty trick - we need to loop the logging into the graph.
        # To do so, we resassign the post values (which are unchanged) to
        # ensure that the logging code is called. TODO: Better way?
        post = da.map_blocks(
            _log_chisq, pre, post, sxds.attrs, dtype=np.float32
        )

        stats_log_xds_list.append(
            sxds.assign(
                {"POSTSOLVE_CHISQ": (("t_chunk", "f_chunk"), post)}
            )
        )

    return stats_log_xds_list


colours = ["23D18B", "FFFF00", "FF8000", "FF0000"]


def log_summary_stats(stats_xds_list):

    from columnar import columnar

    tables = []

    n_sxds = len(stats_xds_list)
    group_size = 6

    sxds_groups = [stats_xds_list[i:i+group_size]
                   for i in range(0, n_sxds, group_size)]

    # This is only approximate as it is not weighted.
    chisq_mean, chisq_std = compute_chisq_mean_and_std(stats_xds_list)

    def colourize_chisq(match_obj):

        match = match_obj.group(0)
        value = float(match)

        deviation = (value - chisq_mean)/chisq_std

        if deviation <= 3:
            i = 0
        elif deviation <= 5:
            i = 1
        elif deviation <= 10:
            i = 2
        else:
            i = 3

        return f"<fg #{colours[i]}>{match}</fg #{colours[i]}>"

    for sxds_group in sxds_groups:

        max_nt_chunk = max([sxds.dims["t_chunk"] for sxds in sxds_group])
        max_nf_chunk = max([sxds.dims["f_chunk"] for sxds in sxds_group])

        ids = [f"T{t}F{f}" for t in range(max_nt_chunk)
                           for f in range(max_nf_chunk)]  # noqa

        attr_fields = ["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"]
        fmt = "FLD{}\nSPW{}\nSCN{}"

        data = []
        headers = ["CHUNK"]

        for sxds in sxds_group:

            frame = np.zeros((max_nt_chunk, max_nf_chunk))

            chisq = sxds.POSTSOLVE_CHISQ.values

            t_coords, f_coords = sxds.t_chunk.values, sxds.f_chunk.values
            t_coords, f_coords = np.meshgrid(t_coords, f_coords)
            t_coords, f_coords = t_coords.ravel(), f_coords.ravel()

            frame[t_coords, f_coords] = chisq[t_coords, f_coords]

            attrs = [sxds.attrs.get(f, "?") for f in attr_fields]

            data.append([f"{v:.2f}" if v else "" for v in frame.ravel()])
            headers.append(fmt.format(*attrs))

        data = [list(x) for x in zip(ids, *data)]

        try:
            columns, _ = os.get_terminal_size()
        except OSError:
            columns = 80  # Fall over to some sensible default.
        finally:
            columns = 80 if columns < 80 else columns  # Don't go too narrow.

        table = columnar(
            data,
            headers=headers,
            justify='l',
            no_borders=True,
            terminal_width=columns
        )

        table = table.replace("nan", "<fg #a0a0a0>nan</fg #a0a0a0>")

        float_re = re.compile(r'\d+\.\d+')

        table = re.sub(float_re, colourize_chisq, table)

        tables.append(table)

    logger.opt(colors=True).info(
        "\nFinal post-solve chi-squared sumary:\n" + "\n".join(tables)
    )

    return


def compute_chisq_mean_and_std(stats_xds_list):

    chisq_vals = []

    for sxds in stats_xds_list:

        chisq = sxds.POSTSOLVE_CHISQ.values

        sel = np.where(np.isfinite(chisq))

        chisq_vals.append(chisq[sel])

    chisq_vals = np.concatenate(chisq_vals)

    return np.mean(chisq_vals), np.std(chisq_vals)
