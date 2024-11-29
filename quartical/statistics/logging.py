import os
import re
import numpy as np
from loguru import logger
from columnar import columnar


# Some hex codes for reused colours.
colours = {
    "green": "23D18B",
    "yellow": "F5F543",
    "orange": "FF8000",
    "red": "F14C4C",
    "grey": "A0A0A0"
}


def log_chisq(pre, post, attrs, block_id=None):
    """Logs an info message per chunk containing chunk and chisq info.

    Args:
        pre: A numpy.ndarray contatining pre-solve chisq values.
        post: A numpy.ndarray contatining post-solve chisq values.
        attrs: xarray.Dataset attrs that contatain useful metadata.
        block_info: A dummy kwarg that tells dask to give us block info.

    Returns:
        post: An exact copy of the input - this is used to ensure the logging
            is embedded. TODO: Improve?
    """

    # Get the chink info (from the first arg), and pull out the location.
    t_chunk, f_chunk, _ = block_id

    ddid = attrs.get("DATA_DESC_ID", "?")
    scan = attrs.get("SCAN_NUMBER", "?")
    field = attrs.get("FIELD_ID", "?")

    msg = "\n    "

    msg += f"FLD: {field} SPW: {ddid} SCN: {scan} "
    msg += f"T_CHUNK: {t_chunk} "
    msg += f"F_CHUNK: {f_chunk} "

    pre = pre.item()
    post = post.item()

    if pre == 0:  # This may be zero in noise-free tests.
        fractional_change = 0
    elif np.isfinite(pre) and np.isfinite(post):
        fractional_change = (post - pre) / pre
    else:
        fractional_change = np.nan

    if np.isfinite(fractional_change):
        if np.abs(fractional_change) < 1e-12:  # Slightly numb to jitter.
            colour = colours['yellow']
        elif fractional_change < 0:
            colour = colours['green']
        elif fractional_change > 0:
            colour = colours['red']
    else:
        colour = colours['grey']

    co, cc = f"<fg #{colour}>", f"</fg #{colour}>"
    msg += f"{co}CHISQ: {pre:.2f} -> {post:.2f}{cc}"

    logger.opt(colors=True).info(msg)


def log_summary_stats(stats_xds_list):

    tables = []

    n_sxds = len(stats_xds_list)
    group_size = 6

    sxds_groups = [stats_xds_list[i:i+group_size]
                   for i in range(0, n_sxds, group_size)]

    for sxds_group in sxds_groups:

        max_nt_chunk = max([sxds.sizes["t_chunk"] for sxds in sxds_group])
        max_nf_chunk = max([sxds.sizes["f_chunk"] for sxds in sxds_group])

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

            # TODO: Chi-squared values of zero disappear here.
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

        tables.append(table)

    # This is only approximate as it is not weighted.
    chisq_mean, chisq_std = compute_chisq_mean_and_std(stats_xds_list)

    def colourise_chisq(match_obj):

        match = match_obj.group(0)
        value = float(match)

        if chisq_std:
            deviation = (value - chisq_mean)/chisq_std
        else:
            deviation = 0

        if deviation <= 3:
            colour = colours["green"]
        elif deviation <= 5:
            colour = colours["yellow"]
        elif deviation <= 10:
            colour = colours["orange"]
        else:
            colour = colours["red"]

        return f"<fg #{colour}>{match}</fg #{colour}>"

    bins = ["<= 3", "<= 5", "<= 10", "> 10"]
    clrs = ["green", "yellow", "orange", "red"]

    msg = "\nFinal post-solve chi-squared summary, colourised by deviation " \
          "from the mean:\n"
    msg += " ".join(f"<fg #{colours[c]}> {b}*sigma </fg #{colours[c]}>"
                    for b, c in zip(bins, clrs))
    msg += "\n".join(tables)

    colour = colours["grey"]
    msg = msg.replace("nan", f"<fg #{colour}>nan</fg #{colour}>")
    msg = re.sub(r'\d+\.\d+', colourise_chisq, msg)

    logger.opt(colors=True).info(msg)


def compute_chisq_mean_and_std(stats_xds_list):

    chisq_vals = []

    for sxds in stats_xds_list:

        chisq = sxds.POSTSOLVE_CHISQ.values

        sel = np.where(np.isfinite(chisq))

        chisq_vals.append(chisq[sel])

    chisq_vals = np.concatenate(chisq_vals)

    return np.mean(chisq_vals), np.std(chisq_vals)
