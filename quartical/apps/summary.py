import argparse
from pathlib import Path
from daskms import xds_from_storage_ms, xds_from_storage_table
from daskms.fsspec_store import DaskMSStore
import numpy as np
import dask.array as da
from loguru import logger
import logging
import sys
from quartical.logging import InterceptHandler
from quartical.data_handling import CORR_TYPES


def configure_loguru(output_dir):
    logging.basicConfig(handlers=[InterceptHandler()], level="WARNING")

    # Put together a formatting string for the logger. Split into pieces in
    # order to improve legibility.

    tim_fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
    lvl_fmt = "<level>{level}</level>"
    src_fmt = "<cyan>{module}</cyan>:<cyan>{function}</cyan>"
    msg_fmt = "<level>\n{message}</level>"

    fmt = " | ".join([tim_fmt, lvl_fmt, src_fmt, msg_fmt])

    output_path = Path(output_dir)
    output_name = Path("{time:YYYYMMDD_HHmmss}.summary.qc")

    config = {
        "handlers": [
            {"sink": sys.stderr,
             "level": "INFO",
             "format": fmt},
            {"sink": str(output_path / output_name),
             "level": "INFO",
             "format": fmt}
        ],
    }

    logger.configure(**config)


def antenna_info(path):

    # NOTE: Assume one dataset for now.
    ant_xds = xds_from_storage_table(path + "::ANTENNA")[0]

    antenna_names = ant_xds.NAME.values
    antenna_mounts = ant_xds.MOUNT.values
    antenna_flags = ant_xds.FLAG_ROW.values

    msg = "Antenna summary:\n"
    msg += "    {:<8} {:<8} {:<8} {:<8}\n".format("INDEX", "NAME", "MOUNT",
                                                  "FLAG")

    zipper = zip(antenna_names, antenna_mounts, antenna_flags)

    for i, vals in enumerate(zipper):
        msg += f"    {i:<8} {vals[0]:<8} {vals[1]:<8} {vals[2]:<8}\n"

    logger.info(msg)


def data_desc_info(path):

    dd_xds_list = xds_from_storage_table(  # noqa
        path + "::DATA_DESCRIPTION",
        group_cols=["__row__"],
        chunks={"row": 1, "chan": -1}
    )

    # Not printing any summary information for this subtable yet - not sure
    # what is relevant.


def feed_info(path):

    feed_xds_list = xds_from_storage_table(
        path + "::FEED",
        group_cols=["SPECTRAL_WINDOW_ID"],
        chunks={"row": -1}
    )

    ant_id_per_xds = [xds.ANTENNA_ID.values for xds in feed_xds_list]

    pol_type_per_xds = [xds.POLARIZATION_TYPE.values for xds in feed_xds_list]

    rec_angle_per_xds = [xds.RECEPTOR_ANGLE.values for xds in feed_xds_list]

    msg = "Feed summary:\n"
    msg += "    {:<4} {:<8} {:<8} {:<16}\n".format("SPW",
                                                   "ANTENNA",
                                                   "POL_TYPE",
                                                   "RECEPTOR_ANGLE")

    zipper = zip(ant_id_per_xds, pol_type_per_xds, rec_angle_per_xds)

    for i, arrs in enumerate(zipper):
        for vals in zip(*arrs):
            msg += f"    {i:<4} {vals[0]:<8} {' '.join(vals[1]):<8} " \
                   f"{'{:.4f} {:.4f}'.format(*vals[2]):<16}\n"

    logger.info(msg)


def flag_cmd_info(path):

    flag_cmd_xds = xds_from_storage_table(path + "::FLAG_CMD")  # noqa

    # Not printing any summary information for this subtable yet - not sure
    # what is relevant.


def field_info(path):

    field_xds = xds_from_storage_table(path + "::FIELD")[0]

    ids = [i for i in field_xds.SOURCE_ID.values]
    names = [n for n in field_xds.NAME.values]
    phase_dirs = [pd for pd in field_xds.PHASE_DIR.values]
    ref_dirs = [rd for rd in field_xds.REFERENCE_DIR.values]
    delay_dirs = [dd for dd in field_xds.REFERENCE_DIR.values]

    msg = "Field summary:\n"
    msg += "    {:<4} {:<16} {:<16} {:<16} {:<16}\n".format("ID", "NAME",
                                                            "PHASE_DIR",
                                                            "REF_DIR",
                                                            "DELAY_DIR")

    zipper = zip(ids, names, phase_dirs, ref_dirs, delay_dirs)

    for vals in zipper:
        msg += f"    {vals[0]:<4} {vals[1]:<16} " \
               f"{'{:.4f} {:.4f}'.format(*vals[2][0]):<16} " \
               f"{'{:.4f} {:.4f}'.format(*vals[3][0]):<16} " \
               f"{'{:.4f} {:.4f}'.format(*vals[4][0]):<16}\n"

    logger.info(msg)


def history_info(path):

    history_xds = xds_from_storage_table(path + "::HISTORY")[0]  # noqa

    # Not printing any summary information for this subtable yet - not sure
    # what is relevant.


def observation_info(path):

    observation_xds = xds_from_storage_table(path + "::OBSERVATION")[0]  # noqa

    # Not printing any summary information for this subtable yet - not sure
    # what is relevant.


def polarization_info(path):

    polarization_xds = xds_from_storage_table(path + "::POLARIZATION")[0]

    corr_types = polarization_xds.CORR_TYPE.values

    readable_corr_types = \
        [[CORR_TYPES.get(ct, '-') for ct in cta] for cta in corr_types]

    msg = "Polarization summary:\n"
    msg += "    {:<8} {:<30}\n".format("INDEX", "CORR_TYPE")

    for i, vals in enumerate(zip(corr_types, readable_corr_types)):
        msg += f"    {i:<8} {'{:} -> {:}'.format(vals[0], vals[1]):<30}\n"

    logger.info(msg)


def processor_info(path):

    processor_xds = xds_from_storage_table(path + "::PROCESSOR")[0]  # noqa

    # Not printing any summary information for this subtable yet - not sure
    # what is relevant.


def spw_info(path):

    spw_xds_list = xds_from_storage_table(
        path + "::SPECTRAL_WINDOW",
        group_cols=["__row__"],
        chunks={"row": 1, "chan": -1}
    )

    n_chan_per_spw = [xds.dims["chan"] for xds in spw_xds_list]

    bw_per_spw = [xds.TOTAL_BANDWIDTH.values.item() for xds in spw_xds_list]

    ref_per_spw = [xds.REF_FREQUENCY.values.item() for xds in spw_xds_list]

    msg = "Spectral window summary:\n"
    msg += "    {:<8} {:<10} {:<16} {:<16}\n".format("INDEX", "CHANNELS",
                                                     "BANDWIDTH", "REF_FREQ")

    for i, vals in enumerate(zip(n_chan_per_spw, bw_per_spw, ref_per_spw)):
        msg += f"    {i:<8} {vals[0]:<10} {vals[1]:<16} {vals[2]:<16}\n"

    logger.info(msg)


def state_info(path):

    state_xds = xds_from_storage_table(path + "::STATE")[0]  # noqa

    # Not printing any summary information for this subtable yet - not sure
    # what is relevant.


def source_info(path):

    # NOTE: Skip reading this for now - it can break dask-ms.
    # source_xds = xds_from_table(path + "::SOURCE")[0]  # noqa

    return

    # Not printing any summary information for this subtable yet - not sure
    # what is relevant.


def pointing_info(path):

    pointing_xds = xds_from_storage_table(path + "::POINTING")[0]  # noqa

    # Not printing any summary information for this subtable yet - not sure
    # what is relevant.


def dimension_summary(xds_list):

    rows_per_xds = [xds.dims["row"] for xds in xds_list]
    chan_per_xds = [xds.dims["chan"] for xds in xds_list]
    corr_per_xds = [xds.dims["corr"] for xds in xds_list]

    utime_per_xds = [np.unique(xds.TIME.values).size for xds in xds_list]

    fields = (
        "DATA_DESC_ID",
        "SCAN_NUMBER",
        "FIELD_ID",
        "ROW",
        "TIME",
        "CHAN",
        "CORR",
    )

    msg = "Dimension summary:\n"
    fmt = "    {:<12} {:<12} {:<12} {:<8} {:<6} {:<6} {:<4}\n"
    msg += fmt.format(*fields)

    for idx, xds in enumerate(xds_list):
        msg += fmt.format(xds.DATA_DESC_ID,
                          xds.SCAN_NUMBER,
                          xds.FIELD_ID,
                          rows_per_xds[idx],
                          utime_per_xds[idx],
                          chan_per_xds[idx],
                          corr_per_xds[idx])

    logger.info(msg)


def flagging_summary(xds_list):

    n_flag_per_xds = []
    n_elem_per_xds = []
    perc_flagged_per_xds = []

    for xds in xds_list:

        flags = xds.FLAG.data | xds.FLAG_ROW.data[:, None, None]

        n_flag = da.sum(flags)
        n_elem = flags.size
        flag_perc = (n_flag/n_elem)*100

        n_flag_per_xds.append(n_flag)
        n_elem_per_xds.append(n_elem)
        perc_flagged_per_xds.append(flag_perc)

    total_n_flag = da.sum(da.stack(n_flag_per_xds))
    total_n_elem = da.sum(da.stack(n_elem_per_xds))
    total_flag_perc = (total_n_flag/total_n_elem)*100

    total_flag_perc, perc_flagged_per_xds = \
        da.compute(total_flag_perc, perc_flagged_per_xds)

    fields = (
        "DATA_DESC_ID",
        "SCAN_NUMBER",
        "FIELD_ID",
        "PERC_FLAGGED"
    )

    msg = "Flagging summary:\n"
    msg += "    {:<12} {:<12} {:<12} {:<12}\n".format(*fields)

    for idx, xds in enumerate(xds_list):
        msg += f"    {xds.DATA_DESC_ID:<12} {xds.SCAN_NUMBER:<12} " \
               f"{xds.FIELD_ID:<12} {perc_flagged_per_xds[idx]:<12.2f}\n"

    fields = ("", "", "TOTAL", total_flag_perc)

    msg += "    {:<12} {:<12} {:<12} {:<12.2f}\n".format(*fields)

    logger.info(msg)


def summary():
    parser = argparse.ArgumentParser(
        description='Print some useful information about target dataset.'
    )

    parser.add_argument(
        'path',
        type=DaskMSStore,
        help='Path to input measurement set, e.g. path/to/dir/foo.MS. Also '
             'accepts valid s3 urls.'
    )
    parser.add_argument(
        'output_dir',
        type=Path,
        help='Path to output directory, e.g. summaries.qc. Local file system '
             'only.'
    )

    args = parser.parse_args()

    path = args.path.url
    output_dir = str(args.output_dir.resolve())

    configure_loguru(output_dir)

    # Get summary info for subtables. TODO: Improve as needed.
    antenna_info(path)
    data_desc_info(path)
    feed_info(path)
    flag_cmd_info(path)
    field_info(path)
    history_info(path)
    observation_info(path)
    polarization_info(path)
    processor_info(path)
    spw_info(path)
    state_info(path)
    source_info(path)
    pointing_info(path)

    # Open the data, grouping by the usual columns. Use these datasets to
    # produce some useful summaries.

    data_xds_list = xds_from_storage_ms(
        path,
        index_cols=("TIME",),
        columns=("TIME", "FLAG", "FLAG_ROW", "DATA"),
        group_cols=("DATA_DESC_ID", "SCAN_NUMBER", "FIELD_ID"),
        chunks={"row": 25000, "chan": 1024, "corr": -1},
    )

    dimension_summary(data_xds_list)
    flagging_summary(data_xds_list)
