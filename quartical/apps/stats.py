import argparse
from pathlib import Path
from daskms import xds_from_ms, xds_from_table
import dask.array as da
import numpy as np
from loguru import logger
import logging
from quartical.logging import InterceptHandler
import sys


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

    ant_xds = xds_from_table(path + "::ANTENNA")[0]  # Assume one for now.

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

    dd_xds_list = xds_from_table(
        path + "::DATA_DESCRIPTION",
        group_cols=["__row__"],
        chunks={"row": 1, "chan": -1}
    )

    # Not printing any summary information for this subtable yet - it is more
    # an implementation detail than useful information.


def feed_info(path):

    feed_xds_list = xds_from_table(
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
                   f"{'{} {}'.format(*vals[2]):<16}\n"

    logger.info(msg)


def spw_info(path):

    spw_xds_list = xds_from_table(
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


def summary():
    parser = argparse.ArgumentParser(
        description='Print some useful information about target dataset.'
    )

    parser.add_argument(
        'path',
        type=Path,
        help='Path to dataset.'
    )
    parser.add_argument(
        'output_dir',
        type=Path,
        help='Path to output directory.'
    )

    args = parser.parse_args()

    path = str(args.path.resolve())
    output_dir = str(args.output_dir.resolve())

    configure_loguru(output_dir)

    antenna_info(path)
    data_desc_info(path)
    feed_info(path)
    spw_info(path)

    # # Determine the number/type of correlations present in the measurement set.
    # pol_xds = xds_from_table(ms_opts.path + "::POLARIZATION")[0]

    # try:
    #     corr_types = [CORR_TYPES[ct] for ct in pol_xds.CORR_TYPE.values[0]]
    # except KeyError:
    #     raise KeyError("Data contains unsupported correlation products.")

    # n_corr = len(corr_types)

    # if n_corr not in (1, 2, 4):
    #     raise ValueError(f"Measurement set contains {n_corr} correlations - "
    #                      f"this is not supported.")

    # logger.info(f"Polarization table indicates {n_corr} correlations are "
    #             f"present in the measurement set - {corr_types}.")

    # # Determine the phase direction from the measurement set. TODO: This will
    # # probably need to be done on a per xds basis. Can probably be accomplished
    # # by merging the field xds grouped by DDID into data grouped by DDID.

    # field_xds = xds_from_table(ms_opts.path + "::FIELD")[0]
    # phase_dir = np.squeeze(field_xds.PHASE_DIR.values)
    # field_names = field_xds.NAME.values

    # logger.info("Field table indicates phase centre is at ({} {}).",
    #             phase_dir[0], phase_dir[1])


    # data_xds_list = xds_from_ms(
    #     path,
    #     index_cols=("TIME",),
    #     columns=("TIME"),
    #     group_cols=("DATA_DESC_ID", "SCAN_NUMBER", "FIELD_ID"),
    #     chunks={"row": "auto", "chan": "auto", "corr": -1},
    #     table_schema=["MS", {**schema}])

    # restored_xds_list = xds_to_table(
    #     zarr_xds_list,
    #     str(ms_path),
    #     columns=(args.column,)
    # )

    # dask.compute(restored_xds_list)
