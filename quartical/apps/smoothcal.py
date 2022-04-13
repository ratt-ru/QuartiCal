import argparse
from daskms import xds_from_ms
from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
from pathlib import Path
import time
import dask
from quartical.interpolation.interpolants import gpr_interpolate_gains


def backup():
    parser = argparse.ArgumentParser(
        description='Smooth gain solutions using Gaussian process regression.'
    )

    parser.add_argument(
        'ms_path',
        type=Path,
        help='Path to input measurement set, e.g. path/to/dir/foo.MS.'
    )
    parser.add_argument(
        'gain_dir',
        type=Path,
        help='Path to gain solutions produced by QuartiCal.'
    )
    parser.add_argument('time-length-scales',
                        type=float,
                        nargs=2,
                        help='Length scales for time dimension for ampltudes '
                        'and phases respectively.'
                        )
    parser.add_argument('freq-length-scales',
                        type=float,
                        nargs=2,
                        help='Length scales for frequency dimension for '
                        'amplitudes and phases respectively.')
    parser.add_argument('noise-inflation',
                        dtype=float,
                        help='Noise inflation factor')

    args = parser.parse_args()

    ms_path = args.ms_path.resolve()
    zarr_dir = args.gain_dir.resolve()

    ms_name = ms_path.name

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    data_xds_list = xds_from_ms(
        ms_path,
        columns=args.column,
        index_cols=("TIME",),
        group_cols=("FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"))

    bkp_xds_list = xds_to_zarr(
        data_xds_list,
        f"{zarr_dir}::{timestamp}-{ms_name}-{args.column}.bkp.qc",
    )

    dask.compute(bkp_xds_list)
