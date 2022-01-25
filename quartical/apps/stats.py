import argparse
from daskms import xds_from_ms, xds_from_table
import dask.array as da
import numpy as np


def stat():
    parser = argparse.ArgumentParser(
        description='Produce .'
    )

    parser.add_argument(
        'zarr_path',
        type=Path,
        help='Path to backup zarr column e.g. '
             'path/to/dir/20211201-154457-foo.MS-FLAG.bkp.qc.'
    )
    parser.add_argument(
        'ms_path',
        type=Path,
        help='Path to measurement set, e.g. path/to/dir/foo.MS.'
    )
    parser.add_argument(
        'column',
        type=str,
        help='Name of column to populate using the backup. Note that this '
             'does not have to be the same column as was used to create the '
             'backup.'
    )

    args = parser.parse_args()

    zarr_path = args.zarr_path.resolve()
    ms_path = args.ms_path.resolve()

    zarr_xds_list = xds_from_zarr(
        f"{zarr_path.parent}::{zarr_path.name}",
    )

    restored_xds_list = xds_to_table(
        zarr_xds_list,
        str(ms_path),
        columns=(args.column,)
    )

    dask.compute(restored_xds_list)
