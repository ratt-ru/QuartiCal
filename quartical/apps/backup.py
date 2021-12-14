import argparse
from daskms import xds_from_ms, xds_to_table
from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
from pathlib import Path
import time
import dask


def backup():
    parser = argparse.ArgumentParser(
        description='Backup any Measurement Set column to zarr. Backups will '
                    'be labelled automatically using the current datetime, '
                    'the Measurement Set name and the column name.'
    )

    parser.add_argument(
        'ms_path',
        type=Path,
        help='Path to input measurement set, e.g. path/to/dir/foo.MS.'
    )
    parser.add_argument(
        'zarr_dir',
        type=Path,
        help='Path to desired backup location. Note that this only allows '
             'the user to specify a directory and not the name of the backup '
             'zarr that will be created, e.g. path/to/dir.'
    )
    parser.add_argument('column',
                        type=str,
                        help='Name of column to be backed up.')

    args = parser.parse_args()

    ms_path = args.ms_path.resolve()
    zarr_dir = args.zarr_dir.resolve()

    ms_name = ms_path.name

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    data_xds_list = xds_from_ms(
        ms_path,
        columns=args.column,
        index_cols=("TIME",),
        group_cols=("DATA_DESC_ID",))

    bkp_xds_list = xds_to_zarr(
        data_xds_list,
        f"{zarr_dir}::{timestamp}-{ms_name}-{args.column}.bkp.qc",
    )

    dask.compute(bkp_xds_list)


def restore():
    parser = argparse.ArgumentParser(
        description='Restore a zarr column backup to a Measurement Set.'
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
