import argparse
from daskms import xds_from_storage_ms, xds_to_storage_table
from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
from daskms.fsspec_store import DaskMSStore
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
        type=DaskMSStore,
        help='Path to input measurement set, e.g. path/to/dir/foo.MS. Also '
             'accepts valid s3 urls.'
    )
    parser.add_argument(
        'zarr_dir',
        type=DaskMSStore,
        help='Path to desired backup location. Note that this only allows '
             'the user to specify a directory and not the name of the backup '
             'zarr that will be created, e.g. path/to/dir. Also '
             'accepts valid s3 urls.'
    )
    parser.add_argument('column',
                        type=str,
                        help='Name of column to be backed up.')

    args = parser.parse_args()

    ms_name = args.ms_path.full_path.rsplit("/", 1)[1]

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    data_xds_list = xds_from_storage_ms(
        args.ms_path,
        columns=args.column,
        index_cols=("TIME",),
        group_cols=("DATA_DESC_ID",))

    for i, ds in enumerate(data_xds_list):
        chunks = {'row':'auto'}
        if 'chan' in ds.dims:
            chunks['chan'] = 'auto'
        data_xds_list[i] = ds.chunk(chunks)

    bkp_xds_list = xds_to_zarr(
        data_xds_list,
        f"{args.zarr_dir.url}::{timestamp}-{ms_name}-{args.column}.bkp.qc",
    )

    dask.compute(bkp_xds_list)


def restore():
    parser = argparse.ArgumentParser(
        description='Restore a zarr column backup to a Measurement Set.'
    )

    parser.add_argument(
        'zarr_path',
        type=DaskMSStore,
        help='Path to backup zarr column e.g. '
             'path/to/dir/20211201-154457-foo.MS-FLAG.bkp.qc. '
             'Also accepts valid s3 urls.'
    )
    parser.add_argument(
        'ms_path',
        type=DaskMSStore,
        help='Path to measurement set, e.g. path/to/dir/foo.MS. '
             'Also accepts valid s3 urls.'
    )
    parser.add_argument(
        'column',
        type=str,
        help='Name of column to populate using the backup. Note that this '
             'does not have to be the same column as was used to create the '
             'backup.'
    )

    args = parser.parse_args()

    zarr_root, zarr_name = args.zarr_path.url.rsplit("/", 1)

    zarr_xds_list = xds_from_zarr(f"{zarr_root}::{zarr_name}")

    restored_xds_list = xds_to_storage_table(
        zarr_xds_list,
        args.ms_path,
        columns=(args.column,),
        rechunk=True
    )

    dask.compute(restored_xds_list)
