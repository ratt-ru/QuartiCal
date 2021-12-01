import argparse
from daskms import xds_from_ms, xds_to_table
from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
from pathlib import Path
import time
import dask


def backup():
    parser = argparse.ArgumentParser(description='Backup MS columns to zarr.')

    parser.add_argument('input_path',
                        type=Path,
                        help='Path to input measurement set.')
    parser.add_argument('output_path',
                        type=Path,
                        help='Path to desired backup location.')
    parser.add_argument('column',
                        type=str,
                        help='Column to be backed up.')

    args = parser.parse_args()

    input_path = args.input_path.resolve()
    output_path = args.output_path.resolve()

    ms_name = input_path.name

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    data_xds_list = xds_from_ms(
        input_path,
        columns=args.column,
        index_cols=("TIME",),
        group_cols=("DATA_DESC_ID",))

    bkp_xds_list = xds_to_zarr(
        data_xds_list,
        f"{output_path}::{timestamp}-{ms_name}-{args.column}.bkp.qc",
    )

    dask.compute(bkp_xds_list)


def restore():
    parser = argparse.ArgumentParser(
        description='Restore zarr column backup to MS.'
    )

    parser.add_argument('input_path',
                        type=Path,
                        help='Path to backup zarr column.')
    parser.add_argument('output_path',
                        type=Path,
                        help='Path to measurement set.')
    parser.add_argument('column',
                        type=str,
                        help='Column to be restored up.')

    args = parser.parse_args()

    input_path = args.input_path.resolve()
    output_path = args.output_path.resolve()

    zarr_xds_list = xds_from_zarr(
        f"{input_path.parent}::{input_path.name}",
    )

    restored_xds_list = xds_to_table(
        zarr_xds_list,
        str(output_path),
        columns=(args.column,)
    )

    dask.compute(restored_xds_list)
