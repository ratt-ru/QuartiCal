# -*- coding: utf-8 -*-
import dask.array as da
import numpy as np
from xarrayms import xds_from_table
from loguru import logger


def read_ms(opts):
    """Reads an input measurement set and generates a number of data sets."""

    # Create an xarray data set containing indexing columns. This is
    # necessary to determine initial chunking over row. TODO: Add blocking
    # based on arbitrary columns/jumps. Figure out behaviour on multi-SPW/field
    # data. Figure out chunking based on a memory budget rather than as an
    # option.

    indexing_xds = xds_from_table(opts.input_ms_name,
                                  columns=("TIME", "INTERVAL"),
                                  index_cols=("TIME",),
                                  group_cols=("SCAN_NUMBER",))

    # row_chunks is a dictionary containing row chunks per data set.

    row_chunks_per_xds = []

    for xds in indexing_xds:

        time_col = xds.TIME.data

        # Compute unique times, indices of their first ocurrence and number of
        # appearances.

        da_utimes, da_utime_inds, da_utime_counts = \
            da.unique(time_col, return_counts=True, return_index=True)

        utimes, utime_inds, utime_counts = da.compute(da_utimes,
                                                      da_utime_inds,
                                                      da_utime_counts)

        # If the chunking interval is a float after preprocessing, we are
        # dealing with a duration rather than a number of intervals. TODO:
        # Need to take resulting chunks and reprocess them based on chunk-on
        # columns and jumps.

        if isinstance(opts.input_ms_time_chunk, float):

            interval_col = indexing_xds[0].INTERVAL.data

            da_cumint = da.cumsum(interval_col[utime_inds])
            da_cumint = da_cumint - da_cumint[0]
            da_cumint_ind = \
                (da_cumint//opts.input_ms_time_chunk).astype(np.int32)
            _, da_utime_per_chunk = \
                da.unique(da_cumint_ind, return_counts=True)
            utime_per_chunk = da_utime_per_chunk.compute()

            cum_utime_per_chunk = np.cumsum(utime_per_chunk)
            cum_utime_per_chunk = cum_utime_per_chunk - cum_utime_per_chunk[0]

        else:

            cum_utime_per_chunk = range(0,
                                        len(utimes),
                                        opts.input_ms_time_chunk)

        chunks = np.add.reduceat(utime_counts, cum_utime_per_chunk).tolist()

        row_chunks_per_xds.append({"row": chunks})

        logger.debug("Scan {}: row chunks: {}", xds.SCAN_NUMBER, chunks)

    # Once we have determined the row chunks from the indexing columns, we set
    # up an xarray data set for the data. Note that we will reload certain
    # indexing columns so that they are consistent with the chunking strategy.

    data_columns = ("TIME", "ANTENNA1", "ANTENNA2", "DATA", "MODEL_DATA")

    data_xds = xds_from_table(opts.input_ms_name,
                              columns=data_columns,
                              index_cols=("TIME",),
                              group_cols=("SCAN_NUMBER",),
                              chunks=row_chunks_per_xds)

    return data_xds


def handle_model(opts):

    pass
