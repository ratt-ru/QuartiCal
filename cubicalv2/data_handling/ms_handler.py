# -*- coding: utf-8 -*-
import dask.array as da
import numpy as np
from daskms import xds_from_ms, xds_from_table, xds_to_table
from cubicalv2.flagging.flagging import update_kwrds, ibfdtype
from uuid import uuid4
from loguru import logger


def read_ms(opts):
    """Reads an input measurement set and generates a number of data sets."""

    # Create an xarray data set containing indexing columns. This is
    # necessary to determine initial chunking over row. TODO: Add blocking
    # based on arbitrary columns/jumps. Figure out behaviour on multi-SPW/field
    # data. Figure out chunking based on a memory budget rather than as an
    # option.

    logger.debug("Setting up indexing xarray dataset.")

    indexing_xds = xds_from_ms(opts.input_ms_name,
                               columns=("TIME", "INTERVAL"),
                               index_cols=("TIME",),
                               group_cols=("SCAN_NUMBER",
                                           "FIELD_ID",
                                           "DATA_DESC_ID"))

    # Read the antenna table and add the number of antennas to the options
    # namespace/dictionary. Leading underscore indiciates that this option is
    # private.

    antenna_xds = xds_from_table(opts.input_ms_name+"::ANTENNA")

    opts._n_ant = antenna_xds[0].dims["row"]

    logger.info("Antenna table indicates {} antennas were present for this "
                "observation.", opts._n_ant)

    # Determine the number of correlations present in the measurement set.

    polarization_xds = xds_from_table(opts.input_ms_name + "::POLARIZATION")

    opts._ms_ncorr = polarization_xds[0].dims["corr"]

    if opts._ms_ncorr not in (1, 2, 4):
        raise ValueError("Measurement set contains {} correlations - this "
                         "is not supported.".format(opts._ms_ncorr))

    logger.info("Polarization table indicates {} correlations are present in "
                "the measurement set.", opts._ms_ncorr)

    # Determine the feed types present in the measurement set.

    feed_xds = xds_from_table(opts.input_ms_name + "::FEED")

    feeds = feed_xds[0].POLARIZATION_TYPE.data.compute()
    unique_feeds = np.unique(feeds)

    if np.all([feed in "XxYy" for feed in unique_feeds]):
        opts._feed_type = "linear"
    elif np.all([feed in "LlRr" for feed in unique_feeds]):
        opts._feed_type = "circular"
    else:
        raise ValueError("Unsupported feed type/configuration.")

    logger.info("Feed table indicates {} ({}) feeds are present in the "
                "measurement set.", unique_feeds, opts._feed_type)

    # Determine the phase direction from the measurement set. TODO: This will
    # probably need to be done on a per xds basis. Can probably be accomplished
    # by merging the field xds grouped by DDID into data grouped by DDID.

    field_xds = xds_from_table(opts.input_ms_name + "::FIELD")[0]
    opts._phase_dir = np.squeeze(field_xds.PHASE_DIR.data.compute())

    logger.info("Field table indicates phase centre is at ({} {}).",
                opts._phase_dir[0], opts._phase_dir[1])

    # Check whether the BITFLAG column exists - if not, we will need to add it
    # or ignore it.

    try:
        xds_from_ms(opts.input_ms_name, columns=("BITFLAG",))
        opts._bitflag_exists = True
        logger.info("BITFLAG column is present.")
    except RuntimeError:
        opts._bitflag_exists = False
        logger.info("BITFLAG column is missing. It will be added.")

    try:
        xds_from_ms(opts.input_ms_name, columns=("BITFLAG_ROW",))
        opts._bitflagrow_exists = True
        logger.info("BITFLAG_ROW column is present.")
    except RuntimeError:
        opts._bitflagrow_exists = False
        logger.info("BITFLAG_ROW column is missing. It will be added.")

    # Check whether the specified weight column exists. If not, log a warning
    # and fall back to unity weights.

    opts._unity_weights = opts.input_ms_weight_column.lower() == "unity"

    if not opts._unity_weights:
        try:
            xds_from_ms(opts.input_ms_name,
                        columns=(opts.input_ms_weight_column))
        except RuntimeError:
            logger.warning("Specified weight column was not found/understood. "
                           "Falling back to unity weights.")
            opts._unity_weights = True

    # row_chunks is a list of dictionaries containing row chunks per data set.

    row_chunks_per_xds = []

    chunk_spec_per_xds = []

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
            cum_utime_per_chunk = [0] + cum_utime_per_chunk[:-1].tolist()

        else:
            n_utimes = len(utimes)
            tc = opts.input_ms_time_chunk
            n_full = n_utimes//tc
            remainder = [(n_utimes - n_full*tc)] if n_utimes % tc else []
            utime_per_chunk = [tc]*n_full + remainder
            cum_utime_per_chunk = list(range(0,
                                             len(utimes),
                                             opts.input_ms_time_chunk))

        chunk_spec_per_xds.append(tuple(utime_per_chunk))

        chunks = np.add.reduceat(utime_counts, cum_utime_per_chunk).tolist()

        row_chunks_per_xds.append({"row": chunks})

        logger.debug("Scan {}: row chunks: {}", xds.SCAN_NUMBER, chunks)

    # Once we have determined the row chunks from the indexing columns, we set
    # up an xarray data set for the data. Note that we will reload certain
    # indexing columns so that they are consistent with the chunking strategy.

    extra_columns = tuple(opts._model_columns)
    extra_columns += ("BITFLAG",) if opts._bitflag_exists else ()
    extra_columns += ("BITFLAG_ROW",) if opts._bitflagrow_exists else ()
    extra_columns += (opts.input_ms_weight_column,) if \
        not opts._unity_weights else ()

    data_columns = ("TIME", "ANTENNA1", "ANTENNA2", "DATA", "FLAG", "FLAG_ROW",
                    "UVW") + extra_columns

    data_xds, col_kwrds = xds_from_ms(opts.input_ms_name,
                                      columns=data_columns,
                                      index_cols=("TIME",),
                                      group_cols=("SCAN_NUMBER",
                                                  "FIELD_ID",
                                                  "DATA_DESC_ID"),
                                      chunks=row_chunks_per_xds,
                                      column_keywords=True)

    # If the BITFLAG and BITFLAG_ROW columns were missing, we simply add
    # appropriately sized dask arrays to the data sets. These can be written
    # to the MS at the end. Note that if we are adding the bitflag column,
    # we initiliase it using the internal dtype. This reduces the memory
    # footprint a little, although it will still ultimately be saved as an
    # int32. TODO: Check whether we can write it as int16 to save space.

    updated_kwrds = update_kwrds(col_kwrds, opts)

    # The use of name below guaratees that dask produces unique arrays and
    # avoids accidental aliasing.

    for xds_ind, xds in enumerate(data_xds):
        xds_updates = {}
        if not opts._bitflag_exists:
            data = da.zeros(xds.FLAG.data.shape,
                            dtype=ibfdtype,
                            chunks=xds.FLAG.data.chunks,
                            name="zeros-" + uuid4().hex)
            schema = ("row", "chan", "corr")
            xds_updates["BITFLAG"] = (schema, data)
        if not opts._bitflagrow_exists:
            data = da.zeros(xds.FLAG_ROW.data.shape,
                            dtype=ibfdtype,
                            chunks=xds.FLAG_ROW.data.chunks,
                            name="zeros-" + uuid4().hex)
            schema = ("row",)
            xds_updates["BITFLAG_ROW"] = (schema, data)
        if xds_updates:
            data_xds[xds_ind] = xds.assign(xds_updates)

    # Add the external bitflag dtype to the opts Namespace. This is necessary
    # as internal bitflags may have a different dtype and we need to reconcile
    # the two. Note that we elect to interpret the input as an unsigned int
    # to avoid issues with negation. TODO: Check/warn that the maximal bit
    # is correct.
    ebfdtype = data_xds[0].BITFLAG.dtype

    if ebfdtype == np.int32:
        opts._ebfdtype = np.uint32
    elif ebfdtype == ibfdtype:
        opts._ebfdtype = ibfdtype
    else:
        raise TypeError("BITFLAG type {} not supported.".format(ebfdtype))

    # Add an attribute to the xds on which we will store the names of fields
    # which must be written to the MS. Also add the attribute which stores
    # the unique times per xds.
    for xds_ind, xds in enumerate(data_xds):
        data_xds[xds_ind] = \
            xds.assign_attrs(WRITE_COLS=[],
                             UTIME_CHUNKS=chunk_spec_per_xds[xds_ind])

    return data_xds, updated_kwrds


def write_columns(xds_list, col_kwrds, opts):

    import daskms.descriptors.ratt_ms  # noqa

    output_cols = tuple(set([cn for xds in xds_list for cn in xds.WRITE_COLS]))
    output_kwrds = {cn: col_kwrds.get(cn, {}) for cn in output_cols}

    logger.info("Outputs will be written to {}.".format(
        ", ".join(output_cols)))

    return xds_to_table(xds_list, opts.input_ms_name,
                        columns=output_cols,
                        column_keywords=output_kwrds,
                        descriptor="ratt_ms(fixed=False)")
