# -*- coding: utf-8 -*-
import dask
import dask.array as da
import numpy as np
from daskms import xds_from_ms, xds_from_table, xds_to_table
from quartical.flagging.flagging import update_kwrds, ibfdtype
from quartical.weights.weights import initialize_weights
from quartical.flagging.flagging import (is_set,
                                         make_bitmask,
                                         initialise_bitflags)
from quartical.data_handling.bda import process_bda_input, process_bda_output
from uuid import uuid4
from loguru import logger


def read_xds_list(opts):
    """Reads a measurement set and generates a list of xarray data sets.

    Args:
        opts: A Namepsace of global options.

    Returns:
        data_xds_list: A list of appropriately chunked xarray datasets.
        updated_kwrds: A dictionary of updated column keywords.
    """

    # Create an xarray data set containing indexing columns. This is
    # necessary to determine initial chunking over row. TODO: Add blocking
    # based on arbitrary columns/jumps. Figure out behaviour on multi-SPW/field
    # data. Figure out chunking based on a memory budget rather than as an
    # option.

    logger.debug("Setting up indexing xarray dataset.")

    indexing_xds_list = xds_from_ms(opts.input_ms_name,
                                    columns=("TIME", "INTERVAL"),
                                    index_cols=("TIME",),
                                    group_cols=(#"SCAN_NUMBER",
                                                "FIELD_ID",
                                                "DATA_DESC_ID"),
                                    taql_where="ANTENNA1 != ANTENNA2",
                                    chunks={"row": -1})

    # Read the antenna table and add the number of antennas to the options
    # namespace/dictionary. Leading underscore indiciates that this option is
    # private and added internally.

    antenna_xds = xds_from_table(opts.input_ms_name + "::ANTENNA")[0]

    n_ant = antenna_xds.dims["row"]

    logger.info("Antenna table indicates {} antennas were present for this "
                "observation.", n_ant)

    # Determine the number of correlations present in the measurement set.

    polarization_xds = xds_from_table(opts.input_ms_name + "::POLARIZATION")[0]

    opts._ms_ncorr = polarization_xds.dims["corr"]

    if opts._ms_ncorr not in (1, 2, 4):
        raise ValueError("Measurement set contains {} correlations - this "
                         "is not supported.".format(opts._ms_ncorr))

    logger.info("Polarization table indicates {} correlations are present in "
                "the measurement set.", opts._ms_ncorr)

    # Determine the feed types present in the measurement set.

    feed_xds = xds_from_table(opts.input_ms_name + "::FEED")[0]

    feeds = feed_xds.POLARIZATION_TYPE.data.compute()
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
    # or ignore it. TODO: Figure out how to prevent this thowing a message
    # wall.

    col_names = list(xds_from_ms(opts.input_ms_name)[0].keys())

    if "BITFLAG" in col_names:
        opts._bitflag_exists = True
        logger.info("BITFLAG column is present.")
    else:
        opts._bitflag_exists = False
        logger.info("BITFLAG column is missing. It will be added.")

    if "BITFLAG_ROW" in col_names:
        opts._bitflagrow_exists = True
        logger.info("BITFLAG_ROW column is present.")
    else:
        opts._bitflagrow_exists = False
        logger.info("BITFLAG_ROW column is missing. It will be added.")

    # Check whether the specified weight column exists. If not, log a warning
    # and fall back to unity weights.

    opts._unity_weights = opts.input_ms_weight_column.lower() == "unity"

    if not opts._unity_weights:
        if opts.input_ms_weight_column in col_names:
            logger.info(f"Using {opts.input_ms_weight_column} for weights.")
        else:
            logger.warning("Specified weight column was not present. "
                           "Falling back to unity weights.")
            opts._unity_weights = True

    # Determine the channels in the measurement set. Or handles unchunked case.
    # TODO: Handle multiple SPWs and specification in bandwidth.

    spw_xds_list = xds_from_table(
        opts.input_ms_name + "::SPECTRAL_WINDOW",
        group_cols=["__row__"],
        columns=["CHAN_FREQ", "CHAN_WIDTH"],
        chunks={"row": 1, "chan": opts.input_ms_freq_chunk or -1})

    # The spectral window xds should be correctly chunked in frequency.

    utime_chunking_per_xds, chunking_per_xds = \
        compute_chunking(indexing_xds_list, spw_xds_list, opts)

    # Once we have determined the row chunks from the indexing columns, we set
    # up an xarray data set for the data. Note that we will reload certain
    # indexing columns so that they are consistent with the chunking strategy.

    extra_columns = tuple(opts._model_columns)
    extra_columns += ("BITFLAG",) if opts._bitflag_exists else ()
    extra_columns += ("BITFLAG_ROW",) if opts._bitflagrow_exists else ()
    extra_columns += (opts.input_ms_weight_column,) if \
        not opts._unity_weights else ()

    data_columns = ("TIME", "INTERVAL", "ANTENNA1", "ANTENNA2", "DATA", "FLAG",
                    "FLAG_ROW", "UVW") + extra_columns

    data_xds_list, col_kwrds = xds_from_ms(
        opts.input_ms_name,
        columns=data_columns,
        index_cols=("TIME",),
        group_cols=(#"SCAN_NUMBER",
                    "FIELD_ID",
                    "DATA_DESC_ID"),
        taql_where="ANTENNA1 != ANTENNA2",
        chunks=chunking_per_xds,
        column_keywords=True,
        table_schema=["MS", {"BITFLAG": {'dims': ('chan', 'corr')}}])

    # Preserve a copy of the xds_list prior to any BDA/assignment. Necessary
    # for undoing BDA.
    ref_xds_list = data_xds_list

    # BDA data needs to be processed into something more manageable.
    if opts.input_ms_is_bda:
        data_xds_list, utime_chunking_per_xds = \
            process_bda_input(data_xds_list, spw_xds_list, opts)

    # Add coordinates to the xarray datasets - this becomes immensely useful
    # down the line.
    data_xds_list = [xds.assign_coords({"corr": np.arange(xds.dims["corr"]),
                                        "chan": np.arange(xds.dims["chan"]),
                                        "ant": np.arange(n_ant)})
                     for xds in data_xds_list]

    # Add the actual channel frequecies to the xds - this is in preparation
    # for solvers which require this information. Also adds the antenna names
    # which will be useful when reference antennas are required.
    data_xds_list = [xds.assign(
        {"CHAN_FREQ":
            (("chan",), spw_xds_list[xds.DATA_DESC_ID].CHAN_FREQ.data[0]),
         "CHAN_WIDTH":
            (("chan",), spw_xds_list[xds.DATA_DESC_ID].CHAN_WIDTH.data[0]),
         "ANT_NAME":
            (("ant",), antenna_xds.NAME.data)}) for xds in data_xds_list]

    # Add an attribute to the xds on which we will store the names of fields
    # which must be written to the MS. Also add the attribute which stores
    # the unique time chunking per xds.

    data_xds_list = \
        [xds.assign_attrs({"WRITE_COLS": [],
                           "UTIME_CHUNKS": list(map(int, utime_chunking_per_xds[xds_ind]))})
         for xds_ind, xds in enumerate(data_xds_list)]

    # We may only want to use some of the input correlation values. xarray
    # has a neat syntax for this. #TODO: This needs to depend on the number of
    # correlations actually present in the MS/on the xds.

    if opts.input_ms_correlation_mode == "diag" and opts._ms_ncorr == 4:
        data_xds_list = [xds.sel(corr=[0, 3]) for xds in data_xds_list]
    elif opts.input_ms_correlation_mode == "full" and opts._ms_ncorr != 4:
        raise ValueError(f"--input-ms-correlation-mode was set to full, "
                         f"but the measurement set only contains "
                         f"{opts._ms_ncorr} correlations")

    # If the BITFLAG and BITFLAG_ROW columns were missing, we simply add
    # appropriately sized dask arrays to the data sets. These can be written
    # to the MS at the end. Note that if we are adding the bitflag column,
    # we initiliase it using the internal dtype. This reduces the memory
    # footprint a little, although it will still ultimately be saved as an
    # int32. TODO: Check whether we can write it as int16 to save space.

    updated_kwrds = update_kwrds(col_kwrds, opts)

    # The use of name below guaratees that dask produces unique arrays and
    # avoids accidental aliasing.

    for xds_ind, xds in enumerate(data_xds_list):
        xds_updates = {}
        if not opts._bitflag_exists:
            data = da.zeros(xds.FLAG.data.shape,
                            dtype=ibfdtype,
                            chunks=xds.FLAG.data.chunks,
                            name="bfzeros-" + uuid4().hex)
            schema = ("row", "chan", "corr")
            xds_updates["BITFLAG"] = (schema, data)
        if not opts._bitflagrow_exists:
            data = da.zeros(xds.FLAG_ROW.data.shape,
                            dtype=ibfdtype,
                            chunks=xds.FLAG_ROW.data.chunks,
                            name="bfrzeros-" + uuid4().hex)
            schema = ("row",)
            xds_updates["BITFLAG_ROW"] = (schema, data)
        if xds_updates:
            data_xds_list[xds_ind] = xds.assign(xds_updates)

    # Add the external bitflag dtype to the opts Namespace. This is necessary
    # as internal bitflags may have a different dtype and we need to reconcile
    # the two. Note that we elect to interpret the input as an unsigned int
    # to avoid issues with negation. TODO: Check/warn that the maximal bit
    # is correct.
    ebfdtype = data_xds_list[0].BITFLAG.dtype

    if ebfdtype == np.int32:
        opts._ebfdtype = np.uint32
    elif ebfdtype == ibfdtype:
        opts._ebfdtype = ibfdtype
    else:
        raise TypeError("BITFLAG type {} not supported.".format(ebfdtype))

    return data_xds_list, ref_xds_list, updated_kwrds


def write_xds_list(xds_list, ref_xds_list, col_kwrds, opts):
    """Writes fields spicified in the WRITE_COLS attribute to the MS.

    Args:
        xds_list: A list of xarray datasets.
        ref_xds_list: A list of reference xarray.Dataset objects.
        col_kwrds: A dictionary of column keywords.
        opts: A Namepsace of global options.

    Returns:
        write_xds_list: A list of xarray datasets indicating success of writes.
    """

    import daskms.descriptors.ratt_ms  # noqa

    # If we selected some correlations, we need to be sure that whatever we
    # attempt to write back to the MS is still consistent. This does this using
    # the magic of reindex. TODO: Check whether it would be better to let
    # dask-ms handle this. This also might need some further consideration,
    # as the fill_value might cause problems.

    if opts._ms_ncorr != xds_list[0].corr.size:
        xds_list = \
            [xds.reindex({"corr": np.arange(opts._ms_ncorr)}, fill_value=0)
             for xds in xds_list]

    output_cols = tuple(set([cn for xds in xds_list for cn in xds.WRITE_COLS]))

    if opts.output_visibility_product:
        # Drop variables from columns we intend to overwrite.
        xds_list = [xds.drop_vars(opts.output_column, errors="ignore")
                    for xds in xds_list]

        vis_prod_map = {"residual": "_RESIDUAL",
                        "corrected_residual": "_CORRECTED_RESIDUAL",
                        "corrected_data": "_CORRECTED_DATA"}
        n_vis_prod = len(opts.output_visibility_product)

        # Rename QuartiCal's underscore prefixed results so that they will be
        # written to the appropriate column.
        xds_list = \
            [xds.rename({vis_prod_map[prod]: opts.output_column[ind]
             for ind, prod in enumerate(opts.output_visibility_product)})
             for xds in xds_list]

        output_cols += tuple(opts.output_column[:n_vis_prod])

    output_kwrds = {cn: col_kwrds.get(cn, {}) for cn in output_cols}

    if opts.input_ms_is_bda:
        xds_list = process_bda_output(xds_list, ref_xds_list, output_cols,
                                      opts)

    logger.info("Outputs will be written to {}.".format(
        ", ".join(output_cols)))

    # TODO: Nasty hack due to bug in daskms. Remove ASAP.
    xds_list = [xds.drop_vars(["ANT_NAME", "CHAN_FREQ", "CHAN_WIDTH"],
                              errors='ignore')
                for xds in xds_list]

    write_xds_list = xds_to_table(xds_list, opts.input_ms_name,
                                  columns=output_cols,
                                  column_keywords=output_kwrds,
                                  descriptor="ratt_ms(fixed=False)")

    return write_xds_list


def preprocess_xds_list(xds_list, col_kwrds, opts):
    """Adds data preprocessing steps - inits flags, weights and fixes bad data.

    Given a list of xarray.DataSet objects, initializes the bitflag data,
    the weight data and fixes bad data points (NaN, inf, etc). TODO: This
    function can likely be improved/extended.

    Args:
        xds_list: A list of xarray.DataSet objects containing MS data.
        col_kwrds: Column keywords - necessary for figuring out bitflags.
        opts: A Namepsace object of global options.

    Returns:
        output_xds_list: A list of xarray.DataSet objects containing MS data
            with preprocessing operations applied.
    """

    # In the event that not all input BITFLAGS are required, generate a mask
    # which can be applied to select the appropriate bits.

    bitmask = make_bitmask(col_kwrds, opts)

    output_xds_list = []

    for xds_ind, xds in enumerate(xds_list):

        # Unpack the data on the xds into variables with understandable names.
        # We create copies of arrays we intend to mutate as otherwise we end
        # up implicitly updating the xds.
        data_col = xds.DATA.data.copy()
        flag_col = xds.FLAG.data
        flag_row_col = xds.FLAG_ROW.data
        bitflag_col = xds.BITFLAG.data
        bitflag_row_col = xds.BITFLAG_ROW.data

        weight_col = initialize_weights(xds, data_col, opts)

        data_bitflags = initialise_bitflags(data_col,
                                            weight_col,
                                            flag_col,
                                            flag_row_col,
                                            bitflag_col,
                                            bitflag_row_col,
                                            bitmask)

        # Selections of the following form generate where operations - this
        # will return a new array and is therefore distribution safe. It might
        # be suboptimal though.

        # If we raised the invalid bitflag, zero those data points.
        data_col[is_set(data_bitflags, "INVALID")] = 0

        # Anywhere we have a full resolution bitflag, we set the weight to 0.
        weight_col[data_bitflags] = 0

        output_xds = xds.assign(
            {"DATA": (("row", "chan", "corr"), data_col),
             "WEIGHT": (("row", "chan", "corr"), weight_col),
             "DATA_BITFLAGS": (("row", "chan", "corr"), data_bitflags)})

        output_xds_list.append(output_xds)

    return output_xds_list


def compute_chunking(indexing_xds_list, spw_xds_list, opts, compute=True):
    """Compute time and frequency chunks for the input data.

    Given a list of indexing xds's, and a list of spw xds's, determines how to
    chunk the data given the chunking parameters.

    Args:
        indexing_xds_list: List of xarray.dataset objects contatining indexing
            information.
        spw_xds_list: List of xarray.dataset objects containing spectral window
            information.
        opts: A Namespace object containing options.
        compute: Boolean indicating whether or not to compute the result.

    Returns:
        A tuple of utime_chunking_per_xds and chunking_per_xds which describe
        the chunking of the data.
    """

    chan_chunks = {i: xds.chunks["chan"] for i, xds in enumerate(spw_xds_list)}

    # row_chunks is a list of dictionaries containing row chunks per data set.

    chunking_per_xds = []

    utime_chunking_per_xds = []

    for xds in indexing_xds_list:

        # If the chunking interval is a float after preprocessing, we are
        # dealing with a duration rather than a number of intervals. TODO:
        # Need to take resulting chunks and reprocess them based on chunk-on
        # columns and jumps.

        # TODO: BDA will assume no chunking, and in general we can skip this
        # bit if the row axis is unchunked.

        if isinstance(opts.input_ms_time_chunk, float):

            def interval_chunking(time_col, interval_col, time_chunk):

                utimes, uinds, ucounts = \
                    np.unique(time_col, return_counts=True, return_index=True)
                cumulative_interval = np.cumsum(interval_col[uinds])
                cumulative_interval -= cumulative_interval[0]
                chunk_map = \
                    (cumulative_interval // time_chunk).astype(np.int32)

                _, utime_chunks = np.unique(chunk_map, return_counts=True)

                chunk_starts = np.zeros(utime_chunks.size, dtype=np.int32)
                chunk_starts[1:] = np.cumsum(utime_chunks)[:-1]

                row_chunks = np.add.reduceat(ucounts, chunk_starts)

                return np.vstack((utime_chunks, row_chunks))

            chunking = da.map_blocks(interval_chunking,
                                     xds.TIME.data,
                                     xds.INTERVAL.data,
                                     opts.input_ms_time_chunk,
                                     chunks=((2,), (np.nan,)),
                                     dtype=np.int32)

        else:

            def integer_chunking(time_col, time_chunk):

                utimes, ucounts = np.unique(time_col, return_counts=True)
                n_utime = utimes.size
                time_chunk = time_chunk or n_utime  # Catch time_chunk == 0.

                utime_chunks = [time_chunk] * (n_utime // time_chunk)
                last_chunk = n_utime % time_chunk

                utime_chunks += [last_chunk] if last_chunk else []
                utime_chunks = np.array(utime_chunks)

                chunk_starts = np.arange(0, n_utime, time_chunk)

                row_chunks = np.add.reduceat(ucounts, chunk_starts)

                return np.vstack((utime_chunks, row_chunks))

            chunking = da.map_blocks(integer_chunking,
                                     xds.TIME.data,
                                     opts.input_ms_time_chunk,
                                     chunks=((2,), (np.nan,)),
                                     dtype=np.int32)

        utime_per_chunk = dask.delayed(tuple)(chunking[0, :])
        row_chunks = dask.delayed(tuple)(chunking[1, :])

        utime_chunking_per_xds.append(utime_per_chunk)

        chunking_per_xds.append({"row": row_chunks,
                                 "chan": chan_chunks[xds.DATA_DESC_ID]})

    if compute:
        return da.compute(utime_chunking_per_xds, chunking_per_xds)
    else:
        return utime_chunking_per_xds, chunking_per_xds
