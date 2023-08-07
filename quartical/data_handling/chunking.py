import dask.delayed as dd
import numpy as np
import dask.array as da
from daskms import xds_from_storage_ms, xds_from_storage_table


def compute_chunking(ms_opts, compute=True):

    # Create an xarray data set containing indexing columns. This is
    # necessary to determine initial chunking over row and chan. TODO: Test
    # multi-SPW/field cases. Implement a memory budget.

    indexing_xds_list = xds_from_storage_ms(
        ms_opts.path,
        columns=("TIME", "INTERVAL"),
        index_cols=("TIME",),
        group_cols=ms_opts.group_by,
        chunks={"row": -1}
    )

    utime_chunking_per_xds, row_chunking_per_xds = row_chunking(
        indexing_xds_list,
        ms_opts.time_chunk,
        compute=False
    )

    spw_xds_list = xds_from_storage_table(
        ms_opts.path + "::SPECTRAL_WINDOW",
        group_cols=["__row__"],
        columns=["CHAN_FREQ", "CHAN_WIDTH"],
        chunks={"row": 1, "chan": -1}
    )

    chan_chunking_per_spw = chan_chunking(
        spw_xds_list,
        ms_opts.freq_chunk,
        compute=False
    )

    chan_chunking_per_xds = [chan_chunking_per_spw[xds.DATA_DESC_ID]
                             for xds in indexing_xds_list]

    zipper = zip(row_chunking_per_xds, chan_chunking_per_xds)
    chunking_per_data_xds = [{"row": r, "chan": c} for r, c in zipper]

    chunking_per_spw_xds = \
        [{"__row__": 1, "chan": c} for c in chan_chunking_per_spw.values()]

    if compute:
        return da.compute(
            utime_chunking_per_xds,
            chunking_per_data_xds,
            chunking_per_spw_xds
        )
    else:
        utime_chunking_per_xds, chunking_per_data_xds, chunking_per_spw_xds


def chan_chunking(
    spw_xds_list,
    freq_chunk,
    compute=True
):
    """Compute frequency chunks for the input data.

    Given a list of indexing xds's, and a list of spw xds's, determines how to
    chunk the data in frequency given the chunking parameters.

    Args:
        indexing_xds_list: List of xarray.dataset objects contatining indexing
            information.
        spw_xds_list: List of xarray.dataset objects containing SPW
            information.
        freq_chunk: Int or float specifying chunking.
        compute: Boolean indicating whether or not to compute the result.

    Returns:
        A list giving the chunking in freqency for each SPW xds.
    """
    chan_chunking_per_spw = {}

    for ddid, xds in enumerate(spw_xds_list):

        # If the chunking interval is a float after preprocessing, we are
        # dealing with a bandwidth rather than a number of channels.

        if isinstance(freq_chunk, float):

            def interval_chunking(chan_widths, freq_chunk):

                chunks = ()
                bin_width = 0
                bin_nchan = 0
                for width in chan_widths:
                    bin_width += width
                    bin_nchan += 1
                    if bin_width > freq_chunk:
                        chunks += (bin_nchan,)
                        bin_width = 0
                        bin_nchan = 0
                if bin_width:
                    chunks += (bin_nchan,)

                return np.array(chunks, dtype=np.int32)

            chunking = da.map_blocks(interval_chunking,
                                     xds.CHAN_WIDTH.data[0],
                                     freq_chunk,
                                     chunks=((np.nan,),),
                                     dtype=np.int32)

        else:

            def integer_chunking(chan_widths, freq_chunk):

                n_chan = chan_widths.size
                freq_chunk = freq_chunk or n_chan  # Catch zero case.

                chunk_starts = np.arange(0, n_chan, freq_chunk)

                chunks = np.array(
                    [
                        freq_chunk if i + freq_chunk < n_chan else n_chan - i
                        for i in chunk_starts
                    ]
                )

                return chunks.astype(np.int32)

            chunking = da.map_blocks(integer_chunking,
                                     xds.CHAN_WIDTH.data[0],
                                     freq_chunk,
                                     chunks=((np.nan,),),
                                     dtype=np.int32)

        # We use delayed to convert to tuples and satisfy daskms/dask.
        chan_chunking_per_spw[ddid] = dd(tuple)(chunking)

    if compute:
        return da.compute(chan_chunking_per_spw)[0]
    else:
        return chan_chunking_per_spw


def row_chunking(
    indexing_xds_list,
    time_chunk,
    compute=True
):
    """Compute time and frequency chunks for the input data.

    Given a list of indexing xds's, and a list of spw xds's, determines how to
    chunk the data given the chunking parameters.

    Args:
        indexing_xds_list: List of xarray.dataset objects contatining indexing
            information.
        time_chunk: Int or float specifying chunking.
        compute: Boolean indicating whether or not to compute the result.

    Returns:
        A tuple of utime_chunking_per_xds and row_chunking_per_xds which
        describe the chunking of the data.
    """

    row_chunking_per_xds = []
    utime_chunking_per_xds = []

    for xds in indexing_xds_list:

        # If the chunking interval is a float after preprocessing, we are
        # dealing with a duration rather than a number of intervals.

        # TODO: BDA will assume no chunking, and in general we can skip this
        # bit if the row axis is unchunked.

        if isinstance(time_chunk, float):

            def interval_chunking(time_col, interval_col, time_chunk):
                """Given a time column, figure out interval chunking."""

                _, uinds, ucounts = \
                    np.unique(time_col, return_counts=True, return_index=True)
                cumulative_interval = np.cumsum(interval_col[uinds])
                cumulative_interval -= cumulative_interval[0]
                chunk_map = \
                    (cumulative_interval // time_chunk).astype(np.int32)

                _, utime_chunks = np.unique(chunk_map, return_counts=True)

                chunk_starts = np.zeros(utime_chunks.size, dtype=np.int32)
                chunk_starts[1:] = np.cumsum(utime_chunks)[:-1]

                row_chunks = np.add.reduceat(ucounts, chunk_starts)

                return np.vstack((utime_chunks, row_chunks)).astype(np.int32)

            chunking = da.map_blocks(
                interval_chunking,
                xds.TIME.data,
                xds.INTERVAL.data,
                time_chunk,
                chunks=((2,), (np.nan,)),
                dtype=np.int32
            )

        else:

            def integer_chunking(time_col, time_chunk):
                """Given a time column, figure out integer chunking."""

                utimes, ucounts = np.unique(time_col, return_counts=True)
                n_utime = utimes.size
                time_chunk = time_chunk or n_utime  # Catch 0.

                chunk_starts = np.arange(0, n_utime, time_chunk)

                utime_chunks = np.array(
                    [
                        time_chunk if i + time_chunk < n_utime else n_utime - i
                        for i in chunk_starts
                    ]
                )

                row_chunks = np.add.reduceat(ucounts, chunk_starts)

                return np.vstack((utime_chunks, row_chunks)).astype(np.int32)

            chunking = da.map_blocks(
                integer_chunking,
                xds.TIME.data,
                time_chunk,
                chunks=((2,), (np.nan,)),
                dtype=np.int32
            )

        # We use delayed to convert to tuples and satisfy daskms/dask.
        # TODO: Why was this necessary and can we avoid it?
        utime_per_chunk = dd(tuple)(chunking[0, :])
        row_chunks = dd(tuple)(chunking[1, :])

        utime_chunking_per_xds.append(utime_per_chunk)
        row_chunking_per_xds.append(row_chunks)

    if compute:
        return da.compute(utime_chunking_per_xds, row_chunking_per_xds)
    else:
        return utime_chunking_per_xds, row_chunking_per_xds
