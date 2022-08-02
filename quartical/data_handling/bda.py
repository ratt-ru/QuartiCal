import numpy as np
import dask.array as da
import xarray
from quartical.utils.dask import blockwise_unique
from quartical.utils.maths import arr_gcd


def time_resampler(tcol, icol, reps, gcd, resample_size):

    resampled_time = np.empty(resample_size, dtype=np.float64)

    offset = 0

    for time, ivl, rep in zip(tcol, icol, reps):

        start = time - 0.5*ivl

        for n in range(1, rep + 1):

            resampled_time[offset] = start + 0.5 * gcd

            start += gcd

            offset += 1

    return np.sort(resampled_time)


def fix_time_col(time_col):

    otc = time_col.copy()

    for ind, (ele1, ele2) in enumerate(zip(otc[:-1], otc[1:])):
        if np.abs(ele2 - ele1) < 0.05:
            otc[ind + 1] = ele1
        else:
            otc[ind + 1] = ele2

    return otc


def process_bda_input(data_xds_list, spw_xds_list, weight_column):
    """Processes BDA xarray.Dataset objects into something more regular.

    Given a list of xarray.Dataset objects, upsamples and merges those which
    share a SCAN_NUMBER. Upsampling is to the highest frequency resolution
    present in the data.

    Args:
        data_xds_list: List of xarray.Datasets containing input BDA data.
        spw_xds_list: List of xarray.Datasets contataining SPW data.
        weight_column: String containing the input weight column.

    Returns:
        bda_xds_list: List of xarray.Dataset objects which contains upsampled
            and merged data.
        utime_per_xds: List of number of unique times per xds.
    """

    # If WEIGHT_SPECTRUM is not in use, BDA data makes no sense. TODO: This is
    # not strictly true. Any weight column with a frequency axis is valid.
    if weight_column != "WEIGHT_SPECTRUM":
        raise ValueError("--input-ms-weight column must be "
                         "WEIGHT_SPECTRUM for BDA data.")

    # Figure out the highest frequency resolution and its DDID.
    spw_dims = {i: xds.dims["chan"] for i, xds in enumerate(spw_xds_list)}
    max_nchan_ddid = max(spw_dims, key=spw_dims.get)
    max_nchan = spw_dims[max_nchan_ddid]

    bda_xds_list = []

    for xds in data_xds_list:

        upsample_factor = max_nchan//xds.dims["chan"]

        weight_col = xds.WEIGHT_SPECTRUM.data

        # Divide the weights by the upsampling factor - this should keep
        # the values consistent.
        bda_xds = xds.assign(
            {"WEIGHT_SPECTRUM": (("row", "chan", "corr"),
                                 weight_col/upsample_factor)})

        # Create a selection which will upsample the frequency axis.
        selection = np.repeat(np.arange(xds.dims["chan"]), upsample_factor)

        bda_xds = bda_xds.sel({"chan": selection})

        bda_xds = bda_xds.assign_attrs({"DATA_DESC_ID": max_nchan_ddid})

        bda_xds_list.append(bda_xds)

    unique_scans = np.unique([xds.SCAN_NUMBER for xds in bda_xds_list])

    # Determine mergeable datasets - they will share scan_number.
    xds_merge_list = \
        [[xds for xds in bda_xds_list if xds.SCAN_NUMBER == sn]
         for sn in unique_scans]

    bda_xds_list = [xarray.concat(xdss, dim="row")
                    for xdss in xds_merge_list]

    bda_xds_list = [xds.chunk({"row": xds.dims["row"]})
                    for xds in bda_xds_list]

    # This should guarantee monotonicity in time (not baseline).
    bda_xds_list = [xds.sortby("ROWID") for xds in bda_xds_list]

    # This is a necessary evil - compute the utimes present on the merged xds.
    _bda_xds_list = []

    for xds in bda_xds_list:

        interval_col = xds.INTERVAL.data
        time_col = xds.TIME.data

        uintervals = blockwise_unique(interval_col)
        gcd = uintervals.map_blocks(lambda x: np.array(arr_gcd(x)),
                                    chunks=(1,))

        # NOTE: Make the interval and time columns "perfect" i.e. ignore small
        # errors in the prior averaging. This is very dodgy and is mostly a
        # workaround.
        interval_col = da.round(interval_col/gcd).astype(np.int64) * gcd
        time_col = da.blockwise(fix_time_col, "r",
                                time_col, "r",
                                dtype=np.float64)

        upsample_reps = da.rint(interval_col/gcd).astype(np.int64)
        upsample_size = da.sum(upsample_reps).compute()

        upsampled_time_col = da.map_blocks(time_resampler,
                                           time_col,
                                           interval_col,
                                           upsample_reps,
                                           gcd,
                                           upsample_size,
                                           dtype=np.float64,
                                           chunks=(upsample_size,))

        # NOTE: Make the upsampled time column perfectly consistent. This is
        # very brittle but is neccessary as we cannot trust the input time
        # and interval values.
        upsampled_time_col = da.blockwise(fix_time_col, "r",
                                          upsampled_time_col, "r",
                                          dtype=np.float64)

        # TODO: This assumes a consistent interval everywhere, as we are still
        # using the GCD logic. This will need to change when we have access to
        # a BDA table which allows us to better restore the time axis.
        upsampled_ivl_col = gcd*da.ones_like(upsampled_time_col)

        row_map = da.map_blocks(
            lambda _col, _reps: np.repeat(np.arange(_col.size), _reps),
            time_col,
            upsample_reps,
            chunks=(upsample_size,))

        row_weights = da.map_blocks(
            lambda _reps: 1./np.repeat(_reps, _reps),
            upsample_reps,
            chunks=(upsample_size,))

        _bda_xds = xds.assign({"UPSAMPLED_TIME": (("urow",),
                              upsampled_time_col),
                               "UPSAMPLED_INTERVAL": (("urow",),
                              upsampled_ivl_col),
                               "ROW_MAP": (("urow",),
                              row_map),
                               "ROW_WEIGHTS": (("urow",),
                              row_weights)})

        _bda_xds_list.append(_bda_xds)

    bda_xds_list = _bda_xds_list

    utime_per_xds = [da.unique(xds.UPSAMPLED_TIME.data)
                     for xds in bda_xds_list]
    utime_per_xds = da.compute(*utime_per_xds)
    utime_per_xds = [ut.shape for ut in utime_per_xds]

    return bda_xds_list, utime_per_xds


def process_bda_output(xds_list, ref_xds_list, output_cols):
    """Processes xarray.Dataset objects back into BDA format.

    Given a list of xarray.Dataset objects, samples and splits into separate
    spectral windows.

    Args:
        xds_list: List of xarray.Datasets containing post-solve data.
        ref_xds_list: List of xarray.Datasets containing original data.
        output_cols: List of column names we expect to write.

    Returns:
        bda_xds_list: List of xarray.Dataset objects which contains BDA data.
    """

    bda_xds_list = []

    xds_dict = {xds.SCAN_NUMBER: xds for xds in xds_list}

    for ref_xds in ref_xds_list:

        xds = xds_dict[ref_xds.SCAN_NUMBER]

        xds = xds.assign_coords({"row": xds.ROWID.data})

        xds = xds.sel(row=ref_xds.ROWID.data)

        bda_xds = xarray.Dataset(coords=ref_xds.coords)

        for col_name in output_cols:

            col = xds[col_name]

            if "chan" in col.dims:
                data = col.data
                dims = col.dims
                shape = list(col.shape)

                chan_ind = dims.index('chan')

                nchan = xds.dims['chan']
                ref_nchan = ref_xds.dims['chan']

                shape[chan_ind: chan_ind + 1] = [ref_xds.dims['chan'], -1]

                data = data.reshape(shape)

                scdtype = np.obj2sctype(data)

                if np.issubdtype(scdtype, np.complexfloating):
                    # Corresponds to a visibility column. Simple average.
                    data = data.sum(axis=chan_ind + 1)/(nchan//ref_nchan)
                elif np.issubdtype(scdtype, np.floating):
                    # This probably isn't used at present and my be wrong.
                    data = data.sum(axis=chan_ind + 1)/(nchan//ref_nchan)
                elif np.issubdtype(scdtype, np.integer):
                    # Corresponds to BITFLAG style column. Bitwise or.
                    data = data.map_blocks(
                        lambda d, a: np.bitwise_or.reduce(d, axis=a),
                        chan_ind + 1,
                        drop_axis=chan_ind + 1)
                elif np.issubdtype(scdtype, np.bool):
                    # Corresponds to FLAG style column.
                    data = data.any(axis=chan_ind + 1)

                bda_xds = bda_xds.assign({col_name: (col.dims, data)})

            else:
                bda_xds = bda_xds.assign({col_name: (col.dims, col.data)})

        bda_xds_list.append(bda_xds)

    return bda_xds_list
