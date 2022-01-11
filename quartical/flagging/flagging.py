import dask.array as da
import numpy as np
from loguru import logger  # noqa
from quartical.utils.xarray import check_fields, check_dims
from quartical.flagging.flagging_kernels import (compute_bl_mad_and_med,
                                                 compute_gbl_mad_and_med,
                                                 compute_chisq,
                                                 compute_mad_flags)


def finalise_flags(xds_list):
    """Finishes processing flags to produce writable flag data.

    Given a list of xarray.Dataset objects, uses the updated flag column to
    create appropriate flags for writing to disk. Removes all temporary flags.

    Args:
        xds_list: A list of xarray datasets.

    Returns:
        writable_xds: A list of xarray datasets.
    """

    writable_xds = []

    for xds in xds_list:

        data_col = xds.DATA.data
        flag_col = xds.FLAG.data

        # Remove QuartiCal's temporary flagging.
        flag_col = da.where(flag_col == -1, 0, flag_col)

        # Reintroduce the correlation axis.
        flag_col = da.broadcast_to(flag_col[:, :, None],
                                   data_col.shape,
                                   chunks=data_col.chunks)

        # Convert back to a boolean array.
        flag_col = flag_col.astype(np.bool)

        # Make the FLAG_ROW column consistent with FLAG.
        flag_row_col = da.all(flag_col, axis=(1, 2))

        updated_xds = xds.assign(
            {
                "FLAG": (xds.DATA.dims, flag_col),
                "FLAG_ROW": (xds.FLAG_ROW.dims, flag_row_col)
            }
        )

        writable_xds.append(updated_xds)

    return writable_xds


def flag_row_to_flag(input_xds_list,
                     flag_row_field="FLAG_ROW",
                     input_flag_field="FLAG",
                     output_flag_field="FLAG"):
    """Combine FLAG_ROW with FLAG.

    Args:
        input_xds_list: A list of xarray.Dataset objects.
        flag_row_field: A str corresponding to FLAG_ROW or equivalent.
        input_flag_field: A str corresponding to input flag field e.g. FLAG.
        output_flag_field: A str corresponding to desired output flag field.

    Returns:
        output_xds_list: A list of xarray.DataSet objects with flag row
            applied to flags and assinged to a existing/new field.
    """

    output_dims = ('row', 'chan', 'corr')

    check_fields(input_xds_list, (flag_row_field, input_flag_field))
    check_dims(input_xds_list, output_dims)

    output_xds_list = []

    for input_xds in input_xds_list:

        input_flag_row = input_xds[flag_row_field].data
        input_flags = input_xds[input_flag_field].data

        output_flags = da.logical_or(input_flags,
                                     input_flag_row[:, None,  None])

        output_xds = \
            input_xds.assign({output_flag_field: (output_dims, output_flags)})

        output_xds_list.append(output_xds)

    return output_xds_list


def flag_zeros(input_xds_list,
               target_field,
               input_flag_field="FLAG",
               output_flag_field="FLAG"):
    """Raise flags on points which are zero.

    Args:
        input_xds_list: A list of xarray.Dataset objects.
        target_field: A str corresponding to fields to check for zeroes.
        input_flag_field: A str corresponding to input flag field e.g. FLAG.
        output_flag_field: A str corresponding to desired output flag field.

    Returns:
        output_xds_list: A list of xarray.DataSet objects with zero values
            flagged and the results assigned to a new/existing field.
    """

    output_dims = ('row', 'chan', 'corr')

    check_fields(input_xds_list, (target_field, input_flag_field))
    check_dims(input_xds_list, output_dims)

    output_xds_list = []

    for input_xds in input_xds_list:

        data = input_xds[target_field].data
        input_flags = input_xds[input_flag_field].data

        output_flags = da.where(data == 0, True, input_flags)

        output_xds = \
            input_xds.assign({output_flag_field: (output_dims, output_flags)})

        output_xds_list.append(output_xds)

    return output_xds_list


def combine_corr_flags(input_xds_list,
                       input_flag_field="FLAG",
                       output_flag_field="FLAG"):
    """Combine flags along the correlation axis.

    Args:
        input_xds_list: A list of xarray.Dataset objects.
        input_flag_field: A str corresponding to input flag field e.g. FLAG.
        output_flag_field: A str corresponding to desired output flag field.

    Returns:
        output_xds_list: A list of xarray.DataSet objects with flag values
            collapsed along correlation using a logical or.
    """

    output_dims = ('row', 'chan')

    check_fields(input_xds_list, (input_flag_field,))
    check_dims(input_xds_list, output_dims)

    output_xds_list = []

    for input_xds in input_xds_list:

        input_flags = input_xds[input_flag_field].data

        output_flags = da.any(input_flags, axis=-1)

        output_xds = \
            input_xds.assign({output_flag_field: (output_dims, output_flags)})

        output_xds_list.append(output_xds)

    return output_xds_list


def flag_autocorrelations(input_xds_list,
                          flag_value=-1,
                          input_flag_field="FLAG",
                          output_flag_field="FLAG"):
    """Flag autocorrelation values - this will promote flag dtype to int8.

    Args:
        input_xds_list: A list of xarray.Dataset objects.
        flag_value: An int to use as a flag value - necessary for temporary
            flagging.
        input_flag_field: A str corresponding to input flag field e.g. FLAG.
        output_flag_field: A str corresponding to desired output flag field.

    Returns:
        output_xds_list: A list of xarray.DataSet objects with autocorrelation
            values flagged and assigned to a new/existing field.
    """

    output_dims = ('row', 'chan')

    check_fields(input_xds_list, (input_flag_field, "ANTENNA1", "ANTENNA2"))
    check_dims(input_xds_list, output_dims)

    output_xds_list = []

    for input_xds in input_xds_list:

        antenna1 = input_xds.ANTENNA1.data
        antenna2 = input_xds.ANTENNA2.data
        input_flags = input_xds[input_flag_field].data

        output_flags = da.where((antenna1 == antenna2)[:, None] & ~input_flags,
                                np.int8(flag_value), input_flags)

        output_xds = \
            input_xds.assign({output_flag_field: (output_dims, output_flags)})

        output_xds_list.append(output_xds)

    return output_xds_list


def flag_uv_range(input_xds_list,
                  uv_min,
                  uv_max,
                  flag_value=-1,
                  input_flag_field="FLAG",
                  output_flag_field="FLAG"):
    """Flag values falling outside of a uv range.

    Args:
        input_xds_list: A list of xarray.Dataset objects.
        uv_min: Float corresponding to minimum of admissable uv range.
        uv_max: Float corresponding to maximum of admissable uv range.
        flag_value: An int to use as a flag value - necessary for temporary
            flagging.
        input_flag_field: A str corresponding to input flag field e.g. FLAG.
        output_flag_field: A str corresponding to desired output flag field.

    Returns:
        output_xds_list: A list of xarray.DataSet objects with values outside
            the uv range flagged and assigned to a new/existing field.
    """

    output_dims = ('row', 'chan')

    check_fields(input_xds_list, (input_flag_field, "UVW"))
    check_dims(input_xds_list, output_dims)

    output_xds_list = []

    for input_xds in input_xds_list:

        uvw = input_xds.UVW.data
        input_flags = input_xds[input_flag_field].data

        uv = da.sqrt(da.sum(uvw[:, :2] ** 2, axis=1))

        uv_sel = (uv_min < uv) & (uv < (uv_max or np.inf))

        output_flags = da.where(~uv_sel[:, None] & ~input_flags,
                                np.int8(flag_value), input_flags)

        output_xds = \
            input_xds.assign({output_flag_field: (output_dims, output_flags)})

        output_xds_list.append(output_xds)

    return output_xds_list


def valid_median(arr):
    return np.median(arr[np.isfinite(arr) & (arr > 0)], keepdims=True)


def add_mad_graph(data_xds_list, mad_opts):

    bl_thresh = mad_opts.threshold_bl
    gbl_thresh = mad_opts.threshold_global
    max_deviation = mad_opts.max_deviation

    flagged_data_xds_list = []

    for xds in data_xds_list:
        residuals = xds._RESIDUAL.data
        weight_col = xds._WEIGHT.data
        flag_col = xds.FLAG.data
        ant1_col = xds.ANTENNA1.data
        ant2_col = xds.ANTENNA2.data
        n_ant = xds.dims["ant"]
        n_t_chunk = residuals.numblocks[0]

        chisq = da.blockwise(compute_chisq, ("rowlike", "chan"),
                             residuals, ("rowlike", "chan", "corr"),
                             weight_col, ("rowlike", "chan", "corr"),
                             dtype=residuals.real.dtype,
                             align_arrays=False,
                             concatenate=True)

        bl_mad_and_med = da.blockwise(
            compute_bl_mad_and_med, ("rowlike", "ant1", "ant2"),
            chisq, ("rowlike", "chan"),
            flag_col, ("rowlike", "chan"),
            ant1_col, ("rowlike",),
            ant2_col, ("rowlike",),
            n_ant, None,
            dtype=chisq.dtype,
            align_arrays=False,
            concatenate=True,
            adjust_chunks={"rowlike": (2,)*n_t_chunk},
            new_axes={"ant1": n_ant,
                      "ant2": n_ant}
        )

        gbl_mad_and_med = da.blockwise(
            compute_gbl_mad_and_med, ("rowlike",),
            chisq, ("rowlike", "chan"),
            flag_col, ("rowlike", "chan"),
            dtype=chisq.dtype,
            align_arrays=False,
            concatenate=True,
            adjust_chunks={"rowlike": (2,)*n_t_chunk}
        )

        row_chunks = residuals.chunks[0]

        mad_flags = da.blockwise(compute_mad_flags, ("rowlike", "chan"),
                                 chisq, ("rowlike", "chan"),
                                 gbl_mad_and_med, ("rowlike",),
                                 bl_mad_and_med, ("rowlike", "ant1", "ant2"),
                                 ant1_col, ("rowlike",),
                                 ant2_col, ("rowlike",),
                                 gbl_thresh, None,
                                 bl_thresh, None,
                                 max_deviation, None,
                                 dtype=np.int8,
                                 align_arrays=False,
                                 concatenate=True,
                                 adjust_chunks={"rowlike": row_chunks},)

        flag_col = da.where(mad_flags, 1, flag_col)

        flagged_data_xds = xds.assign({"FLAG": (("row", "chan"), flag_col)})

        flagged_data_xds_list.append(flagged_data_xds)

    return flagged_data_xds_list
