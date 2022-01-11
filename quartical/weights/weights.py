import dask.array as da
import numpy as np
import uuid
from quartical.utils.xarray import check_fields, check_dims


def sigma_to_weight(sigma_col):

    weight = np.zeros_like(sigma_col)

    sel = sigma_col != 0

    weight[sel] = 1/(sigma_col[sel])**2

    return weight


def initialize_weights_from_column(input_xds_list,
                                   input_field_name,
                                   output_field_name="WEIGHT",
                                   sigma=False):
    """Initialize weight values on a list of xarray datasets.

    This will assign the weights to the specified field, ensuring that they
    have shape (row, chan, corr). If sigma is True, the weights will be
    recalculated assuming the given column is similar to SIGMA/SIGMA_SPECTRUM.

    Args:
        input_xds_list: A list of xarray.Dataset objects.
        input_field_name: A str corresponding to the field from which the
            weights are to be populated.
        output_field_name: A str corresponding to the field to which the
            weights will be assigned.
        sigma: A boolean value indicating whether the input is a sigma column.
            In this case it will be converted into weight values.

    Returns:
        output_xds_list: A list of xarray.DataSet objects with weights
            assigned to a new/existing field.
    """

    output_dims = ('row', 'chan', 'corr')

    check_fields(input_xds_list, (input_field_name,))
    check_dims(input_xds_list, output_dims)

    output_xds_list = []

    for input_xds in input_xds_list:

        column_data = input_xds[input_field_name].data
        column_dims = input_xds[input_field_name].dims

        if sigma:
            weights = column_data.map_blocks(sigma_to_weight)
        else:
            weights = column_data

        if weights.ndim == 2:
            if 'chan' not in column_dims:

                shape = [input_xds.dims[ax] for ax in output_dims]
                chunks = [input_xds.chunks[ax] for ax in output_dims]

                weights = da.broadcast_to(weights[:, None, :], shape, chunks)
            else:
                raise ValueError(f"Weight column has unexpected shape:"
                                 f"{weights.shape}")

        output_xds = \
            input_xds.assign({output_field_name: (output_dims, weights)})

        output_xds_list.append(output_xds)

    return output_xds_list


def initialize_unity_weights(input_xds_list, output_field_name="WEIGHT"):
    """Initialize weight values on a list of xarray datasets.

    Args:
        input_xds_list: A list of xarray.Dataset objects.
        output_field_name: A str corresponding to the field to which the
            weights will be assigned.

    Returns:
        output_xds_list: A list of xarray.DataSet objects with unity weights
            assigned.
    """

    output_dims = ('row', 'chan', 'corr')

    check_dims(input_xds_list, output_dims)

    output_xds_list = []

    for input_xds in input_xds_list:

        shape = [input_xds.dims[ax] for ax in output_dims]
        chunks = [input_xds.chunks[ax] for ax in output_dims]

        weights = da.ones(shape,
                          chunks=chunks,
                          name="weights-" + uuid.uuid4().hex,
                          dtype=np.float32)

        output_xds = \
            input_xds.assign({output_field_name: (output_dims, weights)})

        output_xds_list.append(output_xds)

    return output_xds_list
