import dask.array as da
from quartical.utils.xarray import check_fields


def filter_xds_list(xds_list, fields, ddids):

    filter_fields = {"FIELD_ID": fields,
                     "DATA_DESC_ID": ddids}

    for k, v in filter_fields.items():
        fil = filter(lambda xds: getattr(xds, k) in v, xds_list)
        xds_list = list(fil) if v else xds_list

    if len(xds_list) == 0:
        raise ValueError("Selection of field/ddid has deselected all data.")

    return xds_list


def zero_nonfinite(input_xds_list,
                   input_field="DATA",
                   output_field="DATA"):
    """Zeros nonfinite values from the target field on a list of datasets.

    Args:
        input_xds_list: A list of xarray.Dataset objects.
        input_field: A string corresponding to the input field from which
            to remove non-finite values.
        output_field: A str corresponding to desired output field.

    Returns:
        output_xds_list: A list of xarray.DataSet objects with zeroing applied
            and assigned to a new/existing field.
    """

    check_fields(input_xds_list, (input_field,))

    output_xds_list = []

    for input_xds in input_xds_list:

        input_data = input_xds[input_field].data
        input_dims = input_xds[input_field].dims
        output_data = da.where(da.isfinite(input_data), input_data, 0)

        output_xds = \
            input_xds.assign({output_field: (input_dims, output_data)})

        output_xds_list.append(output_xds)

    return output_xds_list
