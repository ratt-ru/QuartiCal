def check_fields(xds_list, req_fields):
    """Check that the given fields (data_vars) are present on every xds.

    Args:
        input_xds_list: A list of xarray.Dataset objects.
        req_fields: A list of str corresponding to the fields which must be
            present.
    """

    req_fields = set(req_fields)

    for i, xds in enumerate(xds_list):

        xds_fields = set(xds.data_vars.keys())

        missing_fields = req_fields - xds_fields

        assert missing_fields == set(), \
            f"Dataset {i} is missing the following fields: {missing_fields}."


def check_coords(xds_list, req_coords):
    """Check that the given coordinates are present on every xds.

    Args:
        input_xds_list: A list of xarray.Dataset objects.
        req_coords: A list of str corresponding to the coords which must be
            present.
    """

    req_coords = set(req_coords)

    for i, xds in enumerate(xds_list):

        xds_coords = set(xds.coords.keys())

        missing_coords = req_coords - xds_coords

        assert missing_coords == set(), \
            f"Dataset {i} is missing the following coords: {missing_coords}."


def check_dims(xds_list, req_dims):
    """Check that the given dimensions are present on every xds.

    Args:
        input_xds_list: A list of xarray.Dataset objects.
        req_fields: A list of str corresponding to the dimensions which must be
            present.
    """

    req_dims = set(req_dims)

    for i, xds in enumerate(xds_list):

        xds_dims = set(xds.dims.keys())

        missing_dims = req_dims - xds_dims

        assert missing_dims == set(), \
            f"Dataset {i} is missing the following dimensions: {missing_dims}."
