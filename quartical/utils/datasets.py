

def group_by_attr(xdsl, attr, default="?"):
    """Group list of xarray datasets based on value of attribute."""

    attr_vals = {xds.attrs.get(attr, default) for xds in xdsl}

    return {
        f"{attr}_{attr_val}": [
            xds for xds in xdsl if xds.attrs.get(attr, default) == attr_val
        ]
        for attr_val in attr_vals
    }


def _recursive_group_by_attr(partition_dict, attrs):

    attr = attrs.pop(0)

    for k, v in partition_dict.items():
        partition_dict[k] = group_by_attr(v, attr)

        if attrs:
            _recursive_group_by_attr(partition_dict[k], attrs)


def recursive_group_by_attr(xdsl, keys):

    keys = keys.copy()  # Don't destroy input keys.

    group_dict = group_by_attr(xdsl, keys.pop(0))

    if keys:
        _recursive_group_by_attr(group_dict, keys)

    return group_dict
