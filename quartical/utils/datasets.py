from collections.abc import Iterable


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

    for k, v in partition_dict.items():
        partition_dict[k] = group_by_attr(v, attrs[0])

        if len(attrs[1:]):
            _recursive_group_by_attr(partition_dict[k], attrs[1:])


def recursive_group_by_attr(xdsl, keys):

    if not isinstance(keys, Iterable):
        keys = [keys]

    group_dict = group_by_attr(xdsl, keys[0])

    if len(keys[1:]):
        _recursive_group_by_attr(group_dict, keys[1:])

    return group_dict
