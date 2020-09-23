# -*- coding: utf-8 -*-
from numba.extending import overload
from numba import jit, types


# Handy alias for functions that need to be jitted in this way.
injit = jit(nogil=True,
            nopython=True,
            fastmath=True,
            cache=True,
            inline="always")

# TODO: Consider whether these should have true optionals. This will likely
# require them all to be implemented as overloads.


def get_dims(col, row_map):
    """Returns effective column dimensions. This may be larger than col.

    Args:
        col: A numpy.ndrray containing column-like data.
        row_map: A nominal to effective row mapping. If not None, the returned
            shape will have the effective number of rows.

    Returns:
        A shape tuple.
    """
    return


@overload(get_dims, inline="always")
def _get_dims(col, row_map):

    if isinstance(row_map, types.NoneType):
        def impl(col, row_map):
            return col.shape
        return impl
    else:
        if col.ndim == 4:
            def impl(col, row_map):
                _, n_chan, n_dir, n_corr = col.shape
                return (row_map.size, n_chan, n_dir, n_corr)
            return impl
        else:
            def impl(col, row_map):
                _, n_chan, n_corr = col.shape
                return (row_map.size, n_chan, n_corr)
            return impl


def get_row(row_ind, row_map):
    """Gets the current row index. Row map is needed for the BDA case.

    Args:
        row_ind: Integer index of current row.
        row_map: A nominal to effective row mapping.

    Returns:
        Integer index of effective row.
    """
    return


@overload(get_row, inline="always")
def _get_row(row_ind, row_map):

    if isinstance(row_map, types.NoneType):
        def impl(row_ind, row_map):
            return row_ind
        return impl
    else:
        def impl(row_ind, row_map):
            return row_map[row_ind]
        return impl


def mul_rweight(vis, weight, ind):
    """Multiplies the row weight into a visiblity if weight is not None.

    Args:
        vis: An complex valued visibility.
        weight: A float row weight.
        ind: Integer row index for selecting weight.

    Returns:
        Product of visilbity and weight, if weight is not None.
    """
    return


@overload(mul_rweight, inline="always")
def _mul_rweight(vis, weight, ind):

    if isinstance(weight, types.NoneType):
        def impl(vis, weight, ind):
            return vis
        return impl
    else:
        def impl(vis, weight, ind):
            return vis*weight[ind]
        return impl
