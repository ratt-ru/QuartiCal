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


@injit
def get_row(row_ind, row_map):
    """Gets the current row index. Row map is needed for the BDA case."""

    if row_map is None:
        return row_ind
    else:
        return row_map[row_ind]


@injit
def mul_rweight(vis, weight, ind):
    """Multiplies the row weight into a visiblity if weight is not None."""

    if weight is None:
        return vis
    else:
        return vis*weight[ind]
