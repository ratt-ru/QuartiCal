import dask.array as da
import numpy as np
from cubicalv2.flagging.flagging import _set_bitflag, _unset_bitflag


def set_bitflag(bitflag_arr, bitflag_names, selection=None):
    """Convenience function for setting bitflags."""

    return _bitflagger(bitflag_arr, bitflag_names, selection, _set_bitflag)


def unset_bitflag(bitflag_arr, bitflag_names, selection=None):
    """Convenience function for unsetting bitflags."""

    return _bitflagger(bitflag_arr, bitflag_names, selection, _unset_bitflag)


def _bitflagger(bitflag_arr, bitflag_names, selection, setter):
    """Given a dask array, sets or unsets bitflags based on selection.

    Given a dask array of bitflags, sets up the necessary blockwise operation
    to set or unset (using setter) the specified bitflag based on the selection
    argument.

    Args:
        bitflag_arr: Dask array containing btiflags.
        bitflag_names: Name or list of names of bitflag/s to set/unset.
        selection: Dask array or None which determines where set/unset happens.
        setter: Function to use in blockwise call. Either

    """

    if bitflag_arr.ndim == 3:
        bitflag_axes = ("rowlike", "chan", "corr")
    else:
        raise ValueError("BITFLAG is missing one or more dimensions.")

    if selection is None:
        selection_args = []
    elif isinstance(selection, da.Array):
        selection_args = [selection, bitflag_axes[:selection.ndim]]
    else:
        raise ValueError("Invalid selection when attempting to set bitflags.")

    return da.blockwise(setter, bitflag_axes,
                        bitflag_arr, bitflag_axes,
                        bitflag_names, None,
                        *selection_args,
                        dtype=np.object)


def update_bitflag_col_kwrds(col_kwrds):
    """Updates the columns keywords to reflect cubical bitflags."""

    return col_kwrds
