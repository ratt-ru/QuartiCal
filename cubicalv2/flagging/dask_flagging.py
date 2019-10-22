import dask.array as da
import numpy as np
from cubicalv2.flagging.flagging import _set_bitflag, _unset_bitflag
from copy import deepcopy
from loguru import logger


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
        setter: Function to use in blockwise call. Either set or unset.

    Returns:
        Dask array for blockwise flagging.
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
                        dtype=np.uint32)


def update_kwrds(col_kwrds, opts):
    """Updates the columns keywords to reflect cubical bitflags."""

    # Create a deep copy of the column keywords to avoid mutating the input.
    col_kwrds = deepcopy(col_kwrds)

    # If the bitflag column already exists, we assume it is correct. Otherwise
    # we initialise some keywords.
    if opts._bitflag_exists:
        bitflag_kwrds = col_kwrds["BITFLAG"]
        flagsets = set(bitflag_kwrds["FLAGSETS"].split(","))
    else:
        col_kwrds["BITFLAG"] = dict()
        bitflag_kwrds = col_kwrds["BITFLAG"]
        bitflag_kwrds["FLAGSETS"] = str()
        flagsets = set()

    reserved_bits = [0]

    for flagset in flagsets:
        reserved_bit = bitflag_kwrds.get("FLAGSET_{}".format(flagset))
        if reserved_bit is None:
            raise ValueError("Cannot determine reserved bit for flagset"
                             " {}.".format(flagset))
        else:
            reserved_bits.append(reserved_bit)

    available_bits = [bit for bit in range(32) if bit not in reserved_bits]

    opts._init_legacy = False

    try:
        if "legacy" not in flagsets:
            flagsets |= set(("legacy",))
            bitflag_kwrds.update(FLAGSET_legacy=available_bits.pop(0))
            opts._init_legacy = True
            logger.info("LEGACY bitflag will be populated from FLAG/FLAG_ROW.")

        if "cubical" not in flagsets:
            flagsets |= set(("cubical",))
            bitflag_kwrds.update(FLAGSET_cubical=available_bits.pop(0))
    except IndexError:
        raise ValueError("BITFLAG is full - aborting.")

    bitflag_kwrds["FLAGSETS"] = ",".join(flagsets)

    return col_kwrds


def finalise_flags(xds_list, col_kwrds, opts):

    cubical_bit = col_kwrds["BITFLAG"]["FLAGSET_cubical"]
    legacy_bit = col_kwrds["BITFLAG"]["FLAGSET_legacy"]

    writable_xds = []

    for xds in xds_list:

        flag_col = xds.FLAG.data
        flag_row_col = xds.FLAG_ROW.data
        bitflag_col = xds.BITFLAG.data
        cubi_bitflags = xds.CUBI_BITFLAG.data

        # If legacy doesn't exist, it will be added.
        if opts._init_legacy:
            legacy_flags = flag_col | flag_row_col[:, None, None]
            legacy_flags = legacy_flags.astype(np.uint32) << legacy_bit
            bitflag_col |= legacy_flags

        # Set the CubiCal bit in the bitflag column.
        cubi_bitflag = (cubi_bitflags > 0).astype(np.uint32) << cubical_bit
        cubi_bitflag = cubi_bitflag.astype(np.uint32)

        bitflag_col |= cubi_bitflag

        flag_col = bitflag_col > 0

        writable_xds.append(
            xds.assign({"BITFLAG": (xds.BITFLAG.dims,
                                    bitflag_col.astype(np.int32)),
                        "FLAG": (xds.FLAG.dims, flag_col)}))

    return writable_xds
