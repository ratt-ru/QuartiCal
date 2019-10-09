import numpy as np

dtype = np.uint16

bitflags = {
    "PRIOR": dtype(1 << 0),      # prior flags (i.e. from MS)
    "MISSING": dtype(1 << 1),    # missing data or solution
    "INVALID": dtype(1 << 2),    # invalid data (zero, inf, nan)
    "ILLCOND": dtype(1 << 3),    # solution ill conditioned - bad inverse
    "NOCONV": dtype(1 << 4),     # no convergence
    "CHISQ": dtype(1 << 5),      # excessive chisq
    "GOOB": dtype(1 << 6),       # gain solution out of bounds
    "BOOM": dtype(1 << 7),       # gain solution exploded (inf/nan)
    "GNULL": dtype(1 << 8),      # gain solution gone to zero.
    "LOWSNR": dtype(1 << 9),     # prior SNR too low for gain solution
    "GVAR": dtype(1 << 10),      # posterior variance too low for solution
    "INVMODEL": dtype(1 << 11),  # invalid model (zero, inf, nan)
    "INVWGHT": dtype(1 << 12),   # invalid weight (inf or nan)
    "NULLWGHT": dtype(1 << 13),  # null weight
    "MAD": dtype(1 << 14),       # residual exceeds MAD-based threshold
    "SKIPSOL": dtype(1 << 15)    # omit this data point from the solver
}


def _make_flagmask(bitflag_names):
    """Given a bitflag name/names, returns the appropriate mask."""

    if isinstance(bitflag_names, list):
        flag_mask = \
            np.bitwise_or.reduce([bitflags[name] for name in bitflag_names])
    else:
        flag_mask = bitflags[bitflag_names]

    return flag_mask


def _set_bitflag(bitflag_arr, bitflag_names, selection=None):
    """Given bitflag array, sets bitflag_name where selection is True.

    Args:
        bitflag_arr: Array containing bitflags.
        bitflag_names: Name/s of relevant bitflag/s.
        selection: If specificed, sets bitflag_names where selection is True.

    Returns:
        bitflag_arr: Modified version of input bitflag_arr.
    """

    flag_mask = _make_flagmask(bitflag_names)

    if selection is None:
        bitflag_arr |= flag_mask
    elif isinstance(selection, np.ndarray):
        bitflag_arr[np.where(selection)] |= flag_mask

    return bitflag_arr


def _unset_bitflag(bitflag_arr, bitflag_names, selection=None):
    """Given bitflag array, unsets bitflag_names where selection is True.

    Args:
        bitflag_arr: Array containing bitflags.
        bitflag_names: Name/s of relevant bitflag/s.
        selection: If specificed, unsets bitflag_names where selection is True.

    Returns:
        bitflag_arr: Modified version of input bitflag_arr.
    """

    flag_mask = _make_flagmask(bitflag_names)

    if selection is None:
        bitflag_arr &= ~flag_mask
    elif isinstance(selection, np.ndarray):
        bitflag_arr[np.where(selection)] &= ~flag_mask

    return bitflag_arr
