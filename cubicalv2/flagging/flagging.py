import dask.array as da
import numpy as np
from copy import deepcopy
from uuid import uuid4
from loguru import logger
from cubicalv2.flagging.flagging_kernels import madmax, threshold


ibfdtype = np.uint16  # Data type for internal bitflags.

bitflags = {
    "PRIOR": ibfdtype(1 << 0),      # prior flags (i.e. from MS)
    "MISSING": ibfdtype(1 << 1),    # missing data or solution
    "INVALID": ibfdtype(1 << 2),    # invalid data (zero, inf, nan)
    "ILLCOND": ibfdtype(1 << 3),    # solution ill conditioned - bad inverse
    "NOCONV": ibfdtype(1 << 4),     # no convergence
    "CHISQ": ibfdtype(1 << 5),      # excessive chisq
    "GOOB": ibfdtype(1 << 6),       # gain solution out of bounds
    "BOOM": ibfdtype(1 << 7),       # gain solution exploded (inf/nan)
    "GNULL": ibfdtype(1 << 8),      # gain solution gone to zero.
    "LOWSNR": ibfdtype(1 << 9),     # prior SNR too low for gain solution
    "GVAR": ibfdtype(1 << 10),      # posterior variance too low for solution
    "INVMODEL": ibfdtype(1 << 11),  # invalid model (zero, inf, nan)
    "INVWGHT": ibfdtype(1 << 12),   # invalid weight (inf or nan)
    "NULLWGHT": ibfdtype(1 << 13),  # null weight
    "MAD": ibfdtype(1 << 14),       # residual exceeds MAD-based threshold
    "SKIPSOL": ibfdtype(1 << 15)    # omit this data point from the solver
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

    # NOTE: This is a nasty catch for using the distributed scheduler with 
    # multiple workers. Bitflag arrays may become non-writable. TODO: Consider
    # alternatives.
    if not bitflag_arr.flags["W"]:
        print("WRITABLE: False")
        bitflag_arr = bitflag_arr.copy()

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

    # NOTE: This is a nasty catch for using the distributed scheduler with 
    # multiple workers. Bitflag arrays may become non-writable. TODO: Consider
    # alternatives.
    if not bitflag_arr.flags["W"]:
        print("WRITABLE: False")
        bitflag_arr = bitflag_arr.copy()

    flag_mask = _make_flagmask(bitflag_names)

    if selection is None:
        bitflag_arr &= ~flag_mask
    elif isinstance(selection, np.ndarray):
        bitflag_arr[np.where(selection)] &= ~flag_mask

    return bitflag_arr


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
    elif bitflag_arr.ndim == 5:
        bitflag_axes = ("rowlike", "chan", "ant", "dir", "corr")
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
                        dtype=bitflag_arr.dtype)


def update_kwrds(col_kwrds, opts):
    """Updates the columns keywords to reflect cubical bitflags.

    Given the existing column keywords (from the MS), updates them to include
    CubiCal's bitflags and legacy bitflags if necessary.

    Args:
        col_kwrds: A dictionary of column keywords.
        opts: The options Namespace.

    Returns:
        col_kwrds: An updated (copy) of the column keywords dictionary.
    """

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

    # We assume the 0 bit is always unavailable for new bitflags.
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
    """ Combines internal and input bitflags to produce writable flag data.

    Given a list of xds and appropriately updated keywords, combines CubiCal's
    internal bitflags with the input bitflags and creates a new list of xds
    on which the combined flagging data is assigned. Also handles legacy flags.

    Args:
        xds_list: A list of xarray datasets.
        col_kwrds: A dictionary of (updated) column keywowrds.
        opts: The options Namespace.

    Returns:
        writable_xds: A list of xarray datasets.
    """

    ebfdtype = opts._ebfdtype

    cubical_bit = ebfdtype(col_kwrds["BITFLAG"]["FLAGSET_cubical"])
    legacy_bit = ebfdtype(col_kwrds["BITFLAG"]["FLAGSET_legacy"])

    writable_xds = []

    for xds in xds_list:

        # The following may be slightly incorrect as we assume that no
        # information from the current contents of the cubical bitflag needs
        # to be retained. It might be necessary to have a "cubical_to_legacy"
        # option which merges existing cubical flags into legacy so that we
        # are free to only retain the latest round of CubiCal flags.

        flag_col = xds.FLAG.data
        flag_row_col = xds.FLAG_ROW.data
        bitflag_col = xds.BITFLAG.data & ~ebfdtype(1 << cubical_bit)
        bitflag_row_col = xds.BITFLAG_ROW.data & ~ebfdtype(1 << cubical_bit)
        cubi_bitflags = xds.CUBI_BITFLAG.data

        # If legacy doesn't exist, it will be added.
        if opts._init_legacy or opts.flags_reinit_bitflags:
            legacy_flags = flag_col | flag_row_col[:, None, None]
            legacy_flags = legacy_flags.astype(ebfdtype) << legacy_bit
            bitflag_col |= legacy_flags

        # Set the CubiCal bit in the bitflag column.
        cubi_bitflags = unset_bitflag(cubi_bitflags, "PRIOR")
        cubi_bitflag = (cubi_bitflags > 0).astype(ebfdtype) << cubical_bit

        bitflag_col |= cubi_bitflag
        bitflag_row_col = da.map_blocks(np.bitwise_and.reduce,
                                        bitflag_col,
                                        axis=(1, 2),
                                        drop_axis=(1, 2),
                                        dtype=ebfdtype)

        # TODO: This might be slightly incorrect - I may need to reuse the
        # bitmask here, as the old-school flags will be a subset of the
        # bitflags.
        flag_col = bitflag_col > 0
        flag_row_col = da.map_blocks(np.logical_and.reduce,
                                     flag_col,
                                     axis=(1, 2),
                                     drop_axis=(1, 2),
                                     dtype=bool)

        # BITFLAG and BITFLAG_ROW must be written as int32 as the MS doesn't
        # play nicely with uint values. # TODO: Make this more sophisticated
        # in the event that we can write smaller columns.
        updated_xds = \
            xds.assign({"BITFLAG": (xds.BITFLAG.dims,
                                    bitflag_col.astype(np.int32)),
                        "BITFLAG_ROW": (xds.BITFLAG_ROW.dims,
                                        bitflag_row_col.astype(np.int32)),
                        "FLAG": (xds.FLAG.dims, flag_col),
                        "FLAG_ROW": (xds.FLAG_ROW.dims, flag_row_col)})
        updated_xds.attrs["WRITE_COLS"] += \
            ["BITFLAG", "BITFLAG_ROW", "FLAG", "FLAG_ROW"]

        writable_xds.append(updated_xds)

    return writable_xds


def make_bitmask(col_kwrds, opts):
    """Generate a BITFLAG mask in accordance with opts."""

    ebfdtype = opts._ebfdtype

    bflag_sel = opts.flags_apply_precal
    bflag_kwrds = col_kwrds["BITFLAG"]

    # Strip out bitflags which we don't understand.
    bflag_sel = [bf for bf in bflag_sel
                 if bflag_kwrds.get("FLAGSET_" + bf.lstrip("~"))]

    # Check for exclusion - only one exclusion argument permitted.
    exclusion = next(
        (bf.lstrip("~") for bf in bflag_sel if bf.startswith("~")), False)

    # Check for old-school FLAG/FLAG_ROW.
    flagcols_only = next((True for bf in bflag_sel if bf == "FLAG"), False)

    if exclusion:
        logger.info("--flags-apply-precal contains '~' - all bitflags other "
                    "than {} will be applied.".format(exclusion.upper()))
        bitshift = bflag_kwrds.get("FLAGSET_{}".format(exclusion))
        bitmask = ~(ebfdtype(1) << ebfdtype(bitshift))  # TODO: CHECK!!!
    elif flagcols_only:
        logger.info("--flags-apply-precal contains FLAG - no bitflags will "
                    "be applied.")
        bitmask = ebfdtype(0)
    else:
        logger.info("Generating bitmask for {} bitflags. Missing bitflags "
                    "were ignored.".format(", ".join(bflag_sel).upper()))

        bitmask = ebfdtype(0)
        for bf in bflag_sel:
            bitshift = bflag_kwrds.get("FLAGSET_" + bf)
            bitmask |= ebfdtype(1) << ebfdtype(bitshift)

    logger.info("Generated the following bitmask: 0b{0:016b}.".format(bitmask))

    return bitmask


def _is_set(bitflag_arr, flag_mask):

    return (bitflag_arr & flag_mask) > 0


def is_set(bitflag_arr, bitflag_names):

    flag_mask = _make_flagmask(bitflag_names)

    bitflag_axes = ("rowlike", "chan", "corr")

    return da.blockwise(_is_set, bitflag_axes,
                        bitflag_arr, bitflag_axes,
                        flag_mask, None,
                        dtype=np.bool)


def initialise_fullres_bitflags(data_col, weight_col, flag_col, flag_row_col,
                                bitflag_col, bitflag_row_col, bitmask):
    """Given input data, weights and flags, initialise the internal bitflags.

    Populates the internal bitflag array based on existing flags and data
    points/weights which appear invalid.

    Args:
        data_col: A dask.array containing the data.
        weight_col: A dask.array containing the weights.
        flag_col: A dask.array containing the conventional flags.
        flag_row_col: A dask.array containing the conventional row flags.
        bitflag_col: A dask.array containing the bitflags.
        bitflag_row_col: A dask.array containing the row bitflags.
        bitmask: A binary (integer) mask for selecting relevant bitflags.

    Returns:
        fullres_bitflags: A dask.array containing the initialized internal
            bitflags.
    """

    fullres_bitflags = da.zeros(bitflag_col.shape,
                                dtype=ibfdtype,
                                chunks=bitflag_col.chunks,
                                name="zeros-" + uuid4().hex)

    # If no bitmask is generated, we presume that we are using the
    # conventional FLAG and FLAG_ROW columns. TODO: Consider whether this
    # is safe behaviour.
    if bitmask == 0:
        fullres_bitflags = set_bitflag(fullres_bitflags, "PRIOR",
                                       flag_col)
        fullres_bitflags = set_bitflag(fullres_bitflags, "PRIOR",
                                       flag_row_col)
    else:
        fullres_bitflags = set_bitflag(fullres_bitflags, "PRIOR",
                                       (bitflag_col & bitmask) > 0)
        fullres_bitflags = set_bitflag(fullres_bitflags, "PRIOR",
                                       (bitflag_row_col & bitmask) > 0)

    # The following does some sanity checking on the input data and weights.
    # Specifically, we look for bad data points, points with missing data,
    # and points with null weights. These operations implicitly broadcast the
    # weights to the same frequency dimension as the data. This unavoidable as
    # we want to exploit the weights to effectively flag.

    invalid_points = ~da.isfinite(data_col)

    fullres_bitflags = set_bitflag(fullres_bitflags,
                                   "INVALID",
                                   invalid_points)

    # TODO: This is likely incorrect for different number of correlations. We
    # assume that the first and last entries of the correlation axis are the
    # on-diagonal terms. This should be safe provided we don't have
    # off-diagonal only data, although in that case the flagging logic is
    # probablly equally applicable.
    missing_points = da.logical_or(data_col[..., 0] == 0,
                                   data_col[..., -1] == 0)
    missing_points = da.logical_or(missing_points[..., None], data_col == 0)

    fullres_bitflags = set_bitflag(fullres_bitflags,
                                   "MISSING",
                                   missing_points)

    fullres_bitflags = set_bitflag(fullres_bitflags,
                                   "NULLWGHT",
                                   weight_col == 0)

    return fullres_bitflags


def compute_mad_flags(residuals, bitflag_col, ant1_col, ant2_col, n_ant,
                      n_chunks, opts):

    bl_thresh = opts.flags_bl_mad_threshold
    gbl_thresh = opts.flags_global_mad_threshold

    mad_estimate = da.blockwise(madmax, ("rowlike", "ant1", "ant2"),
                                residuals, ("rowlike", "chan", "corr"),
                                bitflag_col, ("rowlike", "chan", "corr"),
                                ant1_col, ("rowlike",),
                                ant2_col, ("rowlike",),
                                n_ant, None,
                                dtype=residuals.real.dtype,
                                align_arrays=False,
                                concatenate=True,
                                adjust_chunks={"rowlike": (1,)*n_chunks},
                                new_axes={"ant1": n_ant,
                                          "ant2": n_ant})

    med_mad_estimate = \
        da.map_blocks(lambda x: np.median(x[np.isfinite(x) & (x > 0)],
                                          keepdims=True),
                      mad_estimate, drop_axis=(1, 2), dtype=np.float64)

    mad_flags = da.blockwise(threshold, ("rowlike", "chan", "corr"),
                             residuals, ("rowlike", "chan", "corr"),
                             bl_thresh*mad_estimate,
                             ("rowlike", "ant1", "ant2"),
                             gbl_thresh*med_mad_estimate,
                             ("rowlike",),
                             bitflag_col, ("rowlike", "chan", "corr"),
                             ant1_col, ("rowlike",),
                             ant2_col, ("rowlike",),
                             dtype=np.bool_,
                             align_arrays=False,
                             concatenate=True,
                             adjust_chunks={"rowlike": residuals.chunks[0]},)

    return mad_flags
