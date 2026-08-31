import numpy as np


# FIELD subtable columns which may hold the direction in which the antennas
# are pointing, in order of preference. Rephasing tools (chgcentre, phaseshift)
# move PHASE_DIR and rotate the uvw coordinates to match, but leave
# REFERENCE_DIR and DELAY_DIR at the pointing. See ratt-ru/QuartiCal#439.
POINTING_DIR_COLUMNS = ("REFERENCE_DIR", "DELAY_DIR", "PHASE_DIR")


def get_pointing_dir(field_xds, field_id=0):
    """Selects the direction in which the antennas are pointing.

    This is the direction on which the primary beam is centred and about which
    the parallactic angles are computed. It is distinct from the phase centre,
    which is where the visibilities are referenced and which a rephase moves.

    Candidate columns are tried in the order given by POINTING_DIR_COLUMNS. A
    candidate is skipped if it is absent or unpopulated, as not every writer
    fills in every direction. Note that (0, 0) is a valid sky position and
    consequently cannot be used to detect an unpopulated column.

    Args:
        field_xds: An xarray.Dataset corresponding to the FIELD subtable.
        field_id: Integer row of the FIELD subtable to select.

    Returns:
        A tuple containing the name of the selected column and its order-zero
        polynomial term, a (2,) numpy.ndarray of radians.

    Raises:
        ValueError: If none of the candidate columns contain a usable
            direction.
    """

    for column in POINTING_DIR_COLUMNS:

        if column not in field_xds:
            continue

        direction = np.asarray(field_xds[column].values[field_id][0])  # poly

        if not np.isfinite(direction).all():
            continue

        return column, direction

    raise ValueError(
        "No usable direction found in the FIELD subtable. Tried "
        f"{POINTING_DIR_COLUMNS}."
    )
