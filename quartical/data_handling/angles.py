import numpy as np
import casacore.measures
import casacore.quanta as pq

from daskms import xds_from_storage_table
import dask.array as da
import threading
from dask.graph_manipulation import clone
from loguru import logger
import xarray
from numba import njit
from numba.extending import overload
from quartical.data_handling.pointing import get_pointing_dir
from quartical.utils.numba import coerce_literal, JIT_OPTIONS
from quartical.utils.dask import blockwise_unique
import quartical.gains.general.factories as factories


# Create thread local storage for the measures server. TODO: This works for
# Simon but I think it may break things for me. Investigate.
_thread_local = threading.local()


def _select_receptor_angles(feedtab, spw_id, n_ant):
    """Select the receptor angles which apply to a spectral window.

    Args:
        feedtab: The FEED subtable as a single, computed xarray.Dataset.
        spw_id: The SPECTRAL_WINDOW_ID of the data.
        n_ant: The number of antennas in the ANTENNA subtable.

    Returns:
        An (n_ant, n_receptor) array of receptor angles in radians, ordered
        by ANTENNA_ID.

    Raises:
        ValueError: If no FEED rows apply to the spectral window, or if they
            do not hold exactly one row per antenna.
    """
    feed_spw_ids = feedtab.SPECTRAL_WINDOW_ID.values

    # Rows with a SPECTRAL_WINDOW_ID of -1 apply to every spectral window
    # which has no rows of its own.
    applicable_rows = feed_spw_ids == spw_id
    if not applicable_rows.any():
        applicable_rows = feed_spw_ids == -1
    if not applicable_rows.any():
        raise ValueError(
            f"No FEED table entry found for SPECTRAL_WINDOW_ID {spw_id} "
            "and no global (-1) entry is present."
        )

    ant_ids = feedtab.ANTENNA_ID.values[applicable_rows]

    if not np.array_equal(np.sort(ant_ids), np.arange(n_ant)):
        raise ValueError(
            "The FEED table entries which apply to SPECTRAL_WINDOW_ID "
            f"{spw_id} do not hold exactly one row per antenna. "
            "Time-dependent and multi-feed FEED tables are not supported."
        )

    receptor_angles = np.empty((n_ant, feedtab.sizes["receptors"]))
    receptor_angles[ant_ids] = feedtab.RECEPTOR_ANGLE.values[applicable_rows]

    return receptor_angles


def assign_parangle_data(ms_path, data_xds_list):
    """Assign the inputs of the parallactic angle calculation to each dataset.

    Each dataset gains its antenna POSITION and its RECEPTOR_ANGLE, selected
    from the FEED rows which apply to its spectral window, along with the
    FEED_TYPE and FIELD_CENTRE attributes. A dataset which is not partitioned
    by FIELD_ID is flagged as MULTI_FIELD (see assert_parangle_supported).

    Args:
        ms_path: Path to the input measurement set.
        data_xds_list: A list of xarray.Dataset objects.

    Returns:
        updated_data_xds_list: A list of xarray.Dataset objects carrying the
            parallactic angle inputs.

    Raises:
        ValueError: If the FEED table is empty, mixes receptor counts, holds
            unsupported or mixed feed types, or does not hold exactly one row
            per antenna which applies to a dataset's spectral window.
    """

    anttab = xds_from_storage_table(ms_path + "::ANTENNA")[0]
    fieldtab = xds_from_storage_table(ms_path + "::FIELD")[0]

    # dask-ms ignores group_cols when reading zarr, and dask-ms convert writes
    # the FEED subtable to zarr with one row per partition. Join whatever
    # partitions are present and select rows per spectral window ourselves.
    feed_partitions = xds_from_storage_table(
        ms_path + "::FEED",
        columns=[
            "ANTENNA_ID",
            "SPECTRAL_WINDOW_ID",
            "NUM_RECEPTORS",
            "POLARIZATION_TYPE",
            "RECEPTOR_ANGLE",
        ],
    )

    if sum(xds.sizes["row"] for xds in feed_partitions) == 0:
        raise ValueError(f"The FEED table in {ms_path} is empty.")

    # The receptor columns are shaped by NUM_RECEPTORS, so a table which mixes
    # receptor counts fails to read or join with an obscure error. Check the
    # counts first, as NUM_RECEPTORS is a scalar column and always reads.
    receptor_counts = np.unique(
        xarray.concat(
            [xds.NUM_RECEPTORS for xds in feed_partitions], dim="row"
        ).values
    )
    if receptor_counts.size > 1:
        raise ValueError(
            "QuartiCal requires every feed to have the same number of "
            "receptors, but FEED.NUM_RECEPTORS holds "
            f"{receptor_counts.tolist()}."
        )

    # The table is small, so materialise it.
    feedtab = xarray.concat(feed_partitions, dim="row").compute()

    unique_feeds = set(feedtab.POLARIZATION_TYPE.values.ravel())

    if np.all([feed in "XxYy" for feed in unique_feeds]):
        feed_type = "linear"
    elif np.all([feed in "LlRr" for feed in unique_feeds]):
        feed_type = "circular"
    else:
        raise ValueError("Unsupported feed type/configuration.")

    # The parallactic angles describe how each antenna is oriented while
    # tracking its target, so they belong at the pointing rather than at the
    # phase centre - the two differ on a rephased measurement set. The FIELD
    # subtable is small, so materialise it and look the pointing up per field.
    fieldtab = fieldtab.compute()
    pointing_columns = set()

    n_ant = anttab.sizes["row"]

    updated_data_xds_list = []
    for xds in data_xds_list:
        receptor_angles = _select_receptor_angles(
            feedtab, int(xds.SPECTRAL_WINDOW_ID), n_ant
        )
        xds = xds.assign(
            {
                "RECEPTOR_ANGLE": (
                    ("ant", "feed"),
                    clone(da.from_array(receptor_angles, chunks=-1))
                ),
                "POSITION": (
                    ("ant", "xyz"),
                    clone(anttab.POSITION.data)
                )
            }
        )
        xds.attrs["FEED_TYPE"] = feed_type
        if "FIELD_ID" in xds.attrs:
            field_id = int(xds.FIELD_ID)
        else:
            # The data was not partitioned by FIELD_ID (see
            # input_ms.group_by), so this dataset may span multiple fields
            # with differing pointings. The parallactic angle machinery
            # assumes a single field centre per dataset (see _make_parangles),
            # so it cannot be supported here. We assign placeholder values so
            # that the unconditional - but lazy - parallactic angle graph
            # construction does not fail, and flag the dataset so that any
            # attempt to actually use the parallactic angles raises a clear
            # error. See assert_parangle_supported.
            field_id = 0
            xds.attrs["FIELD_ID"] = field_id
            xds.attrs["MULTI_FIELD"] = True

        pointing_column, pointing_dir = get_pointing_dir(fieldtab, field_id)
        pointing_columns.add(pointing_column)
        xds.attrs["FIELD_CENTRE"] = tuple(pointing_dir)

        updated_data_xds_list.append(xds)

    for column in sorted(pointing_columns):
        logger.info(f"Computing parallactic angles about FIELD.{column}.")

    return updated_data_xds_list


def assert_parangle_supported(data_xds_list):
    """Ensure parallactic angle correction is valid for the given data.

    Parallactic angles are computed relative to a single field centre per
    dataset (see _make_parangles). When the input data is not partitioned by
    FIELD_ID (see input_ms.group_by), a dataset may contain rows from multiple
    fields with different phase centres, violating this assumption. Rather than
    silently producing incorrect results, we raise.

    Args:
        data_xds_list: A list of xarray.Dataset objects.

    Raises:
        ValueError: If any dataset spans multiple fields.
    """
    if any(xds.attrs.get("MULTI_FIELD", False) for xds in data_xds_list):
        raise ValueError(
            "Parallactic angle correction is not supported when the input "
            "data is not partitioned by FIELD_ID. This affects "
            "input_model.apply_p_jones, output.apply_p_jones_inv and the "
            "parallactic_angle gain term. Add FIELD_ID to input_ms.group_by "
            "to enable these features."
        )


def make_parangle_xds_list(ms_path, data_xds_list):
    """Create a list of xarray.Datasets containing the parallactic angles.

    The angles are computed about each dataset's FIELD_CENTRE, the pointing
    selected by assign_parangle_data, which the parallactic_angle gain term
    also uses.

    Args:
        ms_path: Path to the input measurement set.
        data_xds_list: A list of xarray.Dataset objects which have been
            through assign_parangle_data.

    Returns:
        parangle_xds_list: A list of xarray.Dataset objects containing the
            parallactic angles per unique time, antenna and receptor.
    """

    # This may need to be more sophisticated. TODO: Can we guarantee that
    # this only ever has one element?
    anttab = xds_from_storage_table(ms_path + "::ANTENNA")[0]

    ant_names = anttab.NAME.data

    ant_positions_ecef = anttab.POSITION.data  # ECEF coordinates.

    epoch = "J2000"  # TODO: Should this be configurable?

    parangle_xds_list = []

    for xds in data_xds_list:

        receptor_angles = xds.RECEPTOR_ANGLE.data
        field_centre = np.array(xds.FIELD_CENTRE)

        parangles = da.blockwise(_make_parangles, "tar",
                                 xds.TIME.data, "t",
                                 clone(ant_names), "a",
                                 clone(ant_positions_ecef), "a3",
                                 clone(receptor_angles), "ar",
                                 field_centre, None,
                                 epoch, None,
                                 align_arrays=False,
                                 concatenate=True,
                                 dtype=np.float64,
                                 adjust_chunks={"t": xds.UTIME_CHUNKS})

        parangle_xds = xarray.Dataset(
            {
                "PARALLACTIC_ANGLES": (("utime", "ant", "receptor"), parangles)
            },
            coords={
                "utime": np.arange(sum(xds.UTIME_CHUNKS)),
                "ant": xds.ant,
                "receptor": np.arange(receptor_angles.shape[1])
            },
            attrs={
                "FEED_TYPE": xds.FEED_TYPE,
                "UTIME_CHUNKS": xds.UTIME_CHUNKS
            }
        )

        parangle_xds_list.append(parangle_xds)

    return parangle_xds_list


def _make_parangles(time_col, ant_names, ant_positions_ecef, receptor_angles,
                    field_centre, epoch):
    """Handles the construction of the parallactic angles using measures.

    Args:
        time_col: Array containing time values for each row.
        ant_names: Array of antenna names.
        ant_positions_ecef: Array of antenna positions in ECEF frame.
        receptor_angles: Array of receptor angles (two per ant).
        field_centre: Array containing field centre coordinates.
        epoch: Reference epoch for measures calculations.

    Returns:
        angles: Array of parallactic angles per antenna per unique time.
    """

    try:
        cms = _thread_local.cms
    except AttributeError:
        # Create a measures server.
        _thread_local.cms = cms = casacore.measures.measures()

    if not np.all(np.equal(receptor_angles, receptor_angles[:, :1])):
        raise ValueError("FEED table indicates that some receptors "
                         "are non-orthogonal. This is not yet supported. "
                         "Please raise an issue if you require this "
                         "functionality.")

    n_time = time_col.size
    n_ant = ant_names.size

    # Init angles from receptor angles. TODO: This only works for orthogonal
    # receptors. The more general case needs them to be kept separate.
    angles = np.zeros((n_time, n_ant, 2), dtype=np.float64)
    angles[:] = receptor_angles[np.newaxis, :, :]

    # Assume all antenna are pointed in the same direction.
    field_centre = \
        cms.direction(epoch, *(pq.quantity(fi, 'rad') for fi in field_centre))

    unique_times = np.unique(time_col)
    n_utime = unique_times.size
    angles = np.empty((n_utime, n_ant, 2), dtype=np.float64)
    angles[:] = receptor_angles[None, :, :]

    zenith_azel = cms.direction(
        "AZEL", *(pq.quantity(fi, 'deg') for fi in (0, 90))
    )

    ant_positions_itrf = [
        cms.position(
            'itrf', *(pq.quantity(p, 'm') for p in pos)
        ) for pos in ant_positions_ecef
    ]

    for ti, t in enumerate(unique_times):
        cms.do_frame(cms.epoch("UTC", pq.quantity(t, 's')))
        for rpi, rp in enumerate(ant_positions_itrf):
            cms.do_frame(rp)
            angles[ti, rpi, :] += \
                cms.posangle(field_centre, zenith_azel).get_value("rad")

    return angles


def apply_parangles(data_xds_list, parangle_xds_list, data_var_names,
                    derotate=False):
    """Apply a parallactic angle correction to specific data vars.

    NOTE: RECEPTOR_ANGLE is currently included in the parallactic angle. This
    will work for most common use cases but isn't correct for non-orthogonal
    receptors.

    Args:
        data_xds_list: A list of xarray.Dataset objects contatining
            measurement set data.
        parangle_xds_list: A list of xarray.Dataset objects containing
            parallactic angles.
        data_var_names: The names of the data_vars to which the parallactic
            angle rotation must be applied.
        derotate: If True, apply a derotation rather than a rotation.

    Returns:
        output_data_xds_list: A list of xarray.Dataset objects. Each dataset
            will have parallactic angles applied to the data_vars in in
            data_var_names.
    """

    output_data_xds_list = []

    for xds, pxds in zip(data_xds_list, parangle_xds_list):

        rot_vars = {}

        for data_var_name in data_var_names:

            data_var = xds[data_var_name].data
            time_col = xds.TIME.data
            ant1_col = xds.ANTENNA1.data
            ant2_col = xds.ANTENNA2.data
            corr_mode = data_var.shape[-1]

            feed_type = pxds.FEED_TYPE
            parangles = pxds.PARALLACTIC_ANGLES.data

            # Convert the time column data into indices. Chunks is expected to
            # be a tuple of tuples. utime_ind associates each row with a
            # unique time.
            utime_chunks = xds.UTIME_CHUNKS
            _, utime_ind = blockwise_unique(time_col,
                                            chunks=(utime_chunks,),
                                            return_inverse=True)

            # Negate the angles if the desired output is a derotation.
            parangles = -parangles if derotate else parangles

            rot_vars[data_var_name] = da.blockwise(
                apply_parangle, "rfc",
                data_var, "rfc",
                parangles, "ra2",
                utime_ind, "r",
                ant1_col, "r",
                ant2_col, "r",
                corr_mode, None,
                feed_type, None,
                align_arrays=False,
                concatenate=True,
                dtype=data_var.dtype
            )

        output_data_xds_list.append(
            xds.assign({n: ((xds[n].dims), v) for n, v in rot_vars.items()}))

    return output_data_xds_list


@njit(**JIT_OPTIONS)
def apply_parangle(
    data_col,
    parangles,
    utime_ind,
    ant1_col,
    ant2_col,
    corr_mode,
    feed_type
):
    return apply_parangle_impl(
        data_col,
        parangles,
        utime_ind,
        ant1_col,
        ant2_col,
        corr_mode,
        feed_type
    )


def apply_parangle_impl(
    data_col,
    parangles,
    utime_ind,
    ant1_col,
    ant2_col,
    corr_mode,
    feed_type
):
    return NotImplementedError


@overload(apply_parangle_impl, jit_options=JIT_OPTIONS)
def nb_apply_parangle_rot(data_col, parangles, utime_ind, ant1_col, ant2_col,
                          corr_mode, feed_type):

    coerce_literal(nb_apply_parangle_rot, ["corr_mode", "feed_type"])

    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    v1_imul_v2ct = factories.v1_imul_v2ct_factory(corr_mode)
    valloc = factories.valloc_factory(corr_mode)
    rotmat = factories.rotation_factory(corr_mode, feed_type)

    def impl(data_col, parangles, utime_ind, ant1_col, ant2_col,
             corr_mode, feed_type):

        n_row, n_chan, _ = data_col.shape

        data_col = data_col.copy()

        rot_p = valloc(np.complex128)
        rot_q = valloc(np.complex128)

        for r in range(n_row):
            ut = utime_ind[r]
            a1 = ant1_col[r]
            a2 = ant2_col[r]

            p0, p1 = parangles[ut, a1]
            q0, q1 = parangles[ut, a2]

            rotmat(p0, p1, rot_p)
            rotmat(q0, q1, rot_q)

            for f in range(n_chan):

                data_elem = data_col[r, f]

                v1_imul_v2(rot_p, data_elem, data_elem)
                v1_imul_v2ct(data_elem, rot_q, data_elem)

        return data_col

    return impl
