import numpy as np
import casacore.measures
import casacore.quanta as pq

from daskms import xds_from_storage_table
import dask.array as da
import threading
from dask.graph_manipulation import clone
import xarray
from numba import generated_jit
from quartical.utils.dask import blockwise_unique
import quartical.gains.general.factories as factories
from quartical.utils.numba import coerce_literal


# Create thread local storage for the measures server. TODO: This works for
# Simon but I think it may break things for me. Investigate.
_thread_local = threading.local()


def make_parangle_xds_list(ms_path, data_xds_list):
    """Create a list of xarray.Datasets containing the parallactic angles."""

    # This may need to be more sophisticated. TODO: Can we guarantee that
    # these only ever have one element?
    anttab = xds_from_storage_table(ms_path + "::ANTENNA")[0]
    feedtab = xds_from_storage_table(ms_path + "::FEED")[0]
    fieldtab = xds_from_storage_table(ms_path + "::FIELD")[0]

    # We do this eagerly to make life easier.
    feeds = feedtab.POLARIZATION_TYPE.values
    unique_feeds = np.unique(feeds)

    if np.all([feed in "XxYy" for feed in unique_feeds]):
        feed_type = "linear"
    elif np.all([feed in "LlRr" for feed in unique_feeds]):
        feed_type = "circular"
    else:
        raise ValueError("Unsupported feed type/configuration.")

    phase_dirs = fieldtab.PHASE_DIR.data

    field_centres = [phase_dirs[xds.FIELD_ID, 0] for xds in data_xds_list]

    # TODO: This could be more complicated for arrays with multiple feeds.
    receptor_angles = feedtab.RECEPTOR_ANGLE.data

    ant_names = anttab.NAME.data

    ant_positions_ecef = anttab.POSITION.data  # ECEF coordinates.

    epoch = "J2000"  # TODO: Should this be configurable?

    parangle_xds_list = []

    for xds, field_centre in zip(data_xds_list, field_centres):

        parangles = da.blockwise(_make_parangles, "tar",
                                 xds.TIME.data, "t",
                                 clone(ant_names), "a",
                                 clone(ant_positions_ecef), "a3",
                                 clone(receptor_angles), "ar",
                                 clone(field_centre), "t",
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
                "receptor": np.arange(2)
            },
            attrs={
                "FEED_TYPE": feed_type,
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

            rot_vars[data_var_name] = da.blockwise(py_apply_parangle_rot,
                                                   "rfc",
                                                   data_var, "rfc",
                                                   parangles, "ra2",
                                                   utime_ind, "r",
                                                   ant1_col, "r",
                                                   ant2_col, "r",
                                                   corr_mode, None,
                                                   feed_type, None,
                                                   align_arrays=False,
                                                   concatenate=True,
                                                   dtype=data_var.dtype)

        output_data_xds_list.append(
            xds.assign({n: ((xds[n].dims), v) for n, v in rot_vars.items()}))

    return output_data_xds_list


def py_apply_parangle_rot(data_col, parangles, utime_ind, ant1_col, ant2_col,
                          corr_mode, feed_type):
    """Wrapper for numba function to ensure pickling works."""
    return nb_apply_parangle_rot(data_col, parangles, utime_ind, ant1_col,
                                 ant2_col, corr_mode, feed_type)


@generated_jit(nopython=True, nogil=True, fastmath=True, cache=True)
def nb_apply_parangle_rot(data_col, parangles, utime_ind, ant1_col, ant2_col,
                          corr_mode, feed_type):

    coerce_literal(nb_apply_parangle_rot, ["corr_mode", "feed_type"])

    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    v1_imul_v2ct = factories.v1_imul_v2ct_factory(corr_mode)
    valloc = factories.valloc_factory(corr_mode)
    rotmat = rotation_factory(corr_mode, feed_type)

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


def rotation_factory(corr_mode, feed_type):

    if feed_type.literal_value == "circular":
        if corr_mode.literal_value == 4:
            def impl(rot0, rot1, out):
                out[0] = np.exp(-1j*rot0)
                out[1] = 0
                out[2] = 0
                out[3] = np.exp(1j*rot1)
        elif corr_mode.literal_value == 2:  # TODO: Is this sensible?
            def impl(rot0, rot1, out):
                out[0] = np.exp(-1j*rot0)
                out[1] = np.exp(1j*rot1)
        elif corr_mode.literal_value == 1:  # TODO: Is this sensible?
            def impl(rot0, rot1, out):
                out[0] = np.exp(-1j*rot0)
        else:
            raise ValueError("Unsupported number of correlations.")
    elif feed_type.literal_value == "linear":
        if corr_mode.literal_value == 4:
            def impl(rot0, rot1, out):
                out[0] = np.cos(rot0)
                out[1] = np.sin(rot0)
                out[2] = -np.sin(rot1)
                out[3] = np.cos(rot1)
        elif corr_mode.literal_value == 2:  # TODO: Is this sensible?
            def impl(rot0, rot1, out):
                out[0] = np.cos(rot0)
                out[1] = np.cos(rot1)
        elif corr_mode.literal_value == 1:  # TODO: Is this sensible?
            def impl(rot0, rot1, out):
                out[0] = np.cos(rot0)
        else:
            raise ValueError("Unsupported number of correlations.")
    else:
        raise ValueError("Unsupported feed type.")

    return factories.qcjit(impl)
