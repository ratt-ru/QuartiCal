import dask.array as da
import numpy as np
from uuid import uuid4
from loguru import logger  # noqa
from quartical.flagging.flagging_kernels import (compute_bl_mad_and_med,
                                                 compute_gbl_mad_and_med,
                                                 compute_whitened_residual,
                                                 compute_mad_flags)


def finalise_flags(xds_list):
    """Finishes processing flags to produce writable flag data.

    Given a list of xarray.Dataset objects, uses the updated flag column to
    create appropriate flags for writing to disk. Removes all temporary flags.

    Args:
        xds_list: A list of xarray datasets.

    Returns:
        writable_xds: A list of xarray datasets.
    """

    writable_xds = []

    for xds in xds_list:

        data_col = xds.DATA.data
        flag_col = xds.FLAG.data

        # Remove QuartiCal's temporary flagging.
        flag_col = da.where(flag_col == -1, 0, flag_col)

        # Reintroduce the correlation axis.
        flag_col = da.broadcast_to(flag_col[:, :, None],
                                   data_col.shape,
                                   chunks=data_col.chunks)

        # Convert back to a boolean array.
        flag_col = flag_col.astype(bool)

        # Make the FLAG_ROW column consistent with FLAG.
        flag_row_col = da.all(flag_col, axis=(1, 2))

        updated_xds = xds.assign(
            {
                "FLAG": (xds.DATA.dims, flag_col),
                "FLAG_ROW": (xds.FLAG_ROW.dims, flag_row_col)
            }
        )

        writable_xds.append(updated_xds)

    return writable_xds


def initialise_flags(data_col, weight_col, flag_col, flag_row_col):
    """Given input data, weights and flags, initialise the aggregate flags.

    Populates the internal flag array based on existing flags and data
    points/weights which appear invalid.

    Args:
        data_col: A dask.array containing the data.
        weight_col: A dask.array containing the weights.
        flag_col: A dask.array containing the conventional flags.
        flag_row_col: A dask.array containing the conventional row flags.

    Returns:
        flags: A dask.array containing the initialized aggregate flags.
    """

    return da.blockwise(_initialise_flags, ("rowlike", "chan"),
                        data_col, ("rowlike", "chan", "corr"),
                        weight_col, ("rowlike", "chan", "corr"),
                        flag_col, ("rowlike", "chan", "corr"),
                        flag_row_col, ("rowlike",),
                        dtype=np.int8,
                        name="init_flags-" + uuid4().hex,
                        adjust_chunks=data_col.chunks,
                        align_arrays=False,
                        concatenate=True)


def _initialise_flags(data_col, weight_col, flag_col, flag_row_col):
    """See docstring for initialise_flags."""

    # Combine the flags from both the flag and flag_row columns.
    flags = flag_col | flag_row_col[:, None, None]

    # The following does some sanity checking on the input data and
    # weights. Specifically, we look for points with missing/broken data, and
    # points with null weights. TODO: We can do this with a much smaller
    # memory footprint by passing this into a numba loop which makes these
    # decisions per element.

    # We assume that the first and last entries of the correlation axis
    # are the on-diagonal terms. TODO: This should be safe provided we don't
    # have off-diagonal only data, although in that case the flagging
    # logic is probablly equally applicable.

    n_corr = data_col.shape[-1]
    start = 0
    stop = n_corr
    step = 3 if n_corr == 4 else 1
    corr_sel = slice(start, stop, step)

    missing_points = np.any(data_col[..., corr_sel] == 0, axis=-1)
    flags[missing_points] = True

    noweight_points = np.any(weight_col[..., corr_sel] == 0, axis=-1)
    flags[noweight_points] = True

    # At this point, if any correlation is flagged, flag other correlations.
    flags = np.any(flags, axis=-1).astype(np.int8)

    return flags


def valid_median(arr):
    return np.median(arr[np.isfinite(arr) & (arr > 0)], keepdims=True)


def add_mad_graph(data_xds_list, mad_opts):

    diag_corrs = ['RR', 'LL', 'XX', 'YY']

    bl_thresh = mad_opts.threshold_bl
    gbl_thresh = mad_opts.threshold_global
    max_deviation = mad_opts.max_deviation

    flagged_data_xds_list = []

    for xds in data_xds_list:
        residuals = xds._RESIDUAL.data
        if mad_opts.whitening == "robust":
            weight_col = xds._WEIGHT.data
        elif mad_opts.whitening == "native":
            weight_col = xds.WEIGHT.data
        else:
            weight_col = da.ones_like(
                xds._WEIGHT.data,
                name="unity_weights-" + uuid4().hex
            )
        flag_col = xds.FLAG.data
        ant1_col = xds.ANTENNA1.data
        ant2_col = xds.ANTENNA2.data
        n_ant = xds.sizes["ant"]
        n_bl_w_autos = (n_ant * (n_ant - 1))/2 + n_ant
        n_t_chunk, n_f_chunk, _ = residuals.numblocks

        if mad_opts.use_off_diagonals:
            corr_sel = tuple(np.arange(residuals.shape[-1]))
        else:
            corr_sel = tuple(
                [i for i, c in enumerate(xds.corr.values) if c in diag_corrs]
            )

        wres = da.blockwise(
            compute_whitened_residual, ("rowlike", "chan", "corr"),
            residuals, ("rowlike", "chan", "corr"),
            weight_col, ("rowlike", "chan", "corr"),
            dtype=residuals.dtype,
            align_arrays=False,
            concatenate=True
        )

        bl_mad_and_med_real = da.blockwise(
            compute_bl_mad_and_med,
            ("rowlike", "chan", "bl", "corr", "est"),
            wres.real, ("rowlike", "chan", "corr"),
            flag_col, ("rowlike", "chan"),
            ant1_col, ("rowlike",),
            ant2_col, ("rowlike",),
            n_ant, None,
            dtype=wres.real.dtype,
            align_arrays=False,
            concatenate=True,
            adjust_chunks={"rowlike": (1,)*n_t_chunk,
                           "chan": (1,)*n_f_chunk},
            new_axes={"bl": n_bl_w_autos,
                      "est": 2}
        )

        bl_mad_and_med_imag = da.blockwise(
            compute_bl_mad_and_med,
            ("rowlike", "chan", "bl", "corr", "est"),
            wres.imag, ("rowlike", "chan", "corr"),
            flag_col, ("rowlike", "chan"),
            ant1_col, ("rowlike",),
            ant2_col, ("rowlike",),
            n_ant, None,
            dtype=wres.imag.dtype,
            align_arrays=False,
            concatenate=True,
            adjust_chunks={"rowlike": (1,)*n_t_chunk,
                           "chan": (1,)*n_f_chunk},
            new_axes={"bl": n_bl_w_autos,
                      "est": 2}
        )

        gbl_mad_and_med_real = da.blockwise(
            compute_gbl_mad_and_med, ("rowlike", "chan", "corr", "est"),
            wres.real, ("rowlike", "chan", "corr"),
            flag_col, ("rowlike", "chan"),
            dtype=wres.real.dtype,
            align_arrays=False,
            concatenate=True,
            adjust_chunks={"rowlike": (1,)*n_t_chunk,
                           "chan": (1,)*n_f_chunk},
            new_axes={"est": 2}
        )

        gbl_mad_and_med_imag = da.blockwise(
            compute_gbl_mad_and_med, ("rowlike", "chan", "corr", "est"),
            wres.imag, ("rowlike", "chan", "corr"),
            flag_col, ("rowlike", "chan"),
            dtype=wres.imag.dtype,
            align_arrays=False,
            concatenate=True,
            adjust_chunks={"rowlike": (1,)*n_t_chunk,
                           "chan": (1,)*n_f_chunk},
            new_axes={"est": 2}
        )

        row_chunks = residuals.chunks[0]

        mad_flags = da.blockwise(
            compute_mad_flags, ("rowlike", "chan"),
            wres, ("rowlike", "chan", "corr"),
            gbl_mad_and_med_real, ("rowlike", "chan", "corr", "est"),
            gbl_mad_and_med_imag, ("rowlike", "chan", "corr", "est"),
            bl_mad_and_med_real, ("rowlike", "chan", "bl", "corr", "est"),
            bl_mad_and_med_imag, ("rowlike", "chan", "bl", "corr", "est"),
            ant1_col, ("rowlike",),
            ant2_col, ("rowlike",),
            gbl_thresh, None,
            bl_thresh, None,
            max_deviation, None,
            corr_sel, None,
            n_ant, None,
            dtype=np.int8,
            align_arrays=False,
            concatenate=True,
            adjust_chunks={"rowlike": row_chunks},
        )

        flag_col = da.where(mad_flags, 1, flag_col)

        flagged_data_xds = xds.assign({"FLAG": (("row", "chan"), flag_col)})

        flagged_data_xds_list.append(flagged_data_xds)

    return flagged_data_xds_list
