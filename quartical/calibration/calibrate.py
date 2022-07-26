# -*- coding: utf-8 -*-
import dask.array as da
from quartical.gains.general.generics import (compute_residual,
                                              compute_corrected_residual,
                                              compute_corrected_weights)
from quartical.calibration.constructor import construct_solver
from quartical.calibration.mapping import make_t_maps, make_f_maps, make_d_maps
from quartical.gains.datasets import (make_gain_xds_lod,
                                      compute_dataset_coords,
                                      compute_interval_chunking,
                                      make_net_xds_list,
                                      populate_net_xds_list)
from quartical.interpolation.interpolate import load_and_interpolate_gains
from loguru import logger  # noqa
from collections import namedtuple


# The following supresses the egregious numba pending deprecation warnings.
# TODO: Make sure that the code doesn't break when they finally decprecate
# reflected lists.
from numba.core.errors import NumbaDeprecationWarning
from numba.core.errors import NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


dstat_dims_tup = namedtuple("dstat_dims_tup",
                            "n_utime n_chan n_ant n_t_chunk n_f_chunk")


def dask_residual(data, model, a1, a2, t_map_arr, f_map_arr, d_map_arr,
                  sub_dirs, row_map, row_weights, corr_mode, *gains):
    """Thin wrapper to handle an unknown number of input gains."""

    return compute_residual(data, model, gains, a1, a2, t_map_arr[0],
                            f_map_arr[0], d_map_arr, row_map, row_weights,
                            corr_mode, sub_dirs=sub_dirs)


def dask_corrected_residual(residual, a1, a2, t_map_arr, f_map_arr,
                            d_map_arr, row_map, row_weights, corr_mode,
                            *gains):
    """Thin wrapper to handle an unknown number of input gains."""

    return compute_corrected_residual(residual, gains, a1, a2, t_map_arr[0],
                                      f_map_arr[0], d_map_arr, row_map,
                                      row_weights, corr_mode)


def dask_corrected_weights(weights, a1, a2, t_map_arr, f_map_arr,
                           d_map_arr, row_map, row_weights, corr_mode,
                           *gains):
    """Thin wrapper to handle an unknown number of input gains."""

    return compute_corrected_weights(weights, gains, a1, a2, t_map_arr[0],
                                     f_map_arr[0], d_map_arr, row_map,
                                     row_weights, corr_mode)


def add_calibration_graph(data_xds_list, solver_opts, chain_opts, output_opts):
    """Given data graph and options, adds the steps necessary for calibration.

    Extends the data graph with the steps necessary to perform gain
    calibration and in accordance with the options Namespace.

    Args:
        data_xds_list: A list of xarray data sets/graphs providing input data.
        opts: A Namespace object containing all necessary configuration.

    Returns:
        gain_xds_lod: A list of dicts containing xarray.Datasets housing the
            solved gains.
        net_xds_list: A list of xarray.Datasets containing the effective gains.
        data_xds_list: A list of xarra.Datasets containing the MS data with
            added visibility outputs.
    """
    # Figure out all mappings between data and solution intervals.
    t_bin_list, t_map_list = make_t_maps(data_xds_list, chain_opts)
    f_map_list = make_f_maps(data_xds_list, chain_opts)
    d_map_list = make_d_maps(data_xds_list, chain_opts)

    # Early compute to figure out solution intervals per data chunk.
    tipc_list, fipc_list = compute_interval_chunking(
        data_xds_list,
        t_map_list,
        f_map_list
    )

    # Early compute to figure out coordinates of gain datasets.
    coords_per_xds = compute_dataset_coords(
        data_xds_list,
        t_bin_list,
        f_map_list,
        tipc_list,
        fipc_list,
        solver_opts.terms
    )

    # Create a list of dicts of xarray.Dataset objects which will describe the
    # gains per data xarray.Dataset.
    gain_xds_lod = make_gain_xds_lod(
        data_xds_list,
        tipc_list,
        fipc_list,
        coords_per_xds,
        chain_opts
    )

    # If there are gains to be loaded from disk, this will load an interpolate
    # them to be consistent with this calibration run.
    gain_xds_lod = load_and_interpolate_gains(gain_xds_lod, chain_opts)

    # Poplulate the gain xarray.Datasets with solutions and convergence info.
    gain_xds_lod, data_xds_list = construct_solver(
        data_xds_list,
        gain_xds_lod,
        t_bin_list,
        t_map_list,
        f_map_list,
        d_map_list,
        solver_opts,
        chain_opts
    )

    if output_opts.net_gains:
        # Construct an effective gain per data_xds. This is always at the full
        # time and frequency resolution of the data.
        net_xds_lod = make_net_xds_list(
            data_xds_list,
            coords_per_xds,
            output_opts
        )

        net_xds_lod = populate_net_xds_list(
            net_xds_lod,
            gain_xds_lod,
            t_bin_list,
            f_map_list,
            d_map_list,
            output_opts
        )
    else:
        net_xds_lod = []

    # Update the data xarray.Datasets with visibility outputs.
    data_xds_list = make_visibility_output(
        data_xds_list,
        gain_xds_lod,
        t_map_list,
        f_map_list,
        d_map_list,
        output_opts
    )

    # Return the resulting graphs for the gains and updated xds.
    return gain_xds_lod, net_xds_lod, data_xds_list


def make_visibility_output(data_xds_list, solved_gain_xds_lod, t_map_list,
                           f_map_list, d_map_list, output_opts):
    """Creates dask arrays for possible visibility outputs.

    Given and xds containing data and its assosciated gains, produces
    dask.Array objects containing the possible visibility outputs.

    Args:
        data_xds_list: A list of xarray.Dataset objects containing MS data.
        solved_gain_xds_lod: A list of dicts containing xarray.Dataset objects
            describing the gain terms.
        t_map_list: List of dask.Array objects containing time mappings.
        f_map_list: List of dask.Array objects containing frequency mappings.
        d_map_list: List of dask.Array objects containing direction mappings.

    Returns:
        A dictionary of lists containing graphs which prodcuce a gain array
        per gain term per xarray dataset.

    """

    post_solve_data_xds_list = []

    for xds_ind, data_xds in enumerate(data_xds_list):
        data_col = data_xds.DATA.data
        model_col = data_xds.MODEL_DATA.data
        weight_col = data_xds._WEIGHT.data  # The weights exiting the solver.
        ant1_col = data_xds.ANTENNA1.data
        ant2_col = data_xds.ANTENNA2.data
        gain_terms = solved_gain_xds_lod[xds_ind]
        t_map_arr = t_map_list[xds_ind]
        f_map_arr = f_map_list[xds_ind]
        d_map_arr = d_map_list[xds_ind]
        corr_mode = data_xds.dims["corr"]

        is_bda = hasattr(data_xds, "ROW_MAP")  # We are dealing with BDA.
        row_map = data_xds.ROW_MAP.data if is_bda else None
        row_weights = data_xds.ROW_WEIGHTS.data if is_bda else None

        gain_schema = ("rowlike", "chan", "ant", "dir", "corr")

        # TODO: For gains with n_dir > 1, we can select out the gains we
        # actually want to correct for.
        gain_list = [x for gxds in gain_terms.values()
                     for x in (gxds.gains.data, gain_schema)]

        residual = da.blockwise(
            dask_residual, ("rowlike", "chan", "corr"),
            data_col, ("rowlike", "chan", "corr"),
            model_col, ("rowlike", "chan", "dir", "corr"),
            ant1_col, ("rowlike",),
            ant2_col, ("rowlike",),
            t_map_arr, ("gp", "rowlike", "term"),
            f_map_arr, ("gp", "chan", "term"),
            d_map_arr, None,
            output_opts.subtract_directions, None,
            *((row_map, ("rowlike",)) if is_bda else (None, None)),
            *((row_weights, ("rowlike",)) if is_bda else (None, None)),
            corr_mode, None,
            *gain_list,
            dtype=data_col.dtype,
            align_arrays=False,
            concatenate=True,
            adjust_chunks={"rowlike": data_col.chunks[0],
                           "chan": data_col.chunks[1]})

        corrected_residual = da.blockwise(
            dask_corrected_residual, ("rowlike", "chan", "corr"),
            residual, ("rowlike", "chan", "corr"),
            ant1_col, ("rowlike",),
            ant2_col, ("rowlike",),
            t_map_arr, ("gp", "rowlike", "term"),
            f_map_arr, ("gp", "chan", "term"),
            d_map_arr, None,
            *((row_map, ("rowlike",)) if is_bda else (None, None)),
            *((row_weights, ("rowlike",)) if is_bda else (None, None)),
            corr_mode, None,
            *gain_list,
            dtype=residual.dtype,
            align_arrays=False,
            concatenate=True,
            adjust_chunks={"rowlike": data_col.chunks[0],
                           "chan": data_col.chunks[1]})

        # We can cheat and reuse the corrected residual code - the only
        # difference is whether we supply the residuals or the data.
        corrected_data = da.blockwise(
            dask_corrected_residual, ("rowlike", "chan", "corr"),
            data_col, ("rowlike", "chan", "corr"),
            ant1_col, ("rowlike",),
            ant2_col, ("rowlike",),
            t_map_arr, ("gp", "rowlike", "term"),
            f_map_arr, ("gp", "chan", "term"),
            d_map_arr, None,
            *((row_map, ("rowlike",)) if is_bda else (None, None)),
            *((row_weights, ("rowlike",)) if is_bda else (None, None)),
            corr_mode, None,
            *gain_list,
            dtype=residual.dtype,
            align_arrays=False,
            concatenate=True,
            adjust_chunks={"rowlike": data_col.chunks[0],
                           "chan": data_col.chunks[1]})

        # We may also want to form corrected weights. Not technicallay a
        # visibility output but close enough. TODO: Change calling function.
        corrected_weight = da.blockwise(
            dask_corrected_weights, ("rowlike", "chan", "corr"),
            weight_col, ("rowlike", "chan", "corr"),
            ant1_col, ("rowlike",),
            ant2_col, ("rowlike",),
            t_map_arr, ("gp", "rowlike", "term"),
            f_map_arr, ("gp", "chan", "term"),
            d_map_arr, None,
            *((row_map, ("rowlike",)) if is_bda else (None, None)),
            *((row_weights, ("rowlike",)) if is_bda else (None, None)),
            corr_mode, None,
            *gain_list,
            dtype=weight_col.dtype,
            align_arrays=False,
            concatenate=True,
            adjust_chunks={"rowlike": data_col.chunks[0],
                           "chan": data_col.chunks[1]}
        )

        # QuartiCal will assign these to the xarray.Datasets as the following
        # underscore prefixed data vars. This is done to avoid overwriting
        # input data prematurely.
        visibility_outputs = {
            "_RESIDUAL": residual,
            "_CORRECTED_RESIDUAL": corrected_residual,
            "_CORRECTED_DATA": corrected_data,
            "_CORRECTED_WEIGHT": corrected_weight,
        }

        dims = data_xds.DATA.dims  # All visiblity columns share these dims.
        data_vars = {k: (dims, v) for k, v in visibility_outputs.items()}

        post_solve_data_xds = data_xds.assign(data_vars)

        post_solve_data_xds_list.append(post_solve_data_xds)

    return post_solve_data_xds_list
