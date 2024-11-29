# -*- coding: utf-8 -*-
import numpy as np
import dask.array as da
from quartical.calibration.mapping import make_mapping_datasets
from quartical.gains.general.generics import (compute_residual,
                                              compute_corrected_residual,
                                              compute_corrected_weights)
from quartical.calibration.constructor import construct_solver
from quartical.gains.datasets import (make_gain_xds_lod,
                                      make_net_xds_lod,
                                      populate_net_xds_list)
from quartical.interpolation.interpolate import load_and_interpolate_gains
from quartical.gains.baseline import (compute_baseline_corrections,
                                      apply_baseline_corrections)
from loguru import logger  # noqa


def dask_residual(
    data,
    model,
    a1,
    a2,
    sub_dirs,
    row_map,
    row_weights,
    corr_mode,
    *args
):
    """Thin wrapper to handle an unknown number of input gains."""

    gains = tuple(args[::4])
    time_maps = tuple(args[1::4])
    freq_maps = tuple(args[2::4])
    dir_maps = tuple(args[3::4])

    return compute_residual(
        data,
        model,
        gains,
        a1,
        a2,
        time_maps,
        freq_maps,
        dir_maps,
        row_map,
        row_weights,
        corr_mode,
        sub_dirs=sub_dirs
    )


def dask_corrected_residual(
    residual,
    a1,
    a2,
    row_map,
    row_weights,
    corr_mode,
    *args
):
    """Thin wrapper to handle an unknown number of input gains."""

    gains = tuple(args[::4])
    time_maps = tuple(args[1::4])
    freq_maps = tuple(args[2::4])
    dir_maps = tuple(args[3::4])

    return compute_corrected_residual(
        residual,
        gains,
        a1,
        a2,
        time_maps,
        freq_maps,
        dir_maps,
        row_map,
        row_weights,
        corr_mode
    )


def dask_corrected_weights(
    weights,
    a1,
    a2,
    row_map,
    row_weights,
    corr_mode,
    *args
):
    """Thin wrapper to handle an unknown number of input gains."""

    gains = tuple(args[::4])
    time_maps = tuple(args[1::4])
    freq_maps = tuple(args[2::4])
    dir_maps = tuple(args[3::4])

    return compute_corrected_weights(
        weights,
        gains,
        a1,
        a2,
        time_maps,
        freq_maps,
        dir_maps,
        row_map,
        row_weights,
        corr_mode
    )


def add_calibration_graph(
    data_xds_list,
    stats_xds_list,
    solver_opts,
    chain,
    output_opts
):
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

    # TODO: Does this check belong here or elsewhere?
    have_dd_model = any(xds.sizes['dir'] > 1 for xds in data_xds_list)
    have_dd_chain = any(term.direction_dependent for term in chain)

    if have_dd_model and not have_dd_chain:
        logger.warning(
            "User has specified a direction-dependent model but no gain term "
            "has term.direction_dependent enabled. This is supported but may "
            "indicate user error."
        )

    # Create a list of dicts of xarray.Dataset objects which will describe the
    # gains per data xarray.Dataset.
    gain_xds_lod = make_gain_xds_lod(data_xds_list, chain)

    # Create a list of datasets containing mappings. TODO: Is this the best
    # place to do this?
    mapping_xds_list = make_mapping_datasets(data_xds_list, chain)

    # If there are gains to be loaded from disk, this will load an interpolate
    # them to be consistent with this calibration run. TODO: This needs to
    # be substantially improved to handle term specific behaviour/utilize
    # mappings.
    gain_xds_lod = load_and_interpolate_gains(
        gain_xds_lod,
        chain,
        output_opts.gain_directory
    )

    # Poplulate the gain xarray.Datasets with solutions and convergence info.
    gain_xds_lod, data_xds_list, stats_xds_list = construct_solver(
        data_xds_list,
        mapping_xds_list,
        stats_xds_list,
        gain_xds_lod,
        solver_opts,
        chain
    )

    if output_opts.net_gains:
        # Construct an effective gain per data_xds. This is always at the full
        # time and frequency resolution of the data. Triggers an early compute.
        net_xds_lod = make_net_xds_lod(
            data_xds_list,
            chain,
            output_opts
        )

        net_xds_lod = populate_net_xds_list(
            net_xds_lod,
            gain_xds_lod,
            mapping_xds_list,
            output_opts
        )
    else:
        net_xds_lod = []

    # TODO: This is a very hacky implementation that needs work.
    if output_opts.compute_baseline_corrections:
        bl_corr_xds_list = compute_baseline_corrections(
            data_xds_list,
            gain_xds_lod,
            mapping_xds_list
        )
    else:
        bl_corr_xds_list = None

    # Update the data xarray.Datasets with visibility outputs.
    data_xds_list = make_visibility_output(
        data_xds_list,
        gain_xds_lod,
        mapping_xds_list,
        output_opts
    )

    if output_opts.apply_baseline_corrections:
        data_xds_list = apply_baseline_corrections(
            data_xds_list,
            bl_corr_xds_list
        )

    # Return the resulting graphs for the gains and updated xds.
    return (
        gain_xds_lod,
        net_xds_lod,
        data_xds_list,
        stats_xds_list,
        bl_corr_xds_list
    )


def make_visibility_output(
    data_xds_list,
    solved_gain_xds_lod,
    mapping_xds_list,
    output_opts
):
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

    itr = enumerate(zip(data_xds_list, mapping_xds_list))

    if output_opts.subtract_directions:
        n_dir = data_xds_list[0].sizes['dir']  # Should be the same over xdss.
        requested = set(output_opts.subtract_directions)
        valid = set(range(n_dir))
        invalid = requested - valid
        if invalid:
            raise ValueError(
                f"User has specified output.subtract_directions as "
                f"{requested} but the following directions are not present "
                f"in the model: {invalid}."
            )

    for xds_ind, (data_xds, mapping_xds) in itr:
        data_col = data_xds.DATA.data
        model_col = data_xds.MODEL_DATA.data
        weight_col = data_xds._WEIGHT.data  # The weights exiting the solver.
        ant1_col = data_xds.ANTENNA1.data
        ant2_col = data_xds.ANTENNA2.data
        gain_terms = solved_gain_xds_lod[xds_ind]

        time_maps = tuple(
            [mapping_xds.get(f"{k}_time_map").data for k in gain_terms.keys()]
        )
        freq_maps = tuple(
            [mapping_xds.get(f"{k}_freq_map").data for k in gain_terms.keys()]
        )
        dir_maps = tuple(
            [mapping_xds.get(f"{k}_dir_map").data for k in gain_terms.keys()]
        )

        corr_mode = data_xds.sizes["corr"]

        is_bda = hasattr(data_xds, "ROW_MAP")  # We are dealing with BDA.
        row_map = data_xds.ROW_MAP.data if is_bda else None
        row_weights = data_xds.ROW_WEIGHTS.data if is_bda else None

        gain_schema = ("rowlike", "chan", "ant", "dir", "corr")

        term_args = []

        for gain_idx, gain_xds in enumerate(gain_terms.values()):
            term_args.extend([gain_xds.gains.data, gain_schema])
            term_args.extend([time_maps[gain_idx], ("rowlike",)])
            term_args.extend([freq_maps[gain_idx], ("chan",)])
            term_args.extend([dir_maps[gain_idx], ("dir",)])

        residual = da.blockwise(
            dask_residual, ("rowlike", "chan", "corr"),
            data_col, ("rowlike", "chan", "corr"),
            model_col, ("rowlike", "chan", "dir", "corr"),
            ant1_col, ("rowlike",),
            ant2_col, ("rowlike",),
            output_opts.subtract_directions, None,
            *((row_map, ("rowlike",)) if is_bda else (None, None)),
            *((row_weights, ("rowlike",)) if is_bda else (None, None)),
            corr_mode, None,
            *term_args,
            meta=np.empty((0, 0, 0), dtype=data_col.dtype),
            align_arrays=False,
            concatenate=True,
            adjust_chunks={"rowlike": data_col.chunks[0],
                           "chan": data_col.chunks[1]})

        corrected_residual = da.blockwise(
            dask_corrected_residual, ("rowlike", "chan", "corr"),
            residual, ("rowlike", "chan", "corr"),
            ant1_col, ("rowlike",),
            ant2_col, ("rowlike",),
            *((row_map, ("rowlike",)) if is_bda else (None, None)),
            *((row_weights, ("rowlike",)) if is_bda else (None, None)),
            corr_mode, None,
            *term_args,
            meta=np.empty((0, 0, 0), dtype=residual.dtype),
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
            *((row_map, ("rowlike",)) if is_bda else (None, None)),
            *((row_weights, ("rowlike",)) if is_bda else (None, None)),
            corr_mode, None,
            *term_args,
            meta=np.empty((0, 0, 0), dtype=data_col.dtype),
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
            *((row_map, ("rowlike",)) if is_bda else (None, None)),
            *((row_weights, ("rowlike",)) if is_bda else (None, None)),
            corr_mode, None,
            *term_args,
            meta=np.empty((0, 0, 0), dtype=weight_col.dtype),
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
