# -*- coding: utf-8 -*-
import dask.array as da
from quartical.kernels.generics import (compute_residual,
                                        compute_corrected_residual)
from quartical.statistics.statistics import (assign_interval_stats,
                                             assign_post_solve_chisq,
                                             assign_presolve_data_stats,)
from quartical.calibration.gain_types import term_types
from quartical.calibration.constructor import construct_solver
from quartical.calibration.mapping import make_t_maps, make_f_maps, make_d_maps
from quartical.scheduling import annotate
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
                  row_map, row_weights, corr_mode, *gains):
    """Thin wrapper to handle an unknown number of input gains."""

    return compute_residual(data, model, gains, a1, a2, t_map_arr,
                            f_map_arr, d_map_arr, row_map, row_weights,
                            corr_mode)


def dask_corrected_residual(residual, a1, a2, t_map_arr, f_map_arr,
                            d_map_arr, row_map, row_weights, corr_mode,
                            *gains):
    """Thin wrapper to handle an unknown number of input gains."""

    return compute_corrected_residual(residual, gains, a1, a2, t_map_arr,
                                      f_map_arr, d_map_arr, row_map,
                                      row_weights, corr_mode)


def add_calibration_graph(data_xds_list, col_kwrds, opts):
    """Given data graph and options, adds the steps necessary for calibration.

    Extends the data graph with the steps necessary to perform gain
    calibration and in accordance with the options Namespace.

    Args:
        data_xds_list: A list of xarray data sets/graphs providing input data.
        col_kwrds: A dictionary containing column keywords.
        opts: A Namespace object containing all necessary configuration.

    Returns:
        A dictionary of lists containing graphs which prodcuce a gain array
        per gain term per xarray dataset.
    """

    # Calibrate per xds. This list will likely consist of an xds per SPW, per
    # scan. This behaviour can be changed.

    # data_stats_xds_list = []
    # post_cal_data_xds_list = []

    # Figure out all mappings between data and solution intervals.
    t_bin_list, t_map_list = make_t_maps(data_xds_list, opts)
    f_map_list = make_f_maps(data_xds_list, opts)
    d_map_list = make_d_maps(data_xds_list, opts)

    # Create a list of lists of xarray.Dataset objects which will describe the
    # gains per data xarray.Dataset. This triggers some early compute.
    gain_xds_list = make_gain_xds_list(data_xds_list,
                                       t_map_list,
                                       f_map_list,
                                       opts)

    # Poplulate the gain xarray.Datasets with solutions and convergence info.
    solved_gain_xds_list = construct_solver(data_xds_list,
                                            gain_xds_list,
                                            t_bin_list,
                                            t_map_list,
                                            f_map_list,
                                            d_map_list,
                                            opts)

    # Update the data xrray.Datasets with visiblity outputs.
    post_solve_data_xds_list = \
        make_visibility_output(data_xds_list,
                               solved_gain_xds_list,
                               t_map_list,
                               f_map_list,
                               d_map_list,
                               opts)

    annotate(gain_xds_list)
    annotate(solved_gain_xds_list)
    annotate(post_solve_data_xds_list)

    # for xds_ind, xds in enumerate(data_xds_list):

        # Create and populate xds for statisics at data resolution. Returns
        # some useful arrays required for future computations. TODO: I really
        # dislike this layer. Consider revising.

        # data_stats_xds, unflagged_tfac, avg_abs_sqrd_model = \
        #     assign_presolve_data_stats(xds, utime_ind, utime_per_chunk)

        # Update the gain xds with relevant interval statistics. Used to be
        # very expensive - has been improved. TODO: Broken by massive changes
        # calibration graph code. Needs to be revisited.

        # gain_xds_list, empty_intervals = \
        #     assign_interval_stats(gain_xds_list,
        #                           data_stats_xds,
        #                           unflagged_tfac,
        #                           avg_abs_sqrd_model,
        #                           utime_per_chunk,
        #                           t_bin_arr,
        #                           f_map_arr,
        #                           opts)

        # ---------------------------------------------------------------------

        # data_stats_xds = assign_post_solve_chisq(data_stats_xds,
        #                                          residuals,
        #                                          weight_col,
        #                                          ant1_col,
        #                                          ant2_col,
        #                                          utime_ind,
        #                                          utime_per_chunk,
        #                                          utime_chunks)

        # data_stats_xds_list.append(data_stats_xds)

    # Return the resulting graphs for the gains and updated xds.
    return solved_gain_xds_list, post_solve_data_xds_list


def make_gain_xds_list(data_xds_list, t_map_list, f_map_list, opts):
    """Returns a list of xarray.Dataset objects describing the gain terms.

    For a given input xds containing data, creates an xarray.Dataset object
    per term which describes the term's dimensions.

    Args:
        data_xds_list: A list of xarray.Dataset objects containing MS data.
        t_map_list: List of dask.Array objects containing time mappings.
        f_map_list: List of dask.Array objects containing frequency mappings.
        opts: A Namespace object containing global options.

    Returns:
        gain_xds_list: A list of lists of xarray.Dataset objects describing the
            gain terms assosciated with each data xarray.Dataset.
    """

    tipc_list = []
    fipc_list = []

    for xds_ind, data_xds in enumerate(data_xds_list):

        t_map_arr = t_map_list[xds_ind]
        f_map_arr = f_map_list[xds_ind]

        tipc_per_term = da.map_blocks(lambda arr: arr[-1:, :] + 1,
                                      t_map_arr,
                                      chunks=((1,)*t_map_arr.numblocks[0],
                                              t_map_arr.chunks[1]))

        fipc_per_term = da.map_blocks(lambda arr: arr[-1:, :] + 1,
                                      f_map_arr,
                                      chunks=((1,)*f_map_arr.numblocks[0],
                                              f_map_arr.chunks[1]))

        tipc_list.append(tipc_per_term)
        fipc_list.append(fipc_per_term)

    # This is an early compute which is necessary to figure out the gain dims.
    tipc_list, fipc_list = da.compute(tipc_list, fipc_list)

    gain_xds_list = []

    for xds_ind, data_xds in enumerate(data_xds_list):

        term_xds_list = []

        for term_ind, term_name in enumerate(opts.solver_gain_terms):

            term_type = getattr(opts, "{}_type".format(term_name))

            term_obj = term_types[term_type](term_name,
                                             data_xds,
                                             tipc_list[xds_ind],
                                             fipc_list[xds_ind],
                                             opts)

            term_xds_list.append(term_obj.make_xds())

        gain_xds_list.append(term_xds_list)

    return gain_xds_list


def make_visibility_output(data_xds_list, solved_gain_xds_list, t_map_list,
                           f_map_list, d_map_list, opts):
    """Creates dask arrays for possible visibility outputs.

    Given and xds containing data and its assosciated gains, produces
    dask.Array objects containing the possible visibility outputs.

    Args:
        data_xds_list: A list of xarray.Dataset objects containing MS data.
        solved_gain_xds_list: A list of lists containing xarray.Dataset objects
            describing the gain terms.
        t_map_list: List of dask.Array objects containing time mappings.
        f_map_list: List of dask.Array objects containing frequency mappings.
        d_map_list: List of dask.Array objects containing direction mappings.
        opts: A Namespace object containing all necessary configuration.

    Returns:
        A dictionary of lists containing graphs which prodcuce a gain array
        per gain term per xarray dataset.

    """

    corr_mode = opts.input_ms_correlation_mode
    is_bda = opts.input_ms_is_bda
    post_solve_data_xds_list = []

    for xds_ind, data_xds in enumerate(data_xds_list):
        data_col = data_xds.DATA.data
        model_col = data_xds.MODEL_DATA.data
        ant1_col = data_xds.ANTENNA1.data
        ant2_col = data_xds.ANTENNA2.data
        gain_terms = solved_gain_xds_list[xds_ind]
        t_map_arr = t_map_list[xds_ind]
        f_map_arr = f_map_list[xds_ind]
        d_map_arr = d_map_list[xds_ind]

        row_map = data_xds.ROW_MAP.data if is_bda else None
        row_weights = data_xds.ROW_WEIGHTS.data if is_bda else None

        gain_schema = ("rowlike", "chan", "ant", "dir", "corr")

        # TODO: For gains with n_dir > 1, we can select out the gains we
        # actually want to correct for.
        gain_list = [x for gxds in gain_terms
                     for x in (gxds.gains.data, gain_schema)]

        residual = da.blockwise(
            dask_residual, ("rowlike", "chan", "corr"),
            data_col, ("rowlike", "chan", "corr"),
            model_col, ("rowlike", "chan", "dir", "corr"),
            ant1_col, ("rowlike",),
            ant2_col, ("rowlike",),
            t_map_arr, ("rowlike", "term"),
            f_map_arr, ("chan", "term"),
            d_map_arr, None,
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
            t_map_arr, ("rowlike", "term"),
            f_map_arr, ("chan", "term"),
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
            t_map_arr, ("rowlike", "term"),
            f_map_arr, ("chan", "term"),
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

        # QuartiCal will assign these to the xarray.Datasets as the following
        # underscore prefixed data vars. This is done to avoid overwriting
        # input data prematurely.
        visibility_outputs = {"_RESIDUAL": residual,
                              "_CORRECTED_RESIDUAL": corrected_residual,
                              "_CORRECTED_DATA": corrected_data}

        dims = data_xds.DATA.dims  # All visiblity coloumns share these dims.
        # TODO: This addition of CUBI_BITFLAG should be done elsewhere.
        data_vars = {"CUBI_BITFLAG": (dims, data_xds.DATA_BITFLAGS.data)}
        data_vars.update({k: (dims, v) for k, v in visibility_outputs.items()})

        post_solve_data_xds = data_xds.assign(data_vars)
        post_solve_data_xds.attrs.update(data_xds.attrs)

        post_solve_data_xds_list.append(post_solve_data_xds)

    return post_solve_data_xds_list
