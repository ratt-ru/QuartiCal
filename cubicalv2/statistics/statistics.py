from cubicalv2.statistics.stat_kernels import (estimate_noise_kernel,
                                               compute_chi_squared)
from cubicalv2.utils.intervals import (column_to_abs_tfadc,
                                       column_to_tfac,
                                       sum_intervals,
                                       model_schema,
                                       data_schema)
from dask.core import flatten
from dask.base import tokenize
from operator import getitem
from dask.array.core import HighLevelGraph
import dask.array as da
import numpy as np
import xarray


def create_data_stats_xds(n_time, n_chan, n_ant, n_t_chunk, n_f_chunk):
    """Set up a data stats xarray dataset and define its coordinates."""

    stats_xds = xarray.Dataset(
        coords={"ant": ("ant", np.arange(n_ant, dtype=np.int32)),
                "time": ("time", np.arange(n_time, dtype=np.int32)),
                "chan": ("chan", np.arange(n_chan, dtype=np.int32)),
                "t_chunk": ("t_chunk", np.arange(n_t_chunk, dtype=np.int32)),
                "f_chunk": ("f_chunk", np.arange(n_f_chunk, dtype=np.int32))})

    return stats_xds


def assign_noise_estimates(stats_xds, data_col, data_bitflags, ant1_col,
                           ant2_col):
    """Generates a graph describing the computation of the noise estimates.

    A custom graph descibing the production of a noise estimate and inverse
    variance per channel per chunk of data_col.

    Args:
        stats_xds: An xarray.Dataset object to which we can assign the
            estimates.
        data_col: A chunked dask array containing data (or the residual).
        data_bitflags: An chunked dask array containing bitflags.
        ant1_col: A chunked dask array of antenna values.
        ant2_col: A chunked dask array of antenna values.

    Returns:
        noise_est: Graph which produces noise estimates.
        inv_var_per_chan: Graph which produces inverse variance per channel.
    """

    data_col_keys = list(flatten(data_col.__dask_keys__()))
    data_bitflags_keys = list(flatten(data_bitflags.__dask_keys__()))
    ant1_col_keys = list(flatten(ant1_col.__dask_keys__()))
    ant2_col_keys = list(flatten(ant2_col.__dask_keys__()))

    dask_inputs = (data_col, data_bitflags, ant1_col, ant2_col)

    # Based on the inputs, generate a unique hash which can be used to uniquely
    # identify nodes in the graph.
    token = tokenize(*dask_inputs)

    # These should be guarateed to have the correct dimension in each chunking
    # axis. This is important for key matching.
    n_t_chunks, n_f_chunks, _ = data_col.numblocks

    # Layers are the high level graph structure. Initialise with the dasky
    # inputs. Similarly for the their dependencies.
    layers = {inp.name: inp.__dask_graph__() for inp in dask_inputs}
    deps = {inp.name: inp.__dask_graph__().dependencies for inp in dask_inputs}

    datastats_name = "datastats-" + token

    datastats_dsk = {}

    # Set up the solver graph. Loop over the chunked axes.
    for t in range(n_t_chunks):
        for f in range(n_f_chunks):

            # Convert time and freq chunk indices to a flattened index.
            ind = t*n_f_chunks + f

            # Set up the per-chunk stats. Note how keys are assosciated.
            datastats_dsk[(datastats_name, t, f,)] = \
                (estimate_noise_kernel,
                 data_col_keys[ind],
                 data_bitflags_keys[ind],
                 ant1_col_keys[t],
                 ant2_col_keys[t],
                 stats_xds.ant.size)

    # Add the solver layer and its dependencies.
    layers.update({datastats_name: datastats_dsk})
    deps.update({datastats_name: {inp.name for inp in dask_inputs}})

    noise_est_key = "noiseest-{}".format(token)
    inv_var_per_chan_key = "ivpc-{}".format(token)

    get_noise_est = {(noise_est_key, k[1], k[2]): (getitem, k, 0)
                     for i, k in enumerate(datastats_dsk.keys())}

    layers.update({noise_est_key: get_noise_est})
    deps.update({noise_est_key: {datastats_name}})

    get_inv_var_per_chan = \
        {(inv_var_per_chan_key, k[1], k[2]): (getitem, k, 1)
         for i, k in enumerate(datastats_dsk.keys())}

    layers.update({inv_var_per_chan_key: get_inv_var_per_chan})
    deps.update({inv_var_per_chan_key: {datastats_name}})

    # Turn the layers and dependencies into a high level graph.
    graph = HighLevelGraph(layers, deps)

    noise_est = da.Array(graph,
                         name=noise_est_key,
                         chunks=((1,)*n_t_chunks,
                                 (1,)*n_f_chunks),
                         dtype=np.float32)

    inv_var_per_chan = da.Array(graph,
                                name=inv_var_per_chan_key,
                                chunks=((1,)*n_t_chunks,
                                        data_col.chunks[1]),
                                dtype=np.float32)

    updated_stats_xds = stats_xds.assign(
        {"inv_var_per_chan": (("t_chunk", "chan"), inv_var_per_chan),
         "noise_est": (("t_chunk", "f_chunk"), noise_est)})

    return updated_stats_xds


def assign_tf_stats(stats_xds, data_bitflags, ant1_col, ant2_col, utime_ind,
                    n_utime, chunk_spec):
    """Computes and assigns a host of data resolution statistics.

    Updates an input data stats xds with counts of equations per antenna and
    equations per time-frequency. In addition it computes the chi-squared
    nomalisation factor per time-frequency and for the entire chunk.

    Args:
        stats_xds: An xarray.dataset on which data resolution stats live.
        data_bitflags: dask.array of bitflags at full resolution.
        ant1_col: A dask.array of antenna1 values.
        ant2_col: A dask.array of antenna1 values.
        utime_ind: A dask.array of indices corresponding to unique time values.
        n_utime: A dask.array of number of unique times per chunk.
        n_ant: An integer number of antennas.
        n_chunk: An integer number of chunks in this xarray.dataset.
        n_chan: An integer number of channels.
        chunk_spec: A tuple of integers describing the times per chunk.

    Returns:
        modified_stats_xds: xarray.dataset onto which the stats have been
            assigned.
        unflagged_tfac: dask.array of unflagged points. Returned here to
            avoid wasteful recomputation.
    """

    # Set up some dimensions.

    n_t_chunk, n_f_chunk, n_ant = \
        [stats_xds.dims[d] for d in ["t_chunk", "f_chunk", "ant"]]

    # Get all the unflagged points.

    unflagged = data_bitflags == 0

    # Convert the bitflags (column-like) to a (time, freq, antenna,
    # correlation) array. This is smaller (no baseline dimension) and can
    # be easily manipulated to produce other quantities of interest.

    unflagged_tfac = \
        da.blockwise(column_to_tfac, ("rowlike", "chan", "ant", "corr"),
                     unflagged, ("rowlike", "chan", "corr"),
                     ant1_col, ("rowlike",),
                     ant2_col, ("rowlike",),
                     utime_ind, ("rowlike",),
                     n_utime, ("rowlike",),
                     n_ant, None,
                     dtype=np.int32,
                     concatenate=True,
                     align_arrays=False,
                     new_axes={"ant": n_ant},
                     adjust_chunks={"rowlike": chunk_spec})

    # Determine the number of equations per antenna by summing the appropriate
    # values from the per-row unflagged values. The factor of 2 accounts for
    # the conjugate points.

    eqs_per_ant = da.map_blocks(
        lambda a, **kw: 2*np.sum(a, **kw)[..., 0],
        unflagged_tfac,
        axis=(0, 1, 3),
        keepdims=True,
        drop_axis=3,
        chunks=((1,)*n_t_chunk,
                (1,)*n_f_chunk,
                (n_ant,)))

    # Determine the number of equations per time-frequency slot. The factor of
    # 2 accounts for the conjugate points.
    eqs_per_tf = da.map_blocks(
        lambda a, **kw: 2*np.sum(a, **kw)[..., 0, 0],
        unflagged_tfac,
        axis=(2, 3),
        keepdims=True,
        drop_axis=(2, 3))

    # Determine the normalisation factor as the reciprocal of the equations
    # per time-frequency bin.
    tf_norm_factor = da.map_blocks(silent_divide, 1, eqs_per_tf,
                                   dtype=np.float64)

    # Compute the total number of equations per chunk.
    total_eqs = da.map_blocks(
        lambda a, **kw: np.sum(a, **kw),
        eqs_per_tf,
        keepdims=True,
        dtype=np.int64,
        chunks=((1,)*n_t_chunk,
                (1,)*n_f_chunk))

    # Compute the overall normalisation factor.
    total_norm_factor = da.map_blocks(silent_divide, 1, total_eqs,
                                      dtype=np.float64)

    # Assign the relevant values to the xds.
    modified_stats_xds = stats_xds.assign(
        {"eqs_per_ant": (("t_chunk", "f_chunk", "ant"), eqs_per_ant),
         "eqs_per_tf": (("time", "chan"), eqs_per_tf),
         "tf_norm_factor": (("time", "chan"), tf_norm_factor),
         "tot_norm_factor": (("t_chunk", "f_chunk"), total_norm_factor)})

    return modified_stats_xds, unflagged_tfac


def compute_average_model(stats_xds, model_col, unflagged_tfac, ant1_col,
                          ant2_col, utime_ind, n_utime, chunk_spec):
    """Computes average value of |model|^2.

    Given model data, accumulates |model|^2 into a (time, freq, ant, dir, corr)
    array. This is later used to determine the errors on the gains.

    Args:
        model_col: dask.array of model values.
        unflagged_tfac: A dask.array of flag counts with shape (time, freq,
            ant, corr).
        ant1_col: A dask.array of antenna1 values.
        ant2_col: A dask.array of antenna1 values.
        utime_ind: A dask.array of indices corresponding to unique time values.
        n_utime: A dask.array of number of unique times per chunk.
        n_chan: An integer number of channels.
        chunk_spec: A tuple of integers describing the times per chunk.

    Returns:
        avg_abs_sqrd_model: A dask.array of per-antenna |model|^2 values of
            shape (time, freq, ant, dir, corr).
    """

    # Note that we currently do not use the weights here - this differs from
    # V1 and needs to be discussed. This collapses the model values into a
    # (time, freq, ant, dir, corr) array of abs^2. This is done as a single
    # operation to avoid creating an array the size of the model.

    abs_sqrd_model_tfadc = da.blockwise(
        column_to_abs_tfadc, ("rowlike", "chan", "ant", "dir", "corr"),
        model_col, ("rowlike", "chan", "dir", "corr"),
        ant1_col, ("rowlike",),
        ant2_col, ("rowlike",),
        utime_ind, ("rowlike",),
        n_utime, ("rowlike",),
        stats_xds.ant.size, None,
        dtype=model_col.real.dtype,
        concatenate=True,
        align_arrays=False,
        new_axes={"ant": stats_xds.ant.size},
        adjust_chunks={"rowlike": chunk_spec})

    # Sum over the correlation axis as is done in V1. Note that we retain
    # correlation as a dummy index (D) so that we don't confuse arrays with
    # similar dimensions.

    abs_sqrd_model_tfadD = \
        abs_sqrd_model_tfadc.map_blocks(np.sum, axis=4, drop_axis=4,
                                        new_axis=4, keepdims=True)

    # Sum over the correlation axis as is done in V1. Note that we retain
    # correlation as a dummy index (D) so that we don't confuse arrays with
    # similar dimensions.

    unflagged_tfaD = unflagged_tfac.map_blocks(np.sum, axis=3, drop_axis=3,
                                               new_axis=3, keepdims=True)

    # This is appropriate for the case where we sum over correlation.

    avg_abs_sqrd_model = \
        abs_sqrd_model_tfadD.map_blocks(silent_divide,
                                        unflagged_tfaD[..., None, :])

    # In the event that we want to retain the correlation axis, this code is
    # appropriate.

    # avg_abs_sqrd_model = \
    #     abs_sqrd_model_tfadc.map_blocks(silent_divide,
    #                                     unflagged_tfac[..., None, :])

    return avg_abs_sqrd_model


def assign_interval_stats(gain_xds, data_stats_xds, unflagged_tfac,
                          avg_abs_sqrd_model, ti_chunks, fi_chunks,
                          t_int, f_int, n_utime):
    """Assign interval based statistics to the gain xarray.Dataset.

    Computes and assigns the prior gain error, missing fraction, and
    chisq correction factors per interval and over each block.

    Args:
        gain_xds: xarray.Dataset on which the gains and their stats live.
        data_stats_xds: xarray.Dataset on which the data and their stats live.
        unflagged_tfac: dask.array of unflagged values per (t, f, a, c).
        avg_abs_sqrd_model: dask.array containing average abs squared model.
        ti_chunks: Tuple of integer time chunk values.
        fi_chunks: Tuple of integer frequency chunk values.
        t_int: Integer time interval.
        f_int: Integer frequency interal.
        n_utime: Dask.array of number of unique times per chunk.

    Returns:
        updated_gain_xds: xarray.Dataset with new values.
        flagged_tifia: dask.array of flagged values per (ti, fi, a).
    """
    n_dir = gain_xds.dir.size  # TODO: Add fixed direction logic.
    n_ant = gain_xds.ant.size

    # This creates an (n_t_int, n_f_int, n_ant, n_corr) array of unflagged
    # points by summing over solution interval. Note that V1 did not retain a
    # correlation axis.

    unflagged_tifiac = da.blockwise(sum_intervals, data_schema,
                                    unflagged_tfac, data_schema,
                                    t_int, None,
                                    f_int, None,
                                    dtype=np.int32,
                                    concatenate=True,
                                    align_arrays=False,
                                    adjust_chunks={"rowlike": ti_chunks,
                                                   "chan": fi_chunks})

    # Antennas which have no unflagged points in an interval must be fully
    # flagged. Note that we reduce over the correlation axis here.

    flagged_tifia = da.all(unflagged_tifiac == 0, axis=-1)

    missing_fraction = da.map_blocks(
        lambda x: np.atleast_2d(np.sum(x)/x.size),
        flagged_tifia,
        chunks=((1,)*gain_xds.t_chunk.size,
                (1,)*gain_xds.f_chunk.size),
        drop_axis=2,
        dtype=np.int32)

    # Sum the average abs^2 model over solution intervals.

    if n_dir == 1 and avg_abs_sqrd_model.shape[3] != 1:
        avg_abs_sqrd_model = avg_abs_sqrd_model.map_blocks(np.sum,
                                                           axis=3,
                                                           keepdims=True,
                                                           drop_axis=3,
                                                           new_axis=3)

    avg_abs_sqrd_model_int = \
        da.blockwise(sum_intervals, model_schema,
                     avg_abs_sqrd_model, model_schema,
                     t_int, None,
                     f_int, None,
                     dtype=np.float32,
                     concatenate=True,
                     align_arrays=False,
                     adjust_chunks={"rowlike": ti_chunks,
                                    "chan": fi_chunks})

    # Determine the noise^2 per channel by inverting the varaince^2.

    sigma_sqrd_per_chan = \
        da.map_blocks(silent_divide, 1, data_stats_xds.inv_var_per_chan.data,
                      dtype=np.float64)

    # Map the per channel estimates to per interval estimates.

    sigma_sqrd_per_int = \
        da.blockwise(per_chan_to_per_int, model_schema,
                     sigma_sqrd_per_chan, ("rowlike", "chan"),
                     avg_abs_sqrd_model_int, model_schema,
                     n_utime, ("rowlike",),
                     t_int, None,
                     f_int, None,
                     dtype=np.float32,
                     concatenate=True,
                     align_arrays=False)

    # Sum over the correlation axis.

    unflagged_tifiaD = unflagged_tifiac.map_blocks(np.sum, axis=3, drop_axis=3,
                                                   new_axis=3, keepdims=True)

    # Note the egregious fudge factor of four. This was introduced to be
    # consistent with V1 which abandons doesn't count correlations. TODO:
    # Sit down with Oleg and figure out exactly what we want to happen in V2.

    noise_to_signal_ratio = da.map_blocks(
        silent_divide,
        4*sigma_sqrd_per_int,
        unflagged_tifiaD[:, :, :, None, :]*avg_abs_sqrd_model_int,
        undefined=np.inf,
        dtype=np.float64,
        chunks=(ti_chunks, fi_chunks, (n_ant,), (n_dir,), (1,)))

    # The prior gain error is the square root of the noise to signal ratio.

    prior_gain_error = da.sqrt(noise_to_signal_ratio)

    # Determine the number of equations per interval by collapsing the
    # equations per time-frequency array.

    eqs_per_interval = \
        da.blockwise(sum_intervals, ("rowlike", "chan"),
                     data_stats_xds.eqs_per_tf.data, ("rowlike", "chan"),
                     t_int, None,
                     f_int, None,
                     dtype=np.int32,
                     concatenate=True,
                     align_arrays=False,
                     adjust_chunks={"rowlike": ti_chunks,
                                    "chan": fi_chunks})

    dof_per_ant = 8  # TODO: Should depend on solver mode.

    n_unknowns = dof_per_ant*n_ant*n_dir  # TODO: Add fixed direction logic.

    # Check for intervals with a sufficient number of equations.

    valid_intervals = eqs_per_interval > n_unknowns
    n_valid_intervals = da.map_blocks(
        lambda x: np.atleast_2d(np.sum(x)),
        valid_intervals,
        chunks=((1,)*gain_xds.t_chunk.size,
                (1,)*gain_xds.f_chunk.size),
        dtype=np.int32)

    n_valid_solutions = n_valid_intervals*n_dir

    # Compute chi-squared correction factor for time-frequency and overall
    # data.

    chisq_tf_factor = da.map_blocks(
        silent_divide,
        eqs_per_interval,
        eqs_per_interval - n_unknowns,
        dtype=np.float64)
    chisq_tf_factor[~valid_intervals] = 0

    chisq_tot_factor = da.map_blocks(
        lambda x: np.atleast_2d(np.sum(x)),
        chisq_tf_factor,
        chunks=((1,)*gain_xds.t_chunk.size,
                (1,)*gain_xds.f_chunk.size),
        dtype=np.float32)
    chisq_tot_factor = da.map_blocks(
        silent_divide,
        chisq_tot_factor,
        n_valid_intervals,
        dtype=np.float64)

    # Zero the PGE in intervals which are considered unsolvable.

    prior_gain_error[~valid_intervals[..., None, None, None]] = 0

    updated_gain_xds = gain_xds.assign(
        {"prior_gain_error": (("time_int", "freq_int", "ant", "dir"),
                              prior_gain_error[..., 0]),
         "missing_fraction": (("t_chunk", "f_chunk"), missing_fraction),
         "chisq_tf_correction": (("time_int", "freq_int"), chisq_tf_factor),
         "chisq_tot_correction": (("t_chunk", "f_chunk"), chisq_tot_factor)})

    # TODO: Handle direction pinning. Handle logging/stat reporting.

    return updated_gain_xds, flagged_tifia


def silent_divide(in1, in2, undefined=0):
    """Divides in1 by in2, supressing warnings. Division by zero gives zero."""

    with np.errstate(divide='ignore', invalid='ignore'):
        out_arr = np.where(in2 != 0, in1/in2, undefined)

    return out_arr


def per_chan_to_per_int(sigma_sqrd_per_chan, avg_abs_sqrd_model_int, n_time,
                        t_int, f_int):
    """Converts per channel sigma squared into per interval sigma squared."""

    n_chan = sigma_sqrd_per_chan.shape[1]

    sigma_sqrd_per_int = np.zeros_like(avg_abs_sqrd_model_int,
                                       dtype=sigma_sqrd_per_chan.dtype)

    chan_per_int = np.add.reduceat(sigma_sqrd_per_chan,
                                   np.arange(0, n_chan, f_int),
                                   axis=1)
    time_per_int = np.add.reduceat(np.ones(n_time),
                                   np.arange(0, n_time, t_int))

    sigma_sqrd_per_int[:] = \
        (time_per_int[:, None]*chan_per_int)[..., None, None, None]

    return sigma_sqrd_per_int


def assign_pre_solve_chisq(stats_xds, data_col, model_col, weight_col,
                           ant1_col, ant2_col, utime_ind, n_utime, chunk_spec):
    """See _assign_chisq. Suitable for OTF residual values with no gains."""

    modified_stats_xds = _assign_chisq(stats_xds,
                                       data_col,
                                       model_col,
                                       weight_col,
                                       ant1_col,
                                       ant2_col,
                                       utime_ind,
                                       n_utime,
                                       chunk_spec,
                                       "pre")

    return modified_stats_xds


def assign_post_solve_chisq(stats_xds, residual_col, weight_col,
                            ant1_col, ant2_col, utime_ind, n_utime,
                            chunk_spec):
    """See _assign_chisq. Suitable for pre-computed residual values."""

    modified_stats_xds = _assign_chisq(stats_xds,
                                       residual_col,
                                       None,
                                       weight_col,
                                       ant1_col,
                                       ant2_col,
                                       utime_ind,
                                       n_utime,
                                       chunk_spec,
                                       "post")

    return modified_stats_xds


def _assign_chisq(stats_xds, data_col, model_col, weight_col,
                  ant1_col, ant2_col, utime_ind, n_utime, chunk_spec, prefix):
    """Assigns the value of the chi-squared onto a data stats xds.

    Given an input stats xds, computes the chi-quared from the value of the
    residual. If model_col is None, it is presumed that data_col already
    contains the residual value. Otherwise the resdual is computed on the
    fly as data_col - model_col. Note that this does not incude the weights.

    Args:
        stats_xds: An xarray.Dataset on which data stats live.
        data_col: A dask.array containing the data/residual.
        model_col: A dask.array containing the model or None.
        weight_col: A dask.array containing the weights.
        ant1_col: A dask.array containing antenna indices.
        ant2_col: A dask.array containing antenna indices.
        utime_ind: A dask.array containing unique time indices.
        n_utime: A dask.array containing the number of unique times.
        chunk_spec: A non-dask equivalent of n_utime.
        prefix: String prefix for chi-squared xds fields.

    Returns:
        modified_stats_xds: xarray.Dataset updated with new values.
    """

    # Grab the weights from the xds. Note that these may be worth computing
    # here.
    inv_var_per_chan = stats_xds.inv_var_per_chan.data
    tf_norm_factor = stats_xds.tf_norm_factor.data
    tot_norm_factor = stats_xds.tot_norm_factor.data

    # Account for the model is None case - prior to calibration the residuals
    # can easily be computed as data - model. Thereafter we want to feed in
    # precomputed residual values.
    if model_col is None:
        model_arg = [None, None]
    else:
        model_arg = [model_col.sum(axis=2), ("rowlike", "chan", "corr")]

    # Compute the chi-squared value per time, frequency and antenna.
    chisq_tfa = da.blockwise(
        compute_chi_squared, ("rowlike", "chan", "ant"),
        data_col, ("rowlike", "chan", "corr"),
        *model_arg,
        weight_col, ("rowlike", "chan", "corr"),
        inv_var_per_chan, ("rowlike", "chan"),
        utime_ind, ("rowlike",),
        ant1_col, ("rowlike",),
        ant2_col, ("rowlike",),
        n_utime, ("rowlike",),
        stats_xds.ant.size, None,
        dtype=data_col.real.dtype,
        concatenate=True,
        align_arrays=False,
        new_axes={"ant": stats_xds.ant.size},
        adjust_chunks={"rowlike": chunk_spec})

    # Compute and weight the chi-squared per time and frequency.
    chisq_tf = chisq_tfa.sum(axis=-1) * tf_norm_factor

    # Compute and weight the total chi-squared value.
    chisq = tot_norm_factor * chisq_tfa.map_blocks(
        lambda x: np.atleast_2d(np.sum(x)),
        drop_axis=2,
        chunks=((1,)*stats_xds.t_chunk.size,
                (1,)*stats_xds.f_chunk.size))

    # Assign the relevant chi-squared values with the specified prefix.
    modified_stats_xds = \
        stats_xds.assign(
            {"{}_chisq_tfa".format(prefix): (("time", "chan", "ant"),
                                             chisq_tfa),
             "{}_chisq_tf".format(prefix): (("time", "chan"), chisq_tf),
             "{}_chisq".format(prefix): (("t_chunk", "f_chunk"), chisq)})

    return modified_stats_xds


def assign_presolve_data_stats(data_xds, utime_ind, utime_per_chunk):
    """Conveneience function which computes majority of pre-solve statistics.

    Given an input data xarray.Dataset and unique time information, produces
    and assigns pre-solve statistics to a new xarray.Dataset.

    Args:
        data_xds: A xarray.Dataset object conatining input data.
        utime_ind: A dask.Array containing unique time indices.
        utime_per_chunk: A dask.Array containing the number of unique times
            per chunk.

    Returns:
        stats_xds: xarray.Dataset updated with new values.
        unflagged_tfac:  dask.array of unflagged points. Returned here to
            avoid wasteful recomputation.
        avg_abs_sqrd_model: A dask.array of per-antenna |model|^2 values of
            shape (time, freq, ant, dir, corr).
    """

    # Set up some sensible names for xds fields.
    data_col = data_xds.DATA.data
    model_col = data_xds.MODEL_DATA.data
    ant1_col = data_xds.ANTENNA1.data
    ant2_col = data_xds.ANTENNA2.data
    weight_col = data_xds.WEIGHT.data
    data_bitflags = data_xds.DATA_BITFLAGS.data

    utime_chunks = data_xds.UTIME_CHUNKS

    # Set up some values relating to problem dimensions.
    n_row, n_chan, n_ant, n_dir, n_corr = \
        [data_xds.dims[d] for d in ["row", "chan", "ant", "dir", "corr"]]
    n_t_chunk, n_f_chunk = \
        [len(c) for c in [data_xds.chunks["row"], data_xds.chunks["chan"]]]
    n_utime = sum(utime_chunks)

    # Create an xarray.Dataset object to store the statistics.

    stats_xds = create_data_stats_xds(n_utime,
                                      n_chan,
                                      n_ant,
                                      n_t_chunk,
                                      n_f_chunk)

    # Determine the estimated noise.

    stats_xds = assign_noise_estimates(stats_xds,
                                       data_col - model_col.sum(axis=2),
                                       data_bitflags,
                                       ant1_col,
                                       ant2_col)

    # Compute statistics at time/frequency (data) resolution and return a
    # useful (time, chan, ant, corr) version of flag counts.

    stats_xds, unflagged_tfac = assign_tf_stats(stats_xds,
                                                data_bitflags,
                                                ant1_col,
                                                ant2_col,
                                                utime_ind,
                                                utime_per_chunk,
                                                utime_chunks)

    # Compute the average value of the |model|^2. This is used to compute
    # gain errors.

    avg_abs_sqrd_model = compute_average_model(stats_xds,
                                               model_col,
                                               unflagged_tfac,
                                               ant1_col,
                                               ant2_col,
                                               utime_ind,
                                               utime_per_chunk,
                                               utime_chunks)

    # Compute the pre solve chi-squared values. TODO: Needs to be weighted.

    stats_xds = assign_pre_solve_chisq(stats_xds,
                                       data_col,
                                       model_col,
                                       weight_col,
                                       ant1_col,
                                       ant2_col,
                                       utime_ind,
                                       utime_per_chunk,
                                       utime_chunks)

    return stats_xds, unflagged_tfac, avg_abs_sqrd_model
