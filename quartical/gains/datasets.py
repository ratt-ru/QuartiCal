# -*- coding: utf-8 -*-
from quartical.config.external import Gain
from quartical.config.internal import yield_from
from loguru import logger  # noqa
import numpy as np
import dask.array as da
import pathlib
import shutil
from daskms.experimental.zarr import xds_to_zarr
from quartical.gains import TERM_TYPES
from quartical.utils.dask import blockwise_unique
from quartical.utils.maths import mean_for_index


def make_gain_xds_list(data_xds_list,
                       tipc_list,
                       fipc_list,
                       coords_per_xds,
                       chain_opts):
    """Returns a list of xarray.Dataset objects describing the gain terms.

    For a given input xds containing data, creates an xarray.Dataset object
    per term which describes the term's dimensions.

    Args:
        data_xds_list: A list of xarray.Dataset objects containing MS data.
        t_map_list: List of dask.Array objects containing time mappings.
        t_bin_list: List of dask.Array objects containing time binnings.
            Binnings map unique time to solutiion interval, rather than row.
        f_map_list: List of dask.Array objects containing frequency mappings.
        chain_opts: A Chain config object.

    Returns:
        gain_xds_list: A list of lists of xarray.Dataset objects describing the
            gain terms assosciated with each data xarray.Dataset.
    """

    gain_xds_list = []

    for xds_ind, data_xds in enumerate(data_xds_list):

        term_xds_list = []

        term_coords = coords_per_xds[xds_ind]

        for loop_vars in enumerate(yield_from(chain_opts, "type")):
            term_ind, (term_name, term_type) = loop_vars

            term_t_chunks = tipc_list[xds_ind][:, :, term_ind]
            term_f_chunks = fipc_list[xds_ind][:, :, term_ind]
            term_opts = getattr(chain_opts, term_name)

            term_obj = TERM_TYPES[term_type](term_name,
                                             term_opts,
                                             data_xds,
                                             term_coords,
                                             term_t_chunks,
                                             term_f_chunks)

            term_xds_list.append(term_obj.make_xds())

        gain_xds_list.append(term_xds_list)

    return gain_xds_list


def compute_interval_chunking(data_xds_list, t_map_list, f_map_list):
    '''Compute the per-term chunking of the gains.

    Given a list of data xarray.Datasets as well as information about the
    time and frequency mappings, computes the chunk sizes of the gain terms.

    Args:
        data_xds_list: A list of data-containing xarray.Dataset objects.
        t_map_list: A list of arrays describing how times map to solint.
        f_map_list: A list of arrays describing how freqs map to solint.

    Returns:
        A tuple of lists containing arrays which descibe the chunking.
    '''

    tipc_list = []
    fipc_list = []

    for xds_ind, data_xds in enumerate(data_xds_list):

        t_map_arr = t_map_list[xds_ind]
        f_map_arr = f_map_list[xds_ind]

        tipc_per_term = da.map_blocks(lambda arr: arr[:, -1:, :] + 1,
                                      t_map_arr,
                                      chunks=((2,),
                                              (1,)*t_map_arr.numblocks[1],
                                              t_map_arr.chunks[2]))

        fipc_per_term = da.map_blocks(lambda arr: arr[:, -1:, :] + 1,
                                      f_map_arr,
                                      chunks=((2,),
                                              (1,)*f_map_arr.numblocks[1],
                                              f_map_arr.chunks[2]))

        tipc_list.append(tipc_per_term)
        fipc_list.append(fipc_per_term)

    # This is an early compute which is necessary to figure out the gain dims.
    return da.compute(tipc_list, fipc_list)


def compute_dataset_coords(data_xds_list,
                           t_bin_list,
                           f_map_list,
                           tipc_list,
                           fipc_list,
                           terms):
    '''Compute the cooridnates for the gain datasets.

    Given a list of data xarray.Datasets as well as information about the
    binning along the time and frequency axes, computes the true coordinate
    values for the gain xarray.Datasets.

    Args:
        data_xds_list: A list of data-containing xarray.Dataset objects.
        t_bin_list: A list of arrays describing how times map to solint.
        f_map_list: A list of arrays describing how freqs map to solint.
        tipc_list: A list of arrays contatining the number of time intervals
            per chunk.
        fipc_list: A list of arrays contatining the number of freq intervals
            per chunk.

    Returns:
        A list of dictionaries containing the computed coordinate values.
    '''

    coords_per_xds = []

    for xds_ind, data_xds in enumerate(data_xds_list):

        utime_chunks = list(map(int, data_xds.UTIME_CHUNKS))

        unique_times = blockwise_unique(data_xds.TIME.data,
                                        chunks=(utime_chunks,))
        unique_freqs = data_xds.CHAN_FREQ.data

        coord_dict = {"time": unique_times,  # Doesn't vary with term.
                      "freq": unique_freqs}  # Doesn't vary with term.

        for term_ind, term_name in enumerate(terms):

            # This indexing corresponds to grabbing the info per xds, per term.
            tipc = tipc_list[xds_ind][:, :, term_ind]
            fipc = fipc_list[xds_ind][:, :, term_ind]
            term_t_bins = t_bin_list[xds_ind][:, :, term_ind]
            term_f_map = f_map_list[xds_ind][:, :, term_ind]

            mean_gtimes = da.map_blocks(mean_for_index,
                                        unique_times,
                                        term_t_bins[0],
                                        dtype=unique_times.dtype,
                                        chunks=(tuple(map(int, tipc[0])),))

            mean_ptimes = da.map_blocks(mean_for_index,
                                        unique_times,
                                        term_t_bins[1],
                                        dtype=unique_times.dtype,
                                        chunks=(tuple(map(int, tipc[1])),))

            mean_gfreqs = da.map_blocks(mean_for_index,
                                        unique_freqs,
                                        term_f_map[0],
                                        dtype=unique_freqs.dtype,
                                        chunks=(tuple(map(int, fipc[0])),))

            mean_pfreqs = da.map_blocks(mean_for_index,
                                        unique_freqs,
                                        term_f_map[1],
                                        dtype=unique_freqs.dtype,
                                        chunks=(tuple(map(int, fipc[1])),))

            coord_dict[f"{term_name}_mean_gtime"] = mean_gtimes
            coord_dict[f"{term_name}_mean_ptime"] = mean_ptimes
            coord_dict[f"{term_name}_mean_gfreq"] = mean_gfreqs
            coord_dict[f"{term_name}_mean_pfreq"] = mean_pfreqs

        coords_per_xds.append(coord_dict)

    # We take the hit on a second early compute in order to make loading and
    # interpolating gains a less complicated operation.
    return da.compute(coords_per_xds)[0]


def make_net_gain_xds_list(data_xds_list, coords_per_xds):

    net_gain_xds_list = []

    for data_xds, xds_coords in zip(data_xds_list, coords_per_xds):
        net_coords = {}
        net_coords["time"] = xds_coords["time"]
        net_coords["freq"] = xds_coords["freq"]
        net_coords["NET_mean_gtime"] = xds_coords["time"]
        net_coords["NET_mean_gfreq"] = xds_coords["freq"]
        net_coords["NET_mean_ptime"] = xds_coords["time"]
        net_coords["NET_mean_pfreq"] = xds_coords["freq"]

        net_t_chunks = np.tile(data_xds.UTIME_CHUNKS, 2).reshape(2, -1)
        net_f_chunks = np.tile(data_xds.chunks["chan"], 2).reshape(2, -1)

        net_obj = TERM_TYPES["complex"]("NET",
                                        Gain(),  # May need to know about dd.
                                        data_xds,
                                        net_coords,
                                        net_t_chunks,
                                        net_f_chunks)

        net_gain_xds_list.append(net_obj.make_xds())

    return net_gain_xds_list


def populate_net_gain_xds_list(net_gain_xds_list,
                               solved_gain_xds_lol,
                               t_bin_list,
                               f_map_list,
                               d_map_list):

    populated_net_gain_xds_list = []

    for ind, xds_list in enumerate(solved_gain_xds_lol):

        net_gain_xds = net_gain_xds_list[ind]
        net_shape = (net_gain_xds.dims["gain_t"],
                     net_gain_xds.dims["gain_f"],
                     net_gain_xds.dims["ant"],
                     net_gain_xds.dims["dir"],
                     net_gain_xds.dims["corr"])

        gain_schema = ("time", "chan", "ant", "dir", "corr")

        gains = [x for xds in xds_list for x in (xds.gains.data, gain_schema)]

        net_gain = da.blockwise(
            combine_gains, ("time", "chan", "ant", "dir", "corr"),
            t_bin_list[ind], ("param", "time", "term"),
            f_map_list[ind], ("param", "chan", "term"),
            d_map_list[ind], None,
            net_shape, None,
            *gains,
            dtype=xds_list[0].gains.dtype,
            align_arrays=False,
            concatenate=True,
            adjust_chunks={"time": net_gain_xds.GAIN_SPEC.tchunk,
                           "chan": net_gain_xds.GAIN_SPEC.fchunk}
        )

        net_gain_xds = net_gain_xds.assign(
            {"gains": (net_gain_xds.GAIN_AXES, net_gain)}
        )

        populated_net_gain_xds_list.append(net_gain_xds)

    return populated_net_gain_xds_list


def combine_gains(t_bin_arr, f_map_arr, d_map_arr, net_shape, *gains):

    t_bin_arr = t_bin_arr[0]
    f_map_arr = f_map_arr[0]

    n_time = t_bin_arr.shape[0]
    n_freq = f_map_arr.shape[0]

    _, _, n_ant, n_dir, n_corr = net_shape

    net_gains = np.zeros((n_time, n_freq, n_ant, n_dir, n_corr),
                         dtype=np.complex128)
    net_gains[..., 0] = 1
    net_gains[..., -1] = 1

    n_term = len(gains)

    for t in range(n_time):
        for f in range(n_freq):
            for a in range(n_ant):
                for d in range(n_dir):
                    for gi in range(n_term):
                        tm = t_bin_arr[t, gi]
                        fm = f_map_arr[f, gi]
                        dm = d_map_arr[gi, d]
                        net_gains[t, f, a, d] = \
                            net_gains[t, f, a, d] @ gains[gi][tm, fm, a, dm]

    return net_gains


def write_gain_datasets(gain_xds_lol, output_opts):
    """Write the contents of gain_xds_lol to zarr in accordance with opts."""

    root_path = pathlib.Path().absolute()  # Wherever the script is being run.
    gain_path = root_path.joinpath(output_opts.gain_dir)

    term_names = [xds.NAME for xds in gain_xds_lol[0]]

    # If the directory in which we intend to store a gain already exists, we
    # remove it to make sure that we don't end up with a mix of old and new.
    for term_name in term_names:
        term_path = gain_path.joinpath(term_name)
        if term_path.is_dir():
            logger.info(f"Removing preexisting gain folder {term_path}.")
            try:
                shutil.rmtree(term_path)
            except Exception as e:
                logger.warning(f"Failed to delete {term_path}. Reason: {e}.")

    gain_writes = []

    for ti, term_name in enumerate(term_names):

        term_xds_list = [tl[ti].chunk({dim: -1 for dim in tl[ti].dims})
                         for tl in gain_xds_lol]

        output_path = f"{gain_path}{'::' + term_name}"

        term_writes = xds_to_zarr(term_xds_list, output_path)

        gain_writes.append(term_writes)

    return [list(terms) for terms in zip(*gain_writes)]
