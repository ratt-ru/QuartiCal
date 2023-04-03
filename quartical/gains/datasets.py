# -*- coding: utf-8 -*-
from quartical.config.external import Gain
from quartical.config.internal import yield_from
from loguru import logger  # noqa
import numpy as np
import dask.array as da
from daskms.experimental.zarr import xds_to_zarr
from quartical.gains import TERM_TYPES
from quartical.utils.dask import blockwise_unique
from quartical.utils.maths import mean_for_index
from quartical.gains.general.generics import combine_gains, combine_flags


def make_gain_xds_lod(data_xds_list,
                      tipc_list,
                      fipc_list,
                      coords_per_xds,
                      chain_opts):
    """Returns a list of dicts of xarray.Dataset objects describing the gains.

    For a given input xds containing data, creates an xarray.Dataset object
    per term which describes the term's dimensions.

    Args:
        data_xds_list: A list of xarray.Dataset objects containing MS data.
        tipc_list: List of numpy.ndarray objects containing number of time
            intervals in a chunk.
        fipc_list: List of numpy.ndarray objects containing number of freq
            intervals in a chunk.
        coords_per_xds: A List of Dicts containing coordinates.
        chain_opts: A Chain config object.

    Returns:
        gain_xds_lod: A List of Dicts of xarray.Dataset objects describing the
            gain terms assosciated with each data xarray.Dataset.
    """

    gain_xds_lod = []

    for xds_ind, data_xds in enumerate(data_xds_list):

        term_xds_dict = {}

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

            term_xds_dict[term_name] = term_obj.make_xds()

        gain_xds_lod.append(term_xds_dict)

    return gain_xds_lod


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

    for xds_ind, _ in enumerate(data_xds_list):

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

        # NOTE: Use the BDA version of the column if it is present.
        time_col = data_xds.get("UPSAMPLED_TIME", data_xds.TIME).data

        unique_times = blockwise_unique(time_col,
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


def make_net_xds_list(data_xds_list, coords_per_xds, output_opts):
    """Construct a list of dicts of xarray.Datasets to house the net gains.

    Args:
        data_xds_list: A List of xarray.Dataset objects containing MS data.
        coords_per_xds: A List of Dicts containing dataset coords.
        output_opts: An output config object.

    Returns:
        net_gain_xds_dol: A Dict of Lists of xarray.Dataset objects to house
            the net gains.
    """

    net_names = [f"{''.join(lt)}-net" for lt in output_opts.net_gains]

    net_gain_xds_lod = []

    for data_xds, xds_coords in zip(data_xds_list, coords_per_xds):

        net_t_chunks = np.tile(data_xds.UTIME_CHUNKS, 2).reshape(2, -1)
        net_f_chunks = np.tile(data_xds.chunks["chan"], 2).reshape(2, -1)

        net_dict = {}

        for net_name in net_names:

            # Create a default config object, consistent with the net gain.
            # NOTE: If we have a direction-dependent model, assume the net gain
            # is also direction dependent.
            config = Gain(direction_dependent=bool(data_xds.dims["dir"]))

            net_obj = TERM_TYPES["complex"](net_name,
                                            config,
                                            data_xds,
                                            xds_coords,
                                            net_t_chunks,
                                            net_f_chunks)

            net_dict[net_name] = net_obj.make_xds()

        net_gain_xds_lod.append(net_dict)

    return net_gain_xds_lod


def combine_gains_wrapper(t_bin_arr, f_map_arr, d_map_arr, term_ids, net_shape,
                          corr_mode, *gains):
    """Wrapper to stop dask from getting confused. See issue #99."""

    return combine_gains(t_bin_arr, f_map_arr, d_map_arr, term_ids, net_shape,
                         corr_mode, *gains)


def combine_flags_wrapper(t_bin_arr, f_map_arr, d_map_arr, term_ids, net_shape,
                          corr_mode, *flags):
    """Wrapper to stop dask from getting confused. See issue #99."""

    return combine_flags(t_bin_arr, f_map_arr, d_map_arr, term_ids, net_shape,
                         corr_mode, *flags)


def populate_net_xds_list(
    net_gain_xds_lod,
    solved_gain_xds_lod,
    t_bin_list,
    f_map_list,
    d_map_list,
    output_opts
):
    """Poplulate the list net gain datasets with net gain values.

    Args:
        net_gain_xds_list: A List of xarray.Dataset objects to house the
            net gains.
        solved_gain_xds_lol: A List of Lists of xarray.Dataset objects housing
            the solved gain terms.
        t_bin_list: A List of dask.Arrays containing mappings from unique
            time to solution interval.
        f_map_list: A List of dask.Arrays containing mappings from channel
            to solution interval.
        d_map_list: A List of numpy.ndarrays containing mappings between
            direction dependent terms and direction independent terms.
        output_opts: An output configuration object,

    Returns:
        net_gain_xds_list: A List of xarray.Dataset objects to house the
            net gains.
    """

    net_terms = output_opts.net_gains

    net_names = [f"{''.join(lt)}-net" for lt in net_terms]

    net_map = dict(zip(net_names, net_terms))

    gain_dims = ("gain_t", "gain_f", "ant", "dir", "corr")
    gain_schema = ("time", "chan", "ant", "dir", "corr")
    flag_schema = ("time", "chan", "ant", "dir")
    populated_net_gain_xds_lod = []

    for ind, (solved_gains, net_gains) in enumerate(zip(solved_gain_xds_lod,
                                                        net_gain_xds_lod)):

        gains = [itm for xds in solved_gains.values()
                 for itm in (xds.gains.data, gain_schema)]
        gain_dtype = np.find_common_type(
            [xds.gains.data.dtype for xds in solved_gains.values()], []
        )
        identity_elements = {
            1: np.ones(1, dtype=gain_dtype),
            2: np.ones(2, dtype=gain_dtype),
            4: np.array((1, 0, 0, 1), dtype=gain_dtype)
        }

        flags = [itm for xds in solved_gains.values()
                 for itm in (xds.gain_flags.data, flag_schema)]
        flag_dtype = np.find_common_type(
            [xds.gain_flags.dtype for xds in solved_gains.values()], []
        )

        net_xds_dict = {}

        for net_name, req_terms in net_map.items():

            net_xds = net_gains[net_name]

            net_shape = tuple(net_xds.dims[d] for d in gain_dims)

            corr_mode = net_shape[-1]

            req_term_ids = \
                [list(solved_gains.keys()).index(tn) for tn in req_terms]

            net_gain = da.blockwise(
                combine_gains_wrapper, ("time", "chan", "ant", "dir", "corr"),
                t_bin_list[ind], ("param", "time", "term"),
                f_map_list[ind], ("param", "chan", "term"),
                d_map_list[ind], None,
                req_term_ids, None,
                net_shape, None,
                corr_mode, None,
                *gains,
                dtype=gain_dtype,
                align_arrays=False,
                concatenate=True,
                adjust_chunks={"time": net_xds.GAIN_SPEC.tchunk,
                               "chan": net_xds.GAIN_SPEC.fchunk,
                               "dir": net_xds.GAIN_SPEC.dchunk}
            )

            net_flags = da.blockwise(
                combine_flags_wrapper, ("time", "chan", "ant", "dir"),
                t_bin_list[ind], ("param", "time", "term"),
                f_map_list[ind], ("param", "chan", "term"),
                d_map_list[ind], None,
                req_term_ids, None,
                net_shape[:-1], None,
                *flags,
                dtype=flag_dtype,
                align_arrays=False,
                concatenate=True,
                adjust_chunks={"time": net_xds.GAIN_SPEC.tchunk,
                               "chan": net_xds.GAIN_SPEC.fchunk,
                               "dir": net_xds.GAIN_SPEC.dchunk}
            )

            net_gain = da.blockwise(np.where, "tfadc",
                                    net_flags[..., None], "tfadc",
                                    identity_elements[corr_mode], None,
                                    net_gain, "tfadc")

            net_xds = net_xds.assign(
                {
                    "gains": (net_xds.GAIN_AXES, net_gain),
                    "gain_flags": (net_xds.GAIN_AXES[:-1], net_flags)
                }
            )

            net_xds_dict[net_name] = net_xds

        populated_net_gain_xds_lod.append(net_xds_dict)

    return populated_net_gain_xds_lod


def write_gain_datasets(gain_xds_lod, net_xds_lod, output_opts):
    """Write the contents of gain_xds_lol to zarr in accordance with opts."""

    gain_path = output_opts.gain_directory

    term_names = [xds.NAME for xds in gain_xds_lod[0].values()]

    writable_xds_dol = {tn: [d[tn] for d in gain_xds_lod] for tn in term_names}

    # If net gains have been requested, add them to the writes.
    if output_opts.net_gains:
        net_names = [xds.NAME for xds in net_xds_lod[0].values()]
        net_xds_dol = {tn: [d[tn] for d in net_xds_lod] for tn in net_names}
        term_names.extend(net_names)
        writable_xds_dol.update(net_xds_dol)

    gain_writes_lol = []

    for term_name, term_xds_list in writable_xds_dol.items():

        term_write_xds_list = []

        # The following rechunks to some sensible chunk size. This ensures
        # that the chunks are regular and <2GB, which is necessary for zarr.

        for xds in term_xds_list:

            target_chunks = {}

            if hasattr(xds, "PARAM_AXES"):
                rechunked_params = \
                    xds.params.chunk({ax: "auto" for ax in xds.PARAM_AXES[:2]})
                target_chunks.update(rechunked_params.chunksizes)

            rechunked_gains = \
                xds.gains.chunk({ax: "auto" for ax in xds.GAIN_AXES[:2]})
            target_chunks.update(rechunked_gains.chunksizes)

            rechunked_xds = xds.chunk(target_chunks)

            term_write_xds_list.append(rechunked_xds)

        output_path = f"{gain_path}{'::' + term_name}"

        gain_writes_lol.append(xds_to_zarr(term_write_xds_list, output_path))

    # This converts the interpolated list of lists into a list of dicts.
    write_xds_lod = [{tn: term for tn, term in zip(term_names, terms)}
                     for terms in zip(*gain_writes_lol)]

    return write_xds_lod


from uuid import uuid4


class GainDataset(object):

    def __init__(self):
        pass

    @staticmethod
    def from_data_xds(data_xds, gain_opts):

        # Check whether we are dealing with BDA data.
        if hasattr(data_xds, "UPSAMPLED_TIME"):
            time_col = data_xds.UPSAMPLED_TIME.data
            interval_col = data_xds.UPSAMPLED_INTERVAL.data
        else:
            time_col = data_xds.TIME.data
            interval_col = data_xds.INTERVAL.data

        # If SCAN_NUMBER was a partitioning column it will not be present on
        # the dataset - we reintroduce it for cases where we need to ensure
        # solution intervals don't span scan boundaries.
        if "SCAN_NUMBER" in data_xds.data_vars.keys():
            scan_col = data_xds.SCAN_NUMBER.data
        else:
            scan_col = da.zeros_like(
                time_col,
                dtype=np.int32,
                name="scan_number-" + uuid4().hex
            )

        time_interval = gain_opts.time_interval
        respect_scan_boundaries = gain_opts.respect_scan_boundaries

        # These technically depend on the type of the term in question.
        # TODO: How to combine these scaffolds with the Gain objects?
        # TODO: Add in parameterized case.
        time_bins = GainDataset.make_time_bins(
            time_col,
            interval_col,
            scan_col,
            time_interval,
            respect_scan_boundaries
        )

        time_map = GainDataset.make_time_map(
            time_col,
            time_bins
        )

        time_chunks = GainDataset.make_time_chunks(time_bins)

        chan_freqs = data_xds.CHAN_FREQ.data
        chan_widths = data_xds.CHAN_WIDTH.data
        freq_interval = gain_opts.freq_interval

        freq_map = GainDataset.make_freq_map(
            chan_freqs,
            chan_widths,
            freq_interval
        )

        freq_chunks = GainDataset.make_time_chunks(freq_map)

        n_dir = data_xds.dims["dir"]

        # TODO: Move to separate function for consistency.
        if gain_opts.direction_dependent:
            dir_map = da.arange(
                n_dir,
                name="dirmap-" + uuid4().hex,
                dtype=np.int64
            )
        else:
            dir_map = da.zeros(
                n_dir,
                name="dirmap-" + uuid4().hex,
                dtype=np.int64
            )

        gain_times = da.map_blocks(
            GainDataset._make_mean_coord,
            time_col,
            time_bins,
            dtype=time_col.dtype
        )

        gain_freqs = da.map_blocks(
            GainDataset._make_mean_coord,
            chan_freqs,
            freq_map,
            dtype=time_col.dtype
        )

        direction = dir_map if gain_opts.direction_dependent else dir_map[:1]

        partition_schema = data_xds.__daskms_partition_schema__
        id_attrs = {f: data_xds.attrs[f] for f, _ in partition_schema}

        lazy_dataset = {
            "coords": {
                "gain_time": (("gain_time",), gain_times),
                "gain_freq": (("gain_freq",), gain_freqs),
                "time_chunk": (("time_chunk",), time_chunks),
                "freq_chunk": (("freq_chunk",), freq_chunks),
                "direction": (("direction",), direction),
                "antenna": (("antenna",), data_xds.ant.data),
                "correlation": (("correlation",), data_xds.corr.data)
            },
            "attrs": {
                **id_attrs,
                "FIELD_NAME": data_xds.FIELD_NAME,
                "TYPE": gain_opts.type
            }
        }

        # import dask
        # dask.visualize(
        #     lazy_dataset,
        #     filename='graph.pdf',
        #     optimize_graph=True
        # )
        # import ipdb; ipdb.set_trace()

        return lazy_dataset

    @staticmethod
    def _make_mean_coord(values, bins):

        unique_values = np.unique(values)

        sums = np.zeros(bins.max() + 1, dtype=np.float64)
        counts = np.zeros_like(sums, dtype=np.int64)

        np.add.at(sums, bins, unique_values)
        np.add.at(counts, bins, 1)

        return sums / counts

    @staticmethod
    def make_time_bins(
        time_col,
        interval_col,
        scan_col,
        time_interval,
        respect_scan_boundaries
    ):

        time_bins = da.map_blocks(
            GainDataset._make_time_bins,
            time_col,
            interval_col,
            scan_col,
            time_interval,
            respect_scan_boundaries,
            dtype=np.int64
        )

        return time_bins

    @staticmethod
    def _make_time_bins(
        time_col,
        interval_col,
        scan_col,
        time_interval,
        respect_scan_boundaries
    ):

        utime, utime_ind = np.unique(time_col, return_index=True)

        utime_intervals = interval_col[utime_ind]
        utime_scans = scan_col[utime_ind]

        time_bins = TERM_TYPES["complex"].make_t_bins(
            utime.size,
            utime_intervals,
            utime_scans,
            time_interval,
            respect_scan_boundaries
        )

        return time_bins

    @staticmethod
    def make_time_map(time_col, time_bins):

        time_map = da.map_blocks(
            GainDataset._make_time_map,
            time_col,
            time_bins,
            dtype=np.int64
        )

        return time_map

    @staticmethod
    def _make_time_map(time_col, time_bins):

        _, utime_inv = np.unique(time_col, return_inverse=True)

        return time_bins[utime_inv]

    @staticmethod
    def make_freq_map(chan_freqs, chan_widths, freq_interval):

        freq_map = da.map_blocks(
            GainDataset._make_freq_map,
            chan_freqs,
            chan_widths,
            freq_interval,
            dtype=np.int64
        )

        return freq_map

    @staticmethod
    def _make_freq_map(chan_freqs, chan_widths, freq_interval):

        freq_map = TERM_TYPES["complex"].make_f_maps(
            chan_freqs,
            chan_widths,
            freq_interval,
        )

        return freq_map

    @staticmethod
    def make_time_chunks(time_bins):

        time_chunks = da.map_blocks(
            lambda arr: np.array(arr.max() + 1),
            time_bins,
            chunks=(1,),
            dtype=np.int64
        )

        return time_chunks

    @staticmethod
    def make_freq_chunks(freq_map):

        freq_chunks = da.map_blocks(
            lambda arr: np.array(arr.max() + 1),
            freq_map,
            chunks=(1,),
            dtype=np.int64
        )

        return freq_chunks
