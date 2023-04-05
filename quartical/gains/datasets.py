# -*- coding: utf-8 -*-
from quartical.config.external import Gain
from quartical.config.internal import yield_from
from loguru import logger  # noqa
import numpy as np
import dask.array as da
import xarray
from uuid import uuid4
from daskms.experimental.zarr import xds_to_zarr
from quartical.gains.gain import gain_spec_tup, param_spec_tup
from quartical.gains import TERM_TYPES
from quartical.gains.general.generics import combine_gains, combine_flags


def make_gain_xds_lod(data_xds_list, chain_opts):
    """Returns a list of dicts of xarray.Dataset objects describing the gains.

    For a given input xds containing data, creates an xarray.Dataset object
    per term which describes the term's dimensions.

    Args:
        data_xds_list: A list of xarray.Dataset objects containing MS data.
        chain_opts: A Chain config object.

    Returns:
        gain_xds_lod: A List of Dicts of xarray.Dataset objects describing the
            gain terms assosciated with each data xarray.Dataset.
    """

    gain_obj_list = [
        TERM_TYPES[tt](tn, getattr(chain_opts, tn))
        for tn, tt in yield_from(chain_opts, "type")
    ]

    scaffolds_per_xds = []

    for data_xds in data_xds_list:

        gain_scaffolds = {}

        for gain_obj in gain_obj_list:

            gain_scaffolds[gain_obj.name] = \
                scaffold_from_data_xds(data_xds, gain_obj)

        scaffolds_per_xds.append(gain_scaffolds)

    # NOTE: This triggers an early compute to determine all the chunking info.
    gain_xds_lod = [
        {
            tn: xarray.Dataset(**ts) for tn, ts in gain_scaffolds.items()
        }
        for gain_scaffolds in da.compute(scaffolds_per_xds)[0]
    ]

    # This is a nasty workaround for the one problem with the scaffolds -
    # the chunk specs coming out containing arrays.
    # TODO: Think about a neater approach.
    for gain_xdss in gain_xds_lod:
        for _, gain_xds in gain_xdss.items():
            gain_xds.attrs["GAIN_SPEC"] = \
                gain_spec_tup(*list(map(tuple, gain_xds.GAIN_SPEC)))

    return gain_xds_lod


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
        output_opts: An output configuration object.

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


def scaffold_from_data_xds(data_xds, gain_obj):
    """Produces a scaffold (xarray.Dataset coords and attrs).

    Given an xarray.Dataset containing measurement set data and a gain object,
    produces a scaffold for an xarray.Dataset to hold the gains. A scaffold is
    the necessary coords and attributes represented as dask arrays in a
    dictionary. Once computed, these can be passed to the xarray.Dataset
    constructor.

    Args:
        data_xds: An xarray.Dataset containing measurement set data.
        gain_obj: An Gain object providing configuration information.

    Returns:
        scaffold: A dictionary of dask.arrays which can be used to construct
            an xarray.Dataset.
    """

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

    time_interval = gain_obj.time_interval
    respect_scan_boundaries = gain_obj.respect_scan_boundaries

    # TODO: Add in parameterized case.
    time_bins = gain_obj.make_time_bins(
        time_col,
        interval_col,
        scan_col,
        time_interval,
        respect_scan_boundaries
    )

    time_chunks = gain_obj.make_time_chunks(time_bins)

    chan_freqs = data_xds.CHAN_FREQ.data
    chan_widths = data_xds.CHAN_WIDTH.data
    freq_interval = gain_obj.freq_interval

    freq_map = gain_obj.make_freq_map(
        chan_freqs,
        chan_widths,
        freq_interval
    )

    freq_chunks = gain_obj.make_freq_chunks(freq_map)

    n_dir = data_xds.dims["dir"]

    dir_map = gain_obj.make_dir_map(
        n_dir,
        gain_obj.direction_dependent
    )

    gain_times = gain_obj.make_time_coords(time_col, time_bins)
    gain_freqs = gain_obj.make_freq_coords(chan_freqs, freq_map)

    direction = dir_map if gain_obj.direction_dependent else dir_map[:1]

    partition_schema = data_xds.__daskms_partition_schema__
    id_attrs = {f: data_xds.attrs[f] for f, _ in partition_schema}

    # TODO: Move this the the gain object?
    n_corr = data_xds.dims["corr"]
    n_ant = data_xds.dims["ant"]
    chunk_spec = gain_spec_tup(
        time_chunks, freq_chunks, (n_ant,), (n_dir,), (n_corr,)
    )

    scaffold = {
        "coords": {
            "gain_time": (("gain_time",), gain_times),
            "gain_freq": (("gain_freq",), gain_freqs),
            "time_chunk": (("time_chunk",), da.arange(time_chunks.size)),
            "freq_chunk": (("freq_chunk",), da.arange(freq_chunks.size)),
            "direction": (("direction",), direction),
            "antenna": (("antenna",), data_xds.ant.data),
            "correlation": (("correlation",), data_xds.corr.data)
        },
        "attrs": {
            **id_attrs,
            "FIELD_NAME": data_xds.FIELD_NAME,
            "NAME": gain_obj.name,
            "TYPE": gain_obj.type,
            "GAIN_SPEC": chunk_spec,
            "GAIN_AXES": gain_obj.gain_axes
        }
    }

    return scaffold
