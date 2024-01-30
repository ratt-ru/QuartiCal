# -*- coding: utf-8 -*-
from quartical.config.external import Gain
from loguru import logger  # noqa
import numpy as np
import dask.array as da
import xarray
from uuid import uuid4
from daskms.experimental.zarr import xds_to_zarr
from quartical.gains.gain import gain_spec_tup, param_spec_tup
from quartical.gains import TERM_TYPES
from quartical.gains.general.generics import combine_gains, combine_flags


def make_gain_xds_lod(data_xds_list, chain):
    """Returns a list of dicts of xarray.Dataset objects describing the gains.

    For a given input xds containing data, creates an xarray.Dataset object
    per term which describes the term's dimensions.

    Args:
        data_xds_list: A list of xarray.Dataset objects containing MS data.
        chains: A list of Gain objects.

    Returns:
        gain_xds_lod: A List of Dicts of xarray.Dataset objects describing the
            gain terms assosciated with each data xarray.Dataset.
    """

    scaffolds_per_xds = []

    for data_xds in data_xds_list:

        gain_scaffolds = {}

        for gain_obj in chain:

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
            if "PARAM_SPEC" in gain_xds.attrs:
                gain_xds.attrs["PARAM_SPEC"] = \
                    param_spec_tup(*list(map(tuple, gain_xds.PARAM_SPEC)))

    return gain_xds_lod


def make_net_xds_lod(data_xds_list, chain, output_opts):
    """Construct a list of dicts of xarray.Datasets to house the net gains.

    Args:
        data_xds_list: A List of xarray.Dataset objects containing MS data.
        output_opts: An output config object.

    Returns:
        net_gain_xds_dol: A Dict of Lists of xarray.Dataset objects to house
            the net gains.
    """

    net_names = [f"{''.join(lt)}-net" for lt in output_opts.net_gains]

    direction_dependent = any(t.direction_dependent for t in chain)

    net_configs = [
        Gain(direction_dependent=direction_dependent) for _ in net_names
    ]

    net_chain = [
        TERM_TYPES["complex"](n, c) for n, c in zip(net_names, net_configs)
    ]

    return make_gain_xds_lod(data_xds_list, net_chain)


def combine_gains_wrapper(
    net_shape,
    corr_mode,
    *args
):
    """Wrapper to stop dask from getting confused. See issue #99."""

    gains = tuple(args[::4])
    time_maps = tuple(args[1::4])
    freq_maps = tuple(args[2::4])
    dir_maps = tuple(args[3::4])

    return combine_gains(
        gains,
        time_maps,
        freq_maps,
        dir_maps,
        net_shape,
        corr_mode,
    )


def combine_flags_wrapper(
    net_shape,
    *args
):
    """Wrapper to stop dask from getting confused. See issue #99."""

    flags = tuple(args[::4])
    time_bins = tuple(args[1::4])
    freq_maps = tuple(args[2::4])
    dir_maps = tuple(args[3::4])

    return combine_flags(
        flags,
        time_bins,
        freq_maps,
        dir_maps,
        net_shape
    )


def populate_net_xds_list(
    net_gain_xds_lod,
    solved_gain_xds_lod,
    mapping_xds_list,
    output_opts
):
    """Populate the list net gain datasets with net gain values.

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

    # TODO: Can tenchically get this from the datasets.
    net_names = [f"{''.join(lt)}-net" for lt in net_terms]

    net_map = dict(zip(net_names, net_terms))

    populated_net_gain_xds_lod = []

    # TODO: Move into flag combining function?
    identity_elements = {
        1: np.ones(1, dtype=np.complex128),
        2: np.ones(2, dtype=np.complex128),
        4: np.array((1, 0, 0, 1), dtype=np.complex128)
    }

    itr = zip(solved_gain_xds_lod, net_gain_xds_lod, mapping_xds_list)

    for solved_gains, net_gains, mapping_xds in itr:

        net_xds_dict = {}

        for net_name, req_terms in net_map.items():

            req_time_bins = tuple(
                [mapping_xds.get(f"{k}_time_bins").data for k in req_terms]
            )
            req_freq_maps = tuple(
                [mapping_xds.get(f"{k}_freq_map").data for k in req_terms]
            )
            req_dir_maps = tuple(
                [mapping_xds.get(f"{k}_dir_map").data for k in req_terms]
            )

            net_xds = net_gains[net_name]

            net_shape = tuple(net_xds.sizes[d] for d in net_xds.GAIN_AXES)

            corr_mode = net_shape[-1]

            req_term_gains = [solved_gains[t].gains for t in req_terms]
            req_term_flags = [solved_gains[t].gain_flags for t in req_terms]

            itr = zip(
                req_term_gains,
                req_time_bins,
                req_freq_maps,
                req_dir_maps
            )

            req_args = []

            for g, tb, fm, dm in itr:
                req_args.extend([g.data, g.dims])
                req_args.extend([tb, ("gain_time",)])
                req_args.extend([fm, ("gain_freq",)])
                req_args.extend([dm, ("direction",)])

            net_gain = da.blockwise(
                combine_gains_wrapper, net_xds.GAIN_AXES,
                net_shape, None,
                corr_mode, None,
                *req_args,
                dtype=np.complex128,
                align_arrays=False,
                concatenate=True,
                adjust_chunks={
                    "gain_time": net_xds.GAIN_SPEC.tchunk,
                    "gain_freq": net_xds.GAIN_SPEC.fchunk,
                    "direction": net_xds.GAIN_SPEC.dchunk
                }
            )

            itr = zip(
                req_term_flags,
                req_time_bins,
                req_freq_maps,
                req_dir_maps
            )

            req_args = []

            for f, tb, fm, dm in itr:
                req_args.extend([f.data, f.dims])
                req_args.extend([tb, ("gain_time",)])
                req_args.extend([fm, ("gain_freq",)])
                req_args.extend([dm, ("direction",)])

            net_flags = da.blockwise(
                combine_flags_wrapper, net_xds.GAIN_AXES[:-1],
                net_shape, None,
                *req_args,
                dtype=np.int8,
                align_arrays=False,
                concatenate=True,
                adjust_chunks={
                    "gain_time": net_xds.GAIN_SPEC.tchunk,
                    "gain_freq": net_xds.GAIN_SPEC.fchunk,
                    "direction": net_xds.GAIN_SPEC.dchunk
                }
            )

            net_gain = da.blockwise(
                np.where, net_xds.GAIN_AXES,
                net_flags[..., None], net_xds.GAIN_AXES,
                identity_elements[corr_mode], None,
                net_gain, net_xds.GAIN_AXES
            )

            net_xds = net_xds.assign(
                {
                    "gains": (net_xds.GAIN_AXES, net_gain),
                    "gain_flags": (net_xds.GAIN_AXES[:-1], net_flags)
                }
            )

            net_xds_dict[net_name] = net_xds

        populated_net_gain_xds_lod.append(net_xds_dict)

    return populated_net_gain_xds_lod


def write_gain_datasets(gain_xds_lod, directory, net_xds_lod=None):
    """Write the contents of gain_xds_lol to zarr in accordance with opts."""

    term_names = [xds.NAME for xds in gain_xds_lod[0].values()]

    writable_xds_dol = {tn: [d[tn] for d in gain_xds_lod] for tn in term_names}

    # If net gains have been requested, add them to the writes.
    if net_xds_lod:
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

            if "params" in xds.data_vars.keys():
                rechunked_params = xds.params.chunk(
                    {ax: "auto" for ax in xds.PARAM_AXES[:2]}
                )
                target_chunks.update(rechunked_params.chunksizes)

            if "gains" in xds.data_vars.keys():
                rechunked_gains = xds.gains.chunk(
                    {ax: "auto" for ax in xds.GAIN_AXES[:2]}
                )
                target_chunks.update(rechunked_gains.chunksizes)

            rechunked_xds = xds.chunk(target_chunks)

            term_write_xds_list.append(rechunked_xds)

        output_path = f"{directory}::{term_name}"

        gain_writes = xds_to_zarr(
            term_write_xds_list, output_path, rechunk=True
        )

        gain_writes_lol.append(gain_writes)

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

    n_dir = data_xds.sizes["dir"]

    dir_map = gain_obj.make_dir_map(
        n_dir,
        gain_obj.direction_dependent
    )

    gain_times = gain_obj.make_time_coords(time_col, time_bins)
    gain_freqs = gain_obj.make_freq_coords(chan_freqs, freq_map)

    direction = dir_map if gain_obj.direction_dependent else dir_map[:1]
    n_gdir = direction.size

    partition_schema = data_xds.__daskms_partition_schema__
    id_attrs = {f: data_xds.attrs[f] for f, _ in partition_schema}

    # TODO: Move this the the gain object?
    n_corr = data_xds.sizes["corr"]
    n_ant = data_xds.sizes["ant"]
    chunk_spec = gain_spec_tup(
        time_chunks,
        freq_chunks,
        (n_ant,),
        (n_gdir,),
        (n_corr,)
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

    if hasattr(gain_obj, "param_axes"):
        param_time_bins = gain_obj.make_param_time_bins(
            time_col,
            interval_col,
            scan_col,
            time_interval,
            respect_scan_boundaries
        )

        param_time_chunks = gain_obj.make_param_time_chunks(param_time_bins)

        param_freq_map = gain_obj.make_param_freq_map(
            chan_freqs,
            chan_widths,
            freq_interval
        )

        param_freq_chunks = gain_obj.make_param_freq_chunks(param_freq_map)

        param_times = gain_obj.make_time_coords(time_col, param_time_bins)
        param_freqs = gain_obj.make_freq_coords(chan_freqs, param_freq_map)

        param_names = gain_obj.make_param_names(data_xds.corr.data)
        n_param = len(param_names)

        param_chunk_spec = param_spec_tup(
           param_time_chunks,
           param_freq_chunks,
           (n_ant,),
           (n_gdir,),
           (n_param,)
        )

        scaffold["coords"].update(
            {
                "param_time": (("param_time",), param_times),
                "param_freq": (("param_freq",), param_freqs),
                "param_name": (("param_name",), param_names)
            }
        )
        scaffold["attrs"].update(
            {
                "PARAM_SPEC": param_chunk_spec,
                "PARAM_AXES": gain_obj.param_axes
            }
        )

    return scaffold
