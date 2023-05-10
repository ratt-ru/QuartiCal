# -*- coding: utf-8 -*-
from loguru import logger  # noqa
import numpy as np
import dask.array as da
import xarray
from copy import deepcopy
from daskms.experimental.zarr import xds_from_zarr
from quartical.gains.conversion import Converter
from quartical.interpolation.interpolants import (
    interpolate_missing,
    linear2d_interpolate_gains,
    spline2d_interpolate_gains
)
from quartical.gains.datasets import write_gain_datasets


def load_and_interpolate_gains(gain_xds_lod, chain, output_directory):
    """Load and interpolate gains in accordance with the chain.

    Given the gain datasets which are to be applied/solved for, determine
    whether any are to be loaded from disk. Interpolates on-disk datasets
    to be consistent with the solvable datasets.

    Args:
        gain_xds_lod: List of dicts of xarray.Datasets containing gains.
        chain: A list of Gain objects.

    Returns:
        A list like gain_xds_list with the relevant gains loaded from disk.
    """

    interpolated_xds_lol = []

    for term in chain:

        term_name = term.name
        term_path = term.load_from

        # Pull out all the datasets for the current term into a flat list.
        term_xds_list = [term_dict[term_name] for term_dict in gain_xds_lod]

        # If the gain_path is None, this term doesn't require loading/interp.
        if term_path is None:
            interpolated_xds_lol.append(term_xds_list)
            continue
        else:
            load_path = "::".join(term_path.rsplit('/', 1))

        load_xds_list = xds_from_zarr(load_path)
        load_type = {xds.TYPE for xds in load_xds_list}

        if len(load_type) != 1:
            raise ValueError(
                f"Input gain dataset at {term_path} contains multiple term "
                f"types. This should be impossible and may indicate a bug."
            )
        else:
            load_type = load_type.pop()

        assert load_type == term.type, (
            f"Attempted to load {load_type} term as {term.type} term - "
            f"this behaviour is not supported. Please check {term.name}.type."
        )

        # Choose interpolation targets based on term description. TODO: This
        # may be a little simplistic in the general case. Add to datasets?
        targets = (
            ["params", "param_flags"] if hasattr(term, "param_axes")
            else ["gains", "gain_flags"]
        )

        try:
            merged_xds = xarray.combine_by_coords(
                [xds[targets] for xds in load_xds_list],
                combine_attrs='drop_conflicts'
            )
        except ValueError:
            # If we have overlapping SPWs, the above will fail.
            # TODO: How do we want to handle this case?
            raise ValueError(
                "Overlapping SPWs currently unsupported. Please consider "
                "splitting your data such that no SPWs overlap."
            )

        # Remove time/chan chunking and rechunk by antenna.
        merged_xds = merged_xds.chunk({**merged_xds.dims, "antenna": 1})

        # Create a converter object to handle moving between native and
        # interpolation representations.
        converter = Converter(term)

        # Convert standard representation to a representation which can
        # be interpolated. Replace flags with NaNs.
        merged_xds = convert_native_to_interp(merged_xds, converter)

        # Interpolate onto the given grids. TODO: Add back spline support.
        interpolated_xds_list = [
            interpolate(merged_xds, xds, term) for xds in term_xds_list
        ]

        # Convert from representation which can be interpolated back into
        # native representation.
        interpolated_xds_list = [
            convert_interp_to_native(xds, converter)
            for xds in interpolated_xds_list
        ]

        # Make the interpolated xds consistent with the current run.
        interpolated_xds_list = [
            reindex_and_rechunk(ixds, rxds)
            for ixds, rxds in zip(interpolated_xds_list, term_xds_list)
        ]

        interpolated_xds_lol.append(interpolated_xds_list)

    # This converts the interpolated list of lists into a list of dicts.
    term_names = [t.name for t in chain]

    interpolated_xds_lod = [
        {tn: term for tn, term in zip(term_names, terms)}
        for terms in zip(*interpolated_xds_lol)
    ]

    # This triggers an early compute and replaces the complicated interpolation
    # graph with simple reads from disk.
    interpolated_xds_lod = compute_and_reload(
        output_directory, interpolated_xds_lod
    )

    return interpolated_xds_lod


def convert_native_to_interp(xds, converter):

    parameterized = converter.parameterized
    data_field = "params" if parameterized else "gains"
    flag_field = "param_flags" if parameterized else "gain_flags"

    params = converter.convert(xds[data_field].data)
    param_flags = xds[flag_field].data

    params = da.where(param_flags[..., None], np.nan, params)

    param_dims = xds[data_field].dims[:-1] + ('parameter',)

    interpable_xds = xarray.Dataset(
        {
            "params": (param_dims, params),
            "param_flags": (param_dims[:-1], param_flags)
        },
        coords=xds.coords,
        attrs=xds.attrs
    )

    return interpable_xds


def convert_interp_to_native(xds, converter):

    data_field = "params" if converter.parameterized else "gains"

    native = converter.revert(xds.params.data)

    dims = getattr(xds, 'PARAM_AXES', xds.GAIN_AXES)

    native_xds = xarray.Dataset(
        {
            data_field: (dims, native),
        },
        coords=xds.coords,
        attrs=xds.attrs
    )

    return native_xds


def interpolate(source_xds, target_xds, term):

    filled_params = interpolate_missing(source_xds.params)

    source_xds = source_xds.assign(
        {"params": (source_xds.params.dims, filled_params.data)}
    )

    if term.interp_method == "2dlinear":
        interpolated_xds = linear2d_interpolate_gains(source_xds, target_xds)
    elif term.interp_method == "2dspline":
        interpolated_xds = spline2d_interpolate_gains(source_xds, target_xds)
    else:
        raise ValueError(
            f"Unknown interpolation mode {term.interp_method} on {term.name}."
        )

    axes = getattr(target_xds, "PARAM_AXES", target_xds.GAIN_AXES)

    interpolated_xds = interpolated_xds.assign_coords(
        {axes[-1]: source_xds[axes[-1]]}
    )

    return interpolated_xds


def reindex_and_rechunk(interpolated_xds, reference_xds):

    spec = getattr(reference_xds, "PARAM_SPEC", reference_xds.GAIN_SPEC)
    axes = getattr(reference_xds, "PARAM_AXES", reference_xds.GAIN_AXES)

    axis = axes[-1]

    # If we are loading a term with a differing number of correlations,
    # this should handle selecting them out/padding them in.
    if interpolated_xds.dims[axis] < reference_xds.dims[axis]:
        interpolated_xds = interpolated_xds.reindex(
            {"correlation": reference_xds[axis]}, fill_value=0
        )
    elif interpolated_xds.dims[axis] > reference_xds.dims[axis]:
        interpolated_xds = interpolated_xds.sel(
            {"correlation": reference_xds[axis]}
        )

    # We may be interpolating from one set of axes to another.
    t_t_axis, t_f_axis = axes[:2]

    t_chunks = spec.tchunk
    f_chunks = spec.fchunk

    interpolated_xds = interpolated_xds.chunk(
        {
            t_t_axis: t_chunks,
            t_f_axis: f_chunks,
            "antenna": interpolated_xds.dims["antenna"]
        }
    )

    return interpolated_xds


def compute_and_reload(directory, gain_xds_lod):
    """Reread gains datasets to be consistent with the reference datasets."""

    writes = write_gain_datasets(gain_xds_lod, directory)
    # NOTE: Need to set compute calls up using dask config mechansim.
    da.compute(writes)

    # NOTE: This avoids mutating the inputs i.e. avoids side-effects.
    # TODO: Is this computationally expensive?
    gain_xds_lod = deepcopy(gain_xds_lod)

    gain_names = [g for g in gain_xds_lod[0].keys()]

    gain_xds_dol = {g: [d[g] for d in gain_xds_lod] for g in gain_names}

    reference_chunks = {
        g: [dict(xds.chunks) for xds in xdsl]
        for g, xdsl in gain_xds_dol.items()
    }

    loaded_terms = {
        g: xds_from_zarr(
            directory + f"::{g}",
            chunks=reference_chunks[g]
        ) for g in gain_names
    }

    for lxdsl, rxdsl in zip(loaded_terms.values(), gain_xds_dol.values()):
        for lxds, rxds in zip(lxdsl, rxdsl):
            for k in lxds.data_vars.keys():
                rxds[k] = lxds[k]

    return gain_xds_lod
