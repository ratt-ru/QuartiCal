# -*- coding: utf-8 -*-
from loguru import logger  # noqa
import dask.array as da
import xarray
from daskms.experimental.zarr import xds_from_zarr
from quartical.interpolation.interpolants import (
    interpolate_missing,
    linear2d_interpolate_gains,
    spline2d_interpolate_gains
)


def load_and_interpolate_gains(gain_xds_lod, chain):
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

        # Convert from on-disk representation to a representation which can
        # be interpolated.
        merged_xds = term.to_interpable(merged_xds)

        import ipdb; ipdb.set_trace()

        # Interpolate onto the given grids.
        interpolated_xds_list = [
            term.interpolate(merged_xds, xds, term) for xds in term_xds_list
        ]

        # Convert from representation which can be interpolated back into
        # native representation.
        interpolated_xds_list = [
            term.from_interpable(xds) for xds in interpolated_xds_list
        ]

        import ipdb; ipdb.set_trace()

        interpolated_xds_lol.append(interpolated_xds_list)

    # This converts the interpolated list of lists into a list of dicts.
    term_names = [t.name for t in chain]

    interpolated_xds_lod = [
        {tn: term for tn, term in zip(term_names, terms)}
        for terms in zip(*interpolated_xds_lol)
    ]

    return interpolated_xds_lod


def make_interpolated_xds_list(
    term_xds_list,
    merged_xds,
    term
):
    """Given the merged dataset, interpolate to the desired datasets."""

    interpolating_xds = interpolating_xds.drop_vars(("gains", "gain_flags"))

    interpolating_xds = interpolate_missing(interpolating_xds)

    for term_xds in term_xds_list:

        # This fills in missing values using linear interpolation, or by
        # padding with the last good value (edges). Regions with no good data
        # will be zeroed.

        # Interpolate with various methods.
        if interp_method == "2dlinear":
            interpolated_xds = linear2d_interpolate_gains(
                interpolating_xds, term_xds
            )
        elif interp_method == "2dspline":
            interpolated_xds = spline2d_interpolate_gains(
                interpolating_xds, term_xds
            )

        req_ncorr = term_xds.dims["correlation"]
        # If we are loading a term with a differing number of correlations,
        # this should handle selecting them out/padding them in.
        if interpolated_xds.dims["correlation"] < req_ncorr:
            interpolated_xds = interpolated_xds.reindex(
                {"correlation": term_xds.corr}, fill_value=0
            )
        elif interpolated_xds.dims["correlation"] > req_ncorr:
            interpolated_xds = interpolated_xds.sel(
                {"correlation": term_xds.corr}
            )

        # Convert the interpolated quantities back to gains.
        if interp_mode in ("ampphase", "amp", "phase"):
            amp = interpolated_xds.amp.data
            phase = interpolated_xds.phase.data
            gains = amp*da.exp(1j*phase)
            interpolated_xds = term_xds.assign(
                {"gains": (term_xds.GAIN_AXES, gains)}
            )
        elif interp_mode == "reim":
            re = interpolated_xds.re.data
            im = interpolated_xds.im.data
            gains = re + 1j*im
            interpolated_xds = term_xds.assign(
                {"gains": (term_xds.GAIN_AXES, gains)}
            )

        t_chunks = term_xds.GAIN_SPEC.tchunk
        f_chunks = term_xds.GAIN_SPEC.fchunk

        # We may be interpolating from one set of axes to another.
        t_t_axis, t_f_axis = term_xds.GAIN_AXES[:2]

        interpolated_xds = interpolated_xds.chunk(
            {
                t_t_axis: t_chunks,
                t_f_axis: f_chunks,
                "antenna": interpolated_xds.dims["antenna"]
            }
        )

        interpolated_xds_list.append(interpolated_xds)

    return interpolated_xds_list
