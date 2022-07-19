# -*- coding: utf-8 -*-
from loguru import logger  # noqa
import dask.array as da
import numpy as np
import xarray
from daskms.experimental.zarr import xds_from_zarr
from quartical.config.internal import yield_from
from quartical.interpolation.interpolants import (interpolate_missing,
                                                  linear2d_interpolate_gains,
                                                  spline2d_interpolate_gains)


def load_and_interpolate_gains(gain_xds_lod, chain_opts):
    """Load and interpolate gains in accordance with chain_opts.

    Given the gain datasets which are to be applied/solved for, determine
    whether any are to be loaded from disk. Interpolates on-disk datasets
    to be consistent with the solvable datasets.

    Args:
        gain_xds_lod: List of dicts of xarray.Datasets containing gains.
        chain_opts: A Chain config object.

    Returns:
        A list like gain_xds_list with the relevant gains loaded from disk.
    """

    interp_xds_lol = []

    req_fields = ("load_from", "interp_mode", "interp_method")

    for loop_vars in yield_from(chain_opts, req_fields):

        term_name, term_path, interp_mode, interp_method = loop_vars

        # Pull out all the datasets for the current term into a flat list.
        term_xds_list = [term_dict[term_name] for term_dict in gain_xds_lod]

        # If the gain_path is None, this term doesn't require loading/interp.
        if term_path is None:
            interp_xds_lol.append(term_xds_list)
            continue
        else:
            load_path = "::".join(term_path.rsplit('/', 1))

        load_xds_list = xds_from_zarr(load_path)

        # Ensure that no axes are chunked at this point.
        load_xds_list = [xds.chunk(-1) for xds in load_xds_list]

        # Convert to amp and phase/real and imag. Drop unused data_vars.
        converted_xds_list = convert_and_drop(load_xds_list, interp_mode)

        # Sort the datasets on disk into a list of lists, ordered by time
        # and frequency.
        sorted_xds_lol = sort_datasets(converted_xds_list)

        # Figure out which datasets need to be concatenated.
        concat_xds_list = make_concat_xds_list(term_xds_list,
                                               sorted_xds_lol)

        # Form up list of datasets with interpolated values.
        interp_xds_list = make_interp_xds_list(term_xds_list,
                                               concat_xds_list,
                                               interp_mode,
                                               interp_method)

        interp_xds_lol.append(interp_xds_list)

    # This converts the interpolated list of lists into a list of dicts.
    term_names = [tn for tn in yield_from(chain_opts)]

    interp_xds_lod = [{tn: term for tn, term in zip(term_names, terms)}
                      for terms in zip(*interp_xds_lol)]

    return interp_xds_lod


def convert_and_drop(load_xds_list, interp_mode):
    """Convert complex gain reim/ampphase. Drop unused data_vars."""

    converted_xds_list = []

    for load_xds in load_xds_list:

        dims = load_xds.gains.dims

        if interp_mode == "ampphase":
            # Convert the complex gain into amplitide and phase.
            converted_xds = load_xds.assign(
                {"phase": (dims, da.angle(load_xds.gains.data)),
                 "amp": (dims, da.absolute(load_xds.gains.data))})
            keep_vars = {"phase", "amp", "gain_flags"}
        elif interp_mode == "reim":
            # Convert the complex gain into its real and imaginary parts.
            converted_xds = load_xds.assign(
                {"re": (dims, load_xds.gains.data.real),
                 "im": (dims, load_xds.gains.data.imag)})
            keep_vars = {"re", "im", "gain_flags"}

        # Drop the unecessary dims and data vars. TODO: At present, QuartiCal
        # will always interpolate a gain, not the parameters. This makes it
        # impossible to do a further solve on a parameterised term.
        drop_dims = set(converted_xds.dims) - set(converted_xds.GAIN_AXES)
        converted_xds = converted_xds.drop_dims(drop_dims)
        drop_vars = set(converted_xds.data_vars) - keep_vars
        converted_xds = converted_xds.drop_vars(drop_vars)

        converted_xds_list.append(converted_xds)

    return converted_xds_list


def sort_datasets(load_xds_list):
    """Sort the loaded datasets by time and frequency."""

    # We want to sort according to the gain axes. TODO: Parameterised case?
    t_axis, f_axis = load_xds_list[0].GAIN_AXES[:2]

    time_lb = [xds[t_axis].values[0] for xds in load_xds_list]
    freq_lb = [xds[f_axis].values[0] for xds in load_xds_list]

    n_utime_lb = len(set(time_lb))  # Number of unique lower time bounds.
    n_ufreq_lb = len(set(freq_lb))  # Number of unique lower freq bounds.

    # Sort by the lower bounds of the time and freq axes.
    sort_ind = np.lexsort([freq_lb, time_lb])

    # Reshape the indices so we can split the time and frequency axes.
    try:
        sort_ind = sort_ind.reshape((n_utime_lb, n_ufreq_lb))
    except ValueError as e:
        raise ValueError(f"Gains on disk do not lie on a grid - "
                         f"interpolation not possible. Python error: {e}.")

    sorted_xds_lol = [[load_xds_list[sort_ind[i, j]]
                      for j in range(n_ufreq_lb)]
                      for i in range(n_utime_lb)]

    return sorted_xds_lol


def domain_slice(lb, ub, lbounds, ubounds):
    """Create a slice corresponding to the neighbourhood of domain (lb, ub)."""

    if any(lb >= lbounds):
        slice_lb = len(lbounds) - (lb >= lbounds)[::-1].argmax() - 1
    else:
        slice_lb = 0  # Entirely below input domain.

    if any(ub <= ubounds):
        slice_ub = (ub <= ubounds).argmax()
    else:
        slice_ub = len(ubounds) - 1  # Entirely above input domain.

    return slice(slice_lb, slice_ub + 1)  # Non-inclusive, hence +1.


def make_concat_xds_list(term_xds_list, sorted_xds_lol):
    """Map datasets on disk to the dataset required for calibration."""

    # We want to use the axes of the gain on disk. TODO: Parameterised case?
    ld_t_axis, ld_f_axis = sorted_xds_lol[0][0].GAIN_AXES[:2]

    # Figure out the upper and lower time bounds of each dataset.
    time_lbounds = [sl[0][ld_t_axis].values[0] for sl in sorted_xds_lol]
    time_ubounds = [sl[0][ld_t_axis].values[-1] for sl in sorted_xds_lol]

    # Figure out the upper and lower freq bounds of each dataset.
    freq_lbounds = [xds[ld_f_axis].values[0] for xds in sorted_xds_lol[0]]
    freq_ubounds = [xds[ld_f_axis].values[-1] for xds in sorted_xds_lol[0]]

    concat_xds_list = []

    for term_xds in term_xds_list:

        t_axis, f_axis = term_xds.GAIN_AXES[:2]

        tlb = term_xds[t_axis].data[0]
        tub = term_xds[t_axis].data[-1]
        flb = term_xds[f_axis].data[0]
        fub = term_xds[f_axis].data[-1]

        concat_tslice = domain_slice(tlb, tub, time_lbounds, time_ubounds)
        concat_fslice = domain_slice(flb, fub, freq_lbounds, freq_ubounds)

        fconcat_xds_list = []

        for xds_list in sorted_xds_lol[concat_tslice]:
            fconcat_xds_list.append(xarray.concat(xds_list[concat_fslice],
                                                  ld_f_axis,
                                                  join="exact"))

        # Concatenate gains near the interpolation values.
        concat_xds = xarray.concat(fconcat_xds_list,
                                   ld_t_axis,
                                   join="exact")

        # Remove the chunking from the concatenated datasets.
        concat_xds = concat_xds.chunk({ld_t_axis: -1, ld_f_axis: -1})

        concat_xds_list.append(concat_xds)

    return concat_xds_list


def make_interp_xds_list(term_xds_list, concat_xds_list, interp_mode,
                         interp_method):
    """Given the concatenated datasets, interp to the desired datasets."""

    interp_xds_list = []

    for term_xds, concat_xds in zip(term_xds_list, concat_xds_list):

        if interp_mode == "ampphase":
            amp_sel = da.where(concat_xds.gain_flags.data[..., None],
                               np.nan,
                               concat_xds.amp.data)

            phase_sel = da.where(concat_xds.gain_flags.data[..., None],
                                 np.nan,
                                 concat_xds.phase.data)

            interp_xds = concat_xds.assign(
                {"amp": (concat_xds.amp.dims, amp_sel),
                 "phase": (concat_xds.phase.dims, phase_sel)})

        elif interp_mode == "reim":
            re_sel = da.where(concat_xds.gain_flags.data[..., None],
                              np.nan,
                              concat_xds.re.data)

            im_sel = da.where(concat_xds.gain_flags.data[..., None],
                              np.nan,
                              concat_xds.im.data)

            interp_xds = concat_xds.assign(
                {"re": (concat_xds.re.dims, re_sel),
                 "im": (concat_xds.im.dims, im_sel)})

        interp_xds = interp_xds.drop_vars("gain_flags")

        # This fills in missing values using linear interpolation, or by
        # padding with the last good value (edges). Regions with no good data
        # will be zeroed.
        interp_xds = interpolate_missing(interp_xds)

        # Interpolate with various methods.
        if interp_method == "2dlinear":
            interp_xds = linear2d_interpolate_gains(interp_xds, term_xds)
        elif interp_method == "2dspline":
            interp_xds = spline2d_interpolate_gains(interp_xds, term_xds)

        # If we are loading a term with a differing number of correlations,
        # this should handle selecting them out/padding them in.
        if interp_xds.dims["corr"] < term_xds.dims["corr"]:
            interp_xds = \
                interp_xds.reindex({"corr": term_xds.corr}, fill_value=0)
        elif interp_xds.dims["corr"] > term_xds.dims["corr"]:
            interp_xds = interp_xds.sel({"corr": term_xds.corr})

        # Convert the interpolated quantities back to gains.
        if interp_mode == "ampphase":
            gains = interp_xds.amp.data*da.exp(1j*interp_xds.phase.data)
            interp_xds = term_xds.assign(
                {"gains": (term_xds.GAIN_AXES, gains)}
            )
        elif interp_mode == "reim":
            gains = interp_xds.re.data + 1j*interp_xds.im.data
            interp_xds = term_xds.assign(
                {"gains": (term_xds.GAIN_AXES, gains)}
            )

        t_chunks = term_xds.GAIN_SPEC.tchunk
        f_chunks = term_xds.GAIN_SPEC.fchunk

        # We may be interpolating from one set of axes to another.
        t_t_axis, t_f_axis = term_xds.GAIN_AXES[:2]

        interp_xds = interp_xds.chunk({t_t_axis: t_chunks, t_f_axis: f_chunks})

        interp_xds_list.append(interp_xds)

    return interp_xds_list
