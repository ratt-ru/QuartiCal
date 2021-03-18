# -*- coding: utf-8 -*-
from loguru import logger  # noqa
import dask.array as da
import numpy as np
import xarray
import pathlib
from quartical.interpolation.interpolants import (interpolate_missing,
                                                  spline2d_interpolate_gains,
                                                  csaps2d_interpolate_gains)


def load_and_interpolate_gains(gain_xds_list, opts):
    """Load and interpolate gains in accordance with opts.

    Given the gain datasets which are to be applied/solved for, determine
    whether any are to be loaded from disk. Interpolates on-disk datasets
    to be consistent with the solvable datasets.

    Args:
        gain_xds_list: List of xarray.Datasets containing gains.
        opts: A Namespace of globla options.

    Returns:
        A list like gain_xds_list with the relevant gains loaded from disk.
    """

    interp_xds_lol = []

    for term_ind, term in enumerate(opts.solver_gain_terms):

        gain_path = getattr(opts, f"{term}_load_from", None)
        interp_mode = getattr(opts, f"{term}_interp_mode", None)
        interp_method = getattr(opts, f"{term}_interp_method", None)

        # Pull out all the datasets for the current term into a flat list.
        term_xds_list = [tlist[term_ind] for tlist in gain_xds_list]

        # If the gain_path is None, this term doesn't require loading/interp.
        if gain_path is None:
            interp_xds_lol.append(term_xds_list)
            continue
        else:
            gain_path = pathlib.Path(gain_path)

        load_paths = gain_path.glob(f"{gain_path.stem}*")

        load_xds_list = [xarray.open_zarr(pth) for pth in load_paths]

        # Convert to amp and phase/real and imag. Drop unused data_vars.
        converted_xds_list = convert_and_drop(load_xds_list, interp_mode)

        # Sort the datasets on disk into a list of lists, ordered by time
        # and frequency.
        sorted_xds_lol = sort_datasets(converted_xds_list)

        # Figure out which datasets need to be concatenated. TODO: This may
        # slightly overconcatenate.
        concat_xds_list = make_concat_xds_list(term_xds_list,
                                               sorted_xds_lol)

        interp_xds_list = make_interp_xds_list(term_xds_list,
                                               concat_xds_list,
                                               interp_mode,
                                               interp_method)

        interp_xds_lol.append(interp_xds_list)

    # This return reverts the datasets to the expected ordering/structure.

    return [list(xds_list) for xds_list in zip(*interp_xds_lol)]


def convert_and_drop(load_xds_list, interp_mode):
    """Convert complex gain into amplitude and phase. Drop unused data_vars."""

    converted_xds_list = []

    for load_xds in load_xds_list:

        dims = load_xds.gains.dims

        if interp_mode == "ampphase":
            # Convert the complex gain into amplitide and phase.
            converted_xds = load_xds.assign(
                {"phase": (dims, da.angle(load_xds.gains.data)),
                 "amp": (dims, da.absolute(load_xds.gains.data))})
        elif interp_mode == "reim":
            # Convert the complex gain into amplitide and phase.
            converted_xds = load_xds.assign(
                {"re": (dims, load_xds.gains.data.real),
                 "im": (dims, load_xds.gains.data.imag)})

        # Drop the unecessary dims and data vars. TODO: At present, QuartiCal
        # will always interpolate a gain, not the parameters. This makes it
        # impossible to do a further solve on a parameterised term.
        drop_dims = set(converted_xds.dims) - set(converted_xds.GAIN_AXES)
        converted_xds = converted_xds.drop_dims(drop_dims)
        converted_xds = converted_xds.drop_vars("gains")

        converted_xds_list.append(converted_xds)

    return converted_xds_list


def sort_datasets(load_xds_list):
    """Sort the loaded datasets by time and frequency."""

    # We want to sort according the gain axes. TODO: Parameterised case?
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


def overlap_slice(lb, ub, lbounds, ubounds):
    """Create a slice corresponding to the neighbourhood of domain (lb, ub)."""

    overlaps = ~((ub < lbounds) | (lb > ubounds))

    sel = np.where(overlaps)[0]
    slice_lb = sel[0]
    slice_ub = sel[-1] + 1  # Python indexing is not inclusive.

    # Dilate. If the lower bound is zero, leave as is, else, include lb - 1.
    slice_lb = slice_lb - 1 if slice_lb else slice_lb
    # Upper slice may fall off the end - this is safe.
    slice_ub = slice_ub + 1

    return slice(slice_lb, slice_ub)


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

        concat_tslice = overlap_slice(tlb, tub, time_lbounds, time_ubounds)
        concat_fslice = overlap_slice(flb, fub, freq_lbounds, freq_ubounds)

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

    # TODO: This is dodgy. Ideally we shouldn't be reasoning about the
    # values of the gains - we should just use the gain flags. This is
    # an interim solution.
    for term_xds, concat_xds in zip(term_xds_list, concat_xds_list):

        if interp_mode == "ampphase":
            amp_sel = da.where(concat_xds.amp.data < 1e-6,
                               np.nan,
                               concat_xds.amp.data)

            phase_sel = da.where(concat_xds.amp.data < 1e-6,
                                 np.nan,
                                 concat_xds.phase.data)

            interp_xds = concat_xds.assign(
                {"amp": (concat_xds.amp.dims, amp_sel),
                 "phase": (concat_xds.phase.dims, phase_sel)})
        elif interp_mode == "reim":
            re_sel = da.where((concat_xds.re.data < 1e-6) &
                              (concat_xds.im.data < 1e-6),
                              np.nan,
                              concat_xds.re.data)

            im_sel = da.where((concat_xds.re.data < 1e-6) &
                              (concat_xds.im.data < 1e-6),
                              np.nan,
                              concat_xds.im.data)

            interp_xds = concat_xds.assign(
                {"re": (concat_xds.re.dims, re_sel),
                 "im": (concat_xds.im.dims, im_sel)})

        # This fills in missing values using linear interpolation, or by
        # padding with the last good value (edges). Regions with no good data
        # will be zeroed.
        interp_xds = interpolate_missing(interp_xds)

        # We may be interpolating from one set of axes to another.
        i_t_axis, i_f_axis = interp_xds.GAIN_AXES[:2]
        t_t_axis, t_f_axis = term_xds.GAIN_AXES[:2]

        # Interpolate with various methods.
        if interp_method == "2dlinear":
            interp_xds = interp_xds.interp(
                {i_t_axis: term_xds[t_t_axis].data,
                 i_f_axis: term_xds[t_f_axis].data},
                kwargs={"fill_value": "extrapolate"})
        elif interp_method == "2dspline":
            interp_xds = spline2d_interpolate_gains(interp_xds,
                                                    term_xds)
        elif interp_method == "smoothingspline":
            interp_xds = csaps2d_interpolate_gains(interp_xds,
                                                   term_xds)

        # Convert the interpolated quantities back in gains.
        if interp_mode == "ampphase":
            gains = interp_xds.amp.data*da.exp(1j*interp_xds.phase.data)
            interp_xds = term_xds.assign(
                {"gains": (term_xds.GAIN_AXES, gains)})
        elif interp_mode == "reim":
            gains = interp_xds.re.data + 1j*interp_xds.im.data
            interp_xds = term_xds.assign(
                {"gains": (term_xds.GAIN_AXES, gains)})

        t_chunks = term_xds.GAIN_SPEC.tchunk
        f_chunks = term_xds.GAIN_SPEC.fchunk

        interp_xds = interp_xds.chunk({t_t_axis: t_chunks, t_f_axis: f_chunks})

        interp_xds_list.append(interp_xds)

    return interp_xds_list
