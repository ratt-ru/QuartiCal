# -*- coding: utf-8 -*-
from loguru import logger  # noqa
import dask.array as da
import numpy as np
import xarray
import pathlib
from scipy.interpolate import interp2d
from csaps import csaps


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

        # Drop the unecessary data vars. TODO: Parametrised case?
        drop_vars = ("gains", "conv_perc", "conv_iter")
        converted_xds = converted_xds.drop_vars(drop_vars)

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

    # We want to sort according the gain axes. TODO: Parameterised case?
    t_axis, f_axis = sorted_xds_lol[0][0].GAIN_AXES[:2]

    # Figure out the upper and lower time bounds of each dataset.
    time_lbounds = [sl[0][t_axis].values[0] for sl in sorted_xds_lol]
    time_ubounds = [sl[0][t_axis].values[-1] for sl in sorted_xds_lol]

    # Figure out the upper and lower freq bounds of each dataset.
    freq_lbounds = [xds[f_axis].values[0] for xds in sorted_xds_lol[0]]
    freq_ubounds = [xds[f_axis].values[-1] for xds in sorted_xds_lol[0]]

    concat_xds_list = []

    for term_xds in term_xds_list:

        tlb = term_xds[t_axis].data[0]
        tub = term_xds[t_axis].data[-1]
        flb = term_xds[f_axis].data[0]
        fub = term_xds[f_axis].data[-1]

        concat_tslice = overlap_slice(tlb, tub, time_lbounds, time_ubounds)
        concat_fslice = overlap_slice(flb, fub, freq_lbounds, freq_ubounds)

        fconcat_xds_list = []

        for xds_list in sorted_xds_lol[concat_tslice]:
            fconcat_xds_list.append(xarray.concat(xds_list[concat_fslice],
                                                  f_axis,
                                                  join="exact"))

        # Concatenate gains near the interpolation values.
        concat_xds = xarray.concat(fconcat_xds_list,
                                   t_axis,
                                   join="exact")

        # Remove the chunking from the concatenated datasets.
        concat_xds = concat_xds.chunk({t_axis: -1, f_axis: -1})

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

        # TODO: This is INSANELY slow. Omitting until I come up with
        # a better solution.
        # interp_xds = interp_xds.interpolate_na("f_int",
        #                                        method="pchip")

        # This is a fast alternative to the above but it is definitely less
        # correct. Forward/back propagates values over missing entries.
        t_axis, f_axis = term_xds.GAIN_AXES[:2]
        interp_xds = interp_xds.ffill(t_axis)
        interp_xds = interp_xds.bfill(t_axis)
        interp_xds = interp_xds.ffill(f_axis)
        interp_xds = interp_xds.bfill(f_axis)
        # If an entry is STILL missing, there is no data from which to
        # interp/propagate. Set to zero.
        interp_xds = interp_xds.fillna(0)

        # Interpolate with various methods.
        if interp_method == "2dlinear":
            interp_xds = interp_xds.interp(
                {t_axis: term_xds[t_axis].data,
                 f_axis: term_xds[f_axis].data},
                kwargs={"fill_value": "extrapolate"})
        elif interp_method == "2dspline":
            interp_xds = spline2d_interpolate_gains(interp_xds,
                                                    term_xds,
                                                    interp_mode)
        elif interp_method == "smoothingspline":
            interp_xds = csaps2d_interpolate_gains(interp_xds,
                                                   term_xds,
                                                   interp_mode)

        # Convert the interpolated quantities back in gains.
        if interp_mode == "ampphase":
            gains = interp_xds.amp.data*da.exp(1j*interp_xds.phase.data)
            interp_xds = term_xds.assign(
                {"gains": (interp_xds.amp.dims, gains)})
        elif interp_mode == "reim":
            gains = interp_xds.re.data + 1j*interp_xds.im.data
            interp_xds = term_xds.assign(
                {"gains": (interp_xds.re.dims, gains)})

        t_chunks = term_xds.GAIN_SPEC.tchunk
        f_chunks = term_xds.GAIN_SPEC.fchunk

        interp_xds = interp_xds.chunk({t_axis: t_chunks, f_axis: f_chunks})

        interp_xds_list.append(interp_xds)

    return interp_xds_list


def spline2d(x, y, z, xx, yy):

    n_t, n_f, n_a, n_d, n_c = z.shape
    n_ti, n_fi = xx.size, yy.size

    zz = np.zeros((n_ti, n_fi, n_a, n_d, n_c), dtype=z.dtype)

    # NOTE: x are the column coordinates and y and row coordinates.
    for a in range(n_a):
        for d in range(n_d):
            for c in range(n_c):
                z_sel = z[:, :, a, d, c]
                if not np.any(z_sel):
                    continue
                interp_func = interp2d(y, x, z_sel, kind="cubic")
                zz[:, :, a, d, c] = interp_func(yy, xx).reshape(n_ti, n_fi)

    return zz


def spline2d_interpolate_gains(interp_xds, term_xds, interp_mode):

    if interp_mode == "ampphase":
        data_fields = ["amp", "phase"]
    elif interp_mode == "reim":
        data_fields = ["re", "im"]

    output_xds = term_xds
    t_axis, f_axis = output_xds.GAIN_AXES[:2]

    for data_field in data_fields:
        interp = da.blockwise(spline2d, "tfadc",
                              interp_xds[t_axis].values, None,
                              interp_xds[f_axis].values, None,
                              interp_xds[data_field].data, "tfadc",
                              term_xds[t_axis].values, None,
                              term_xds[f_axis].values, None,
                              dtype=np.float64,
                              adjust_chunks={"t": term_xds.dims[t_axis],
                                             "f": term_xds.dims[f_axis]})

        output_xds = output_xds.assign(
            {data_field: (interp_xds[data_field].dims, interp)})

    return output_xds


def csaps2d(x, y, z, xx, yy):

    n_t, n_f, n_a, n_d, n_c = z.shape
    n_ti, n_fi = xx.size, yy.size

    zz = np.zeros((n_ti, n_fi, n_a, n_d, n_c), dtype=z.dtype)

    for a in range(n_a):
        for d in range(n_d):
            for c in range(n_c):
                z_sel = z[:, :, a, d, c]
                if not np.any(z_sel):
                    continue
                interp_vals = csaps([x, y], z_sel, [xx, yy]).values
                zz[:, :, a, d, c] = interp_vals.reshape(n_ti, n_fi)

    return zz


def csaps2d_interpolate_gains(interp_xds, term_xds, interp_mode):

    if interp_mode == "ampphase":
        data_fields = ["amp", "phase"]
    elif interp_mode == "reim":
        data_fields = ["re", "im"]

    output_xds = term_xds
    t_axis, f_axis = output_xds.GAIN_AXES[:2]

    for data_field in data_fields:
        interp = da.blockwise(csaps2d, "tfadc",
                              interp_xds[t_axis].values, None,
                              interp_xds[f_axis].values, None,
                              interp_xds[data_field].data, "tfadc",
                              term_xds[t_axis].values, None,
                              term_xds[f_axis].values, None,
                              dtype=np.float64,
                              adjust_chunks={"t": term_xds.dims[t_axis],
                                             "f": term_xds.dims[f_axis]})

        output_xds = output_xds.assign(
            {data_field: (interp_xds[data_field].dims, interp)})

    return output_xds
