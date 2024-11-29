# -*- coding: utf-8 -*-
from loguru import logger  # noqa
import dask.array as da
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from numba import njit
from quartical.utils.numba import JIT_OPTIONS


def linear2d_interpolate_gains(source_xds, target_xds):
    """Interpolate from source_xds to target_xds linearly.

    Args:
        source_xds: xarray.Dataset containing the data to interpolate from.
        target_xds: xarray.Dataset onto which to interpolate.

    Returns:
        output_xds: xarray.Dataset containing interpolated values
    """

    if hasattr(target_xds, "PARAM_SPEC"):
        i_t_axis, i_f_axis = source_xds.PARAM_AXES[:2]
        t_t_axis, t_f_axis = target_xds.PARAM_AXES[:2]
    else:
        i_t_axis, i_f_axis = source_xds.GAIN_AXES[:2]
        t_t_axis, t_f_axis = target_xds.GAIN_AXES[:2]

    i_t_dim = source_xds.sizes[i_t_axis]
    i_f_dim = source_xds.sizes[i_f_axis]

    interp_axes = {}

    if i_t_dim > 1:
        interp_axes[i_t_axis] = target_xds[t_t_axis].data
    if i_f_dim > 1:
        interp_axes[i_f_axis] = target_xds[t_f_axis].data

    # NOTE: The below is the path of least resistance but may not be the most
    # efficient method for this mixed-mode interpolation - it may be possible
    # to do better using multiple RegularGridInterpoator objects.

    # Interpolate using linear interpolation, filling points outside the
    # domain with NaNs.
    in_domain_xda = source_xds.params.interp(
        interp_axes,
        kwargs={"fill_value": np.nan}
    )

    # Interpolate using nearest-neighbour interpolation, extrapolating points
    # outside the domain.
    out_domain_xda = source_xds.params.interp(
        interp_axes,
        method="nearest",
        kwargs={"fill_value": "extrapolate"}
    )

    # Combine the linear and nearest neighbour interpolation done above i.e.
    # use use linear interpolation inside the domain and nearest-neighbour
    # interpolation anywhere extrapolation was required.
    target_xda = in_domain_xda.where(
        da.isfinite(in_domain_xda), out_domain_xda
    )

    if i_t_dim == 1:
        target_xda = target_xda.reindex(
            {i_t_axis: target_xds[t_t_axis].data},
            method="nearest"
        )
    if i_f_dim == 1:
        target_xda = target_xda.reindex(
            {i_f_axis: target_xds[t_f_axis].data},
            method="nearest"
        )

    target_xds = target_xds.assign(
        {'params': (target_xda.dims, target_xda.data)}
    )

    return target_xds


def spline2d(x, y, z, xx, yy):
    """Constructs a 2D spline using (x,y,z) and evaluates it at (xx,yy)."""

    n_t, n_f, n_a, n_d, n_c = z.shape
    n_ti, n_fi = xx.size, yy.size

    zz = np.zeros((n_ti, n_fi, n_a, n_d, n_c), dtype=z.dtype)

    xxg, yyg = np.meshgrid(xx, yy, indexing="ij")

    for a in range(n_a):
        for d in range(n_d):
            for c in range(n_c):
                z_sel = z[:, :, a, d, c]
                if not np.any(z_sel):
                    continue
                interp_func = RGI(
                    (x, y),
                    z_sel,
                    method="cubic",
                    bounds_error=False,
                    fill_value=None
                )
                zz[:, :, a, d, c] = interp_func((xxg, yyg))

    return zz


def spline2d_interpolate_gains(source_xds, target_xds):
    """Interpolate from interp_xds to term_xds using a 2D spline.

    Args:
        interp_xds: xarray.Dataset containing the data to interpolate from.
        term_xds: xarray.Dataset onto which to interpolate.

    Returns:
        output_xds: xarray.Dataset containing interpolated values
    """
    if hasattr(target_xds, "PARAM_SPEC"):
        i_t_axis, i_f_axis = source_xds.PARAM_AXES[:2]
        t_t_axis, t_f_axis = target_xds.PARAM_AXES[:2]
    else:
        i_t_axis, i_f_axis = source_xds.GAIN_AXES[:2]
        t_t_axis, t_f_axis = target_xds.GAIN_AXES[:2]

    if source_xds.sizes[i_t_axis] < 4 or source_xds.sizes[i_f_axis] < 4:
        raise ValueError(
            f"Cubic spline interpolation requires at least four "
            f"values along an axis. After concatenation, the "
            f"(time, freq) dimensions of the interpolating dataset were "
            f"{(source_xds.sizes[i_t_axis], source_xds.sizes[i_f_axis])}."
        )

    interp_arr = da.blockwise(
        spline2d, "tfadc",
        source_xds[i_t_axis].values, None,
        source_xds[i_f_axis].values, None,
        source_xds.params.data, "tfadc",
        target_xds[t_t_axis].values, None,
        target_xds[t_f_axis].values, None,
        dtype=np.float64,
        adjust_chunks={
            "t": target_xds.sizes[t_t_axis],
            "f": target_xds.sizes[t_f_axis]
        }
    )

    output_xds = target_xds.assign(
        {"params": (source_xds.params.dims, interp_arr)}
    )

    return output_xds


@njit(**JIT_OPTIONS)
def map_ax_itp_to_ax(ax, ax_itp):
    """Given ax_itp, compute indices of closest left-hand points in ax."""

    out = np.empty_like(ax_itp, dtype=np.int64)

    i = 0
    itp_i = 0

    while itp_i < ax_itp.size:
        ax_itp_el = ax_itp[itp_i]
        if ax_itp_el <= ax[0]:  # On or below lower bound.
            out[itp_i] = 0
        elif ax_itp_el >= ax[-1]:  # On or above upper bound.
            out[itp_i] = ax.size - 1
        else:
            while ax_itp_el - ax[i + 1] >= 0:  # NOTE: Assumes ordering.
                i += 1
            out[itp_i] = i
        itp_i += 1

    return out


@njit(**JIT_OPTIONS)
def bilinear_interp(x, y, f, x_itp, y_itp):

    f_itp = np.zeros((x_itp.size, y_itp.size), dtype=f.dtype)

    x_itp_map = map_ax_itp_to_ax(x, x_itp)
    y_itp_map = map_ax_itp_to_ax(y, y_itp)

    nx, ny = x.size, y.size

    for i, (_x, x1m) in enumerate(zip(x_itp, x_itp_map)):
        x1m, x2m = (x1m, x1m + 1) if x1m + 1 < nx - 1 else (x1m - 1, x1m)
        x1, x2 = x[x1m], x[x2m]
        x_diff = x2 - x1
        x1_dist = _x - x1
        x2_dist = x2 - _x

        for j, (_y, y1m) in enumerate(zip(y_itp, y_itp_map)):
            y1m, y2m = (y1m, y1m + 1) if y1m + 1 < ny - 1 else (y1m - 1, y1m)
            y1, y2 = y[y1m], y[y2m]
            y_diff = y2 - y1
            y1_dist = _y - y1
            y2_dist = y2 - _y

            fx1y1 = f[x1m, y1m]
            fx1y2 = f[x1m, y2m]
            fx2y1 = f[x2m, y1m]
            fx2y2 = f[x2m, y2m]

            coeff = 1 / (x_diff * y_diff)

            f_itp[i, j] = coeff * (
                (fx1y1 * x2_dist + fx2y1 * x1_dist) * y2_dist +
                (fx1y2 * x2_dist + fx2y2 * x1_dist) * y1_dist
            )

    return f_itp


@njit(**{**JIT_OPTIONS, "fastmath": False})  # No fastmath due to nans.
def _interpolate_missing(x1, x2, y):
    """Interpolate/extend data y along x1 and x2 to fill in missing values."""

    n_t, n_f, n_a, n_d, n_c = y.shape

    yy = y.copy()

    for f in range(n_f):
        for a in range(n_a):
            for d in range(n_d):
                for c in range(n_c):
                    y_sel = y[:, f, a, d, c]
                    good_data = np.where(np.isfinite(y_sel))
                    if len(good_data[0]) == 0:
                        continue

                    yy[:, f, a, d, c] = linterp(
                        x1, x1[good_data], y_sel[good_data]
                    )

    for t in range(n_t):
        for a in range(n_a):
            for d in range(n_d):
                for c in range(n_c):
                    y_sel = yy[t, :, a, d, c]
                    good_data = np.where(np.isfinite(y_sel))
                    if len(good_data[0]) == 0:
                        # If there is no good data along frequency after
                        # interpolating in time, we have no information
                        # from which to interpolate - we zero these locations.
                        yy[t, :, a, d, c] = 0
                        continue

                    yy[t, :, a, d, c] = linterp(
                        x2, x2[good_data], y_sel[good_data]
                    )

    return yy


@njit(**JIT_OPTIONS)
def linterp(x_itp, x_data, y_data):
    """Basic linear interpolation. Extrapolates with closest good value.

    Given a 1D array of x-values and a 1D array of y-values, perform linear
    interpolation to produce y-values for a 1D array of (different) x-values.

    Args:
        x_itp: A 1D array of x-values at which it interpolate.
        x_data: A 1D array of x-values for which there are y-values.
        y_data: A 1D array of y-values at the x_data locations.

    Returns:
        y_itp: A 1D array of y-values at the x_itp locations.
    """

    i_data = int(0)
    i_itp = int(0)

    y_itp = np.zeros(x_itp.shape, dtype=y_data.dtype)
    n_itp = x_itp.size

    while i_itp < n_itp:
        x_itp_el = x_itp[i_itp]
        x_el = x_data[i_data]
        if x_itp_el == x_el:  # Use existing y value.
            y_itp[i_itp] = y_data[i_data]
            i_itp += 1
        elif x_itp_el < x_data[0]:  # Constant extrapolation on left edge.
            y_itp[i_itp] = y_data[0]
            i_itp += 1
        elif x_itp_el > x_data[-1]:  # Constant extrapolation on right edge.
            y_itp[i_itp] = y_data[-1]
            i_itp += 1
        elif (x_itp_el > x_el) & (x_itp_el < x_data[i_data + 1]):
            y_diff = (y_data[i_data + 1] - y_data[i_data])
            x_diff = (x_data[i_data + 1] - x_el)
            slope = y_diff / x_diff
            y_itp[i_itp] = slope * (x_itp_el - x_el) + y_data[i_data]
            i_itp += 1
        else:
            i_data += 1

    return y_itp


@njit(**{**JIT_OPTIONS, "fastmath": False})  # No fastmath due to nans.
def _interpolate_missing_phase(x1, x2, y):
    """Interpolate/extend data y along x1 and x2 to fill in missing values."""

    n_t, n_f, n_a, n_d, n_c = y.shape

    yy = y.copy()

    for f in range(n_f):
        for a in range(n_a):
            for d in range(n_d):
                for c in range(n_c):
                    y_sel = y[:, f, a, d, c]
                    good_data = np.where(np.isfinite(y_sel))
                    if len(good_data[0]) == 0:
                        continue

                    yy[:, f, a, d, c] = phase_interp(
                        x1, x1[good_data], y_sel[good_data]
                    )

    for t in range(n_t):
        for a in range(n_a):
            for d in range(n_d):
                for c in range(n_c):
                    y_sel = yy[t, :, a, d, c]
                    good_data = np.where(np.isfinite(y_sel))
                    if len(good_data[0]) == 0:
                        # If there is no good data along frequency after
                        # interpolating in time, we have no information
                        # from which to interpolate - we zero these locations.
                        yy[t, :, a, d, c] = 0
                        continue

                    yy[t, :, a, d, c] = phase_interp(
                        x2, x2[good_data], y_sel[good_data]
                    )

    return yy


@njit(**JIT_OPTIONS)
def phase_interp(x_itp, x_data, y_data):
    """Basic phase interpolation. Extrapolates with closest good value.

    Given a 1D array of x-values and a 1D array of y-values, perform phase
    interpolation to produce y-values for a 1D array of (different) x-values.
    This differs from basic linear interpolation in that it understands phase
    wraps and interpolates under the assumption that the minimum angle should
    always be preferred.

    Args:
        x_itp: A 1D array of x-values at which it interpolate.
        x_data: A 1D array of x-values for which there are y-values.
        y_data: A 1D array of y-values at the x_data locations.

    Returns:
        y_itp: A 1D array of y-values at the x_itp locations.
    """

    i_data = int(0)
    i_itp = int(0)

    y_itp = np.zeros(x_itp.shape, dtype=y_data.dtype)
    n_itp = x_itp.size

    while i_itp < n_itp:
        x_itp_el = x_itp[i_itp]
        x_el = x_data[i_data]
        if x_itp_el == x_el:  # Use existing y value.
            y_itp[i_itp] = y_data[i_data]
            i_itp += 1
        elif x_itp_el < x_data[0]:  # Constant extrapolation on left edge.
            y_itp[i_itp] = y_data[0]
            i_itp += 1
        elif x_itp_el > x_data[-1]:  # Constant extrapolation on right edge.
            y_itp[i_itp] = y_data[-1]
            i_itp += 1
        elif (x_itp_el > x_el) & (x_itp_el < x_data[i_data + 1]):
            w = (x_itp_el - x_el)/(x_data[i_data + 1] - x_el)
            cos_0 = (1 - w) * np.cos(y_data[i_data])
            cos_1 = w * np.cos(y_data[i_data + 1])
            sin_0 = (1 - w) * np.sin(y_data[i_data])
            sin_1 = w * np.sin(y_data[i_data + 1])
            y_itp[i_itp] = np.arctan2(sin_0 + sin_1, cos_0 + cos_1)
            i_itp += 1
        else:
            i_data += 1

    return y_itp


def interpolate_missing(input_xda, mode="normal"):
    """Linear interpolate missing values in the given xarray dataset.

    Args:
        interp_xds: xarray.Dataset containing the data.

    Returns:
        output_xds: xarray.Dataset containing the data after interpolation.
    """

    if mode == "normal":
        interp_func = _interpolate_missing
    elif mode == "phase":
        interp_func = _interpolate_missing_phase
    else:
        raise ValueError(f"Unsupported mode '{mode}' in interpolate_missing.")

    i_t_axis, i_f_axis = input_xda.dims[:2]

    interp = da.blockwise(
        interp_func, "tfadc",
        input_xda[i_t_axis].values, None,
        input_xda[i_f_axis].values, None,
        input_xda.data, "tfadc",
        dtype=np.float64
    )

    return input_xda.copy(data=interp)
