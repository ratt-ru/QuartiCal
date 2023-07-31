from collections import defaultdict
import numpy as np
import dask.array as da
from daskms.experimental.zarr import xds_from_zarr
from scipy.interpolate import RegularGridInterpolator
import sympy as sm
from sympy.utilities.lambdify import lambdify
from sympy.parsing.sympy_parser import parse_expr
from ducc0.wgridder.experimental import dirty2vis
from quartical.utils.collections import freeze_default_dict


def _degrid(time, freq, uvw, component):

    name = component.name
    npix_x = component.npix_x  # Will be used in interpolation mode.
    npix_y = component.npix_y  # Will be used in interpolation mode.
    cellsize_x = component.cellsize_x
    cellsize_y = component.cellsize_y
    centre_x = component.centre_x
    centre_y = component.centre_y
    integrations_per_image = component.integrations_per_image
    channels_per_image = component.channels_per_image

    # TODO: Dodgy pattern, as this will produce dask arrays which will
    # need to be evaluated i.e. task-in-task which is a bad idea. OK for
    # the purposes of a prototype.
    model_xds = xds_from_zarr(name)[0]

    # TODO: We want to go from the model to an image cube appropriate for this
    # chunks of data. The dimensions of the image cube will be determined
    # by the time and freqency dimensions of the chunk in conjunction with
    # the integrations per chunk and channels per chunk arguments.

    utime, utime_inv = np.unique(time, return_inverse=True)
    n_utime = utime.size
    ipi = integrations_per_image or n_utime
    n_mean_times = int(np.ceil(n_utime / ipi))

    n_freq = freq.size
    cpi = channels_per_image or n_freq
    n_mean_freqs = int(np.ceil(n_freq / cpi))

    # Let's start off by simply reconstructing the model image as generated
    # by pfb-clean.

    native_npix_x = model_xds.npix_x
    native_npix_y = model_xds.npix_y
    native_image = np.zeros((native_npix_x, native_npix_y), dtype=float)

    # Sey up sympy symbols for expression evaluation.
    params = sm.symbols(('t', 'f'))
    params += sm.symbols(tuple(model_xds.params.values))
    symexpr = parse_expr(model_xds.parametrisation)
    model_func = lambdify(params, symexpr)
    time_expr = parse_expr(model_xds.texpr)
    time_func = lambdify(params[0], time_expr)
    freq_expr = parse_expr(model_xds.fexpr)
    freq_func = lambdify(params[1], freq_expr)

    pixel_xs = model_xds.location_x.values
    pixel_ys = model_xds.location_y.values
    pixel_coeffs = model_xds.coefficients.values

    # TODO: How do we handle the correlation axis neatly?
    vis = np.empty((time.size, freq.size, 4), dtype=np.complex128)

    for ti in range(n_mean_times):
        for fi in range(n_mean_freqs):

            freq_sel = slice(fi * cpi, (fi + 1) * cpi)
            degrid_freq = freq[freq_sel]
            mean_freq = degrid_freq.mean()

            time_sel = slice(ti * ipi, (ti + 1) * ipi)
            degrid_time = utime[time_sel]
            mean_time = degrid_time.mean()

            native_image[pixel_xs, pixel_ys] = model_func(
                time_func(mean_time), freq_func(mean_freq), *pixel_coeffs
            )

            # NOTE: Select out appropriate rows for ipi and make selection
            # consistent.
            row_sel = slice(None)

            # Degrid
            dirty2vis(
                vis=vis[row_sel, freq_sel, 0],
                uvw=uvw,
                freq=degrid_freq,
                dirty=native_image,
                pixsize_x=cellsize_x or model_xds.cell_rad_x,
                pixsize_y=cellsize_y or model_xds.cell_rad_y,
                center_x=centre_x or model_xds.center_x,
                center_y=centre_y or model_xds.center_y,
                epsilon=1e-7,  # TODO: Is this too high?
                do_wgridding=True,  # Should be ok to leave True.
                divide_by_n=False,  # Until otherwise informed.
                nthreads=6  # Should be equivalent to solver threads.
            )

            # Zero the image array between image slices as a precaution.
            native_image[:, :] = 0

    # Degridder only produces I - will need to be more sophisticated.
    vis[..., -1] = vis[..., 0]

    return vis

    # TODO: This was omitted for the sake of simplicity but we ultimately will
    # want this functionality. Need to be very cautious with regard to which
    # parameters get used.

    cellxi, cellyi = model_xds.cell_rad_x, model_xds.cell_rad_y
    x0i, y0i = model_xds.center_x, model_xds.center_y

    xin = (-(nxi//2) + np.arange(nxi))*cellxi + x0i
    yin = (-(nyi//2) + np.arange(nyi))*cellyi + y0i
    xo = (-(nxo//2) + np.arange(nxo))*cellxo + x0o
    yo = (-(nyo//2) + np.arange(nyo))*cellyo + y0o

    # how many pixels to pad by to extrapolate with zeros
    xldiff = xin.min() - xo.min()
    if xldiff > 0.0:
        npadxl = int(np.ceil(xldiff/cellxi))
    else:
        npadxl = 0
    yldiff = yin.min() - yo.min()
    if yldiff > 0.0:
        npadyl = int(np.ceil(yldiff/cellyi))
    else:
        npadyl = 0

    xudiff = xo.max() - xin.max()
    if xudiff > 0.0:
        npadxu = int(np.ceil(xudiff/cellxi))
    else:
        npadxu = 0
    yudiff = yo.max() - yin.max()
    if yudiff > 0.0:
        npadyu = int(np.ceil(yudiff/cellyi))
    else:
        npadyu = 0

    do_pad = npadxl > 0
    do_pad |= npadxu > 0
    do_pad |= npadyl > 0
    do_pad |= npadyu > 0
    if do_pad:
        image_in = np.pad(
            image_in,
            ((npadxl, npadxu), (npadyl, npadyu)),
            mode='constant'
        )

        xin = (-(nxi//2+npadxl) + np.arange(nxi + npadxl + npadxu))*cellxi + x0i
        nxi = nxi + npadxl + npadxu
        yin = (-(nyi//2+npadyl) + np.arange(nyi + npadyl + npadyu))*cellyi + y0i
        nyi = nyi + npadyl + npadyu

    do_interp = cellxi != cellxo
    do_interp |= cellyi != cellyo
    do_interp |= x0i != x0o
    do_interp |= y0i != y0o
    do_interp |= nxi != nxo
    do_interp |= nyi != nyo
    if do_interp:
        interpo = RegularGridInterpolator((xin, yin), image_in,
                                          bounds_error=True, method='linear')
        xx, yy = np.meshgrid(xo, yo, indexing='ij')
        return interpo((xx, yy))
    # elif (nxi != nxo) or (nyi != nyo):
    #     # only need the overlap in this case
    #     _, idx0, idx1 = np.intersect1d(xin, xo, assume_unique=True, return_indices=True)
    #     _, idy0, idy1 = np.intersect1d(yin, yo, assume_unique=True, return_indices=True)
    #     return image[idx0, idy0]
    else:
        return image_in


def degrid(data_xds_list, model_vis_recipe, ms_path, model_opts):

    degrid_models = model_vis_recipe.ingredients.degrid_models

    degrid_list = []

    for data_xds in data_xds_list:

        model_vis = defaultdict(list)

        for degrid_model in degrid_models:

            degrid_vis = da.blockwise(
                _degrid, ("rowlike", "chan", "corr"),
                data_xds.TIME.data, ("rowlike",),
                data_xds.CHAN_FREQ.data, ("chan",),
                data_xds.UVW.data, ("rowlike", "uvw"),
                degrid_model, None,
                concatenate=True,
                align_arrays=False,
                meta=np.empty([0, 0, 0], dtype=np.complex128),
                new_axes={"corr": 4}  # TODO: Shouldn't be hardcoded.
            )

            model_vis[degrid_model].append(degrid_vis)

        degrid_list.append(freeze_default_dict(model_vis))

    return degrid_list
