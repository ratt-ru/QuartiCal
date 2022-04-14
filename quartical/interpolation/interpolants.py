# -*- coding: utf-8 -*-
from loguru import logger  # noqa
import dask.array as da
import numpy as np
from scipy.interpolate import interp2d
from numba import jit
from functools import partial
from quartical.utils.dask import Blocker
import gc
from quartical.utils.maths import (kron_matvec, kron_tensorvec,
                                   fit_hyperplane, pcg, matern52)


def linear2d_interpolate_gains(interp_xds, term_xds):
    """Interpolate from interp_xds to term_xds linearly.

    Args:
        interp_xds: xarray.Dataset containing the data to interpolate from.
        term_xds: xarray.Dataset onto which to interpolate.

    Returns:
        output_xds: xarray.Dataset containing interpolated values
    """
    i_t_axis, i_f_axis = interp_xds.GAIN_AXES[:2]
    t_t_axis, t_f_axis = term_xds.GAIN_AXES[:2]

    i_t_dim = interp_xds.dims[i_t_axis]
    i_f_dim = interp_xds.dims[i_f_axis]

    interp_axes = {}

    if i_t_dim > 1:
        interp_axes[i_t_axis] = term_xds[t_t_axis].data
    if i_f_dim > 1:
        interp_axes[i_f_axis] = term_xds[t_f_axis].data

    output_xds = interp_xds.interp(
        interp_axes,
        kwargs={"fill_value": "extrapolate"}
    )

    if i_t_dim == 1:
        output_xds = output_xds.reindex(
            {i_t_axis: term_xds[t_t_axis].data},
            method="nearest"
        )
    if i_f_dim == 1:
        output_xds = output_xds.reindex(
            {i_f_axis: term_xds[t_f_axis].data},
            method="nearest"
        )

    return output_xds


def spline2d(x, y, z, xx, yy):
    """Constructs a 2D spline using (x,y,z) and evaluates it at (xx,yy)."""

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


def spline2d_interpolate_gains(interp_xds, term_xds):
    """Interpolate from interp_xds to term_xds using a 2D spline.

    Args:
        interp_xds: xarray.Dataset containing the data to interpolate from.
        term_xds: xarray.Dataset onto which to interpolate.

    Returns:
        output_xds: xarray.Dataset containing interpolated values
    """
    i_t_axis, i_f_axis = interp_xds.GAIN_AXES[:2]
    t_t_axis, t_f_axis = term_xds.GAIN_AXES[:2]

    output_xds = term_xds

    if interp_xds.dims[i_t_axis] < 4 or interp_xds.dims[i_f_axis] < 4:
        raise ValueError(
            f"Cubic spline interpolation requires at least four "
            f"values along an axis. After concatenation, the "
            f"(time, freq) dimensions of the interpolating dataset were "
            f"{(interp_xds.dims[i_t_axis], interp_xds.dims[i_f_axis])}"
        )

    for data_field in interp_xds.data_vars.keys():
        interp = da.blockwise(spline2d, "tfadc",
                              interp_xds[i_t_axis].values, None,
                              interp_xds[i_f_axis].values, None,
                              interp_xds[data_field].data, "tfadc",
                              term_xds[t_t_axis].values, None,
                              term_xds[t_f_axis].values, None,
                              dtype=np.float64,
                              adjust_chunks={"t": term_xds.dims[t_t_axis],
                                             "f": term_xds.dims[t_f_axis]})

        output_xds = output_xds.assign(
            {data_field: (term_xds.GAIN_AXES, interp)})

    return output_xds


@jit(nopython=True, nogil=True, cache=True)
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

                    yy[:, f, a, d, c] = linterp(x1,
                                                x1[good_data],
                                                y_sel[good_data])

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

                    yy[t, :, a, d, c] = linterp(x2,
                                                x2[good_data],
                                                y_sel[good_data])

    return yy


@jit(nopython=True, nogil=True, cache=True)
def linterp(xx, x, y):
    """Basic linear interpolation. Extrapolates with closest good value."""

    xi = 0
    xxi = 0

    yy = np.zeros(xx.shape, dtype=y.dtype)
    xxn = len(xx)

    while xxi < xxn:
        xxel = xx[xxi]
        xel = x[xi]
        if xxel == xel:
            yy[xxi] = y[xi]
            xxi += 1
        elif xxel < x[0]:
            yy[xxi] = y[0]
            xxi += 1
        elif xxel > x[-1]:
            yy[xxi] = y[-1]
            xxi += 1
        elif (xxel > xel) & (xxel < x[xi + 1]):
            slope = (y[xi + 1] - y[xi]) / (x[xi + 1] - xel)
            yy[xxi] = slope * (xxel - xel) + y[xi]
            xxi += 1
        else:
            xi += 1

    return yy


def interpolate_missing(interp_xds):
    """Linear interpolate missing values in the given xarray dataset.

    Args:
        interp_xds: xarray.Dataset containing the data.

    Returns:
        output_xds: xarray.Dataset containing the data after interpolation.
    """

    i_t_axis, i_f_axis = interp_xds.GAIN_AXES[:2]

    output_xds = interp_xds

    for data_field in interp_xds.data_vars:

        interp = da.blockwise(_interpolate_missing, "tfadc",
                              interp_xds[i_t_axis].values, None,
                              interp_xds[i_f_axis].values, None,
                              interp_xds[data_field].data, "tfadc",
                              dtype=np.float64)

        output_xds = output_xds.assign(
            {data_field: (interp_xds[data_field].dims, interp)})

    return output_xds


def gpr_interpolate_gains(input_xds, output_xds, gpr_params):
    """
    Interpolate from input_xds to output_xds using GPR.

    TODO - add spatial axis if more than one direction is present

    Args:
        input_xds: xarray.Dataset containing the data to interpolate from.
        output_xds: list of xarray.Datasets onto which to interpolate.
        gpr_params: hyper-parameters for GPR

    Returns:
        output_xds: list of xarray.Datasets containing interpolated values.
    """
    # We first solve for Kyinv y required to perform the interpolation
    t = input_xds.gain_t.values
    f = input_xds.gain_f.values
    jhj = input_xds.jhj.data.rechunk({0:-1, 1:-1, 2: 1, 3:-1, 4:-1})
    gain = input_xds.gains.data.rechunk({0:-1, 1:-1, 2: 1, 3:-1, 4:-1})
    flag = input_xds.gain_flags.data.rechunk({0:-1, 1:-1, 2: 1, 3:-1})
    gref = gain[:,:,-1]
    _, _, nant, _, _ = gain.shape
    p = da.arange(nant, chunks=1)

    lts = gpr_params[0]
    lfs = gpr_params[1]
    ninflates = gpr_params[2]

    # need Blocker here since expecting multiple outputs
    inversion_blocker = Blocker(solve_gpr, 'a')
    inversion_blocker.add_input("gain", gain, 'tfadc')
    inversion_blocker.add_input("jhj", jhj, 'tfadc')
    inversion_blocker.add_input("flag", flag, 'tfad')
    inversion_blocker.add_input("gref", gref, 'tfdc')
    inversion_blocker.add_input("pant", p, 'a')
    inversion_blocker.add_input("t", t)
    inversion_blocker.add_input("f", f)
    inversion_blocker.add_input("lt", lts)
    inversion_blocker.add_input("lf", lfs)
    inversion_blocker.add_input("noise_inflation", ninflates)

    # the final entry in chunks is for amplitudes and phases
    inversion_blocker.add_output("Kyinvs", "tfadcx",
                                 gain.chunks + ((2,),),
                                 np.float64)
    # the first entry here is determined by the number of
    # coordinates we are interpolating
    theta_chunks = ((3,),) + gain.chunks[2:] + ((2,),)
    inversion_blocker.add_output("thetas", "nadcx",
                                 theta_chunks,
                                 np.float64)
    sigmasqs_chunks = gain.chunks[2:] + ((2,),)
    inversion_blocker.add_output("sigmafsqs", 'adcx',
                                 sigmasqs_chunks,
                                 np.float64)

    out_dict = inversion_blocker.get_dask_outputs()

    # iterate over output datasets interpolating each
    # onto target coordiantes
    out_ds = []
    for ds in output_xds:
        tp = ds.gain_t.values
        fp = ds.gain_f.values
        gain = da.blockwise(interp_gpr, 'tfadc',
                            out_dict['Kyinvs'], 'tfadcx',
                            out_dict['thetas'], 'nadcx',
                            out_dict['sigmafsqs'], 'adcx',
                            t, None,
                            f, None,
                            tp, None,
                            fp, None,
                            lts, None,
                            lfs, None,
                            new_axes={'n': 3, 'x': 2},
                            adjust_chunks={'t': tp.size, 'f': fp.size},
                            dtype=np.complex128,
                            meta=np.empty((0,0,0,0,0), dtype=np.complex128))
        dso = ds.assign(**{'gains': (ds.GAIN_AXES, gain.rechunk({2:-1}))})
        out_ds.append(dso)

    return out_ds


def solve_gpr(gain, jhj, flag, gref, pant, t, f, lt, lf,
               noise_inflation):
    '''
    Solves (K + Sigma)inv y for each antenna
    '''
    jhj = jhj.real

    f = f/1e6  # convert to MHz
    Kamp = (matern52(t, t, 1.0, lt[0]), matern52(f, f, 1.0, lf[0]))
    Kphase = (matern52(t, t, 1.0, lt[1]), matern52(f, f, 1.0, lf[1]))

    tt, ff = np.meshgrid(t, f, indexing='ij')
    x = np.vstack((tt.flatten(), ff.flatten()))

    D, N = x.shape
    X = np.vstack((x, np.ones((1,N))))

    # operator that needs to be inverted
    def Kyop(x, K, Sigma, sigmafsq):
        # sigmafsq is actually part of K but we want to set it
        # per antenna and correlation so we cheat this way
        return kron_matvec(K, sigmafsq*x) + Sigma*x

    nt, nf, nant, ndir, ncorr = jhj.shape
    Kyinvs = np.zeros((nt, nf, nant, ndir, ncorr, 2), dtype=np.float64)
    thetas = np.zeros((D+1, nant, ndir, ncorr, 2), dtype=np.float64)
    sigmafsqs = np.zeros((nant, ndir, ncorr, 2), dtype=np.float64)
    for p in range(nant):
        for d in range(ndir):
            for c in range(ncorr):
                # find valid data
                invalid = np.logical_or(flag[:, :, p, d],
                                        jhj[:, :, p, d, c]==0).ravel()
                ival = ~invalid
                xval = x[:, ival]

                # amplitudes
                g = np.abs(gain[:, :, p, d, c])
                yval = g.ravel()[ival]
                theta = fit_hyperplane(xval, yval)
                # subtract plane approx
                plane_approx = X.T.dot(theta).reshape(nt, nf)
                y = g - plane_approx
                sigmafsq = np.var(y.ravel()[ival])
                # this gives small weight to flagged data
                # need to check that it's sensible
                weight = jhj[:, :, p, d, c]
                Sigma = np.where(weight>0, noise_inflation/weight, 1e10)
                Ky = partial(Kyop, K=Kamp, Sigma=Sigma, sigmafsq=sigmafsq)
                Kyinv, success, eps = pcg(Ky,
                                          y,
                                          y,  # init at ML solution
                                          tol=1e-6,
                                          maxit=500)
                Kyinvs[:, :, p, d, c, 0] = Kyinv
                thetas[:, p, d, c, 0] = theta
                sigmafsqs[p, d, c, 0] = sigmafsq

                if not success:
                    print(f"Amplitude failed at ant/corr {pant}/{c} with eps of {eps}")

                # phases
                weight *= g**2  # scale weights for uncertainty propagation
                g = np.angle(gain[:, :, p, d, c] * np.conj(gref[:, :, d, c]))
                g = np.unwrap(np.unwrap(g, axis=0), axis=1)
                # don't smooth the reference antenna
                if not g.any():
                    continue

                yval = g.ravel()[ival]
                theta = fit_hyperplane(xval, yval)
                # subtract plane approx
                plane_approx = X.T.dot(theta).reshape(nt, nf)
                y = g - plane_approx
                sigmafsq = np.var(y.ravel()[ival])
                # this gives small weight to flaged data
                # need to check that it's sensible
                Sigma = np.where(weight>0, noise_inflation/weight, 1e10)
                Ky = partial(Kyop, K=Kphase, Sigma=Sigma, sigmafsq=sigmafsq)
                Kyinv, success, eps = pcg(Ky,
                                          y,
                                          y,  # init at ML solution
                                          tol=1e-6,
                                          maxit=500)
                Kyinvs[:, :, p, d, c, 1] = Kyinv
                thetas[:, p, d, c, 1] = theta
                sigmafsqs[p, d, c, 1] = sigmafsq

                if not success:
                    print(f"Phase failed at ant/corr {pant}/{c} with eps of {eps}")
    result_dict = {}
    result_dict['Kyinvs'] = Kyinvs
    result_dict['thetas'] = thetas
    result_dict['sigmafsqs'] = sigmafsqs
    gc.collect()
    return result_dict


def interp_gpr(Kyinvs, thetas, sigmafsqs,
               t, f, tp, fp, lt, lf):
    return _interp_gpr(Kyinvs[0], thetas[0][0], sigmafsqs[0],
                       t, f, tp, fp, lt, lf)


def _interp_gpr(Kyinvs, thetas, sigmafsqs,
                t, f, tp, fp, lt, lf):
    '''
    Interpolates the GPR solutions onto tp and fp
    '''
    f = f/1e6  # convert to MHz
    fp = fp/1e6

    # for plane approx
    ttp, ffp = np.meshgrid(tp, fp, indexing='ij')
    xp = np.vstack((ttp.flatten(), ffp.flatten()))
    D, Np = xp.shape
    Xp = np.vstack((xp, np.ones((1,Np))))
    ntp = tp.size
    nfp = fp.size
    _, _, nant, ndir, ncorr, _ = Kyinvs.shape
    sol = np.zeros((ntp, nfp, nant, ndir, ncorr), dtype=np.complex128)


    Kamp = (matern52(tp, t, 1.0, lt[0]), matern52(fp, f, 1.0, lf[0]))
    Kphase = (matern52(tp, t, 1.0, lt[1]), matern52(fp, f, 1.0, lf[1]))

    for p in range(nant):
        for d in range(ndir):
            for c in range(ncorr):
                # amplitudes
                Kyinv = Kyinvs[:, :, p, d, c, 0]
                theta = thetas[:, p, d, c, 0]
                sigmafsq = sigmafsqs[p, d, c, 0]
                amp = (kron_tensorvec(Kamp, sigmafsq * Kyinv) +
                       Xp.T.dot(theta).reshape(ntp, nfp))

                # phases
                Kyinv = Kyinvs[:, :, p, d, c, 1]
                theta = thetas[:, p, d, c, 1]
                sigmafsq = sigmafsqs[p, d, c, 1]
                phase = (kron_tensorvec(Kphase, sigmafsq * Kyinv) +
                         Xp.T.dot(theta).reshape(ntp, nfp))

                sol[:, :, p, d, c] = amp * np.exp(1.0j*phase)
    return sol
