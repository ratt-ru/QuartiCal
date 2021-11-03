# -*- coding: utf-8 -*-
from loguru import logger  # noqa
import dask.array as da
import numpy as np
from scipy.interpolate import interp2d
from numba import jit


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


# GPR related utils below this line
def matern52(x, xp, sigmaf, l):
    N = x.size
    Np = xp.size
    xxp = np.abs(np.tile(x, (Np, 1)).T - np.tile(xp, (N, 1)))
    return sigmaf**2*np.exp(-np.sqrt(5)*xxp/l) * (1 +
                            np.sqrt(5)*xxp/l + 5*xxp**2/(3*l**2))

def kron_matvec(A, b):
    D = len(A)
    N = b.size
    x = b.ravel()

    for d in range(D):
        Gd = A[d].shape[0]
        NGd = N // Gd
        X = np.reshape(x, (Gd, NGd))
        Z = A[d].dot(X).T
        x = Z.ravel()
    return x.reshape(b.shape)

# kron_matvec for non-sqaure matrices
def kron_tensorvec(A, b):
    D = len(A)
    G = np.zeros(D, dtype=np.int32)
    M = np.zeros(D, dtype=np.int32)
    for d in range(D):
        M[d], G[d] = A[d].shape
    x = b
    for d in range(D):
        Gd = G[d]
        rem = np.prod(np.delete(G, d))
        X = np.reshape(x, (Gd, rem))
        Z = (A[d].dot(X)).T
        x = Z.ravel()
        G[d] = M[d]
    return x.reshape(tuple(M))

from ducc0.misc import vdot  # np.vdot can be very inaccurate for large arrays
def pcg(A, b, x0, M=None, tol=1e-5, maxit=150):

    if M is None:
        def M(x): return x

    r = A(x0) - b
    y = M(r)
    p = -y
    rnorm = vdot(r, y)
    if np.isnan(rnorm) or rnorm == 0.0:
        eps0 = 1.0
    else:
        eps0 = rnorm
    k = 0
    x = x0
    eps = 1.0
    stall_count = 0
    while eps > tol and k < maxit:
        xp = x.copy()
        rp = r.copy()
        Ap = A(p)
        rnorm = vdot(r, y)
        alpha = rnorm / vdot(p, Ap)
        x = xp + alpha * p
        r = rp + alpha * Ap
        y = M(r)
        rnorm_next = vdot(r, y)
        beta = rnorm_next / rnorm
        p = beta * p - y
        rnorm = rnorm_next
        k += 1

        # using worst case convergence criteria
        epsx = np.linalg.norm(x - xp) / np.linalg.norm(x)
        epsn = rnorm / eps0
        eps = np.maximum(epsx, epsn)

    if k >= maxit:
        print(f"Max iters reached. eps = {eps}")
    else:
        print(f"Success, converged after {k} iters")
    return x

def _interp_gpr(t, f, gain, jhj, tp, fp, sigmaf, lt, lf):
    '''
    GPR interpolation assuming data lies on a grid.
    No hyper-parameter optimisation.

    t - time coordinates where we have data
    f - freq coordinates where we have data
    gain - (tfadc) array of complex valued data
    jhj- (tfadc) array of real valued "weights"
    tp - times we want to interpolate to
    fp - freqs we want to interpolate to
    sigmaf - signal variance
    lt - time length scale
    lf - freq length scale

    The interpolation is performed by using the conjugate
    gradient algorithm to invert the Hessian of

    (y - Lx).H W (y - Lx) + x.H x

    where L is the Cholesky factor of the covariance matrix
    (i.e. whitened Wiener filter) and W consists of the diagonal
    elements of jhj.
    '''
    from functools import partial

    # data prior covariance
    nt = t.size
    K = matern52(t, t, sigmaf, lt)
    Lt = np.linalg.cholesky(K + 1e-10*np.eye(nt))

    nf = f.size
    K = matern52(f, f, 1.0, lf)
    Lf = np.linalg.cholesky(K + 1e-10*np.eye(nf))

    L = (Lt, Lf)
    LH = (Lt.T, Lf.T)

    # cross covariance
    Kp = (matern52(tp, t, sigmaf, lt), matern52(fp, f, 1.0, lf))

    # operator that needs to be inverted
    def hess(x, W, L, LH):
        res = kron_matvec(L, x)
        res *= W
        return kron_matvec(LH, res) + x

    _, _, nant, ndir, ncorr = jhj.shape
    sol = np.zeros((tp.size, fp.size, nant, ndir, ncorr), dtype=gain.dtype)
    for p in range(nant):
        for d in range(ndir):
            for c in range(ncorr):
                W = jhj[:, :, p, d, c]
                y = gain[:, :, p, d, c]
                ymean = np.mean(y)
                # remove mean in case there are large gaps in the data
                y -= ymean

                A = partial(hess,
                            W=W, L=L, LH=LH)

                b = W * y
                x = pcg(A,
                        kron_matvec(LH, b),
                        np.zeros((nt, nf), dtype=gain.dtype),
                        tol=1e-6,
                        maxit=500)
                x = W * kron_matvec(L, x)
                x = b - x
                sol[:, :, p, d, c] = kron_tensorvec(Kp, x)  + ymean

    return sol

def interp_gpr(t, f, gain, jhj, tp, fp, sigmaf, lt, lf):
    smooth_gain = blockwise(_interp_gpr, 'tfadc',
                            t, 't',
                            f, 'f',
                            gain, 'tfadc',
                            jhj, 'tfadc',
                            tp, 't',
                            fp, 'f',
                            sigmaf, None,
                            lt, None,
                            lf, None,
                            align_arrays=False,
                            adjust_chunks={'t': tp.chunks[0],
                                           'f': fp.chunks[0]},
                            dtype=gain.dtype)
    return smooth_gain
