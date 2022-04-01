import numba as nb
import numpy as np
from numba import jit
import math


@nb.vectorize([nb.float64(nb.complex128), nb.float32(nb.complex64)])
def cabs2(x):
    """Fast abs^2 for complex numbers which returns a contiguous array."""
    return x.real**2 + x.imag**2


@nb.vectorize([nb.float64(nb.complex128), nb.float32(nb.complex64)])
def cabs(x):
    """Fast abs for complex numbers which returns a contiguous array."""
    return np.sqrt(x.real**2 + x.imag**2)


def gcd(a, b):
    """Find the greatest common divisor of two floating point numbers.

    Adapted from code originally authored by Nikita Tiwari.
    https://www.geeksforgeeks.org/program-find-gcd-floating-point-numbers/

    """
    if (a < b):
        return gcd(b, a)

    # base case
    if (abs(b) < 0.00001):
        return a
    else:
        return (gcd(b, a - math.floor(a / b) * b))


def arr_gcd(arr):
    """Find the greatest common divisor of an array of floats."""

    if arr.ndim != 1:
        raise ValueError("Only 1D arrays are supported.")

    if arr.size < 2:
        return arr[0]  # GCD of a single number is itself.

    net_gcd = gcd(arr[0], arr[1])

    for i in range(2, arr.size):
        net_gcd = gcd(net_gcd, arr[i])

    if not np.all((arr/net_gcd - np.round(arr/net_gcd)) < 1e-8):
        raise ValueError(f"No GCD was found for {arr}.")

    return net_gcd


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def mean_for_index(arr, inds):

    sums = np.zeros(np.max(inds) + 1, dtype=arr.dtype)
    counts = np.zeros_like(sums)

    for i in range(arr.size):
        ind = inds[i]
        sums[ind] += arr[i]
        counts[ind] += 1

    return sums/counts


def fit_hyperplane(x, y):
    """Approximate a surface by a hyperplane in D dimensions

    inputs:
        x - D x N array of coordinates.
        y - N array of (possibly noisy) observations.
            Can be complex valued.

    outputs:
        theta - a vector of coefficients suct that X.T.dot(theta)
                is the hyperplane approximation of y and X is x
                with a row of ones appended as the final axis
    """
    D, N = x.shape
    y = y.squeeze()[None, :]
    z = np.vstack((x, y))
    centroid = np.zeros((D+1, 1), dtype=y.dtype)
    for d in range(D+1):
        if d < D:
            centroid[d, 0] = np.sum(x[d])/N
        else:
            centroid[d, 0] = np.sum(y)/N
    diff = z - centroid
    cov = diff.dot(diff.conj().T)
    s, V = np.linalg.eigh(cov)
    n = V[:, 0].conj()  # defines normal to the plane
    theta = np.zeros(D+1, dtype=y.dtype)
    for d in range(D+1):
        if d < D:
            theta[d] = -n[d]/n[-1]
        else:
            # we need to take the mean here because y can be noisy
            # i.e. we do not have a point exactly in the plane
            theta[d] = np.mean(n[None, 0:-1].dot(x)/n[-1] + y)
    return theta


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


def pcg(A, b, x0, M=None, tol=1e-5, maxit=150):

    if M is None:
        def M(x): return x

    r = A(x0) - b
    y = M(r)
    p = -y
    rnorm = np.vdot(r, y)
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
        rnorm = np.vdot(r, y).real
        alpha = rnorm / np.vdot(p, Ap)
        x = xp + alpha * p
        r = rp + alpha * Ap
        y = M(r)
        rnorm_next = np.vdot(r, y).real
        beta = rnorm_next / rnorm
        p = beta * p - y
        rnorm = rnorm_next
        k += 1
        eps = rnorm / eps0

    if k >= maxit:
        print(f"Max iters reached. eps = {eps}")
    else:
        print(f"Success, converged after {k} iters")
    return x
