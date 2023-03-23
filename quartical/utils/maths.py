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


def float_gcd(a, b, rtol=1e-05, atol=1e-08):
    t = min(abs(a), abs(b))
    while abs(b) > rtol * t + atol:
        a, b = b, a % b
    return a


def arr_gcd(arr):
    """Find the greatest common divisor of an array of floats."""

    if arr.ndim != 1:
        raise ValueError("Only 1D arrays are supported.")

    if arr.size < 2:
        return arr[0]  # GCD of a single number is itself.

    # NOTE: May need to tune precision here.
    net_gcd = float_gcd(arr[0], arr[1], rtol=1e-3, atol=1e-3)

    for i in range(2, arr.size):
        net_gcd = float_gcd(net_gcd, arr[i], rtol=1e-3, atol=1e-3)

    if not np.all((arr/net_gcd - np.round(arr/net_gcd)) < 1e-3):
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
