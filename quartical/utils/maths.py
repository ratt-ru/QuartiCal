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
