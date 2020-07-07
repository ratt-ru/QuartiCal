import numba as nb
import numpy as np
from numba import jit


@nb.vectorize([nb.float64(nb.complex128), nb.float32(nb.complex64)])
def cabs2(x):
    """Fast abs^2 for complex numbers which returns a contiguous array."""
    return x.real**2 + x.imag**2


@nb.vectorize([nb.float64(nb.complex128), nb.float32(nb.complex64)])
def cabs(x):
    """Fast abs for complex numbers which returns a contiguous array."""
    return np.sqrt(x.real**2 + x.imag**2)

