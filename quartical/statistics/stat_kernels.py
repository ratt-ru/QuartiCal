import numpy as np
from numba import jit, prange


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def compute_mean_presolve_chisq(data, model, weight, flags):

    n_rows, n_chan, n_dir, n_corr = model.shape

    chisq = 0
    counts = 0

    tmp = np.empty((n_corr,), dtype=np.complex128)

    for row in prange(n_rows):
        for chan in range(n_chan):
            tmp[:] = data[row, chan]
            for dir in range(n_dir):
                for corr in range(n_corr):
                    tmp[corr] -= model[row, chan, dir, corr]
            for corr in range(n_corr):
                if flags[row, chan] != 1:
                    w = weight[row, chan, corr]
                    r = tmp[corr]
                    chisq += (r.conjugate() * w * r).real
                    counts += 1

    if counts:
        chisq /= counts
    else:
        chisq = np.nan

    return np.array([[chisq]], dtype=np.float64)


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def compute_mean_postsolve_chisq(residual, weight, flags):

    n_rows, n_chan, n_corr = residual.shape

    chisq = 0
    counts = 0

    for row in prange(n_rows):
        for chan in range(n_chan):
            for corr in range(n_corr):
                if flags[row, chan] != 1:
                    w = weight[row, chan, corr]
                    r = residual[row, chan, corr]
                    chisq += (r.conjugate() * w * r).real
                    counts += 1

    if counts:
        chisq /= counts
    else:
        chisq = np.nan

    return np.array([[chisq]], dtype=np.float64)
