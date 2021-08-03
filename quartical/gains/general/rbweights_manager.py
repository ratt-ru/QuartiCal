# -*- coding: utf-8 -*-
from numba import  jit, prange
import numpy as np
import quartical.gains.general.rbweights_kernel as weights_kernel


@jit(fastmath=True, nopython=True, cache=True, nogil=True)
def py_fast_digamma(x):
    # "Faster digamma function assumes x > 0."
    r = 0
    while (x<=5):
        r -= 1/x
        x += 1
    f = 1/(x*x)
    t = f*(-1/12.0 + f*(1/120.0 + f*(-1/252.0 + f*(1/240.0 + f*(-1/132.0
        + f*(691/32760.0 + f*(-1/12.0 + f*3617/8160.0)))))))
    return r + np.log(x) - 0.5/x + t

@jit(fastmath=True, nopython=True, cache=True, nogil=True, parallel=True)
def vfunc(wn, a, npol, m):
    vval = 0
    nt, nf, _ = wn.shape
    vgamma = py_fast_digamma(a+npol) - np.log(a+npol) -  py_fast_digamma(a) + np.log(a) + 1
    for i in prange(nt):
        for j in prange(nf):
            w = wn[i, j, 0]
            if w != 0:
                vval += np.log(w) - w
    
    vval *= 1./m
    vval += vgamma
    return vval

@jit(fastmath=True, nopython=True, cache=True, nogil=True, parallel=True)
def brute_solve_v(wn, npol, m):
    vvals = [3, 4, 5, 6, 7, 8, 9, 10, 13, 16, 20, 25, 30, 40]
    nvals = len(vvals)
    root = 2
    fval0 = np.abs(vfunc(wn, 2, npol, m))
    for ival in prange(nvals):
        fval = np.abs(vfunc(wn, vvals[ival], npol, m))
        if fval<fval0:
            root = ival
            fval0 = fval
    return root

@jit(fastmath=True, nopython=True, cache=True, nogil=True, parallel=True)
def get_number_of_unflaggedw(arr, nc):
    flattened = arr.ravel()
    sum_ = 0
    for i in prange(flattened.size):
        sum_ += flattened[i] != 0
    return sum_/nc

@jit(fastmath=True, nopython=True, cache=True, nogil=True)
def normalise_cov(ompstd, Nvis, cov_thresh):
	
    ompstd /= (Nvis + 1e-8)
    
    if ompstd[0,0].real>cov_thresh:
        fixed_v = 1
    elif ompstd[1,1].real>cov_thresh:
        fixed_v = 1
    elif ompstd[2,2].real>cov_thresh:
        fixed_v = 1
    elif ompstd[3,3].real>cov_thresh:
        fixed_v = 1
    elif Nvis < 1:
        fixed_v = 1
    else:
        fixed_v = 0

    return fixed_v

@jit(fastmath=True, cache=True, nogil=True, nopython=True) 
def compute_covinv(gains, residual,
                    weights, t_map_arr, f_map_arr, active_term, row_map, n_valid, cov_thresh, corr_mode):
        
    ompstd = np.zeros((4,4), dtype=residual.dtype)
    
    weights_kernel.compute_cov(gains, residual, ompstd, 
             weights, t_map_arr, f_map_arr, active_term, row_map, corr_mode)

    fixed_v = normalise_cov(ompstd, n_valid, cov_thresh)
    covinv = np.eye(4, dtype=ompstd.dtype)
    eps = 1e-8
    
    if fixed_v:		
        covinv[0,0] *= 1./cov_thresh
        covinv[1,1] *= 1./cov_thresh
        covinv[2,2] *= 1./cov_thresh
        covinv[3,3] *= 1./cov_thresh
    else:
        covinv[0,0] *= 1/(ompstd[0,0] + eps)
        covinv[1,1] *= 1/(ompstd[1,1] + eps)  # don't trust the cross correlations
        covinv[2,2] *= 1/(ompstd[2,2] + eps)  # 
        covinv[3,3] *= 1/(ompstd[3,3] + eps)

    if corr_mode != "full":
        covinv[1,1] = 0
        covinv[2,2] = 0
	
    return covinv, fixed_v

@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def update_weights(gains, residual, v,
                     weights, t_map_arr, f_map_arr, active_term, row_map, cov_thresh, robust_thresh, corr_mode):
    
    npol = 4 if corr_mode=="full" else 2
    n_valid = get_number_of_unflaggedw(weights, npol)
    
    icov, fixed_v = compute_covinv(gains, residual,
                    weights, t_map_arr, f_map_arr, active_term, row_map, n_valid, cov_thresh, corr_mode)

    wsum = np.array([0], dtype=weights.dtype)
    weights_kernel.compute_weights(gains, residual, icov, v,
                    weights, t_map_arr, f_map_arr, active_term, row_map, robust_thresh, wsum, corr_mode)
    
    #----------------normalise the weights----------------------------------------------#
    n_valid = get_number_of_unflaggedw(weights, npol)
    norm = wsum[0].real/(n_valid+1e-8)
    if norm !=0:
        weights /= norm

    #-----------compute the v parameter---------------------#
    # TODO make this computation only after a certain number of iterations (maybe 5)

    if fixed_v:
        v = 2
    else:
        v = brute_solve_v(weights, npol, n_valid)
    
   
    return v






