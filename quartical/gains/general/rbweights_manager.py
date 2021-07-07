# -*- coding: utf-8 -*-
from numba import prange, literally, generated_jit, jit
import numpy as np
from quartical.gains.general.convenience import (get_row,
                                                 get_chan_extents,
                                                 get_row_extents)
import quartical.gains.general.factories as factories
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

@jit(fastmath=True, nopython=True, cache=True, nogil=True)
def vfunc(wn, a, npol):
    m = len(wn)
    vval =  py_fast_digamma(a+npol) - np.log(a+npol) -  py_fast_digamma(a) + np.log(a) + (1./m)*np.sum(np.log(wn) - wn) + 1
    return vval

@jit(fastmath=True, nopython=True, cache=True, nogil=True)
def brute_solve_v(wn, fixed_v, npol):

    if fixed_v:
        return 2
    else:
        root = 2
        fval0 = np.abs(vfunc(wn.real, 2, npol))
        for ival in range(3, 41):
            fval = np.abs(vfunc(wn.real, ival, npol))
            if fval<fval0:
                root = ival
                fval0 = fval
        
        return root

@jit(fastmath=True, nopython=True, cache=True, nogil=True)
def get_number_of_unflaggedw(wn):
    
    nt, nf, nc = wn.shape
    ww = wn.real!=0
    Nvis = np.sum(ww)/nc

    return Nvis

@jit(fastmath=True, nopython=True, cache=True, nogil=True)
def normalise_cov(ompstd, Nvis):
	eps = 1e-6
	cov_thresh = 200
	ompstd /=Nvis 
	ompstd += eps**2

	cov_thresh = 200
	if ompstd[0,0].real>cov_thresh:
		fixed_v = 1
	elif ompstd[1,1].real>cov_thresh:
		fixed_v = 1
	elif ompstd[2,2].real>cov_thresh:
		fixed_v = 1
	elif ompstd[3,3].real>cov_thresh:
		fixed_v = 1
	else:
		fixed_v = 0
	
	return fixed_v

@jit(fastmath=True, cache=True, nogil=True, nopython=True) 
def compute_covinv(model, gains, residual,
                    weights, t_map_arr, f_map_arr, active_term, row_map, n_valid, corr_mode):
        
    # TODO implement the option to pass option and do all the other traditional stuffs
    ompstd = np.zeros((4,4), dtype=residual.dtype)

    weights_kernel.compute_cov(model, gains, residual, ompstd, 
             weights, t_map_arr, f_map_arr, active_term, row_map, corr_mode)

    # print(ompstd, "-> corrmode ", corr_mode, " normalsied cov", n_valid)

    fixed_v = normalise_cov(ompstd, n_valid)
    covinv = np.eye(4, dtype=ompstd.dtype)

    # print(ompstd, "-> corrmode ", corr_mode, " normalsied cov", n_valid)
    
    if fixed_v:	
        cov_thresh = 200	
        covinv[0,0] *= 1./cov_thresh
        covinv[1,1] *= 1./cov_thresh
        covinv[2,2] *= 1./cov_thresh
        covinv[3,3] *= 1./cov_thresh
    else:
        covinv[0,0] *= 1/ompstd[0,0]
        covinv[1,1] *= 1/ompstd[1,1]
        covinv[2,2] *= 1/ompstd[2,2]
        covinv[3,3] *= 1/ompstd[3,3]

    if corr_mode != "full":
        covinv[1,1] = 0
        covinv[2,2] = 0
	
    return covinv, fixed_v

@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def update_weights(model, gains, residual, v,
                     weights, t_map_arr, f_map_arr, active_term, row_map, n_valid, corr_mode):
    
    icov, fixed_v = compute_covinv(model, gains, residual,
                    weights, t_map_arr, f_map_arr, active_term, row_map, n_valid, corr_mode)

    weights_kernel.compute_weights(model, gains, residual, icov, v,
                    weights, t_map_arr, f_map_arr, active_term, row_map, corr_mode)

    w_sum = np.sum(weights[:,:,0])
    norm = w_sum.real/n_valid  # normlaise by average of the valid visibilities

    weights/= norm

    #-----------computing the v parameter---------------------#
    # TODO make this computation only after a certain number of iterations (maybe 5)
    npol = 4 if corr_mode=="full" else 2
    ww = weights[:,:,0].flatten()
    v = brute_solve_v(ww[ww!=0], fixed_v, npol) # remove zero weights
    
    # todo implement the weight flagging
    # self.not_all_flagged = self.flag_weights()
  
    return v






