# -*- coding: utf-8 -*-
from numba import jit
from cubicalv2.kernels.gjones_chain import invert_gains
from cubicalv2.kernels.gjones_chain import residual_full
import numpy as np


@jit(nopython=True, fastmath=True, parallel=False, cache=False, nogil=True)
def chain_solver(model, gain_tuple, inverse_gain_tuple, data, a1, a2, weights,
                 t_map_arr, f_map_arr, d_map_arr, compute_jhj_and_jhr,
                 compute_update):

    for gain_ind in range(len(gain_tuple)):

        invert_gains(gain_tuple, inverse_gain_tuple)

        dd_term = gain_tuple[gain_ind].shape[3] > 1

        for i in range(20):

            if dd_term:
                residual = residual_full(data, model, gain_tuple, a1, a2,
                                         t_map_arr, f_map_arr, d_map_arr)
            else:
                residual = data

            jhj, jhr = compute_jhj_and_jhr(model,
                                           gain_tuple,
                                           inverse_gain_tuple,
                                           residual,
                                           a1,
                                           a2,
                                           weights,
                                           t_map_arr,
                                           f_map_arr,
                                           d_map_arr,
                                           gain_ind)

            update = compute_update(jhj, jhr)

            if dd_term:
                gain_tuple[gain_ind][:] = gain_tuple[gain_ind][:] + update/2
            else:
                gain_tuple[gain_ind][:] = (gain_tuple[gain_ind] + update)/2

    return gain_tuple
