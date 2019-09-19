# -*- coding: utf-8 -*-
import numpy as np
from numba import jit, objmode
from numba.typed import List
from cubicalv2.kernels.gjones_chain import invert_gains


@jit(nopython=True, fastmath=True, parallel=False, cache=False, nogil=True)
def chain_solver(model, gain_tuple, inverse_gain_tuple, residual, a1, a2,
                 t_map_list, f_map_list, compute_jhj_and_jhr, compute_update):

    for gain_ind in range(len(gain_tuple)):

        invert_gains(gain_tuple, inverse_gain_tuple)

        for i in range(20):

            jhj, jhr = compute_jhj_and_jhr(model,
                                           gain_tuple,
                                           residual,
                                           a1,
                                           a2,
                                           t_map_list,
                                           f_map_list,
                                           gain_ind,
                                           inverse_gain_tuple)

            update = compute_update(jhj, jhr)

            gain_tuple[gain_ind][:] = (gain_tuple[gain_ind] + update)/2

    return gain_tuple
