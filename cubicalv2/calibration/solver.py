# -*- coding: utf-8 -*-
import numpy as np
from numba import jit, objmode
from numba.typed import List
from cubicalv2.kernels.gjones_chain import invert_gains
import time


@jit(nopython=True, fastmath=True, parallel=False, cache=False, nogil=True)
def solver(model, gains, residual, a1, a2, t_map, f_map, compute_jhj_and_jhr,
           compute_update):

    for i in range(20):

        jhj, jhr = compute_jhj_and_jhr(model, gains, residual, a1, a2, t_map,
                                       f_map)

        update = compute_update(jhj, jhr)

        gains = (gains + update)/2

    return gains


@jit(nopython=True, fastmath=True, parallel=False, cache=False, nogil=True)
def init_gains(gain_shapes, dtype=np.complex128):

    gains = []#List()

    for shape in gain_shapes:
        gain = np.zeros(shape, dtype=dtype)
        gain[..., ::3] = 1
        gains.append(gain)

    return gains


@jit(nopython=True, fastmath=True, parallel=False, cache=False, nogil=True)
def chain_solver(model, gain_shapes, residual, a1, a2, t_map_list, f_map_list,
                 compute_jhj_and_jhr, compute_update):

    gain_list = init_gains(gain_shapes)

    for gain_ind in range(len(gain_list)):

        inverse_gain_list = invert_gains(gain_list)

        for i in range(20):

            jhj, jhr = compute_jhj_and_jhr(model,
                                           gain_list,
                                           residual,
                                           a1,
                                           a2,
                                           t_map_list,
                                           f_map_list,
                                           gain_ind,
                                           inverse_gain_list)
            # with objmode():
            #     print(time.time())

            update = compute_update(jhj, jhr)

            # with objmode():
            #     print(time.time())

            gain_list[gain_ind][:] = (gain_list[gain_ind] + update)/2

    return gain_list
