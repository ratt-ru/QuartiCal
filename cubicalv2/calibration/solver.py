# -*- coding: utf-8 -*-
import numpy as np
from numba import jit


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
def chain_solver(model, gains, residual, a1, a2, t_map, f_map,
                 compute_jhj_and_jhr, compute_update):

    for i in range(20):
        for gain_ind in range(len(gains)):

            jhj, jhr = compute_jhj_and_jhr(model, gains[gain_ind], residual,
                                           a1, a2, t_map, f_map)

            update = compute_update(jhj, jhr)

            gains[gain_ind] = (gains[gain_ind] + update)/2

    return gains
