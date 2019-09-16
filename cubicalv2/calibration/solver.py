# -*- coding: utf-8 -*-
import numpy as np
from numba import jit
from numba.typed import List


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

    gains = List()

    for shape in gain_shapes:
        gain = np.zeros(shape, dtype=dtype)
        gain[..., ::3] = 1
        gains.append(gain)

    return gains


@jit(nopython=True, fastmath=True, parallel=False, cache=False, nogil=True)
def chain_solver(model, gain_shapes, residual, a1, a2, t_map, f_map,
                 compute_jhj_and_jhr, compute_update):

    gains = init_gains(gain_shapes)

    for gain_ind in range(len(gains)):
        for i in range(20):
            jhj, jhr = compute_jhj_and_jhr(model, gains[gain_ind], residual,
                                           a1, a2, t_map[gain_ind],
                                           f_map[gain_ind])

            update = compute_update(jhj, jhr)

            # Note the slicing here - erratic behaviour without it.
            gains[gain_ind] = (gains[gain_ind] + update)/2

    return gains
