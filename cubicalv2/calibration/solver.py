# -*- coding: utf-8 -*-
import numpy as np
from numba import jit


@jit(nopython=True, fastmath=True, parallel=False, cache=False, nogil=True)
def solver(model, gains, residual, a1, a2, t_map, f_map, f1, f2):

    compute_jhj_and_jhr = f1
    compute_update = f2

    for i in range(10):

        jhj, jhr = compute_jhj_and_jhr(model, gains, residual, a1, a2, t_map,
                                       f_map)

        update = compute_update(jhj, jhr)

        gains = (gains + update)/2

    return gains
