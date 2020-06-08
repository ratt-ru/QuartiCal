# -*- coding: utf-8 -*-
import numpy as np
from numba.extending import overload
from numba import jit, prange, literally


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def invert_gains(gain_list, inverse_gain_list, mode):

    for gain_ind, gain in enumerate(gain_list):

        n_tint, n_fint, n_ant, n_dir, n_corr = gain.shape

        inverse_gain = inverse_gain_list[gain_ind]

        for t in range(n_tint):
            for f in range(n_fint):
                for a in range(n_ant):
                    for d in range(n_dir):

                        _invert(gain[t, f, a, d, :],
                                inverse_gain[t, f, a, d, :],
                                literally(mode))


def _invert(gain, inverse_gain, mode):
    pass


@overload(_invert, inline='always')
def _invert_impl(gain, inverse_gain, mode):

    if mode.literal_value == "diag":
        def _invert_impl(gain, inverse_gain, mode):

            g00 = gain[..., 0]
            g11 = gain[..., 1]

            det = g00*g11

            if np.abs(det) < 1e-6:
                inverse_gain[..., 0] = 0
                inverse_gain[..., 1] = 0
            else:
                inverse_gain[..., 0] = 1/g00
                inverse_gain[..., 1] = 1/g11

        return _invert_impl

    else:
        def _invert_impl(gain, inverse_gain, mode):

            g00 = gain[..., 0]
            g01 = gain[..., 1]
            g10 = gain[..., 2]
            g11 = gain[..., 3]

            det = (g00*g11 - g01*g10)

            if np.abs(det) < 1e-6:
                inverse_gain[..., 0] = 0
                inverse_gain[..., 1] = 0
                inverse_gain[..., 2] = 0
                inverse_gain[..., 3] = 0
            else:
                inverse_gain[..., 0] = g11/det
                inverse_gain[..., 1] = -g01/det
                inverse_gain[..., 2] = -g10/det
                inverse_gain[..., 3] = g00/det

        return _invert_impl


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def compute_residual(data, model, gain_list, a1, a2, t_map_arr,
                     f_map_arr, d_map_arr, mode):

    return _compute_residual(data, model, gain_list, a1, a2, t_map_arr,
                             f_map_arr, d_map_arr, literally(mode))


def _compute_residual(data, model, gain_list, a1, a2, t_map_arr,
                      f_map_arr, d_map_arr, mode):
    pass


@overload(_compute_residual, inline='always')
def _compute_residual_impl(data, model, gain_list, a1, a2, t_map_arr,
                           f_map_arr, d_map_arr, mode):

    if mode.literal_value == "diag":
        return residual_diag
    else:
        return residual_full


def residual_diag(data, model, gain_list, a1, a2, t_map_arr, f_map_arr,
                  d_map_arr, mode):

    residual = data.copy()

    n_rows, n_chan, n_dir, _ = model.shape
    n_gains = len(gain_list)

    for row in range(n_rows):

        a1_m, a2_m = a1[row], a2[row]

        for f in range(n_chan):
            for d in range(n_dir):

                m00 = model[row, f, d, 0]
                m11 = model[row, f, d, 1]

                for g in range(n_gains-1, -1, -1):

                    d_m = d_map_arr[g, d]  # Broadcast dir.
                    t_m = t_map_arr[row, g]
                    f_m = f_map_arr[f, g]

                    gain = gain_list[g]

                    g00 = gain[t_m, f_m, a1_m, d_m, 0]
                    g11 = gain[t_m, f_m, a1_m, d_m, 1]

                    gh00 = gain[t_m, f_m, a2_m, d_m, 0].conjugate()
                    gh11 = gain[t_m, f_m, a2_m, d_m, 1].conjugate()

                    r00 = g00*m00*gh00
                    r11 = g11*m11*gh11

                    m00 = r00
                    m11 = r11

                residual[row, f, 0] -= r00
                residual[row, f, 1] -= r11

    return residual


def residual_full(data, model, gain_list, a1, a2, t_map_arr, f_map_arr,
                  d_map_arr, mode):

    residual = data.copy()

    n_rows, n_chan, n_dir, _ = model.shape
    n_gains = len(gain_list)

    for row in prange(n_rows):

        a1_m, a2_m = a1[row], a2[row]

        for f in range(n_chan):
            for d in range(n_dir):

                m00 = model[row, f, d, 0]
                m01 = model[row, f, d, 1]
                m10 = model[row, f, d, 2]
                m11 = model[row, f, d, 3]

                for g in range(n_gains-1, -1, -1):

                    d_m = d_map_arr[g, d]  # Broadcast dir.
                    t_m = t_map_arr[row, g]
                    f_m = f_map_arr[f, g]

                    gain = gain_list[g]

                    g00 = gain[t_m, f_m, a1_m, d_m, 0]
                    g01 = gain[t_m, f_m, a1_m, d_m, 1]
                    g10 = gain[t_m, f_m, a1_m, d_m, 2]
                    g11 = gain[t_m, f_m, a1_m, d_m, 3]

                    gh00 = gain[t_m, f_m, a2_m, d_m, 0].conjugate()
                    gh01 = gain[t_m, f_m, a2_m, d_m, 2].conjugate()
                    gh10 = gain[t_m, f_m, a2_m, d_m, 1].conjugate()
                    gh11 = gain[t_m, f_m, a2_m, d_m, 3].conjugate()

                    gm00 = (g00*m00 + g01*m10)
                    gm01 = (g00*m01 + g01*m11)
                    gm10 = (g10*m00 + g11*m10)
                    gm11 = (g10*m01 + g11*m11)

                    r00 = (gm00*gh00 + gm01*gh10)
                    r01 = (gm00*gh01 + gm01*gh11)
                    r10 = (gm10*gh00 + gm11*gh10)
                    r11 = (gm10*gh01 + gm11*gh11)

                    m00 = r00
                    m01 = r01
                    m10 = r10
                    m11 = r11

                residual[row, f, 0] -= r00
                residual[row, f, 1] -= r01
                residual[row, f, 2] -= r10
                residual[row, f, 3] -= r11

    return residual


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def compute_convergence(gain, last_gain):

    n_tint, n_fint, n_ant, n_dir, n_corr = gain.shape

    tmp_abs2 = 0
    tmp_diff_abs2 = 0

    n_cnvgd = 0

    for ti in range(n_tint):
        for fi in range(n_fint):
            for d in range(n_dir):

                tmp_abs2 = 0
                tmp_diff_abs2 = 0

                for a in range(n_ant):
                    for c in range(n_corr):

                        gsel = gain[ti, fi, a, d, c]
                        lgsel = last_gain[ti, fi, a, d, c]

                        diff = lgsel - gsel

                        tmp_abs2 += gsel.real**2 + gsel.imag**2
                        tmp_diff_abs2 += diff.real**2 + diff.imag**2

                if tmp_abs2 == 0:
                    n_cnvgd += 1
                else:
                    n_cnvgd += tmp_diff_abs2/tmp_abs2 < 1e-6**2

    return n_cnvgd/(n_tint*n_fint*n_dir)