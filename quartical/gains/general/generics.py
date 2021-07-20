# -*- coding: utf-8 -*-
import numpy as np
from numba import prange, literally, generated_jit, types, jit
from numba.extending import register_jitable
from numba.typed import List
from quartical.utils.numba import coerce_literal
import quartical.gains.general.factories as factories
from quartical.gains.general.convenience import (get_dims,
                                                 get_row,
                                                 old_mul_rweight)


qcgjit = generated_jit(nopython=True,
                       fastmath=True,
                       parallel=False,
                       cache=True,
                       nogil=True)


@qcgjit
def invert_gains(gain_list, inverse_gains, corr_mode):

    coerce_literal(invert_gains, ["corr_mode"])

    compute_det = factories.compute_det_factory(corr_mode)
    iinverse = factories.iinverse_factory(corr_mode)

    def impl(gain_list, inverse_gains, corr_mode):
        for gain_ind, gain in enumerate(gain_list):

            n_tint, n_fint, n_ant, n_dir, n_corr = gain.shape

            igain = inverse_gains[gain_ind]

            for t in range(n_tint):
                for f in range(n_fint):
                    for a in range(n_ant):
                        for d in range(n_dir):

                            gain_sel = gain[t, f, a, d]
                            igain_sel = igain[t, f, a, d]

                            det = compute_det(gain_sel)

                            if np.abs(det) < 1e-6:
                                igain_sel[:] = 0
                            else:
                                iinverse(gain_sel, det, igain_sel)

    return impl


@qcgjit
def compute_residual(data, model, gain_list, a1, a2, t_map_arr, f_map_arr,
                     d_map_arr, row_map, row_weights, corr_mode):

    coerce_literal(compute_residual, ["corr_mode"])

    imul_rweight = factories.imul_rweight_factory(corr_mode, row_weights)
    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    v1_imul_v2ct = factories.v1_imul_v2ct_factory(corr_mode)
    isub = factories.isub_factory(corr_mode)
    iunpack = factories.iunpack_factory(corr_mode)
    valloc = factories.valloc_factory(corr_mode)

    def impl(data, model, gain_list, a1, a2, t_map_arr, f_map_arr,
             d_map_arr, row_map, row_weights, corr_mode):

        residual = data.copy()

        n_rows, n_chan, n_dir, _ = get_dims(model, row_map)
        n_gains = len(gain_list)

        for row_ind in prange(n_rows):

            row = get_row(row_ind, row_map)
            a1_m, a2_m = a1[row], a2[row]
            v = valloc(np.complex128)  # Hold GMGH.

            for f in range(n_chan):

                r = residual[row, f]
                m = model[row, f]

                for d in range(n_dir):

                    iunpack(v, m[d])

                    for g in range(n_gains - 1, -1, -1):

                        t_m = t_map_arr[row_ind, g]
                        f_m = f_map_arr[f, g]
                        d_m = d_map_arr[g, d]  # Broadcast dir.

                        gain = gain_list[g][t_m, f_m]
                        gain_p = gain[a1_m, d_m]
                        gain_q = gain[a2_m, d_m]

                        v1_imul_v2(gain_p, v, v)
                        v1_imul_v2ct(v, gain_q, v)

                    imul_rweight(v, v, row_weights, row_ind)
                    isub(r, v)

        return residual

    return impl


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def compute_corrected_residual(residual, gain_list, a1, a2, t_map_arr,
                               f_map_arr, d_map_arr, row_map, row_weights,
                               mode):

    if not isinstance(mode, types.Literal):
        return lambda residual, gain_list, a1, a2, t_map_arr, f_map_arr, \
                      d_map_arr, row_map, row_weights, mode: literally(mode)

    if mode.literal_value == "diag":
        impl = corrected_residual_diag
    else:
        impl = corrected_residual_full

    return impl


@register_jitable
def corrected_residual_diag(residual, gain_list, a1, a2, t_map_arr,
                            f_map_arr, d_map_arr, row_map, row_weights, mode):

    corrected_residual = np.zeros_like(residual)

    inverse_gain_list = List()

    for gain_term in gain_list:
        inverse_gain_list.append(np.empty_like(gain_term))

    invert_gains(gain_list, inverse_gain_list, mode)

    n_rows, n_chan, _ = get_dims(residual, row_map)
    n_gains = len(gain_list)

    for row_ind in range(n_rows):

        row = get_row(row_ind, row_map)
        a1_m, a2_m = a1[row], a2[row]

        for f in range(n_chan):

            cr00 = residual[row, f, 0]
            cr11 = residual[row, f, 1]

            for g in range(n_gains):

                t_m = t_map_arr[row_ind, g]
                f_m = f_map_arr[f, g]

                gain = inverse_gain_list[g]

                g00 = gain[t_m, f_m, a1_m, 0, 0]
                g11 = gain[t_m, f_m, a1_m, 0, 1]

                gh00 = gain[t_m, f_m, a2_m, 0, 0].conjugate()
                gh11 = gain[t_m, f_m, a2_m, 0, 1].conjugate()

                cr00 = g00*cr00*gh00
                cr11 = g11*cr11*gh11

            corrected_residual[row, f, 0] += \
                old_mul_rweight(cr00, row_weights, row_ind)
            corrected_residual[row, f, 1] += \
                old_mul_rweight(cr11, row_weights, row_ind)

    return corrected_residual


@register_jitable
def corrected_residual_full(residual, gain_list, a1, a2, t_map_arr,
                            f_map_arr, d_map_arr, row_map, row_weights, mode):

    corrected_residual = np.zeros_like(residual)

    inverse_gain_list = List()

    for gain_term in gain_list:
        inverse_gain_list.append(np.empty_like(gain_term))

    invert_gains(gain_list, inverse_gain_list, mode)

    n_rows, n_chan, _ = get_dims(residual, row_map)
    n_gains = len(gain_list)

    for row_ind in range(n_rows):

        row = get_row(row_ind, row_map)
        a1_m, a2_m = a1[row], a2[row]

        for f in range(n_chan):

            cr00 = residual[row, f, 0]
            cr01 = residual[row, f, 1]
            cr10 = residual[row, f, 2]
            cr11 = residual[row, f, 3]

            for g in range(n_gains):

                t_m = t_map_arr[row_ind, g]
                f_m = f_map_arr[f, g]

                gain = inverse_gain_list[g]

                g00 = gain[t_m, f_m, a1_m, 0, 0]
                g01 = gain[t_m, f_m, a1_m, 0, 1]
                g10 = gain[t_m, f_m, a1_m, 0, 2]
                g11 = gain[t_m, f_m, a1_m, 0, 3]

                gh00 = gain[t_m, f_m, a2_m, 0, 0].conjugate()
                gh01 = gain[t_m, f_m, a2_m, 0, 2].conjugate()
                gh10 = gain[t_m, f_m, a2_m, 0, 1].conjugate()
                gh11 = gain[t_m, f_m, a2_m, 0, 3].conjugate()

                gr00 = (g00*cr00 + g01*cr10)
                gr01 = (g00*cr01 + g01*cr11)
                gr10 = (g10*cr00 + g11*cr10)
                gr11 = (g10*cr01 + g11*cr11)

                cr00 = (gr00*gh00 + gr01*gh10)
                cr01 = (gr00*gh01 + gr01*gh11)
                cr10 = (gr10*gh00 + gr11*gh10)
                cr11 = (gr10*gh01 + gr11*gh11)

            corrected_residual[row, f, 0] += \
                old_mul_rweight(cr00, row_weights, row_ind)
            corrected_residual[row, f, 1] += \
                old_mul_rweight(cr01, row_weights, row_ind)
            corrected_residual[row, f, 2] += \
                old_mul_rweight(cr10, row_weights, row_ind)
            corrected_residual[row, f, 3] += \
                old_mul_rweight(cr11, row_weights, row_ind)

    return corrected_residual


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


@generated_jit(nopython=True, nogil=True, fastmath=True, cache=True)
def combine_gains(t_bin_arr, f_map_arr, d_map_arr, net_shape, mode, *gains):

    if not isinstance(mode, types.Literal):
        return lambda t_bin_arr, f_map_arr, d_map_arr, \
                      net_shape, mode, *gains: literally(mode)

    v1_imul_v2 = factories.v1_imul_v2_factory(mode)

    def impl(t_bin_arr, f_map_arr, d_map_arr, net_shape, mode, *gains):
        t_bin_arr = t_bin_arr[0]
        f_map_arr = f_map_arr[0]

        n_time = t_bin_arr.shape[0]
        n_freq = f_map_arr.shape[0]

        _, _, n_ant, n_dir, n_corr = net_shape

        net_gains = np.zeros((n_time, n_freq, n_ant, n_dir, n_corr),
                             dtype=np.complex128)
        net_gains[..., 0] = 1
        net_gains[..., -1] = 1

        n_term = len(gains)

        for t in range(n_time):
            for f in range(n_freq):
                for a in range(n_ant):
                    for d in range(n_dir):
                        for gi in range(n_term):
                            tm = t_bin_arr[t, gi]
                            fm = f_map_arr[f, gi]
                            dm = d_map_arr[gi, d]
                            v1_imul_v2(net_gains[t, f, a, d],
                                       gains[gi][tm, fm, a, dm],
                                       net_gains[t, f, a, d])

        return net_gains

    return impl
