# -*- coding: utf-8 -*-
import numpy as np
from numba import prange, generated_jit, jit
from numba.typed import List
from quartical.utils.numba import coerce_literal
import quartical.gains.general.factories as factories
from quartical.gains.general.convenience import get_dims, get_row


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


@qcgjit
def compute_corrected_residual(residual, gain_list, a1, a2, t_map_arr,
                               f_map_arr, d_map_arr, row_map, row_weights,
                               corr_mode):

    coerce_literal(compute_corrected_residual, ["corr_mode"])

    imul_rweight = factories.imul_rweight_factory(corr_mode, row_weights)
    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    v1_imul_v2ct = factories.v1_imul_v2ct_factory(corr_mode)
    iadd = factories.iadd_factory(corr_mode)
    iunpack = factories.iunpack_factory(corr_mode)
    valloc = factories.valloc_factory(corr_mode)

    def impl(residual, gain_list, a1, a2, t_map_arr, f_map_arr,
             d_map_arr, row_map, row_weights, corr_mode):

        corrected_residual = np.zeros_like(residual)

        inverse_gain_list = List()

        for gain_term in gain_list:
            inverse_gain_list.append(np.empty_like(gain_term))

        invert_gains(gain_list, inverse_gain_list, corr_mode)

        n_rows, n_chan, _ = get_dims(residual, row_map)
        n_gains = len(gain_list)

        r = valloc(np.complex128)

        for row_ind in range(n_rows):

            row = get_row(row_ind, row_map)
            a1_m, a2_m = a1[row], a2[row]

            for f in range(n_chan):

                iunpack(r, residual[row, f])
                cr = corrected_residual[row, f]

                for g in range(n_gains):

                    t_m = t_map_arr[row_ind, g]
                    f_m = f_map_arr[f, g]

                    igain = inverse_gain_list[g][t_m, f_m]
                    igain_p = igain[a1_m, 0]  # Only correct in direction 0.
                    igain_q = igain[a2_m, 0]

                    v1_imul_v2(igain_p, r, r)
                    v1_imul_v2ct(r, igain_q, r)

                imul_rweight(r, r, row_weights, row_ind)
                iadd(cr, r)

        return corrected_residual

    return impl


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def compute_convergence(gain, last_gain, criteria):

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
                    n_cnvgd += tmp_diff_abs2/tmp_abs2 < criteria**2

    return n_cnvgd/(n_tint*n_fint*n_dir)


@qcgjit
def combine_gains(t_bin_arr, f_map_arr, d_map_arr, net_shape, corr_mode,
                  *gains):

    coerce_literal(combine_gains, ["corr_mode"])

    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)

    def impl(t_bin_arr, f_map_arr, d_map_arr, net_shape, corr_mode, *gains):
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


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def per_array_jhj_jhr(jhj, jhr):
    """This manipulates the entries of jhj and jhr to be over all antennas."""

    def impl(jhj, jhr):

        n_tint, n_fint, n_ant = jhj.shape[:3]

        for t in range(n_tint):
            for f in range(n_fint):
                for a in range(1, n_ant):
                    jhj[t, f, 0] += jhj[t, f, a]
                    jhr[t, f, 0] += jhr[t, f, a]

                for a in range(1, n_ant):
                    jhj[t, f, a] = jhj[t, f, 0]
                    jhr[t, f, a] = jhr[t, f, 0]

    return impl
