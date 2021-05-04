# -*- coding: utf-8 -*-
import numpy as np
from numba import prange, literally, generated_jit, types
from quartical.kernels.generics import (invert_gains,
                                        compute_residual,
                                        compute_convergence)
from quartical.kernels.convenience import (get_row,
                                           get_chan_extents,
                                           get_row_extents)
import quartical.kernels.factories as factories
from collections import namedtuple


# This can be done without a named tuple now. TODO: Add unpacking to
# constructor.
stat_fields = {"conv_iters": np.int64,
               "conv_perc": np.float64}

term_conv_info = namedtuple("term_conv_info", " ".join(stat_fields.keys()))


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def delay_solver(model, data, a1, a2, weights, t_map_arr, f_map_arr,
                 d_map_arr, corr_mode, active_term, inverse_gains,
                 gains, flags, params, chan_freqs, row_map, row_weights,
                 t_bin_arr):
    """Solve for a delay.

    Note that the paramter vector is ordered as [offset slope].

    """

    if not isinstance(corr_mode, types.Literal):
        return lambda model, data, a1, a2, weights, t_map_arr, f_map_arr, \
                      d_map_arr, corr_mode, active_term, inverse_gains, \
                      gains, flags, params, chan_freqs, row_map, row_weights, \
                      t_bin_arr: literally(corr_mode)

    compute_jhj_jhr = jhj_jhr
    compute_update = update
    finalize_update = finalize

    def impl(model, data, a1, a2, weights, t_map_arr, f_map_arr, d_map_arr,
             corr_mode, active_term, inverse_gains, gains, flags, params,
             chan_freqs, row_map, row_weights, t_bin_arr):

        param_shape = params.shape
        n_tint, n_fint, n_ant, n_dir, n_param, n_corr = param_shape
        n_ppa = 4  # This is always the case.

        invert_gains(gains, inverse_gains, corr_mode)

        dd_term = n_dir > 1

        last_gain = gains[active_term].copy()

        cnv_perc = 0.

        real_dtype = gains[active_term].real.dtype

        # TODO: This n_param**2 component can be optimised but that may
        # introduce unecessary complexity.
        jhj_shape = (n_tint, n_fint, n_ant, n_dir, n_ppa, n_ppa)
        jhj = np.empty(jhj_shape, dtype=real_dtype)
        jhr_shape = (n_tint, n_fint, n_ant, n_dir, n_ppa)
        jhr = np.empty(jhr_shape, dtype=real_dtype)
        update = np.empty(jhr_shape, dtype=real_dtype)

        for i in range(20):

            if dd_term:
                residual = compute_residual(data, model, gains, a1, a2,
                                            t_map_arr, f_map_arr, d_map_arr,
                                            row_map, row_weights,
                                            corr_mode)
            else:
                residual = data

            compute_jhj_jhr(jhj,
                            jhr,
                            model,
                            gains,
                            inverse_gains,
                            chan_freqs,
                            residual,
                            a1,
                            a2,
                            weights,
                            t_map_arr,
                            f_map_arr,
                            d_map_arr,
                            row_map,
                            row_weights,
                            active_term,
                            corr_mode)

            compute_update(update,
                           jhj,
                           jhr,
                           corr_mode)

            finalize_update(update,
                            params,
                            gains[active_term],
                            chan_freqs,
                            t_bin_arr,
                            f_map_arr,
                            d_map_arr,
                            dd_term,
                            corr_mode,
                            active_term)

            # Check for gain convergence. TODO: This can be affected by the
            # weights. Currently unsure how or why, but using unity weights
            # leads to monotonic convergence in all solution intervals.

            cnv_perc = compute_convergence(gains[active_term][:], last_gain)

            last_gain[:] = gains[active_term][:]

            if cnv_perc > 0.99:
                break

        return term_conv_info(i, cnv_perc)

    return impl


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def jhj_jhr(jhj, jhr, model, gains, inverse_gains, chan_freqs,
            residual, a1, a2, weights, t_map_arr, f_map_arr, d_map_arr,
            row_map, row_weights, active_term, corr_mode):

    imul_rweight = factories.imul_rweight_factory(corr_mode, row_weights)
    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    v1_imul_v2ct = factories.v1_imul_v2ct_factory(corr_mode)
    v1ct_imul_v2 = factories.v1ct_imul_v2_factory(corr_mode)
    iunpack = factories.iunpack_factory(corr_mode)
    iunpackct = factories.iunpackct_factory(corr_mode)
    iwmul = factories.iwmul_factory(corr_mode)
    valloc = factories.valloc_factory(corr_mode)
    loop_var = factories.loop_var_factory(corr_mode)
    set_identity = factories.set_identity_factory(corr_mode)
    accumulate_jhr = jhr_factory(corr_mode)
    jhmul = special_jh_mul_factory(corr_mode)
    jhwjmul = special_jh_wmul_j_factory(corr_mode)

    def impl(jhj, jhr, model, gains, inverse_gains, chan_freqs,
             residual, a1, a2, weights, t_map_arr, f_map_arr, d_map_arr,
             row_map, row_weights, active_term, corr_mode):
        _, n_chan, n_dir, n_corr = model.shape

        jhj[:] = 0
        jhr[:] = 0

        n_tint, n_fint, n_ant, n_gdir, n_ppa = jhr.shape
        n_int = n_tint*n_fint

        complex_dtype = gains[active_term].dtype

        n_gains = len(gains)

        # Determine the starts and stops of the rows and channels associated
        # with each solution interval. This could even be moved out for speed.
        row_starts, row_stops = get_row_extents(t_map_arr,
                                                active_term,
                                                n_tint)

        chan_starts, chan_stops = get_chan_extents(f_map_arr,
                                                   active_term,
                                                   n_fint,
                                                   n_chan)

        all_terms, gt_active, lt_active = loop_var(n_gains, active_term)

        # Parallel over all solution intervals.
        for i in prange(n_int):

            ti = i//n_fint
            fi = i - ti*n_fint

            rs = row_starts[ti]
            re = row_stops[ti]
            fs = chan_starts[fi]
            fe = chan_stops[fi]

            m_vec = valloc(complex_dtype)
            mh_vec = valloc(complex_dtype)
            r_vec = valloc(complex_dtype)
            rh_vec = valloc(complex_dtype)

            gains_a = valloc(complex_dtype, leading_dims=(n_gains,))
            gains_b = valloc(complex_dtype, leading_dims=(n_gains,))

            lmul_op_a = valloc(complex_dtype)
            lmul_op_b = valloc(complex_dtype)

            tmp_jh_p = np.empty((n_gdir, n_ppa, n_ppa), dtype=complex_dtype)
            tmp_jh_q = np.empty((n_gdir, n_ppa, n_ppa), dtype=complex_dtype)

            for row_ind in range(rs, re):

                row = get_row(row_ind, row_map)
                a1_m, a2_m = a1[row], a2[row]

                for f in range(fs, fe):

                    r = residual[row, f]
                    w = weights[row, f]  # Consider a map?

                    tmp_jh_p[:, :, :] = 0
                    tmp_jh_q[:, :, :] = 0

                    nu = chan_freqs[f]

                    for d in range(n_dir):

                        set_identity(lmul_op_a)
                        set_identity(lmul_op_b)

                        # Construct a small contiguous gain array.
                        for gi in range(n_gains):
                            d_m = d_map_arr[gi, d]  # Broadcast dir.
                            t_m = t_map_arr[row_ind, gi]
                            f_m = f_map_arr[f, gi]

                            gain = gains[gi][t_m, f_m]

                            iunpack(gains_a[gi], gain[a1_m, d_m])
                            iunpack(gains_b[gi], gain[a2_m, d_m])

                        imul_rweight(r, r_vec, row_weights, row_ind)
                        iwmul(r_vec, w)
                        iunpackct(rh_vec, r_vec)

                        m = model[row, f, d]
                        imul_rweight(m, m_vec, row_weights, row_ind)
                        iunpackct(mh_vec, m_vec)

                        for g in all_terms:     # Unchanged

                            gb = gains_b[g]
                            v1_imul_v2(gb, mh_vec, mh_vec)

                            ga = gains_a[g]
                            v1_imul_v2(ga, m_vec, m_vec)

                        for g in gt_active:     # Unchanged

                            ga = gains_a[g]
                            v1_imul_v2ct(mh_vec, ga, mh_vec)

                            gb = gains_b[g]
                            v1_imul_v2ct(m_vec, gb, m_vec)

                        for g in lt_active:

                            ga = gains_a[g]
                            v1ct_imul_v2(ga, lmul_op_a, lmul_op_a)

                            gb = gains_b[g]
                            v1ct_imul_v2(gb, lmul_op_b, lmul_op_b)

                        t_m = t_map_arr[row_ind, active_term]
                        f_m = f_map_arr[f, active_term]
                        out_d = d_map_arr[active_term, d]

                        ga = gains_a[active_term]
                        accumulate_jhr(ga, r_vec, mh_vec, lmul_op_a,
                                       jhr[t_m, f_m, a1_m, out_d], nu)

                        jhmul(lmul_op_a, mh_vec, ga, nu, tmp_jh_p[out_d])

                        gb = gains_b[active_term]
                        accumulate_jhr(gb, rh_vec, m_vec, lmul_op_b,
                                       jhr[t_m, f_m, a2_m, out_d], nu)

                        jhmul(lmul_op_b, m_vec, gb, nu, tmp_jh_q[out_d])

                    for d in range(n_gdir):

                        jhp = tmp_jh_p[d]
                        jhj_sel = jhj[t_m, f_m, a1_m, d]
                        jhwjmul(jhp, w, jhj_sel)

                        jhq = tmp_jh_q[d]
                        jhj_sel = jhj[t_m, f_m, a2_m, d]
                        jhwjmul(jhq, w, jhj_sel)
        return
    return impl


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def update(update, jhj, jhr, corr_mode):

    if corr_mode.literal_value in ["full", "mixed"]:
        def impl(update, jhj, jhr, corr_mode):
            n_tint, n_fint, n_ant, n_dir, _, _ = jhj.shape

            ident = np.eye(4)

            for t in range(n_tint):
                for f in range(n_fint):
                    for a in range(n_ant):
                        for d in range(n_dir):

                            jhj_sel = jhj[t, f, a, d]

                            det = np.linalg.det(jhj_sel)

                            if det.real < 1e-6 or ~np.isfinite(det):
                                jhj_inv = np.zeros_like(jhj_sel)
                            else:
                                jhj_inv = np.linalg.solve(jhj_sel, ident)

                            update[t, f, a, d] = jhj_inv.dot(jhr[t, f, a, d])
    else:
        def impl(update, jhj, jhr, corr_mode):

            n_tint, n_fint, n_ant, n_dir, _, _ = jhj.shape

            for t in range(n_tint):
                for f in range(n_fint):
                    for a in range(n_ant):
                        for d in range(n_dir):
                            for sl in (slice(0, 2), slice(2, 4)):

                                jhj_sel = jhj[t, f, a, d, sl, sl]

                                jhj00 = jhj_sel[0, 0]
                                jhj01 = jhj_sel[0, 1]
                                jhj10 = jhj_sel[1, 0]
                                jhj11 = jhj_sel[1, 1]

                                det = (jhj00*jhj11 - jhj01*jhj10)

                                if det.real < 1e-6:
                                    jhjinv00 = 0
                                    jhjinv01 = 0
                                    jhjinv10 = 0
                                    jhjinv11 = 0
                                else:
                                    jhjinv00 = jhj11/det
                                    jhjinv01 = -jhj01/det
                                    jhjinv10 = -jhj10/det
                                    jhjinv11 = jhj00/det

                                jhr_sel = jhr[t, f, a, d, sl]

                                jhr_0 = jhr_sel[0]
                                jhr_1 = jhr_sel[1]

                                upd_sel = update[t, f, a, d, sl]

                                upd_sel[0] = jhjinv00*jhr_0 + jhjinv01*jhr_1
                                upd_sel[1] = jhjinv10*jhr_0 + jhjinv11*jhr_1
    return impl


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def finalize(update, params, gain, chan_freqs, t_bin_arr, f_map_arr,
             d_map_arr, dd_term, corr_mode, active_term):

    def impl(update, params, gain, chan_freqs, t_bin_arr, f_map_arr,
             d_map_arr, dd_term, corr_mode, active_term):
        params[:, :, :, :, 0, 0] += update[:, :, :, :, 0]/2
        params[:, :, :, :, 0, -1] += update[:, :, :, :, 2]/2
        params[:, :, :, :, 1, 0] += update[:, :, :, :, 1]/2
        params[:, :, :, :, 1, -1] += update[:, :, :, :, 3]/2

        n_tint, n_fint, n_ant, n_dir, n_param, n_corr = params.shape

        n_time, n_freq, _, _, _ = gain.shape

        for t in range(n_time):
            for f in range(n_freq):
                for a in range(n_ant):
                    for d in range(n_dir):

                        t_m = t_bin_arr[t, active_term]
                        f_m = f_map_arr[f, active_term]
                        d_m = d_map_arr[d, active_term]

                        inter0 = params[t_m, f_m, a, d_m, 0, 0]
                        inter1 = params[t_m, f_m, a, d_m, 0, -1]
                        delay0 = params[t_m, f_m, a, d_m, 1, 0]
                        delay1 = params[t_m, f_m, a, d_m, 1, -1]

                        cf = chan_freqs[f]

                        gain[t, f, a, d, 0] = np.exp(1j*(cf*delay0 + inter0))
                        gain[t, f, a, d, -1] = np.exp(1j*(cf*delay1 + inter1))
    return impl


def jhr_factory(mode):

    unpack = factories.unpack_factory(mode)
    unpackct = factories.unpackct_factory(mode)
    v1_imul_v2 = factories.v1_imul_v2_factory(mode)

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(gain, res, rop, lop, jhr, nu):

            v1_imul_v2(res, rop, res)
            v1_imul_v2(lop, res, res)

            g_00, g_01, g_10, g_11 = unpackct(gain)
            v_00, v_01, v_10, v_11 = unpack(res)

            upd00 = (-1j*g_00*v_00).real
            upd11 = (-1j*g_11*v_11).real

            jhr[0] += upd00
            jhr[1] += nu*upd00
            jhr[2] += upd11
            jhr[3] += nu*upd11
    else:
        def impl(gain, res, rop, lop, jhr, nu):

            v1_imul_v2(res, rop, res)

            g_00, g_11 = unpackct(gain)
            v_00, v_11 = unpack(res)

            upd00 = (-1j*g_00*v_00).real
            upd11 = (-1j*g_11*v_11).real

            jhr[0] += upd00
            jhr[1] += nu*upd00
            jhr[2] += upd11
            jhr[3] += nu*upd11
    return factories.qcjit(impl)


def special_jh_mul_factory(mode):

    unpack = factories.unpack_factory(mode)
    unpackct = factories.unpackct_factory(mode)

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(lop, rop, gain, nu, jh):
            l_00, l_01, l_10, l_11 = unpack(lop)
            r_00, r_10, r_01, r_11 = unpack(rop)  # Note the "transpose".
            g_00, g_01, g_10, g_11 = unpackct(gain)

            # This implements a special kronecker product which simultaneously
            # multiplies in derivative terms. This doesn't generalize to all
            # terms.

            # Top row.
            rkl_00 = r_00*l_00
            rkl_01 = r_00*l_01
            rkl_02 = r_01*l_00
            rkl_03 = r_01*l_01
            # Bottom row.
            rkl_30 = r_10*l_10
            rkl_31 = r_10*l_11
            rkl_32 = r_11*l_10
            rkl_33 = r_11*l_11

            # Coefficients generated by the derivative.
            drv_00 = -1j*g_00
            drv_10 = drv_00*nu
            drv_23 = -1j*g_11
            drv_33 = drv_23*nu

            # (param_per_antenna, 4) result of special kronecker product.
            jh[0, 0] += rkl_00*drv_00
            jh[0, 1] += rkl_01*drv_00
            jh[0, 2] += rkl_02*drv_00
            jh[0, 3] += rkl_03*drv_00
            jh[1, 0] += rkl_00*drv_10
            jh[1, 1] += rkl_01*drv_10
            jh[1, 2] += rkl_02*drv_10
            jh[1, 3] += rkl_03*drv_10
            jh[2, 0] += rkl_30*drv_23
            jh[2, 1] += rkl_31*drv_23
            jh[2, 2] += rkl_32*drv_23
            jh[2, 3] += rkl_33*drv_23
            jh[3, 0] += rkl_30*drv_33
            jh[3, 1] += rkl_31*drv_33
            jh[3, 2] += rkl_32*drv_33
            jh[3, 3] += rkl_33*drv_33
    else:
        def impl(lop, rop, gain, nu, jh):
            r_00, r_11 = unpack(rop)
            g_00, g_11 = unpackct(gain)

            # This implements a special kronecker product which simultaneously
            # multiplies in derivative terms. This doesn't generalize to all
            # terms.

            # Coefficients generated by the derivative.
            drv_00 = -1j*g_00
            drv_10 = drv_00*nu
            drv_23 = -1j*g_11
            drv_33 = drv_23*nu

            # (param_per_antenna, 4) result of special kronecker product.
            jh[0, 0] += r_00*drv_00
            jh[1, 0] += r_00*drv_10
            jh[2, 3] += r_11*drv_23
            jh[3, 3] += r_11*drv_33
    return factories.qcjit(impl)


def special_jh_wmul_j_factory(mode):

    unpack = factories.unpack_factory(mode)
    unpackct = factories.unpackct_factory(mode)

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(jh, w, jhj):
            w1_00, w1_01, w1_10, w1_11 = unpack(w)

            n_ppa, n_corr = jh.shape

            for i in range(n_ppa):
                jh_0, jh_1, jh_2, jh_3 = unpack(jh[i])
                jhw_0 = jh_0*w1_00
                jhw_1 = jh_1*w1_00
                jhw_2 = jh_2*w1_11
                jhw_3 = jh_3*w1_11
                for j in range(n_ppa):
                    # Note "transpose" as I am abusing unpack.
                    j_0, j_2, j_1, j_3 = unpackct(jh[j])

                    jhj[i, j] += (jhw_0*j_0 +
                                  jhw_1*j_1 +
                                  jhw_2*j_2 +
                                  jhw_3*j_3).real
    else:
        def impl(jh, w, jhj):
            w1_00, w1_11 = unpack(w)

            jh_00 = jh[0, 0]
            jh_10 = jh[1, 0]
            jh_23 = jh[2, 3]
            jh_33 = jh[3, 3]

            # Conjugate transpose terms.
            j_00 = jh_00.conjugate()
            j_01 = jh_10.conjugate()
            j_32 = jh_23.conjugate()
            j_33 = jh_33.conjugate()

            # Multiply in the weights.
            jh_00 *= w1_00
            jh_10 *= w1_00
            jh_23 *= w1_11
            jh_33 *= w1_11

            jhj[0, 0] += (jh_00*j_00).real
            jhj[0, 1] += (jh_00*j_01).real
            jhj[1, 0] += (jh_10*j_00).real
            jhj[1, 1] += (jh_10*j_01).real

            jhj[2, 2] += (jh_23*j_32).real
            jhj[2, 3] += (jh_23*j_33).real
            jhj[3, 2] += (jh_33*j_32).real
            jhj[3, 3] += (jh_33*j_33).real
    return factories.qcjit(impl)
