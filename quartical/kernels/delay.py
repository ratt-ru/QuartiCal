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

        jhj_shape = (n_tint, n_fint, n_ant, n_dir, n_ppa, n_ppa)
        jhj = np.empty(jhj_shape, dtype=real_dtype)
        jhr_shape = (n_tint, n_fint, n_ant, n_dir, n_ppa)
        jhr = np.empty(jhr_shape, dtype=real_dtype)
        update = np.empty(jhr_shape, dtype=real_dtype)

        for i in range(50):

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
def compute_jhj_jhr(jhj, jhr, model, gains, inverse_gains, chan_freqs,
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
    make_loop_vars = factories.loop_var_factory(corr_mode)
    set_identity = factories.set_identity_factory(corr_mode)
    accumulate_jhr = accumulate_jhr_factory(corr_mode)
    compute_jh = compute_jh_factory(corr_mode)
    compute_jhwj = compute_jhwj_factory(corr_mode)

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

        # Determine loop variables based on where we are in the chain.
        # gt means greater than (n>j) and lt means less than (n<j).
        all_terms, gt_active, lt_active = make_loop_vars(n_gains, active_term)

        # Parallel over all solution intervals.
        for i in prange(n_int):

            ti = i//n_fint
            fi = i - ti*n_fint

            rs = row_starts[ti]
            re = row_stops[ti]
            fs = chan_starts[fi]
            fe = chan_stops[fi]

            rop_pq = valloc(complex_dtype)  # Right-multiply operator for pq.
            rop_qp = valloc(complex_dtype)  # Right-multiply operator for qp.
            lop_pq = valloc(complex_dtype)  # Left-multiply operator for pq.
            lop_qp = valloc(complex_dtype)  # Left-multiply operator for qp.
            r_pq = valloc(complex_dtype)
            r_qp = valloc(complex_dtype)

            gains_p = valloc(complex_dtype, leading_dims=(n_gains,))
            gains_q = valloc(complex_dtype, leading_dims=(n_gains,))

            tmp_jh_p = np.empty((n_gdir, n_ppa, n_corr), dtype=complex_dtype)
            tmp_jh_q = np.empty((n_gdir, n_ppa, n_corr), dtype=complex_dtype)

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

                        set_identity(lop_pq)
                        set_identity(lop_qp)

                        # Construct a small contiguous gain array. This makes
                        # the single term case fractionally slower.
                        for gi in range(n_gains):
                            d_m = d_map_arr[gi, d]  # Broadcast dir.
                            t_m = t_map_arr[row_ind, gi]
                            f_m = f_map_arr[f, gi]

                            gain = gains[gi][t_m, f_m]

                            iunpack(gains_p[gi], gain[a1_m, d_m])
                            iunpack(gains_q[gi], gain[a2_m, d_m])

                        imul_rweight(r, r_pq, row_weights, row_ind)
                        iwmul(r_pq, w)
                        iunpackct(r_qp, r_pq)

                        m = model[row, f, d]
                        imul_rweight(m, rop_qp, row_weights, row_ind)
                        iunpackct(rop_pq, rop_qp)

                        for g in all_terms:

                            g_q = gains_q[g]
                            v1_imul_v2(g_q, rop_pq, rop_pq)

                            g_p = gains_p[g]
                            v1_imul_v2(g_p, rop_qp, rop_qp)

                        for g in gt_active:

                            g_p = gains_p[g]
                            v1_imul_v2ct(rop_pq, g_p, rop_pq)

                            g_q = gains_q[g]
                            v1_imul_v2ct(rop_qp, g_q, rop_qp)

                        for g in lt_active:

                            g_p = gains_p[g]
                            v1ct_imul_v2(g_p, lop_pq, lop_pq)

                            g_q = gains_q[g]
                            v1ct_imul_v2(g_q, lop_qp, lop_qp)

                        t_m = t_map_arr[row_ind, active_term]
                        f_m = f_map_arr[f, active_term]
                        out_d = d_map_arr[active_term, d]

                        g_p = gains_p[active_term]
                        accumulate_jhr(g_p, r_pq, rop_pq, lop_pq,
                                       jhr[t_m, f_m, a1_m, out_d], nu)

                        compute_jh(lop_pq, rop_pq, g_p, nu, tmp_jh_p[out_d])

                        g_q = gains_q[active_term]
                        accumulate_jhr(g_q, r_qp, rop_qp, lop_qp,
                                       jhr[t_m, f_m, a2_m, out_d], nu)

                        compute_jh(lop_qp, rop_qp, g_q, nu, tmp_jh_q[out_d])

                    for d in range(n_gdir):

                        jh_p = tmp_jh_p[d]
                        jhj_p = jhj[t_m, f_m, a1_m, d]
                        compute_jhwj(jh_p, w, jhj_p)

                        jh_q = tmp_jh_q[d]
                        jhj_q = jhj[t_m, f_m, a2_m, d]
                        compute_jhwj(jh_q, w, jhj_q)
        return
    return impl


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def compute_update(update, jhj, jhr, corr_mode):

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

                            # TODO: This complains as linalg.solve produces
                            # F_CONTIGUOUS output.
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
def finalize_update(update, params, gain, chan_freqs, t_bin_arr, f_map_arr,
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
                        d_m = d_map_arr[active_term, d]

                        inter0 = params[t_m, f_m, a, d_m, 0, 0]
                        inter1 = params[t_m, f_m, a, d_m, 0, -1]
                        delay0 = params[t_m, f_m, a, d_m, 1, 0]
                        delay1 = params[t_m, f_m, a, d_m, 1, -1]

                        cf = chan_freqs[f]

                        gain[t, f, a, d, 0] = np.exp(1j*(cf*delay0 + inter0))
                        gain[t, f, a, d, -1] = np.exp(1j*(cf*delay1 + inter1))
    return impl


def accumulate_jhr_factory(mode):

    unpack = factories.unpack_factory(mode)
    unpackct = factories.unpackct_factory(mode)
    v1_imul_v2 = factories.v1_imul_v2_factory(mode)

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(gain, res, rop, lop, jhr, nu):

            v1_imul_v2(res, rop, res)
            v1_imul_v2(lop, res, res)

            g_00, g_01, g_10, g_11 = unpackct(gain)
            v_00, v_01, v_10, v_11 = unpack(res)

            upd_00 = (-1j*g_00*v_00).real
            upd_11 = (-1j*g_11*v_11).real

            jhr[0] += upd_00
            jhr[1] += nu*upd_00
            jhr[2] += upd_11
            jhr[3] += nu*upd_11
    else:
        def impl(gain, res, rop, lop, jhr, nu):

            v1_imul_v2(res, rop, res)

            g_00, g_11 = unpackct(gain)
            v_00, v_11 = unpack(res)

            upd_00 = (-1j*g_00*v_00).real
            upd_11 = (-1j*g_11*v_11).real

            jhr[0] += upd_00
            jhr[1] += nu*upd_00
            jhr[2] += upd_11
            jhr[3] += nu*upd_11
    return factories.qcjit(impl)


def compute_jh_factory(mode):

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
            jh[2, 1] += r_11*drv_23
            jh[3, 1] += r_11*drv_33
    return factories.qcjit(impl)


def compute_jhwj_factory(mode):

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
            jh_21 = jh[2, 1]
            jh_31 = jh[3, 1]

            # Conjugate transpose terms.
            j_00 = jh_00.conjugate()
            j_01 = jh_10.conjugate()
            j_12 = jh_21.conjugate()
            j_13 = jh_31.conjugate()

            # Multiply in the weights.
            jh_00 *= w1_00
            jh_10 *= w1_00
            jh_21 *= w1_11
            jh_31 *= w1_11

            jhj[0, 0] += (jh_00*j_00).real
            jhj[0, 1] += (jh_00*j_01).real
            jhj[1, 0] += (jh_10*j_00).real
            jhj[1, 1] += (jh_10*j_01).real

            jhj[2, 2] += (jh_21*j_12).real
            jhj[2, 3] += (jh_21*j_13).real
            jhj[3, 2] += (jh_31*j_12).real
            jhj[3, 3] += (jh_31*j_13).real
    return factories.qcjit(impl)
