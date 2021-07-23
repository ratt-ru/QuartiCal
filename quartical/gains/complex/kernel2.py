# -*- coding: utf-8 -*-
import numpy as np
from numba import prange, generated_jit
from numpy.lib import unique
from quartical.utils.numba import coerce_literal
from quartical.gains.general.generics import (invert_gains,
                                              compute_residual,
                                              compute_convergence)
from quartical.gains.general.convenience import (get_row,
                                                 get_chan_extents,
                                                 get_row_extents)
import quartical.gains.general.factories as factories
from collections import namedtuple


# This can be done without a named tuple now. TODO: Add unpacking to
# constructor.
stat_fields = {"conv_iters": np.int64,
               "conv_perc": np.float64}

term_conv_info = namedtuple("term_conv_info", " ".join(stat_fields.keys()))

complex_args = namedtuple("complex_args", ())


@generated_jit(nopython=True,
               fastmath=True,
               parallel=False,
               cache=True,
               nogil=True)
def complex_solver(base_args, term_args, meta_args, corr_mode):

    coerce_literal(complex_solver, ["corr_mode"])

    jhj_dims = jhj_dim_factory(corr_mode)

    def impl(base_args, term_args, meta_args, corr_mode):

        model = base_args.model
        data = base_args.data
        a1 = base_args.a1
        a2 = base_args.a2
        weights = base_args.weights
        t_map_arr = base_args.t_map_arr
        f_map_arr = base_args.f_map_arr
        d_map_arr = base_args.d_map_arr
        inverse_gains = base_args.inverse_gains
        gains = base_args.gains
        flags = base_args.flags
        row_map = base_args.row_map
        row_weights = base_args.row_weights

        active_term = meta_args.active_term

        n_tint, n_fint, n_ant, n_dir, n_corr = gains[active_term].shape

        t_map_arr = t_map_arr[0]  # We don't need the parameter mappings.
        f_map_arr = f_map_arr[0]  # We don't need the parameter mappings.

        invert_gains(gains, inverse_gains, corr_mode)

        dd_term = n_dir > 1

        last_gain = gains[active_term].copy()

        cnv_perc = 0.

        jhj = np.empty((n_tint, n_fint, n_ant, n_dir, *jhj_dims()),
                       dtype=gains[active_term].dtype)
        # jhj = np.empty_like(gains[active_term])
        jhr = np.empty_like(gains[active_term])
        update = np.empty_like(gains[active_term])

        for i in range(meta_args.iters):

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
                            gains[active_term],
                            i,
                            dd_term,
                            corr_mode)

            # Check for gain convergence. TODO: This can be affected by the
            # weights. Currently unsure how or why, but using unity weights
            # leads to monotonic convergence in all solution intervals.

            cnv_perc = compute_convergence(gains[active_term][:], last_gain)

            last_gain[:] = gains[active_term][:]

            if cnv_perc > 0.99:
                break

        print(i+1, cnv_perc)

        return jhj, term_conv_info(i + 1, cnv_perc)

    return impl


@generated_jit(nopython=True,
               fastmath=True,
               parallel=True,
               cache=True,
               nogil=True)
def compute_jhj_jhr(jhj, jhr, model, gains, inverse_gains, residual, a1,
                    a2, weights, t_map_arr, f_map_arr, d_map_arr, row_map,
                    row_weights, active_term, corr_mode):

    imul_rweight = factories.imul_rweight_factory(corr_mode, row_weights)
    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    v1_imul_v2ct = factories.v1_imul_v2ct_factory(corr_mode)
    v1ct_imul_v2 = factories.v1ct_imul_v2_factory(corr_mode)
    iunpack = factories.iunpack_factory(corr_mode)
    iunpackct = factories.iunpackct_factory(corr_mode)
    iwmul = factories.iwmul_factory(corr_mode)
    iadd = factories.iadd_factory(corr_mode)
    valloc = factories.valloc_factory(corr_mode)
    make_loop_vars = factories.loop_var_factory(corr_mode)
    set_identity = factories.set_identity_factory(corr_mode)
    accumulate_jhr = accumulate_jhr_factory(corr_mode)
    compute_jhwj = compute_jhwj_factory(corr_mode)
    jhj_dims = jhj_dim_factory(corr_mode)

    def impl(jhj, jhr, model, gains, inverse_gains, residual, a1, a2, weights,
             t_map_arr, f_map_arr, d_map_arr, row_map, row_weights,
             active_term, corr_mode):
        _, n_chan, n_dir, n_corr = model.shape

        jhj[:] = 0
        jhr[:] = 0

        n_tint, n_fint, n_ant, n_gdir, n_corr = jhr.shape
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

        tmp_jhj_dims = (n_ant, n_gdir, *jhj_dims())

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

            lop_pq_arr = np.empty((n_gdir, n_corr), dtype=complex_dtype)
            rop_pq_arr = np.empty((n_gdir, n_corr), dtype=complex_dtype)
            lop_qp_arr = np.empty((n_gdir, n_corr), dtype=complex_dtype)
            rop_qp_arr = np.empty((n_gdir, n_corr), dtype=complex_dtype)

            tmp_jhr = np.zeros((n_ant, n_gdir, n_corr), dtype=complex_dtype)
            # The shape of this matrix needs to distinguish between the
            # scalar/diag and 2x2 cases.
            tmp_jhj = np.zeros(tmp_jhj_dims, dtype=complex_dtype)

            for row_ind in range(rs, re):

                row = get_row(row_ind, row_map)
                a1_m, a2_m = a1[row], a2[row]

                for f in range(fs, fe):

                    r = residual[row, f]
                    w = weights[row, f]  # Consider a map?

                    lop_pq_arr[:] = 0
                    rop_pq_arr[:] = 0
                    lop_qp_arr[:] = 0
                    rop_qp_arr[:] = 0

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

                        accumulate_jhr(r_pq, rop_pq, lop_pq,
                                       tmp_jhr[a1_m, out_d])

                        iadd(lop_pq_arr[out_d], lop_pq)
                        iadd(rop_pq_arr[out_d], rop_pq)

                        accumulate_jhr(r_qp, rop_qp, lop_qp,
                                       tmp_jhr[a2_m, out_d])

                        iadd(lop_qp_arr[out_d], lop_qp)
                        iadd(rop_qp_arr[out_d], rop_qp)

                    for d in range(n_gdir):

                        lop_pq_d = lop_pq_arr[d]
                        rop_pq_d = rop_pq_arr[d]

                        compute_jhwj(lop_pq_d, rop_pq_d, w,
                                     tmp_jhj[a1_m, d])

                        lop_qp_d = lop_qp_arr[d]
                        rop_qp_d = rop_qp_arr[d]

                        compute_jhwj(lop_qp_d, rop_qp_d, w,
                                     tmp_jhj[a2_m, d])

            jhr[t_m, f_m] = tmp_jhr
            jhj[t_m, f_m] = tmp_jhj
        return
    return impl


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def compute_update(update, jhj, jhr, corr_mode):

    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    compute_det = factories.compute_det_factory(corr_mode)
    iinverse = factories.iinverse_factory(corr_mode)
    a_mul_vecb = a_mul_vecb_factory(corr_mode)

    def impl(update, jhj, jhr, corr_mode):
        n_tint, n_fint, n_ant, n_dir, _, _ = jhj.shape

        ident = np.eye(4, dtype=jhj.dtype)

        for t in range(n_tint):
            for f in range(n_fint):
                for a in range(n_ant):
                    for d in range(n_dir):

                        jhj_sel = jhj[t, f, a, d]

                        det = np.linalg.det(jhj_sel)

                        if np.abs(det) < 1e-6 or ~np.isfinite(det):
                            jhj_inv = np.zeros_like(jhj_sel)
                        else:
                            jhj_inv = np.linalg.solve(jhj_sel, ident)

                        # TODO: This complains as linalg.solve produces
                        # F_CONTIGUOUS output.
                        a_mul_vecb(jhj_inv,
                                   jhr[t, f, a, d],
                                   update[t, f, a, d])

    return impl


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def finalize_update(update, gain, i_num, dd_term, corr_mode):

    def impl(update, gain, i_num, dd_term, corr_mode):
        if dd_term:
            gain[:] = gain[:] + update/2
        elif i_num % 2 == 0:
            gain[:] = update
        else:
            gain[:] = (gain[:] + update)/2
    return impl


def accumulate_jhr_factory(corr_mode):

    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    iadd = factories.iadd_factory(corr_mode)

    def impl(res, rop, lop, jhr):

        v1_imul_v2(lop, res, res)
        v1_imul_v2(res, rop, res)
        iadd(jhr, res)

    return factories.qcjit(impl)


def compute_jhwj_factory(corr_mode):

    atkb = aT_kron_b_factory(corr_mode)
    unpack = factories.unpack_factory(corr_mode)
    unpackct = factories.unpackct_factory(corr_mode)

    def impl(lop, rop, w, jhj):

        kprod = np.empty_like(jhj)
        atkb(rop, lop, kprod)

        w_00, w_01, w_10, w_11 = unpack(w)  # NOTE: XX, XY, YX, YY

        for i in range(4):
            jh_0, jh_1, jh_2, jh_3 = unpack(kprod[i])
            jhw_0 = jh_0*w_00  # XX
            jhw_1 = jh_1*w_10  # YX
            jhw_2 = jh_2*w_01  # XY
            jhw_3 = jh_3*w_11  # YY
            for j in range(4):
                # NOTE: "undoing" transpose as I am abusing unpack.
                j_0, j_2, j_1, j_3 = unpackct(kprod[j])

                jhj[i, j] += (jhw_0*j_0 + jhw_1*j_1 + jhw_2*j_2 + jhw_3*j_3)

        # TODO: I think I am beginning to understand. vec(JHR) is not the same
        # as a flattened/raveled JHR. vec implies stacked columns whereas a
        # ravel/flattened version stacks rows. This is now important as we are
        # moving between representations.

    return factories.qcjit(impl)


def aT_kron_b_factory(corr_mode):

    unpack = factories.unpack_factory(corr_mode)

    def impl(a, b, out):

        a00, a10, a01, a11 = unpack(a)  # Effectively transpose.
        b00, b01, b10, b11 = unpack(b)

        out[0, 0] = a00 * b00
        out[0, 1] = a00 * b01
        out[0, 2] = a01 * b00
        out[0, 3] = a01 * b01
        out[1, 0] = a00 * b10
        out[1, 1] = a00 * b11
        out[1, 2] = a01 * b10
        out[1, 3] = a01 * b11
        out[2, 0] = a10 * b00
        out[2, 1] = a10 * b01
        out[2, 2] = a11 * b00
        out[2, 3] = a11 * b01
        out[3, 0] = a10 * b10
        out[3, 1] = a10 * b11
        out[3, 2] = a11 * b10
        out[3, 3] = a11 * b11

    return factories.qcjit(impl)


def a_mul_vecb_factory(corr_mode):

    unpack = factories.unpack_factory(corr_mode)

    def impl(a, b, out):

        b_0, b_2, b_1, b_3 = unpack(b)  # NOTE: b is XX XY YX YY

        for ii, oi in enumerate((0, 2, 1, 3)):
            a_0, a_1, a_2, a_3 = unpack(a[ii])

            out[oi] = (a_0*b_0 + a_1*b_1 + a_2*b_2 + a_3*b_3)

    return factories.qcjit(impl)


def jhj_dim_factory(corr_mode):

    if corr_mode.literal_value == 4:
        def impl():
            return (4, 4)
    else:
        def impl():
            return (corr_mode.literal_value,)

    return factories.qcjit(impl)
