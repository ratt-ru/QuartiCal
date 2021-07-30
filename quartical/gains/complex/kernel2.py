# -*- coding: utf-8 -*-
import numpy as np
from numba import prange, generated_jit
from quartical.utils.numba import coerce_literal
from quartical.gains.general.generics import (compute_residual,
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

    get_jhj_dims = get_jhj_dims_factory(corr_mode)

    def impl(base_args, term_args, meta_args, corr_mode):

        model = base_args.model
        data = base_args.data
        a1 = base_args.a1
        a2 = base_args.a2
        weights = base_args.weights
        t_map_arr = base_args.t_map_arr[0]  # Ignore parameter mappings.
        f_map_arr = base_args.f_map_arr[0]  # Ignore parameter mappings.
        d_map_arr = base_args.d_map_arr
        gains = base_args.gains
        flags = base_args.flags
        row_map = base_args.row_map
        row_weights = base_args.row_weights

        active_term = meta_args.active_term
        iters = meta_args.iters

        active_gain = gains[active_term]

        dd_term = np.any(d_map_arr[active_term])

        last_gain = active_gain.copy()

        cnv_perc = 0.

        jhj = np.empty(get_jhj_dims(active_gain), dtype=active_gain.dtype)
        jhr = np.empty_like(active_gain)
        update = np.zeros_like(active_gain)

        for i in range(iters):

            if dd_term:
                residual = compute_residual(data,
                                            model,
                                            gains,
                                            a1,
                                            a2,
                                            t_map_arr,
                                            f_map_arr,
                                            d_map_arr,
                                            row_map,
                                            row_weights,
                                            corr_mode)
            else:
                residual = data

            compute_jhj_jhr(jhj,
                            jhr,
                            model,
                            gains,
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

        print(i + 1, cnv_perc)

        return jhj, term_conv_info(i + 1, cnv_perc)

    return impl


@generated_jit(nopython=True,
               fastmath=True,
               parallel=True,
               cache=True,
               nogil=True)
def compute_jhj_jhr(jhj, jhr, model, gains, residual, a1, a2, weights,
                    t_map_arr, f_map_arr, d_map_arr, row_map, row_weights,
                    active_term, corr_mode):

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

    def impl(jhj, jhr, model, gains, residual, a1, a2, weights,
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

        tmp_jhj_dims = jhj.shape[2:]  # Doesn't need time/freq dim.

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

            lop_pq_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
            rop_pq_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
            lop_qp_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
            rop_qp_arr = valloc(complex_dtype, leading_dims=(n_gdir,))

            kprod = np.zeros((4, 4), dtype=complex_dtype)
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
                                     kprod, tmp_jhj[a1_m, d])

                        lop_qp_d = lop_qp_arr[d]
                        rop_qp_d = rop_qp_arr[d]

                        compute_jhwj(lop_qp_d, rop_qp_d, w,
                                     kprod, tmp_jhj[a2_m, d])

            # This is the spigot point. A parameterised solver can do the
            # extra operations it requires here. Technically, we could also
            # compute the update directly, allowing us to avoid storing the
            # entirety of jhj and jhr.
            jhr[t_m, f_m] = tmp_jhr
            jhj[t_m, f_m] = tmp_jhj
        return
    return impl


@generated_jit(nopython=True,
               fastmath=True,
               parallel=True,
               cache=True,
               nogil=True)
def compute_update(update, jhj, jhr, corr_mode):

    mat_mul_vec = mat_mul_vec_factory(corr_mode)
    vecct_mul_vec = vecct_mul_vec_factory(corr_mode)
    diag_add = diag_add_factory(corr_mode)
    vec_iadd_svec = vec_iadd_svec_factory(corr_mode)
    vec_isub_svec = vec_isub_svec_factory(corr_mode)

    def impl(update, jhj, jhr, corr_mode):
        n_tint, n_fint, n_ant, n_dir, n_param, n_param = jhj.shape

        n_int = n_tint * n_fint

        for i in prange(n_int):

            t = i // n_fint
            f = i - t * n_fint

            r = np.zeros(n_param, dtype=jhr.dtype)
            p = np.zeros(n_param, dtype=jhr.dtype)
            Ap = np.zeros(n_param, dtype=jhr.dtype)
            Ax = np.zeros(n_param, dtype=jhr.dtype)
            A = np.zeros((n_param, n_param), dtype=jhj.dtype)

            for a in range(n_ant):
                for d in range(n_dir):

                    # diag_add(jhj[t, f, a, d], 1e-6, A)
                    A = jhj[t, f, a, d]
                    b = jhr[t, f, a, d]
                    x = update[t, f, a, d]

                    mat_mul_vec(A, x, Ax)
                    r[:] = b
                    r -= Ax
                    p[:] = r
                    r_k = vecct_mul_vec(r, r)

                    for _ in range(n_param):
                        mat_mul_vec(A, p, Ap)
                        alpha_denom = vecct_mul_vec(p, Ap)
                        if np.abs(alpha_denom) == 0:
                            x[:] = 0
                            break
                        alpha = r_k / alpha_denom
                        vec_iadd_svec(x, alpha, p)
                        vec_isub_svec(r, alpha, Ap)
                        r_kplus1 = vecct_mul_vec(r, r)
                        if r_kplus1.real < 1e-16:
                            break
                        p *= (r_kplus1 / r_k)
                        p += r
                        r_k = r_kplus1

    return impl


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def finalize_update(update, gain, i_num, dd_term, corr_mode):

    def impl(update, gain, i_num, dd_term, corr_mode):
        if dd_term:
            update /= 2
            gain += update
        elif i_num % 2 == 0:
            gain[:] = update
        else:
            gain += update
            gain /= 2

    return impl


def accumulate_jhr_factory(corr_mode):

    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    iadd = factories.iadd_factory(corr_mode)

    if corr_mode.literal_value == 4:
        def impl(res, rop, lop, jhr):
            v1_imul_v2(lop, res, res)
            v1_imul_v2(res, rop, res)
            iadd(jhr, res)
    elif corr_mode.literal_value in (1, 2):
        def impl(res, rop, lop, jhr):  # lop is a no-op in scalar/diag case.
            v1_imul_v2(res, rop, res)
            iadd(jhr, res)
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def compute_jhwj_factory(corr_mode):

    atkb = aT_kron_b_factory(corr_mode)
    unpack = factories.unpack_factory(corr_mode)
    unpackc = factories.unpackc_factory(corr_mode)

    if corr_mode.literal_value == 4:
        def impl(lop, rop, w, kprod, jhj):

            atkb(rop, lop, kprod)  # NOTE: This is includes a shuffle!

            w_00, w_01, w_10, w_11 = unpack(w)  # NOTE: XX, XY, YX, YY

            for i in range(4):

                jh_0, jh_1, jh_2, jh_3 = unpack(kprod[i])

                jhw_0 = jh_0*w_00  # XX
                jhw_1 = jh_1*w_01  # XY
                jhw_2 = jh_2*w_10  # YX
                jhw_3 = jh_3*w_11  # YY

                for j in range(i):
                    jhj[i, j] = jhj[j, i].conjugate()

                for j in range(i, 4):
                    j_0, j_1, j_2, j_3 = unpackc(kprod[j])
                    jhj[i, j] += (jhw_0*j_0 + jhw_1*j_1 +
                                  jhw_2*j_2 + jhw_3*j_3)

            # NOTE: vec(JHR) is not the same as a flattened/raveled JHR.
            # vec implies stacked columns whereas a ravel/flattened version
            # stacks rows (with C conventions). This is now important as we are
            # moving between representations.
    elif corr_mode.literal_value == 2:
        def impl(lop, rop, w, kprod, jhj):
            jh_00, jh_11 = unpack(rop)
            j_00, j_11 = unpackc(rop)
            w_00, w_11 = unpack(w)

            jhj[0] += jh_00*w_00*j_00
            jhj[1] += jh_11*w_11*j_11
    elif corr_mode.literal_value == 1:
        def impl(lop, rop, w, kprod, jhj):
            jh_00 = unpack(rop)
            j_00 = unpackc(rop)
            w_00 = unpack(w)

            jhj[0] += jh_00*w_00*j_00
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def aT_kron_b_factory(corr_mode):

    unpack = factories.unpack_factory(corr_mode)

    def impl(a, b, out):

        a00, a10, a01, a11 = unpack(a)  # Effectively transpose.
        b00, b01, b10, b11 = unpack(b)

        # NOTE: At present this is not merely B^T kron A (definition in Latex
        # document). It is S(B^T kron A) where S is a shuffle matrix. This is
        # necessary as the ravelled visibilities are ordered as [XX XY YX YY],
        # while the vectorised form (vec(V_pq)) would be ordered [XX YX XY YY].
        out[0, 0] = a00 * b00
        out[0, 1] = a00 * b01
        out[0, 2] = a01 * b00
        out[0, 3] = a01 * b01
        out[1, 0] = a10 * b00
        out[1, 1] = a10 * b01
        out[1, 2] = a11 * b00
        out[1, 3] = a11 * b01
        out[2, 0] = a00 * b10
        out[2, 1] = a00 * b11
        out[2, 2] = a01 * b10
        out[2, 3] = a01 * b11
        out[3, 0] = a10 * b10
        out[3, 1] = a10 * b11
        out[3, 2] = a11 * b10
        out[3, 3] = a11 * b11

    return factories.qcjit(impl)


def mat_mul_vec_factory(corr_mode):

    def impl(mat, vec, out):

        n_row, n_col = mat.shape

        out[:] = 0

        for i in range(n_row):
            for j in range(n_col):
                out[i] += mat[i, j] * vec[j]

    return factories.qcjit(impl)


def vecct_mul_mat_factory(corr_mode):

    def impl(vec, mat, out):

        n_row, n_col = mat.shape

        out[:] = 0

        for i in range(n_col):
            for j in range(n_row):
                out[i] += vec[i].conjugate() * mat[i, j]

    return factories.qcjit(impl)


def vecct_mul_vec_factory(corr_mode):

    def impl(vec1, vec2):

        n_ele = vec1.size

        out = 0

        for i in range(n_ele):
            out += vec1[i].conjugate() * vec2[i]

        return out

    return factories.qcjit(impl)


def diag_add_factory(corr_mode):

    def impl(mat, scalar, out):

        n_ele, _ = mat.shape

        out[:] = mat

        for i in range(n_ele):
            out[i, i] += scalar

    return factories.qcjit(impl)


def vec_iadd_svec_factory(corr_mode):

    def impl(vec1, scalar, vec2):

        n_ele = vec1.size

        for i in range(n_ele):
            vec1[i] += scalar * vec2[i]

    return factories.qcjit(impl)


def vec_isub_svec_factory(corr_mode):

    def impl(vec1, scalar, vec2):

        n_ele = vec1.size

        for i in range(n_ele):
            vec1[i] -= scalar * vec2[i]

    return factories.qcjit(impl)


def get_jhj_dims_factory(corr_mode):

    if corr_mode.literal_value == 4:
        def impl(gain):
            return gain.shape[:4] + (4, 4)
    elif corr_mode.literal_value in (1, 2):
        def impl(gain):
            return gain.shape
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)
