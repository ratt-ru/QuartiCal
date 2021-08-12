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
from quartical.gains.general.inversion import (invert_factory,
                                               inversion_buffer_factory)
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
    imul = factories.imul_factory(corr_mode)
    iadd = factories.iadd_factory(corr_mode)
    valloc = factories.valloc_factory(corr_mode)
    make_loop_vars = factories.loop_var_factory(corr_mode)
    set_identity = factories.set_identity_factory(corr_mode)
    compute_jhwj_jhwr_elem = compute_jhwj_jhwr_elem_factory(corr_mode)

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

            tmp_kprod = np.zeros((4, 4), dtype=complex_dtype)
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

                        out_d = d_map_arr[active_term, d]

                        iunpack(lop_pq_arr[out_d], lop_pq)
                        iadd(rop_pq_arr[out_d], rop_pq)

                        iunpack(lop_qp_arr[out_d], lop_qp)
                        iadd(rop_qp_arr[out_d], rop_qp)

                    for d in range(n_gdir):

                        imul_rweight(r, r_pq, row_weights, row_ind)
                        imul(r_pq, w)
                        iunpackct(r_qp, r_pq)

                        lop_pq_d = lop_pq_arr[d]
                        rop_pq_d = rop_pq_arr[d]

                        compute_jhwj_jhwr_elem(lop_pq_d,
                                               rop_pq_d,
                                               w,
                                               tmp_kprod,
                                               r_pq,
                                               tmp_jhr[a1_m, d],
                                               tmp_jhj[a1_m, d])

                        lop_qp_d = lop_qp_arr[d]
                        rop_qp_d = rop_qp_arr[d]

                        compute_jhwj_jhwr_elem(lop_qp_d,
                                               rop_qp_d,
                                               w,
                                               tmp_kprod,
                                               r_qp,
                                               tmp_jhr[a2_m, d],
                                               tmp_jhj[a2_m, d])

            # This is the spigot point. A parameterised solver can do the
            # extra operations it requires here. Technically, we could also
            # compute the update directly, allowing us to avoid storing the
            # entirety of jhj and jhr.
            jhr[ti, fi] = tmp_jhr
            jhj[ti, fi] = tmp_jhj
        return
    return impl


@generated_jit(nopython=True,
               fastmath=True,
               parallel=True,
               cache=True,
               nogil=True)
def compute_update(update, jhj, jhr, corr_mode):

    generalised = jhj.ndim == 6
    inversion_buffer = inversion_buffer_factory(generalised=generalised)
    invert = invert_factory(corr_mode, generalised=generalised)

    def impl(update, jhj, jhr, corr_mode):
        n_tint, n_fint, n_ant, n_dir, n_param = jhr.shape

        n_int = n_tint * n_fint

        result_dtype = jhr.dtype

        for i in prange(n_int):

            t = i // n_fint
            f = i - t * n_fint

            buffers = inversion_buffer(n_param, result_dtype)

            for a in range(n_ant):
                for d in range(n_dir):

                    # TODO: This check needs to go away - slow and bad.
                    # Should rather process the updates.
                    det = np.linalg.det(jhj[t, f, a, d])

                    if np.abs(det) < 1e-6 or ~np.isfinite(det):
                        update[t, f, a, d] = 0
                    else:
                        invert(jhj[t, f, a, d],
                               jhr[t, f, a, d],
                               update[t, f, a, d],
                               buffers)

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


def compute_jhwj_jhwr_elem_factory(corr_mode):

    iadd = factories.iadd_factory(corr_mode)
    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    a_kron_bt = a_kron_bt_factory(corr_mode)
    unpack = factories.unpack_factory(corr_mode)
    unpackc = factories.unpackc_factory(corr_mode)

    if corr_mode.literal_value == 4:
        def impl(lop, rop, w, tmp_kprod, res, jhr, jhj):

            # Accumulate an element of jhwr.
            v1_imul_v2(lop, res, res)
            v1_imul_v2(res, rop, res)
            iadd(jhr, res)

            # Accumulate and element of jhwj.

            # WARNING: In this instance we are using the row-major
            # version of the kronecker product identity. This is because the
            # MS stores the correlations in row-major order (XX, XY, YX, YY),
            # whereas the standard maths assumes column-major ordering
            # (XX, YX, XY, YY). This subtle change means we can use the MS
            # data directly without worrying about swapping elements around.
            a_kron_bt(lop, rop, tmp_kprod)

            w_0, w_1, w_2, w_3 = unpack(w)  # NOTE: XX, XY, YX, YY

            for i in range(4):

                jh_0, jh_1, jh_2, jh_3 = unpack(tmp_kprod[i])

                jhw_0 = jh_0*w_0  # XX
                jhw_1 = jh_1*w_1  # XY
                jhw_2 = jh_2*w_2  # YX
                jhw_3 = jh_3*w_3  # YY

                for j in range(i):
                    jhj[i, j] = jhj[j, i].conjugate()

                for j in range(i, 4):
                    j_0, j_1, j_2, j_3 = unpackc(tmp_kprod[j])
                    jhj[i, j] += (jhw_0*j_0 + jhw_1*j_1 +
                                  jhw_2*j_2 + jhw_3*j_3)

    elif corr_mode.literal_value == 2:
        def impl(lop, rop, w, tmp_kprod, res, jhr, jhj):
            jh_00, jh_11 = unpack(rop)
            j_00, j_11 = unpackc(rop)
            w_00, w_11 = unpack(w)

            jhj[0] += jh_00*w_00*j_00
            jhj[1] += jh_11*w_11*j_11
    elif corr_mode.literal_value == 1:
        def impl(lop, rop, w, tmp_kprod, res, jhr, jhj):
            jh_00 = unpack(rop)
            j_00 = unpackc(rop)
            w_00 = unpack(w)

            jhj[0] += jh_00*w_00*j_00
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def a_kron_bt_factory(corr_mode):

    unpack = factories.unpack_factory(corr_mode)

    def impl(a, b, out):

        a00, a01, a10, a11 = unpack(a)
        b00, b10, b01, b11 = unpack(b)  # Effectively transpose.

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
