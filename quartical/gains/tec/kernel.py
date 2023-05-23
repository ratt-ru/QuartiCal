# -*- coding: utf-8 -*-
import numpy as np
from numba import generated_jit
from quartical.utils.numba import coerce_literal
from quartical.gains.general.generics import (native_intermediaries,
                                              upsampled_itermediaries,
                                              per_array_jhj_jhr,
                                              resample_solints,
                                              downsample_jhj_jhr)
from quartical.gains.general.flagging import (flag_intermediaries,
                                              update_gain_flags,
                                              finalize_gain_flags,
                                              apply_gain_flags,
                                              update_param_flags)
from quartical.gains.general.convenience import (get_row,
                                                 get_extents)

# from quartical.gains.delay.kernel import (compute_jhj_jhr,
                                        #   compute_update,
                                        #   finalize_update)
from collections import namedtuple
import ipdb

buffers = namedtuple("buffers", "Ap Ax p r")


# This can be done without a named tuple now. TODO: Add unpacking to
# constructor.
stat_fields = {"conv_iters": np.int64,
               "conv_perc": np.float64}

term_conv_info = namedtuple("term_conv_info", " ".join(stat_fields.keys()))

tec_args = namedtuple(
    "tec_args",
    (
        "params",
        "param_flags",
        "chan_freqs",
        "t_bin_arr"
    )
)


def get_identity_params(corr_mode):

    if corr_mode.literal_value in (2, 4):
        return np.zeros((4,), dtype=np.float64)
    elif corr_mode.literal_value == 1:
        return np.zeros((2,), dtype=np.float64)
    else:
        raise ValueError("Unsupported number of correlations.")


# @generated_jit(nopython=True,
#                fastmath=True,
#                parallel=False,
#                cache=True,
#                nogil=True)
def tec_solver(base_args, term_args, meta_args, corr_mode):

    # NOTE: This just reuses delay solver functionality.
    # coerce_literal(tec_solver, ["corr_mode"])
    # identity_params = get_identity_params(corr_mode)

    # import ipdb; ipdb.set_trace()
    gains = base_args.gains
    gain_flags = base_args.gain_flags

    active_term = meta_args.active_term
    max_iter = meta_args.iters
    solve_per = meta_args.solve_per
    dd_term = meta_args.dd_term
    n_thread = meta_args.threads

    active_gain = gains[active_term]
    active_gain_flags = gain_flags[active_term]
    active_params = term_args.params[active_term]

    # Set up some intemediaries used for flagging. TODO: Move?
    km1_gain = active_gain.copy()
    km1_abs2_diffs = np.zeros_like(active_gain_flags, dtype=np.float64)
    abs2_diffs_trend = np.zeros_like(active_gain_flags, dtype=np.float64)
    flag_imdry = \
        flag_intermediaries(km1_gain, km1_abs2_diffs, abs2_diffs_trend)

    # Set up some intemediaries used for solving.
    real_dtype = active_gain.real.dtype
    param_shape = active_params.shape

    active_t_map_g = base_args.t_map_arr[0, :, active_term]
    active_f_map_p = base_args.f_map_arr[1, :, active_term]

    # Create more work to do in paralllel when needed, else no-op.
    resampler = resample_solints(active_t_map_g, param_shape, n_thread)

    # Determine the starts and stops of the rows and channels associated
    # with each solution interval.
    extents = get_extents(resampler.upsample_t_map, active_f_map_p)

    upsample_shape = resampler.upsample_shape
    upsampled_jhj = np.empty(upsample_shape + (upsample_shape[-1],),
                                dtype=real_dtype)
    upsampled_jhr = np.empty(upsample_shape, dtype=real_dtype)
    jhj = upsampled_jhj[:param_shape[0]]
    jhr = upsampled_jhr[:param_shape[0]]
    update = np.zeros(param_shape, dtype=real_dtype)

    upsampled_imdry = upsampled_itermediaries(upsampled_jhj, upsampled_jhr)
    native_imdry = native_intermediaries(jhj, jhr, update)

    scaled_icf = term_args.chan_freqs.copy()  # Don't mutate.
    min_freq = np.min(scaled_icf)
    scaled_icf = 2 * np.pi * min_freq/scaled_icf  # Scale freqs to avoid precision.
    active_params[..., 1::2] /= min_freq  # Scale consistently with freq.

    for loop_idx in range(max_iter):

        compute_jhj_jhr(base_args,
                        term_args,
                        meta_args,
                        upsampled_imdry,
                        extents,
                        scaled_icf,
                        corr_mode)

        #Assuming I do not require downsampling for now - CR??
        if resampler.active:
            downsample_jhj_jhr(upsampled_imdry, resampler.downsample_t_map)
            # temporary higher resolution gain grid
            # it is creating fake solution intervals    
        if solve_per == "array":
            per_array_jhj_jhr(native_imdry)
        
        # import ipdb; ipdb.set_trace()
        compute_update(native_imdry,
                        corr_mode)

        finalize_update(base_args,
                        term_args,
                        meta_args,
                        native_imdry,
                        scaled_icf,
                        loop_idx,
                        corr_mode)
        
        # Check for gain convergence. Produced as a side effect of
        # flagging. The converged percentage is based on unflagged
        # intervals.
        conv_perc = update_gain_flags(base_args,
                                        term_args,
                                        meta_args,
                                        flag_imdry,
                                        loop_idx,
                                        corr_mode,
                                        numbness=1e9)

        # Propagate gain flags to parameter flags.
        update_param_flags(base_args,
                            term_args,
                            meta_args,
                            identity_params)

        if conv_perc >= meta_args.stop_frac:
            break

    # NOTE: Removes soft flags and flags points which have bad trends.
    finalize_gain_flags(base_args,
                        meta_args,
                        flag_imdry,
                        corr_mode)

    # Call this one last time to ensure points flagged by finialize are
    # propagated (in the DI case).
    if not dd_term:
        apply_gain_flags(base_args,
                            meta_args)

    active_params[..., 1::2] *= min_freq  # Undo scaling for SI units.
    
    np.save("/home/russeeawon/testing/testing_tecest_quartical/jhj.npy", native_imdry.jhj)
    return native_imdry.jhj, term_conv_info(loop_idx + 1, conv_perc)


def compute_jhj_jhr(
    base_args,
    term_args,
    meta_args,
    upsampled_imdry,
    extents,
    scaled_cf,
    corr_mode
):
    
    #to be used for numba - CR
    # We want to dispatch based on this field so we need its type.
    # row_weight_type = base_args[base_args.fields.index('row_weights')]

    # imul_rweight = factories.imul_rweight_factory(corr_mode, row_weight_type)
    # v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    # v1_imul_v2ct = factories.v1_imul_v2ct_factory(corr_mode)
    # v1ct_imul_v2 = factories.v1ct_imul_v2_factory(corr_mode)
    # absv1_idiv_absv2 = factories.absv1_idiv_absv2_factory(corr_mode)
    # iunpack = factories.iunpack_factory(corr_mode)
    # iunpackct = factories.iunpackct_factory(corr_mode)
    # imul = factories.imul_factory(corr_mode)
    # iadd = factories.iadd_factory(corr_mode)
    # isub = factories.isub_factory(corr_mode)
    # valloc = factories.valloc_factory(corr_mode)
    # make_loop_vars = factories.loop_var_factory(corr_mode)
    # set_identity = factories.set_identity_factory(corr_mode)
    # compute_jhwj_jhwr_elem = compute_jhwj_jhwr_elem_factory(corr_mode)

    # import ipdb; ipdb.set_trace()

    active_term = meta_args.active_term
    data = base_args.data
    model = base_args.model
    weights = base_args.weights
    flags = base_args.flags
    antenna1 = base_args.a1
    antenna2 = base_args.a2
    row_map = base_args.row_map
    row_weights = base_args.row_weights

    gains = base_args.gains
    t_map_arr = base_args.t_map_arr[0]  # We only need the gain mappings.
    f_map_arr_g = base_args.f_map_arr[0]
    d_map_arr = base_args.d_map_arr

    jhj = upsampled_imdry.jhj
    jhr = upsampled_imdry.jhr

    row_starts = extents.row_starts
    row_stops = extents.row_stops
    chan_starts = extents.chan_starts
    chan_stops = extents.chan_stops

    _, n_chan, n_dir, n_corr = model.shape

    jhj[:] = 0
    jhr[:] = 0

    n_tint, n_fint, n_ant, n_gdir, n_param = jhr.shape
    n_int = n_tint*n_fint

    complex_dtype = gains[active_term].dtype
    weight_dtype = weights.dtype

    n_gains = len(gains)

    # Determine loop variables based on where we are in the chain.
    # gt means greater than (n>j) and lt means less than (n<j).
    # all_terms, gt_active, lt_active = make_loop_vars(n_gains, active_term)

    ##trying out only one term
    # if n_corr == 4:
        # all_terms = np.arange(n_gains - 1, -1, -1)
    #     gt_active = np.arange(n_gains - 1, active_term, -1)
    #     lt_active = np.arange(active_term)

    import ipdb; ipdb.set_trace()
    # Parallel over all solution intervals.
    #prange parallel range by numba
    # for i in prange(n_int):
    for i in range(n_int):

        ti = i//n_fint
        fi = i - ti*n_fint

        rs = row_starts[ti]
        re = row_stops[ti]
        fs = chan_starts[fi]
        fe = chan_stops[fi]

        # rop_pq = valloc(complex_dtype)  # Right-multiply operator for pq.
        # rop_qp = valloc(complex_dtype)  # Right-multiply operator for qp.
        # lop_pq = valloc(complex_dtype)  # Left-multiply operator for pq.
        # lop_qp = valloc(complex_dtype)  # Left-multiply operator for qp.

        rop_pq = valloc(complex_dtype, n_corr)
        rop_qp = valloc(complex_dtype, n_corr)
        lop_pq = valloc(complex_dtype, n_corr)
        lop_qp = valloc(complex_dtype, n_corr)


        # w = valloc(weight_dtype)
        # r_pq = valloc(complex_dtype)
        # wr_pq = valloc(complex_dtype)
        # wr_qp = valloc(complex_dtype)
        # v_pqd = valloc(complex_dtype)
        # v_pq = valloc(complex_dtype)

        w = valloc(weight_dtype, n_corr)
        r_pq = valloc(complex_dtype, n_corr)
        wr_pq = valloc(complex_dtype, n_corr)
        wr_qp = valloc(complex_dtype, n_corr)
        v_pqd = valloc(complex_dtype, n_corr)
        v_pq = valloc(complex_dtype, n_corr)



        # gains_p = valloc(complex_dtype, leading_dims=(n_gains,))
        # gains_q = valloc(complex_dtype, leading_dims=(n_gains,))
        gains_p = valloc(complex_dtype, n_corr, leading_dims=n_gains)
        gains_q = valloc(complex_dtype, n_corr, leading_dims=n_gains)


        
        # lop_pq_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
        # rop_pq_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
        # lop_qp_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
        # rop_qp_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
        lop_pq_arr = valloc(complex_dtype, n_corr, leading_dims=n_gdir)
        rop_pq_arr = valloc(complex_dtype, n_corr, leading_dims=n_gdir)
        lop_qp_arr = valloc(complex_dtype, n_corr, leading_dims=n_gdir)
        rop_qp_arr = valloc(complex_dtype, n_corr, leading_dims=n_gdir)


        # norm_factors = valloc(complex_dtype)
        norm_factors = valloc(complex_dtype, n_corr)
        
        jhr_tifi = jhr[ti, fi]
        jhj_tifi = jhj[ti, fi]

        for row_ind in range(rs, re):

            row = get_row(row_ind, row_map)
            a1_m, a2_m = antenna1[row], antenna2[row]

            for f in range(fs, fe):

                if flags[row, f]:  # Skip flagged data points.
                    continue

                # Apply row weights in the BDA case, otherwise a no-op.
                # imul_rweight(weights[row, f], w, row_weights, row_ind)
                # iunpack(r_pq, data[row, f])
                imul_rweight(n_corr, weights[row, f], w, row_weights, row_ind)
                iunpack(n_corr, r_pq, data[row, f])

                lop_pq_arr[:] = 0
                rop_pq_arr[:] = 0
                lop_qp_arr[:] = 0
                rop_qp_arr[:] = 0
                v_pq[:] = 0

                for d in range(n_dir):

                    # set_identity(lop_pq)
                    # set_identity(lop_qp)
                    set_identity(n_corr, lop_pq)
                    set_identity(n_corr, lop_qp)

                    # Construct a small contiguous gain array. This makes
                    # the single term case fractionally slower.
                    for gi in range(n_gains):
                        d_m = d_map_arr[gi, d]  # Broadcast dir.
                        t_m = t_map_arr[row_ind, gi]
                        f_m = f_map_arr_g[f, gi]

                        gain = gains[gi][t_m, f_m]

                        # iunpack(gains_p[gi], gain[a1_m, d_m])
                        # iunpack(gains_q[gi], gain[a2_m, d_m])
                        iunpack(n_corr, gains_p[gi], gain[a1_m, d_m])
                        iunpack(n_corr, gains_q[gi], gain[a2_m, d_m])


                    m = model[row, f, d]
                    # iunpack(rop_qp, m)
                    # iunpackct(rop_pq, rop_qp)
                    iunpack(n_corr, rop_qp, m)
                    iunpackct(n_corr, rop_pq, rop_qp)

                    #Do I need the following? - CR
                    # for g in all_terms:

                    #     g_q = gains_q[g]
                    #     v1_imul_v2(g_q, rop_pq, rop_pq)

                    #     g_p = gains_p[g]
                    #     v1_imul_v2(g_p, rop_qp, rop_qp)

                    # for g in gt_active:

                    #     g_p = gains_p[g]
                    #     v1_imul_v2ct(rop_pq, g_p, rop_pq)

                    #     g_q = gains_q[g]
                    #     v1_imul_v2ct(rop_qp, g_q, rop_qp)

                    # for g in lt_active:

                    #     g_p = gains_p[g]
                    #     v1ct_imul_v2(g_p, lop_pq, lop_pq)

                    #     g_q = gains_q[g]
                    #     v1ct_imul_v2(g_q, lop_qp, lop_qp)

                    g_q = gains_q[0]
                    v1_imul_v2(n_corr, g_q, rop_pq, rop_pq)

                    g_p = gains_p[0]
                    v1_imul_v2(n_corr, g_p, rop_qp, rop_qp)

                    out_d = d_map_arr[active_term, d]

                    # iunpack(lop_pq_arr[out_d], lop_pq)
                    # iadd(rop_pq_arr[out_d], rop_pq)
                    iunpack(n_corr, lop_pq_arr[out_d], lop_pq)
                    iadd(n_corr, rop_pq_arr[out_d], rop_pq)

                    
                    # iunpack(lop_qp_arr[out_d], lop_qp)
                    # iadd(rop_qp_arr[out_d], rop_qp)
                    iunpack(n_corr, lop_qp_arr[out_d], lop_qp)
                    iadd(n_corr, rop_qp_arr[out_d], rop_qp)


                    
                    # v1ct_imul_v2(lop_pq, gains_p[active_term], v_pqd)
                    # v1_imul_v2ct(v_pqd, rop_pq, v_pqd)
                    # iadd(v_pq, v_pqd)

                    v1ct_imul_v2(n_corr, lop_pq, gains_p[active_term], v_pqd)
                    v1_imul_v2ct(n_corr, v_pqd, rop_pq, v_pqd)
                    iadd(n_corr, v_pq, v_pqd)

                    # import ipdb; ipdb.set_trace()
                # absv1_idiv_absv2(v_pq, r_pq, norm_factors)
                # imul(r_pq, norm_factors)
                # isub(r_pq, v_pq)

                
                absv1_idiv_absv2(n_corr, v_pq, r_pq, norm_factors)
                imul(n_corr, r_pq, norm_factors)
                isub(n_corr, r_pq, v_pq)


                nu = scaled_cf[f]

                for d in range(n_gdir):

                    # iunpack(wr_pq, r_pq)
                    # imul(wr_pq, w)
                    # iunpackct(wr_qp, wr_pq)
                    
                    iunpack(n_corr, wr_pq, r_pq)
                    imul(n_corr, wr_pq, w)
                    iunpackct(n_corr, wr_qp, wr_pq)

                    lop_pq_d = lop_pq_arr[d]
                    rop_pq_d = rop_pq_arr[d]
                    
                    compute_jhwj_jhwr_elem(n_corr, lop_pq_d,
                                            rop_pq_d,
                                            w,
                                            norm_factors,
                                            nu,
                                            gains_p[active_term],
                                            wr_pq,
                                            jhr_tifi[a1_m, d],
                                            jhj_tifi[a1_m, d])

                    lop_qp_d = lop_qp_arr[d]
                    rop_qp_d = rop_qp_arr[d]

                    compute_jhwj_jhwr_elem(n_corr, lop_qp_d,
                                            rop_qp_d,
                                            w,
                                            norm_factors,
                                            nu,
                                            gains_q[active_term],
                                            wr_qp,
                                            jhr_tifi[a2_m, d],
                                            jhj_tifi[a2_m, d])

                                            
    return 


#------------------------------------------------------------
#Some factories
def valloc(dtype, n_corr, leading_dims=None):
    if leading_dims:
        return np.empty((leading_dims, n_corr), dtype=dtype)
    else:
        return np.empty((n_corr), dtype=dtype)


def unpack(n_corr, invec):
    if n_corr == 4:
        return invec[0], invec[1], invec[2], invec[3]
    elif n_corr == 2:
        return invec[0], invec[1]
    elif n_corr == 1:
        return invec[0]
    else:
        raise ValueError("Unsupported number of correlations.")


def unpackct(n_corr, invec):
    if n_corr == 4:
        return np.conjugate(invec[0]), np.conjugate(invec[1]), np.conjugate(invec[2]), np.conjugate(invec[3])
    elif n_corr == 2:
        return np.conjugate(invec[0]), np.conjugate(invec[1])
    elif n_corr == 1:
        return np.conjugate(invec[0])
    else:
        raise ValueError("Unsupported number of correlations.")


def imul_rweight(n_corr, invec, outvec, weights, ind):
    if weights is None:
        if n_corr == 4:
            outvec[0] = invec[0]
            outvec[1] = invec[1]
            outvec[2] = invec[2]
            outvec[3] = invec[3]
        elif n_corr == 2:
            outvec[0] = invec[0]
            outvec[1] = invec[1]
        elif n_corr == 1:
            outvec[0] = invec[0]
        else:
            raise ValueError("Unsupported number of correlations.")

    else:
        w = weights[ind]
        if n_corr == 4:
            v00, v01, v10, v11 = unpack(n_corr, invec)
            outvec[0] = w*v00
            outvec[1] = w*v01
            outvec[2] = w*v10
            outvec[3] = w*v11
        elif n_corr == 2:
            v00, v11 = unpack(n_corr, invec)
            outvec[0] = w*v00
            outvec[1] = w*v11
        elif n_corr == 1:
            v00 = unpack(n_corr, invec)
            outvec[0] = w*v00
        else:
            raise ValueError("Unsupported number of correlations.")


def iunpack(n_corr, outvec, invec):
    if n_corr == 4:
        outvec[0] = invec[0]
        outvec[1] = invec[1]
        outvec[2] = invec[2]
        outvec[3] = invec[3]
    elif n_corr == 2:
        outvec[0] = invec[0]
        outvec[1] = invec[1]
    elif n_corr == 1:
        outvec[0] = invec[0]
    else:
        raise ValueError("Unsupported number of correlations.")

def unpackc(n_corr, invec):

    if n_corr == 4:
        return invec[0].conjugate(), \
                invec[1].conjugate(), \
                invec[2].conjugate(), \
                invec[3].conjugate()
    elif n_corr == 2:
        return invec[0].conjugate(), invec[1].conjugate()
    elif n_corr == 1:
        return invec[0].conjugate()
    else:
        raise ValueError("Unsupported number of correlations.")

#unpack conjugate transpose
def iunpackct(n_corr, outvec, invec):
    if n_corr == 4:
        outvec[0] = np.conjugate(invec[0])
        outvec[1] = np.conjugate(invec[1])
        outvec[2] = np.conjugate(invec[2])
        outvec[3] = np.conjugate(invec[3])
    elif n_corr == 2:
        outvec[0] = np.conjugate(invec[0])
        outvec[1] = np.conjugate(invec[1])
    elif n_corr == 1:
        outvec[0] = np.conjugate(invec[0])
    else:
        raise ValueError("Unsupported number of correlations.")


#set to identity matrix
def set_identity(n_corr, vec):
    if n_corr == 4:
        vec[0] = 1
        vec[1] = 0
        vec[2] = 0
        vec[3] = 1    
    elif n_corr == 2:
        vec[0] = 1
        vec[1] = 1
    elif n_corr == 1:
        vec[0] = 1
    else:
        raise ValueError("Unsupported number of correlations.")


def iadd(n_corr, outvec, invec):
    if n_corr == 4:
        outvec[0] += invec[0]
        outvec[1] += invec[1]
        outvec[2] += invec[2]
        outvec[3] += invec[3]
    elif n_corr == 2:
        outvec[0] += invec[0]
        outvec[1] += invec[1]
    elif n_corr == 1:
        outvec[0] += invec[0]
    else:
        raise ValueError("Unsupported number of correlations.")


def template_corr_func(n_corr, v1, v2, func1, func2):
    if n_corr == 4:
        v1_00, v1_01, v1_10, v1_11 = func1(n_corr, v1)
        v2_00, v2_01, v2_10, v2_11 = func2(n_corr, v2)

        v3_00 = (v1_00*v2_00 + v1_01*v2_10)
        v3_01 = (v1_00*v2_01 + v1_01*v2_11)
        v3_10 = (v1_10*v2_00 + v1_11*v2_10)
        v3_11 = (v1_10*v2_01 + v1_11*v2_11)
        return v3_00, v3_01, v3_10, v3_11
    elif n_corr == 2:
        v1_00, v1_11 = func1(n_corr, v1)
        v2_00, v2_11 = func2(n_corr, v2)

        v3_00 = v1_00*v2_00
        v3_11 = v1_11*v2_11
        return v3_00, v3_11
    elif n_corr == 1:
        v1_00 = func1(n_corr, v1)
        v2_00 = func2(n_corr, v2)

        v3_00 = v1_00*v2_00
        return v3_00
    else:
        raise ValueError("Unsupported number of correlations.")


def itemplate_corr_func(n_corr, v1, v2, o1, func1, func2):
    if n_corr == 4:
        v1_00, v1_01, v1_10, v1_11 = func1(n_corr, v1)
        v2_00, v2_01, v2_10, v2_11 = func2(n_corr, v2)

        o1[0] = (v1_00*v2_00 + v1_01*v2_10)
        o1[1] = (v1_00*v2_01 + v1_01*v2_11)
        o1[2] = (v1_10*v2_00 + v1_11*v2_10)
        o1[3] = (v1_10*v2_01 + v1_11*v2_11)
    elif n_corr == 2:
        v1_00, v1_11 = func1(n_corr, v1)
        v2_00, v2_11 = func2(n_corr, v2)

        o1[0] = v1_00*v2_00
        o1[1] = v1_11*v2_11
    elif n_corr == 1:
        v1_00 = func1(n_corr, v1)
        v2_00 = func2(n_corr, v2)

        o1[0] = v1_00*v2_00
    else:
        raise ValueError("Unsupported number of correlations.")


def v1_mul_v2(n_corr, v1, v2):
    template_corr_func(n_corr, v1, v2, unpack, unpack)


def v1_imul_v2(n_corr, v1, v2, o1):
    itemplate_corr_func(n_corr, v1, v2, o1, unpack, unpack)


def v1ct_imul_v2(n_corr, v1, v2, o1):
    itemplate_corr_func(n_corr, v1, v2, o1, unpackct, unpack)


def v1_imul_v2ct(n_corr, v1, v2, o1):
    itemplate_corr_func(n_corr, v1, v2, o1, unpack, unpackct)


def absv1_idiv_absv2(n_corr, v1, v2, o1):
    if n_corr == 4:
        o1[0] = 0 if v2[0] == 0 else np.sqrt(
            (v1[0].real**2 + v1[0].imag**2)/(v2[0].real**2 + v2[0].imag**2)
        )
        o1[1] = 0 if v2[1] == 0 else np.sqrt(
            (v1[1].real**2 + v1[1].imag**2)/(v2[1].real**2 + v2[1].imag**2)
        )
        o1[2] = 0 if v2[2] == 0 else np.sqrt(
            (v1[2].real**2 + v1[2].imag**2)/(v2[2].real**2 + v2[2].imag**2)
        )
        o1[3] = 0 if v2[3] == 0 else np.sqrt(
            (v1[3].real**2 + v1[3].imag**2)/(v2[3].real**2 + v2[3].imag**2)
        )
    elif n_corr == 2:
        o1[0] = 0 if v2[0] == 0 else np.sqrt(
            (v1[0].real**2 + v1[0].imag**2)/(v2[0].real**2 + v2[0].imag**2)
        )
        o1[1] = 0 if v2[1] == 0 else np.sqrt(
            (v1[1].real**2 + v1[1].imag**2)/(v2[1].real**2 + v2[1].imag**2)
        )
    elif n_corr == 1:
        o1[0] = 0 if v2[0] == 0 else np.sqrt(
            (v1[0].real**2 + v1[0].imag**2)/(v2[0].real**2 + v2[0].imag**2)
        )
    else:
        raise ValueError("Unsupported number of correlations.")


def imul(n_corr, outvec, invec):
    if n_corr == 4:
        outvec[0] *= invec[0]
        outvec[1] *= invec[1]
        outvec[2] *= invec[2]
        outvec[3] *= invec[3]
    elif n_corr == 2:
        outvec[0] *= invec[0]
        outvec[1] *= invec[1]
    elif n_corr == 1:
        outvec[0] *= invec[0]
    else:
        raise ValueError("Unsupported number of correlations.")


def isub(n_corr, outvec, invec):
    if n_corr == 4:
        outvec[0] -= invec[0]
        outvec[1] -= invec[1]
        outvec[2] -= invec[2]
        outvec[3] -= invec[3]
    elif n_corr == 2:
        outvec[0] -= invec[0]
        outvec[1] -= invec[1]
    elif n_corr == 1:
        outvec[0] -= invec[0]
    else:
        raise ValueError("Unsupported number of correlations.")


def iabsdivsq(n_corr, v1):
    if n_corr == 4:
        v1_0, v1_1, v1_2, v1_3 = unpack(n_corr, v1)

        v1[0] = 0 if v1_0 == 0 else 1/(v1_0.real**2 + v1_0.imag**2)
        v1[1] = 0 if v1_1 == 0 else 1/(v1_1.real**2 + v1_1.imag**2)
        v1[2] = 0 if v1_2 == 0 else 1/(v1_2.real**2 + v1_2.imag**2)
        v1[3] = 0 if v1_3 == 0 else 1/(v1_3.real**2 + v1_3.imag**2)
    elif n_corr == 2:
        v1_0, v1_1 = unpack(n_corr, v1)

        v1[0] = 0 if v1_0 == 0 else 1/(v1_0.real**2 + v1_0.imag**2)
        v1[1] = 0 if v1_1 == 0 else 1/(v1_1.real**2 + v1_1.imag**2)
    elif n_corr == 1:
        v1_0 = unpack(n_corr, v1)

        v1[0] = 0 if v1_0 == 0 else 1/(v1_0.real**2 + v1_0.imag**2)
    else:
        raise ValueError("Unsupported number of correlations.")


def compute_jhwj_jhwr_elem(n_corr, lop, rop, w, normf, nu, gain, res, jhr, jhj):

    # v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    # unpack = factories.unpack_factory(corr_mode)
    # unpackc = factories.unpackc_factory(corr_mode)
    # iunpack = factories.iunpack_factory(corr_mode)
    # iabsdivsq = factories.iabsdivsq_factory(corr_mode)
    # imul = factories.imul_factory(corr_mode)

    if n_corr == 4:

        # Effectively apply zero weight to off-diagonal terms.
        # TODO: Can be tidied but requires moving other weighting code.
        res[1] = 0
        res[2] = 0

        # Compute normalization factor.
        # v1_imul_v2(lop, rop, normf)
        # iabsdivsq(normf)
        # imul(res, normf)  # Apply normalization factor to r.

        v1_imul_v2(n_corr, lop, rop, normf)
        iabsdivsq(n_corr, normf)
        imul(n_corr, res, normf)

        # Accumulate an element of jhwr.
        # v1_imul_v2(res, rop, res)
        # v1_imul_v2(lop, res, res)

        v1_imul_v2(n_corr, res, rop, res)
        v1_imul_v2(n_corr, lop, res, res)


        # Accumulate an element of jhwj.

        # r_0, _, _, r_3 = unpack(res)  # NOTE: XX, XY, YX, YY

        # _, _, _, g_3 = unpack(gain)
        # gc_0, _, _, gc_3 = unpackc(gain)

        r_0, _, _, r_3 = unpack(n_corr, res)
        _, _, _, g_3 = unpack(n_corr, gain)
        gc_0, _, _, gc_3 = unpackc(n_corr, gain)

        drv_00 = -1j*gc_0
        drv_23 = -1j*gc_3

        upd_00 = (drv_00*r_0).real
        upd_11 = (drv_23*r_3).real

        jhr[0] += upd_00
        jhr[1] += nu*upd_00
        jhr[2] += upd_11
        jhr[3] += nu*upd_11

        # w_0, _, _, w_3 = unpack(w)  # NOTE: XX, XY, YX, YY
        # n_0, _, _, n_3 = unpack(normf)

        w_0, _, _, w_3 = unpack(n_corr, w)  # NOTE: XX, XY, YX, YY
        n_0, _, _, n_3 = unpack(n_corr, normf)

        # Apply normalisation factors by scaling w. # Neglect (set weight
        # to zero) off diagonal terms.
        w_0 = n_0 * w_0
        w_3 = n_3 * w_3

        # lop_00, lop_01, lop_10, lop_11 = unpack(lop)
        # rop_00, rop_10, rop_01, rop_11 = unpack(rop)  # "Transpose"

        lop_00, lop_01, lop_10, lop_11 = unpack(n_corr, lop)
        rop_00, rop_10, rop_01, rop_11 = unpack(n_corr, rop)  # "Transpose"

        jh_00 = lop_00 * rop_00
        jh_03 = lop_01 * rop_01

        j_00 = jh_00.conjugate()
        j_03 = jh_03.conjugate()

        jh_30 = lop_10 * rop_10
        jh_33 = lop_11 * rop_11

        j_30 = jh_30.conjugate()
        j_33 = jh_33.conjugate()

        jhwj_00 = jh_00*w_0*j_00 + jh_03*w_3*j_03
        jhwj_03 = jh_00*w_0*j_30 + jh_03*w_3*j_33
        jhwj_33 = jh_30*w_0*j_30 + jh_33*w_3*j_33

        nusq = nu * nu

        tmp_0 = jhwj_00.real
        jhj[0, 0] += tmp_0
        jhj[0, 1] += tmp_0*nu
        tmp_1 = (jhwj_03*gc_0*g_3).real
        jhj[0, 2] += tmp_1
        jhj[0, 3] += tmp_1*nu

        jhj[1, 0] = jhj[0, 1]
        jhj[1, 1] += tmp_0*nusq
        jhj[1, 2] = jhj[0, 3]
        jhj[1, 3] += tmp_1*nusq

        jhj[2, 0] = jhj[0, 2]
        jhj[2, 1] = jhj[1, 2]
        tmp_2 = jhwj_33.real
        jhj[2, 2] += tmp_2
        jhj[2, 3] += tmp_2*nu

        jhj[3, 0] = jhj[0, 3]
        jhj[3, 1] = jhj[1, 3]
        jhj[3, 2] = jhj[2, 3]
        jhj[3, 3] += tmp_2*nusq

    elif n_corr == 2:
        # Compute normalization factor.
        # iunpack(normf, rop)
        # iabsdivsq(normf)
        # imul(res, normf)  # Apply normalization factor to r.

        iunpack(n_corr, normf, rop)
        iabsdivsq(n_corr, normf)
        imul(n_corr, res, normf)

        # Accumulate an element of jhwr.
        # v1_imul_v2(res, rop, res)
        v1_imul_v2(n_corr, res, rop, res)

        # r_0, r_1 = unpack(res)
        # gc_0, gc_1 = unpackc(gain)

        r_0, r_1 = unpack(n_corr, res)
        gc_0, gc_1 = unpackc(n_corr, gain)

        drv_00 = -1j*gc_0
        drv_23 = -1j*gc_1

        upd_00 = (drv_00*r_0).real
        upd_11 = (drv_23*r_1).real

        jhr[0] += upd_00
        jhr[1] += upd_00/nu
        jhr[2] += upd_11
        jhr[3] += upd_11/nu

        # Accumulate an element of jhwj.
        # jh_00, jh_11 = unpack(rop)
        # j_00, j_11 = unpackc(rop)
        # w_00, w_11 = unpack(w)
        # n_00, n_11 = unpack(normf)

        jh_00, jh_11 = unpack(n_corr, rop)
        j_00, j_11 = unpackc(n_corr, rop)
        w_00, w_11 = unpack(n_corr, w)
        n_00, n_11 = unpack(n_corr, normf)

        nusq = nu*nu

        tmp = (jh_00*n_00*w_00*j_00).real
        jhj[0, 0] += tmp
        jhj[0, 1] += tmp/nu
        jhj[1, 0] += tmp/nu
        jhj[1, 1] += tmp/nusq

        tmp = (jh_11*n_11*w_11*j_11).real
        jhj[2, 2] += tmp
        jhj[2, 3] += tmp/nu
        jhj[3, 2] += tmp/nu
        jhj[3, 3] += tmp/nusq

    elif n_corr == 1:
        # Compute normalization factor.
        # iunpack(normf, rop)
        # iabsdivsq(normf)
        # imul(res, normf)  # Apply normalization factor to r.

        iunpack(n_corr, normf, rop)
        iabsdivsq(n_corr, normf)
        imul(n_corr, res, normf)

        # Accumulate an element of jhwr.
        # v1_imul_v2(res, rop, res)
        v1_imul_v2(n_corr, res, rop, res)

        # r_0 = unpack(res)
        # gc_0 = unpackc(gain)

        r_0 = unpack(n_corr, res)
        gc_0 = unpackc(n_corr, gain)

        drv_00 = -1j*gc_0

        upd_00 = (drv_00*r_0).real

        jhr[0] += upd_00
        jhr[1] += upd_00/nu

        # Accumulate an element of jhwj.
        # jh_00 = unpack(rop)
        # j_00 = unpackc(rop)
        # w_00 = unpack(w)
        # n_00 = unpack(normf)

        jh_00 = unpack(n_corr, rop)
        j_00 = unpackc(n_corr, rop)
        w_00 = unpack(n_corr, w)
        n_00 = unpack(n_corr, normf)

        nusq = nu*nu

        tmp = (jh_00*n_00*w_00*j_00).real
        jhj[0, 0] += tmp
        jhj[0, 1] += tmp/nu
        jhj[1, 0] += tmp/nu
        jhj[1, 1] += tmp/nusq
    else:
        raise ValueError("Unsupported number of correlations.")


def inversion_buffer(n_param, dtype, generalised=False):
    if generalised:
        r = np.zeros(n_param, dtype=dtype)
        p = np.zeros(n_param, dtype=dtype)
        Ap = np.zeros(n_param, dtype=dtype)
        Ax = np.zeros(n_param, dtype=dtype)

        return buffers(Ap, Ax, p, r)
    else:
        return

def invert(n_corr, A, b, x, buffers, generalised=False):
    if generalised:
        # mat_mul_vec = mat_mul_vec_factory(corr_mode)
        # vecct_mul_vec = vecct_mul_vec_factory(corr_mode)
        # vec_iadd_svec = vec_iadd_svec_factory(corr_mode)
        # vec_isub_svec = vec_isub_svec_factory(corr_mode)


        Ap, Ax, p, r = buffers

        mat_mul_vec(n_corr, A, x, Ax)
        r[:] = b
        r -= Ax
        p[:] = r
        r_k = vecct_mul_vec(n_corr, r, r)

        # If the resdiual is exactly zero, x is exact or missing.
        if r_k == 0:
            return

        for _ in range(x.size):
            mat_mul_vec(n_corr, A, p, Ap)
            alpha_denom = vecct_mul_vec(n_corr, p, Ap)
            alpha = r_k / alpha_denom
            vec_iadd_svec(n_corr, x, alpha, p)
            vec_isub_svec(n_corr, r, alpha, Ap)
            r_kplus1 = vecct_mul_vec(n_corr, r, r)
            if r_kplus1.real < 1e-30:
                break
            p *= (r_kplus1 / r_k)
            p += r
            r_k = r_kplus1

    else:

        # v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
        # compute_det = factories.compute_det_factory(corr_mode)
        # iinverse = factories.iinverse_factory(corr_mode)
        import ipdb; ipdb.set_trace()
        det = compute_det(n_corr, A)
        ##try using numpy.linalg.det()
        # det = np.linalg.det(A)

        if det.real.any() < 1e-6:
            x[:] = 0
        else:
            iinverse(n_corr, A, det, x)

        v1_imul_v2(n_corr, b, x, x)


def mat_mul_vec(n_corr, mat, vec, out):

    n_row, n_col = mat.shape

    out[:] = 0

    for i in range(n_row):
        for j in range(n_col):
            out[i] += mat[i, j] * vec[j]


def vecct_mul_mat(n_corr, vec, mat, out):
    n_row, n_col = mat.shape

    out[:] = 0

    for i in range(n_col):
        for j in range(n_row):
            out[i] += vec[i].conjugate() * mat[i, j]


def vec_iadd_svec(n_corr, vec1, scalar, vec2):
    n_ele = vec1.size

    for i in range(n_ele):
        vec1[i] += scalar * vec2[i]


def vec_isub_svec(n_corr, vec1, scalar, vec2):
    n_ele = vec1.size

    for i in range(n_ele):
        vec1[i] -= scalar * vec2[i]


def compute_det(n_corr, v1):
    if n_corr == 4:
        return v1[0]*v1[3] - v1[1]*v1[2]
    elif n_corr == 2:
        return v1[0]*v1[1]
    elif n_corr == 1:
        return v1[0]
    else:
        raise ValueError("Unsupported number of correlations.")


def iinverse(n_corr, v1, det, o1):

    if n_corr == 4:
        v1_00, v1_01, v1_10, v1_11 = unpack(n_corr, v1)

        o1[0] = v1_11/det
        o1[1] = -v1_01/det
        o1[2] = -v1_10/det
        o1[3] = v1_00/det
    elif n_corr == 2:
        v1_00, v1_11 = unpack(n_corr, v1)

        o1[0] = v1_11/det
        o1[1] = v1_00/det 
    elif n_corr == 1:  # TODO: Is this correct?
        v1_00 = unpack(n_corr, v1)

        o1[0] = 1.0/v1_00
    else:
        raise ValueError("Unsupported number of correlations.")



def param_to_gain(n_corr, params, chanfreq, gain):

    if n_corr == 4:
        gain[0] = np.exp(1j*(chanfreq*params[1] + params[0]))
        gain[3] = np.exp(1j*(chanfreq*params[3] + params[2]))
    elif n_corr == 2:
        gain[0] = np.exp(1j*(chanfreq*params[1] + params[0]))
        gain[1] = np.exp(1j*(chanfreq*params[3] + params[2]))
    elif n_corr == 1:
        gain[0] = np.exp(1j*(chanfreq*params[1] + params[0]))
    else:
        raise ValueError("Unsupported number of correlations.")



#-------------------------------------------------------
#not a factory!
# def downsample_jhj_jhr(upsampled_imdry, downsample_t_map):
#     jhj = upsampled_imdry.jhj
#     jhr = upsampled_imdry.jhr

#     n_tint, n_fint, n_ant, n_dir = jhj.shape[:4]

#     prev_out_ti = -1

#     for ti in range(n_tint):

#         out_ti = downsample_t_map[ti]

#         for fi in range(n_fint):
#             for a in range(n_ant):
#                 for d in range(n_dir):

#                     if prev_out_ti != out_ti:
#                         jhj[out_ti, fi, a, d] = jhj[ti, fi, a, d]
#                         jhr[out_ti, fi, a, d] = jhr[ti, fi, a, d]
#                     else:
#                         jhj[out_ti, fi, a, d] += jhj[ti, fi, a, d]
#                         jhr[out_ti, fi, a, d] += jhr[ti, fi, a, d]

#         prev_out_ti = out_ti


def per_array_jhj_jhr(solver_imdry):
    """This manipulates the entries of jhj and jhr to be over all antennas."""

    jhj = solver_imdry.jhj
    jhr = solver_imdry.jhr

    n_tint, n_fint, n_ant = jhj.shape[:3]

    for t in range(n_tint):
        for f in range(n_fint):
            for a in range(1, n_ant):
                jhj[t, f, 0] += jhj[t, f, a]
                jhr[t, f, 0] += jhr[t, f, a]

            for a in range(1, n_ant):
                jhj[t, f, a] = jhj[t, f, 0]
                jhr[t, f, a] = jhr[t, f, 0]



def compute_update(native_imdry, corr_mode):
    # import ipdb; ipdb.set_trace()
    # We want to dispatch based on this field so we need its type.
    # jhj = native_imdry[native_imdry.fields.index('jhj')] 
    # << error: native_imdry does not have attribute 'fields'
    import ipdb; ipdb.set_trace()
    jhj = native_imdry.jhj

    generalised = jhj.ndim == 6
    # inversion_buffer = inversion_buffer_factory(generalised=generalised)
    # invert = invert_factory(corr_mode, generalised=generalised)

    jhj = native_imdry.jhj
    jhr = native_imdry.jhr
    update = native_imdry.update

    n_tint, n_fint, n_ant, n_dir, n_param = jhr.shape

    n_int = n_tint * n_fint

    result_dtype = jhr.dtype

    # for i in prange(n_int):
    for i in range(n_int):

        t = i // n_fint
        f = i - t * n_fint

        buffers = inversion_buffer(n_param, result_dtype, generalised=generalised)

        for a in range(n_ant):
            for d in range(n_dir):

                invert(corr_mode, jhj[t, f, a, d],
                        jhr[t, f, a, d],
                        update[t, f, a, d],
                        buffers)


def finalize_update(base_args, term_args, meta_args, native_imdry, scaled_cf,
                    loop_idx, corr_mode):

    # set_identity = factories.set_identity_factory(corr_mode)
    # param_to_gain = param_to_gain_factory(corr_mode)

    if n_corr in (1, 2, 4):
        # def impl(base_args, term_args, meta_args, native_imdry, scaled_cf,
        #          loop_idx, corr_mode):

        active_term = meta_args.active_term

        gain = base_args.gains[active_term]
        gain_flags = base_args.gain_flags[active_term]
        f_map_arr_p = base_args.f_map_arr[1, :, active_term]
        d_map_arr = base_args.d_map_arr[active_term, :]

        params = term_args.params[active_term]

        update = native_imdry.update

        update /= 2
        params += update

        n_time, n_freq, n_ant, n_dir, _ = gain.shape

        for t in range(n_time):
            for f in range(n_freq):
                cf = scaled_cf[f]
                f_m = f_map_arr_p[f]
                for a in range(n_ant):
                    for d in range(n_dir):

                        d_m = d_map_arr[d]
                        g = gain[t, f, a, d]
                        fl = gain_flags[t, f, a, d]
                        p = params[t, f_m, a, d_m]

                        if fl == 1:
                            set_identity(g)
                        else:
                            param_to_gain(corr_mode, p, cf, g)
    else:
        raise ValueError("Unsupported number of correlations.")


#--------------------------------------------------------------------------
#Specific to flagging
def finalize_gain_flags(base_args, meta_args, flag_imdry, corr_mode):
    """Removes soft flags and flags points which failed to converge.

    Given the gains, assosciated gain flags and the trend of abosolute
    differences, remove soft flags which were never hardened and hard flag
    points which have positive trend values. This corresponds to points
    which have bad solutions when convergence/maximum iterations are reached.

    Args:
        gain: A (ti, fi, a, d, c) array of gain values.
        gain_flags: A (ti, fi, a, d) array of flag values.
        ab2_diffs_trends: An array containing the accumulated trend values of
            the absolute difference between gains at each iteration. Positive
            values correspond to points which are nowhere near convergence.
    """

    # set_identity = factories.set_identity_factory(corr_mode)

    active_term = meta_args.active_term

    gain = base_args.gains[active_term]
    gain_flags = base_args.gain_flags[active_term]

    abs2_diffs_trend = flag_imdry.abs2_diffs_trend

    n_tint, n_fint, n_ant, n_dir = gain_flags.shape

    for ti in range(n_tint):
        for fi in range(n_fint):
            for a in range(n_ant):
                for d in range(n_dir):
                    if abs2_diffs_trend[ti, fi, a, d] > 1e-6:
                        gain_flags[ti, fi, a, d] = 1
                        set_identity(corr_mode, gain[ti, fi, a, d])
                    elif gain_flags[ti, fi, a, d] == -1:
                        gain_flags[ti, fi, a, d] = 0

    
def update_gain_flags(base_args, term_args, meta_args, flag_imdry, loop_idx,
                      corr_mode, numbness=1e-6):
    """Update the current state of the gain flags.

    Uses the current (km0) and previous (km1) gains to identify diverging
    soultions. This implements trendy flagging - see overleaf document.
    TODO: Add link.

    Args:
        gain: A (ti, fi, a, d, c) array of gain values.
        km1_gain: A (ti, fi, a, d, c) array of gain values at prev iteration.
        gain_flags: A (ti, fi, a, d) array of flag values.
        km1_abs2_diffs: A (ti, fi, a, d) itemediary array to store the
            previous absolute difference in the gains.
        ab2_diffs_trends: A (ti, fi, a, d) itemediary array to store the
            accumulated trend values of the differences between absolute
            differences of the gains.
        critera: A float value below which a gain is considered converged.
        corr_mode: An int which controls how we handle coreelations.
        iteration: An int containing the iteration number.
    """

    # coerce_literal(update_gain_flags, ['corr_mode'])

    # set_identity = factories.set_identity_factory(corr_mode)

    # def impl(
    #     base_args,
    #     term_args,
    #     meta_args,
    #     flag_imdry,
    #     loop_idx,
    #     corr_mode,
    #     numbness=1e-6
    # ):

    active_term = meta_args.active_term
    max_iter = meta_args.iters
    stop_frac = meta_args.stop_frac
    stop_crit2 = meta_args.stop_crit**2

    gain = base_args.gains[active_term]
    gain_flags = base_args.gain_flags[active_term]

    km1_gain = flag_imdry.km1_gain
    km1_abs2_diffs = flag_imdry.km1_abs2_diffs
    abs2_diffs_trend = flag_imdry.abs2_diffs_trend

    n_tint, n_fint, n_ant, n_dir, n_corr = gain.shape

    n_cnvgd = 0
    n_flagged = 0

    for ti in range(n_tint):
        for fi in range(n_fint):
            for a in range(n_ant):
                for d in range(n_dir):

                    # We can skip points which are already hard flagged.
                    if gain_flags[ti, fi, a, d] == 1:
                        n_flagged += 1
                        continue

                    # Relative difference: (|g_k-1 - g_k|/|g_k-1|)^2.
                    km1_abs2 = 0
                    km0_abs2_diff = 0

                    for c in range(n_corr):

                        km0_g = gain[ti, fi, a, d, c]
                        km1_g = km1_gain[ti, fi, a, d, c]

                        diff = km1_g - km0_g

                        km1_abs2 += km1_g.real**2 + km1_g.imag**2
                        km0_abs2_diff += diff.real**2 + diff.imag**2

                    if km1_abs2 == 0:  # TODO: Precaution, not ideal.
                        gain_flags[ti, fi, a, d] = 1
                        set_identity(corr_mode, gain[ti, fi, a, d])
                        n_flagged += 1
                        continue

                    # Grab absolute difference squared at k-1 and update.
                    km1_abs2_diff = km1_abs2_diffs[ti, fi, a, d]
                    km1_abs2_diffs[ti, fi, a, d] = km0_abs2_diff

                    # We cannot flag on the first few iterations.
                    if loop_idx < 2:
                        continue

                    # Grab trend at k-1 and update.
                    km1_trend = abs2_diffs_trend[ti, fi, a, d]
                    km0_trend = km1_trend + km0_abs2_diff - km1_abs2_diff

                    abs2_diffs_trend[ti, fi, a, d] = km0_trend

                    # This if-else ladder aims to do the following:
                    # 1) If a point has converged, ensure it is unflagged.
                    # 2) If a point is strictly converging, it should have
                    #    no flags. Note we allow a small epsilon of
                    #    "numbness" - this is important if our initial
                    #    estimate is very close to the solution.
                    # 3) If a point strictly diverging, it should be soft
                    #    flagged. If it continues to diverge (twice in a
                    #    row) it should be hard flagged and reset.

                    if km0_abs2_diff/km1_abs2 < stop_crit2:
                        # Unflag points which converged.
                        gain_flags[ti, fi, a, d] = 0
                        n_cnvgd += 1
                    elif km0_trend < km1_trend < numbness:
                        gain_flags[ti, fi, a, d] = 0
                    elif km0_trend > km1_trend > numbness:
                        gain_flags[ti, fi, a, d] = \
                            1 if gain_flags[ti, fi, a, d] else -1

                    if gain_flags[ti, fi, a, d] == 1:
                        n_flagged += 1
                        set_identity(corr_mode, gain[ti, fi, a, d])

    n_solvable = (n_tint*n_fint*n_ant*n_dir - n_flagged)

    if n_solvable:
        conv_perc = n_cnvgd/n_solvable
    else:
        conv_perc = 0.

    # Update the k-1 gain if not converged/on final iteration.
    if (conv_perc < stop_frac) and (loop_idx < max_iter - 1):
        km1_gain[:] = gain

    return conv_perc


def update_param_flags(base_args, term_args, meta_args, identity_params):
    """Propagate gain flags into parameter flags.

    Given the gain flags, parameter flags and the relevant mappings, propagate
    gain flags into parameter flags. NOTE: This may not be the best approach.
    We could flag on the parameters directly but this is difficult due to
    having a variable set of identitiy paramters and no reason to believe that
    the parameters live on the same scale.

    Args:
        gain_flags: A (gti, gfi, a, d) array of gain flags.
        param_flags: A (pti, pfi, a, d) array of paramter flag values.
        t_bin_arr: A (2, n_utime, n_term) array of utime to solint mappings.
        f_map_arr: A (2, n_ufreq, n_term) array of ufreq to solint mappings.
        d_map_arr: A (n_term, n_dir) array of direction mappings.
        """


    active_term = meta_args.active_term

    # NOTE: We don't yet let params and gains have different direction
    # maps but this will eventually be neccessary.
    t_bin_arr = term_args.t_bin_arr[:, :, active_term]
    f_map_arr = base_args.f_map_arr[:, :, active_term]

    gain_flags = base_args.gain_flags[active_term]
    param_flags = term_args.param_flags[active_term]
    params = term_args.params[active_term]

    _, _, n_ant, n_dir = gain_flags.shape

    param_flags[:] = 1

    for gt, pt in zip(t_bin_arr[0], t_bin_arr[1]):
        for gf, pf in zip(f_map_arr[0], f_map_arr[1]):
            for a in range(n_ant):
                for d in range(n_dir):

                    flag = gain_flags[gt, gf, a, d] == 1
                    param_flags[pt, pf, a, d] &= flag

    n_tint, n_fint, n_ant, n_dir = param_flags.shape

    for ti in range(n_tint):
        for fi in range(n_fint):
            for a in range(n_ant):
                for d in range(n_dir):
                    if param_flags[ti, fi, a, d] == 1:
                        params[ti, fi, a, d] = identity_params


def apply_gain_flags(base_args, meta_args):
    """Apply gain_flags to flag_col."""

    active_term = meta_args.active_term

    gain_flags = base_args.gain_flags[active_term]
    flag_col = base_args.flags
    ant1_col = base_args.a1
    ant2_col = base_args.a2

    # Select out just the mappings we need.
    t_map_arr = base_args.t_map_arr[0, :, active_term]
    f_map_arr = base_args.f_map_arr[0, :, active_term]

    n_row, n_chan = flag_col.shape

    for row in range(n_row):
        a1, a2 = ant1_col[row], ant2_col[row]
        t_m = t_map_arr[row]
        for f in range(n_chan):
            f_m = f_map_arr[f]

            # NOTE: We only care about the DI case for now.
            flag_col[row, f] |= gain_flags[t_m, f_m, a1, 0] == 1
            flag_col[row, f] |= gain_flags[t_m, f_m, a2, 0] == 1


    
