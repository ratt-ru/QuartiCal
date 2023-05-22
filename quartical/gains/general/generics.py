# -*- coding: utf-8 -*-
import numpy as np
from numba import prange, njit
from numba.typed import List
from collections import namedtuple
from numba.extending import overload
from quartical.utils.numba import JIT_OPTIONS, coerce_literal
import quartical.gains.general.factories as factories
from quartical.gains.general.convenience import get_dims, get_row


native_intermediaries = namedtuple(
    "native_intermediaries",
    (
        "jhj",
        "jhr",
        "update"
    )
)

solver_intermediaries = namedtuple(
    "solver_intermediaries",
    (
        "jhj",
        "jhr",
        "update"
    )
)

upsampled_itermediaries = namedtuple(
    "upsampled_intermediaries",
    (
        "jhj",
        "jhr"
    )
)

resample_outputs = namedtuple(
    "resample_outputs",
    (
        "active",
        "upsample_shape",
        "upsample_t_map",
        "downsample_t_map"
    )
)


@njit(**JIT_OPTIONS)
def invert_gains(gain_list, inverse_gains, corr_mode):
    return invert_gains_impl(gain_list, inverse_gains, corr_mode)


def invert_gains_impl(gain_list, inverse_gains, corr_mode):
    raise NotImplementedError


@overload(invert_gains_impl, jit_options=JIT_OPTIONS)
def nb_invert_gains_impl(gain_list, inverse_gains, corr_mode):

    coerce_literal(nb_invert_gains_impl, ["corr_mode"])

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


@njit(**JIT_OPTIONS)
def compute_residual(
    data,
    model,
    gain_tuple,
    a1,
    a2,
    time_maps,
    freq_maps,
    dir_maps,
    row_map,
    row_weights,
    corr_mode,
    sub_dirs=None
):
    return compute_residual_impl(
        data,
        model,
        gain_tuple,
        a1,
        a2,
        time_maps,
        freq_maps,
        dir_maps,
        row_map,
        row_weights,
        corr_mode,
        sub_dirs
    )


def compute_residual_impl(
    data,
    model,
    gain_tuple,
    a1,
    a2,
    time_maps,
    freq_maps,
    dir_maps,
    row_map,
    row_weights,
    corr_mode,
    sub_dirs=None
):
    raise NotImplementedError


@overload(compute_residual_impl, jit_options=JIT_OPTIONS)
def nb_compute_residual_impl(
    data,
    model,
    gain_tuple,
    a1,
    a2,
    time_maps,
    freq_maps,
    dir_maps,
    row_map,
    row_weights,
    corr_mode,
    sub_dirs=None
):

    coerce_literal(nb_compute_residual_impl, ["corr_mode"])

    imul_rweight = factories.imul_rweight_factory(corr_mode, row_weights)
    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    v1_imul_v2ct = factories.v1_imul_v2ct_factory(corr_mode)
    isub = factories.isub_factory(corr_mode)
    iunpack = factories.iunpack_factory(corr_mode)
    valloc = factories.valloc_factory(corr_mode)

    def impl(
        data,
        model,
        gain_tuple,
        a1,
        a2,
        time_maps,
        freq_maps,
        dir_maps,
        row_map,
        row_weights,
        corr_mode,
        sub_dirs=None
    ):

        residual = data.copy()

        n_rows, n_chan, n_dir, _ = get_dims(model, row_map)
        n_gains = len(gain_tuple)

        if sub_dirs is None:
            dir_loop = np.arange(n_dir)
        else:
            dir_loop = np.array(sub_dirs)

        for row_ind in prange(n_rows):

            row = get_row(row_ind, row_map)
            a1_m, a2_m = a1[row], a2[row]
            v = valloc(np.complex128)  # Hold GMGH.

            for f in range(n_chan):

                r = residual[row, f]
                m = model[row, f]

                for d in dir_loop:

                    iunpack(v, m[d])

                    for g in range(n_gains - 1, -1, -1):

                        t_m = time_maps[g][row_ind]
                        f_m = freq_maps[g][f]
                        d_m = dir_maps[g][d]  # Broadcast dir.

                        gain = gain_tuple[g][t_m, f_m]
                        gain_p = gain[a1_m, d_m]
                        gain_q = gain[a2_m, d_m]

                        v1_imul_v2(gain_p, v, v)
                        v1_imul_v2ct(v, gain_q, v)

                    imul_rweight(v, v, row_weights, row_ind)
                    isub(r, v)

        return residual

    return impl


@njit(**JIT_OPTIONS)
def compute_corrected_residual(
    residual,
    gain_tuple,
    a1,
    a2,
    time_maps,
    freq_maps,
    dir_maps,
    row_map,
    row_weights,
    corr_mode
):
    return compute_corrected_residual_impl(
        residual,
        gain_tuple,
        a1,
        a2,
        time_maps,
        freq_maps,
        dir_maps,
        row_map,
        row_weights,
        corr_mode
    )


def compute_corrected_residual_impl(
    residual,
    gain_tuple,
    a1,
    a2,
    time_maps,
    freq_maps,
    dir_maps,
    row_map,
    row_weights,
    corr_mode
):
    raise NotImplementedError


@overload(compute_corrected_residual_impl, jit_options=JIT_OPTIONS)
def nb_compute_corrected_residual_impl(
    residual,
    gain_tuple,
    a1,
    a2,
    time_maps,
    freq_maps,
    dir_maps,
    row_map,
    row_weights,
    corr_mode
):

    coerce_literal(nb_compute_corrected_residual_impl, ["corr_mode"])

    imul_rweight = factories.imul_rweight_factory(corr_mode, row_weights)
    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    v1_imul_v2ct = factories.v1_imul_v2ct_factory(corr_mode)
    iadd = factories.iadd_factory(corr_mode)
    iunpack = factories.iunpack_factory(corr_mode)
    valloc = factories.valloc_factory(corr_mode)

    def impl(
        residual,
        gain_tuple,
        a1,
        a2,
        time_maps,
        freq_maps,
        dir_maps,
        row_map,
        row_weights,
        corr_mode
    ):

        corrected_residual = np.zeros_like(residual)

        inverse_gain_list = List()

        for gain_term in gain_tuple:
            inverse_gain_list.append(np.empty_like(gain_term))

        invert_gains(gain_tuple, inverse_gain_list, corr_mode)

        n_rows, n_chan, _ = get_dims(residual, row_map)
        n_gains = len(gain_tuple)

        r = valloc(np.complex128)

        for row_ind in range(n_rows):

            row = get_row(row_ind, row_map)
            a1_m, a2_m = a1[row], a2[row]

            for f in range(n_chan):

                iunpack(r, residual[row, f])
                cr = corrected_residual[row, f]

                for g in range(n_gains):

                    t_m = time_maps[g][row_ind]
                    f_m = freq_maps[g][f]

                    igain = inverse_gain_list[g][t_m, f_m]
                    igain_p = igain[a1_m, 0]  # Only correct in direction 0.
                    igain_q = igain[a2_m, 0]

                    v1_imul_v2(igain_p, r, r)
                    v1_imul_v2ct(r, igain_q, r)

                imul_rweight(r, r, row_weights, row_ind)
                iadd(cr, r)

        return corrected_residual

    return impl


@njit(**JIT_OPTIONS)
def compute_corrected_weights(
    weights,
    gain_tuple,
    a1,
    a2,
    time_maps,
    freq_maps,
    dir_maps,
    row_map,
    row_weights,
    corr_mode
):
    return compute_corrected_weights_impl(
        weights,
        gain_tuple,
        a1,
        a2,
        time_maps,
        freq_maps,
        dir_maps,
        row_map,
        row_weights,
        corr_mode
    )


def compute_corrected_weights_impl(
    weights,
    gain_tuple,
    a1,
    a2,
    time_maps,
    freq_maps,
    dir_maps,
    row_map,
    row_weights,
    corr_mode
):
    raise NotImplementedError


@overload(compute_corrected_weights_impl, jit_options=JIT_OPTIONS)
def nb_compute_corrected_weights_impl(
    weights,
    gain_tuple,
    a1,
    a2,
    time_maps,
    freq_maps,
    dir_maps,
    row_map,
    row_weights,
    corr_mode
):

    coerce_literal(nb_compute_corrected_weights_impl, ["corr_mode"])

    imul_rweight = factories.imul_rweight_factory(corr_mode, row_weights)
    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    v1ct_imul_v2 = factories.v1ct_imul_v2_factory(corr_mode)
    iadd = factories.iadd_factory(corr_mode)
    iunpack = factories.iunpack_factory(corr_mode)
    iunpackct = factories.iunpackct_factory(corr_mode)
    valloc = factories.valloc_factory(corr_mode)
    corrected_weights_buffer = corrected_weights_buffer_factory(corr_mode)
    corrected_weights_inner = corrected_weights_inner_factory(corr_mode)

    def impl(
        weights,
        gain_tuple,
        a1,
        a2,
        time_maps,
        freq_maps,
        dir_maps,
        row_map,
        row_weights,
        corr_mode
    ):

        corrected_weights = np.zeros_like(weights)

        n_rows, n_chan, _ = get_dims(weights, row_map)
        n_gains = len(gain_tuple)

        w = valloc(np.float64)

        gp = valloc(np.complex128)
        gq = valloc(np.complex128)

        buffer = corrected_weights_buffer()

        for row_ind in range(n_rows):

            row = get_row(row_ind, row_map)
            a1_m, a2_m = a1[row], a2[row]

            for f in range(n_chan):

                iunpack(w, weights[row, f])
                cw = corrected_weights[row, f]

                for g in range(n_gains):

                    t_m = time_maps[g][row_ind]
                    f_m = freq_maps[g][f]
                    gain = gain_tuple[g][t_m, f_m]

                    if g == 0:
                        iunpack(gp, gain[a1_m, 0])
                        iunpackct(gq, gain[a2_m, 0])
                    else:
                        v1_imul_v2(gp, gain[a1_m, 0], gp)
                        v1ct_imul_v2(gain[a2_m, 0], gq, gq)

                corrected_weights_inner(w, gp, gq, buffer)

                imul_rweight(w, w, row_weights, row_ind)
                iadd(cw, w)

        return corrected_weights

    return impl


def corrected_weights_inner_factory(corr_mode):

    a_kron_bt = factories.a_kron_bt_factory(corr_mode)
    unpack = factories.unpack_factory(corr_mode)

    if corr_mode.literal_value == 4:
        def impl(weights, gain_p, gain_q, buffer):

            a_kron_bt(gain_p, gain_q, buffer)

            w_0, w_1, w_2, w_3 = unpack(weights)

            for i in range(4):

                k_0, k_1, k_2, k_3 = unpack(buffer[:, i])

                weights[i] = (k_0.conjugate() * w_0 * k_0).real + \
                             (k_1.conjugate() * w_1 * k_1).real + \
                             (k_2.conjugate() * w_2 * k_2).real + \
                             (k_3.conjugate() * w_3 * k_3).real

    elif corr_mode.literal_value == 2:
        def impl(weights, gain_p, gain_q, buffer):
            gp_00, gp_11 = unpack(gain_p)
            gq_00, gq_11 = unpack(gain_q)
            w_00, w_11 = unpack(weights)

            gpgq_00 = gp_00*gq_00
            gpgq_11 = gp_11*gq_11

            weights[0] = (gpgq_00.conjugate() * w_00 * gpgq_00).real
            weights[1] = (gpgq_11.conjugate() * w_11 * gpgq_11).real
    elif corr_mode.literal_value == 1:
        def impl(weights, gain_p, gain_q, buffer):
            gp_00 = unpack(gain_p)
            gq_00 = unpack(gain_q)
            w_00 = unpack(weights)

            gpgq_00 = gp_00*gq_00

            weights[0] = (gpgq_00.conjugate() * w_00 * gpgq_00).real
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def corrected_weights_buffer_factory(corr_mode):

    if corr_mode.literal_value == 4:
        def impl():
            return np.zeros((4, 4), dtype=np.complex128)
    elif corr_mode.literal_value == 2:
        def impl():
            return np.zeros((2,), dtype=np.complex128)
    elif corr_mode.literal_value == 1:
        def impl():
            return np.zeros((1,), dtype=np.complex128)
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


@njit(**JIT_OPTIONS)
def compute_convergence(gain, last_gain, criteria):

    n_tint, n_fint, n_ant, n_dir, n_corr = gain.shape

    n_cnvgd = 0

    for ti in range(n_tint):
        for fi in range(n_fint):
            for a in range(n_ant):
                for d in range(n_dir):

                    gain_abs2 = 0
                    gain_diff_abs2 = 0

                    for c in range(n_corr):

                        gsel = gain[ti, fi, a, d, c]
                        lgsel = last_gain[ti, fi, a, d, c]

                        diff = lgsel - gsel

                        gain_abs2 += gsel.real**2 + gsel.imag**2
                        gain_diff_abs2 += diff.real**2 + diff.imag**2

                    if gain_abs2 == 0:
                        n_cnvgd += 1
                    else:
                        n_cnvgd += gain_diff_abs2/gain_abs2 < criteria**2

    return n_cnvgd/(n_tint*n_fint*n_ant*n_dir)


@njit(**JIT_OPTIONS)
def combine_gains(
    gains,
    time_bins,
    freq_maps,
    dir_maps,
    net_shape,
    corr_mode
):
    return combine_gains_impl(
        gains,
        time_bins,
        freq_maps,
        dir_maps,
        net_shape,
        corr_mode
    )


def combine_gains_impl(
    gains,
    time_bins,
    freq_maps,
    dir_maps,
    net_shape,
    corr_mode
):
    raise NotImplementedError


@overload(combine_gains_impl, jit_options=JIT_OPTIONS)
def nb_combine_gains_impl(
    gains,
    time_bins,
    freq_maps,
    dir_maps,
    net_shape,
    corr_mode
):

    coerce_literal(nb_combine_gains_impl, ["corr_mode"])

    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)

    def impl(
        gains,
        time_bins,
        freq_maps,
        dir_maps,
        net_shape,
        corr_mode
    ):

        n_term = len(gains)
        n_time = time_bins[0].size  # Same for all terms.
        n_freq = freq_maps[0].size  # Same for all terms.
        _, _, n_ant, n_dir, n_corr = net_shape

        shape = (n_time, n_freq, n_ant, n_dir, n_corr)

        net_gains = np.zeros(shape, dtype=np.complex128)
        net_gains[..., 0] = 1
        net_gains[..., -1] = 1

        for t in range(n_time):
            for f in range(n_freq):
                for a in range(n_ant):
                    for d in range(n_dir):
                        for gi in range(n_term):

                            tm = time_bins[gi][t]
                            fm = freq_maps[gi][f]
                            dm = dir_maps[gi][d]

                            v1_imul_v2(
                                net_gains[t, f, a, d],
                                gains[gi][tm, fm, a, dm],
                                net_gains[t, f, a, d]
                            )

        return net_gains

    return impl


@njit(**JIT_OPTIONS)
def combine_flags(
    flags,
    time_bins,
    freq_maps,
    dir_maps,
    net_shape
):

    n_term = len(flags)
    n_time = time_bins[0].size  # Same for all terms.
    n_freq = freq_maps[0].size  # Same for all terms.
    _, _, n_ant, n_dir, _ = net_shape

    shape = (n_time, n_freq, n_ant, n_dir)

    net_flags = np.zeros(shape, dtype=np.int8)

    for t in range(n_time):
        for f in range(n_freq):
            for a in range(n_ant):
                for d in range(n_dir):
                    for gi in range(n_term):

                        tm = time_bins[gi][t]
                        fm = freq_maps[gi][f]
                        dm = dir_maps[gi][d]

                        net_flags[t, f, a, d] |= flags[gi][tm, fm, a, dm]

    return net_flags


@njit(**JIT_OPTIONS)
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


@njit(**JIT_OPTIONS)
def resample_solints(native_map, native_shape, n_thread):

    n_tint, n_fint = native_shape[:2]
    n_int = n_tint * n_fint

    if n_int < n_thread:  # TODO: Maybe put some integer factor here?

        active = True

        remap_factor = np.ceil(n_thread/n_int)

        target_n_tint = int(n_tint * remap_factor)

        upsample_map = np.empty_like(native_map)
        downsample_map = np.empty(target_n_tint, dtype=np.int32)
        remap_id = 0
        offset = 0

        for i in range(n_tint):

            sel = np.where(native_map == i)

            sel_n_row = sel[0].size

            upsample_n_row = int(np.ceil(sel_n_row/remap_factor))

            for start in range(offset, offset + sel_n_row, upsample_n_row):

                stop = min(start + upsample_n_row, offset + sel_n_row)

                upsample_map[start:stop] = remap_id
                downsample_map[remap_id] = i

                remap_id += 1

            offset += sel_n_row

        upsample_shape = (target_n_tint,) + native_shape[1:]

    else:

        active = False
        upsample_map = native_map
        downsample_map = np.empty(0, dtype=np.int32)
        upsample_shape = native_shape

    return resample_outputs(
        active, upsample_shape, upsample_map, downsample_map
    )


@njit(**JIT_OPTIONS)
def downsample_jhj_jhr(upsampled_imdry, downsample_t_map):

    jhj = upsampled_imdry.jhj
    jhr = upsampled_imdry.jhr

    n_tint, n_fint, n_ant, n_dir = jhj.shape[:4]

    prev_out_ti = -1

    for ti in range(n_tint):

        out_ti = downsample_t_map[ti]

        for fi in range(n_fint):
            for a in range(n_ant):
                for d in range(n_dir):

                    if prev_out_ti != out_ti:
                        jhj[out_ti, fi, a, d] = jhj[ti, fi, a, d]
                        jhr[out_ti, fi, a, d] = jhr[ti, fi, a, d]
                    else:
                        jhj[out_ti, fi, a, d] += jhj[ti, fi, a, d]
                        jhr[out_ti, fi, a, d] += jhr[ti, fi, a, d]

        prev_out_ti = out_ti
