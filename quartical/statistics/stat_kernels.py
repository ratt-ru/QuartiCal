import numpy as np
from numba import njit, prange
from numba.extending import overload
from quartical.utils.numba import (coerce_literal,
                                   JIT_OPTIONS,
                                   PARALLEL_JIT_OPTIONS)
from quartical.gains.general.convenience import get_dims, get_row
import quartical.gains.general.factories as factories


@njit(**JIT_OPTIONS)
def compute_mean_presolve_chisq(
    data,
    model,
    weights,
    flags,
    row_map,
    row_weights,
    corr_mode
):
    return compute_mean_presolve_chisq_impl(
        data,
        model,
        weights,
        flags,
        row_map,
        row_weights,
        corr_mode
    )


def compute_mean_presolve_chisq_impl(
    data,
    model,
    weights,
    flags,
    row_map,
    row_weights,
    corr_mode
):
    return NotImplementedError


@overload(compute_mean_presolve_chisq_impl, jit_options=PARALLEL_JIT_OPTIONS)
def nb_compute_mean_presolve_chisq_impl(
    data,
    model,
    weights,
    flags,
    row_map,
    row_weights,
    corr_mode
):

    coerce_literal(nb_compute_mean_presolve_chisq_impl, ["corr_mode"])

    imul_rweight = factories.imul_rweight_factory(corr_mode, row_weights)
    isub = factories.isub_factory(corr_mode)
    iunpack = factories.iunpack_factory(corr_mode)
    valloc = factories.valloc_factory(corr_mode)

    def impl(
        data,
        model,
        weights,
        flags,
        row_map,
        row_weights,
        corr_mode
    ):

        n_rows, n_chan, n_dir, n_corr = get_dims(model, row_map)

        chisq = 0
        counts = 0

        for row_ind in prange(n_rows):

            row = get_row(row_ind, row_map)
            r = valloc(np.complex128)  # Hold data.
            v = valloc(np.complex128)  # Hold GMGH.

            for f in range(n_chan):

                if flags[row, f]:
                    continue

                iunpack(r, data[row, f])
                imul_rweight(r, r, row_weights, row_ind)
                m = model[row, f]
                w = weights[row, f]

                for d in range(n_dir):

                    iunpack(v, m[d])
                    imul_rweight(v, v, row_weights, row_ind)
                    isub(r, v)

                for c in range(n_corr):
                    chisq += (r[c].conjugate() * w[c] * r[c]).real
                    counts += 1

        if counts:
            chisq /= counts
        else:
            chisq = np.nan

        return np.array([[chisq]], dtype=np.float64)

    return impl


@njit(**JIT_OPTIONS)
def compute_mean_postsolve_chisq(
    data,
    model,
    weights,
    flags,
    a1,
    a2,
    row_map,
    row_weights,
    gains,
    time_maps,
    freq_maps,
    dir_maps,
    corr_mode
):
    return compute_mean_postsolve_chisq_impl(
        data,
        model,
        weights,
        flags,
        a1,
        a2,
        row_map,
        row_weights,
        gains,
        time_maps,
        freq_maps,
        dir_maps,
        corr_mode
    )


def compute_mean_postsolve_chisq_impl(
    data,
    model,
    weights,
    flags,
    a1,
    a2,
    row_map,
    row_weights,
    gains,
    time_maps,
    freq_maps,
    dir_maps,
    corr_mode
):
    return NotImplementedError


@overload(compute_mean_postsolve_chisq_impl, jit_options=PARALLEL_JIT_OPTIONS)
def nb_compute_mean_postsolve_chisq_impl(
    data,
    model,
    weights,
    flags,
    a1,
    a2,
    row_map,
    row_weights,
    gains,
    time_maps,
    freq_maps,
    dir_maps,
    corr_mode
):

    coerce_literal(nb_compute_mean_postsolve_chisq_impl, ["corr_mode"])

    imul_rweight = factories.imul_rweight_factory(corr_mode, row_weights)
    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    v1_imul_v2ct = factories.v1_imul_v2ct_factory(corr_mode)
    isub = factories.isub_factory(corr_mode)
    iunpack = factories.iunpack_factory(corr_mode)
    valloc = factories.valloc_factory(corr_mode)

    def impl(
        data,
        model,
        weights,
        flags,
        a1,
        a2,
        row_map,
        row_weights,
        gains,
        time_maps,
        freq_maps,
        dir_maps,
        corr_mode
    ):

        n_rows, n_chan, n_dir, n_corr = get_dims(model, row_map)
        n_gains = len(gains)

        chisq = 0
        counts = 0

        for row_ind in prange(n_rows):

            row = get_row(row_ind, row_map)
            a1_m, a2_m = a1[row], a2[row]
            r = valloc(np.complex128)  # Hold data.
            v = valloc(np.complex128)  # Hold GMGH.

            for f in range(n_chan):

                if flags[row, f]:
                    continue

                iunpack(r, data[row, f])
                imul_rweight(r, r, row_weights, row_ind)
                m = model[row, f]
                w = weights[row, f]

                for d in range(n_dir):

                    iunpack(v, m[d])

                    for g in range(n_gains - 1, -1, -1):

                        t_m = time_maps[g][row_ind]
                        f_m = freq_maps[g][f]
                        d_m = dir_maps[g][d]  # Broadcast dir.

                        gain = gains[g][t_m, f_m]
                        gain_p = gain[a1_m, d_m]
                        gain_q = gain[a2_m, d_m]

                        v1_imul_v2(gain_p, v, v)
                        v1_imul_v2ct(v, gain_q, v)

                    imul_rweight(v, v, row_weights, row_ind)
                    isub(r, v)

                for c in range(n_corr):
                    chisq += (r[c].conjugate() * w[c] * r[c]).real
                    counts += 1

        if counts:
            chisq /= counts
        else:
            chisq = np.nan

        return np.array([[chisq]], dtype=np.float64)

    return impl
