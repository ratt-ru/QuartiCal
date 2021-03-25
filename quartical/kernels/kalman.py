# -*- coding: utf-8 -*-
import numpy as np
from numba import literally, generated_jit, types
from numba.extending import register_jitable
from quartical.kernels.generics import (invert_gains,
                                        compute_residual)
from quartical.kernels.complex import jhj_jhr_diag, jhj_jhr_full
from collections import namedtuple


# This can be done without a named tuple now. TODO: Add unpacking to
# constructor.
stat_fields = {"conv_iters": np.int64,
               "conv_perc": np.float64}

term_conv_info = namedtuple("term_conv_info", " ".join(stat_fields.keys()))


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def kalman_solver(model, data, a1, a2, weights, t_map_arr, f_map_arr,
                  d_map_arr, corr_mode, active_term, inverse_gains,
                  gains, flags, row_map, row_weights):

    if not isinstance(corr_mode, types.Literal):
        return lambda model, data, a1, a2, weights, t_map_arr, f_map_arr, \
                   d_map_arr, corr_mode, active_term, inverse_gains, \
                   gains, flags, row_map, row_weights: literally(corr_mode)

    if corr_mode.literal_value == "diag":
        compute_jhj_jhr = jhj_jhr_diag
        compute_update = update_diag
        compute_smoother_update = smoother_update_diag

    else:
        compute_jhj_jhr = jhj_jhr_full
        compute_update = update_full
        compute_smoother_update = smoother_update_full

    def impl(model, data, a1, a2, weights, t_map_arr, f_map_arr,
             d_map_arr, corr_mode, active_term, inverse_gains,
             gains, flags, row_map, row_weights):

        n_tint, n_fint, n_ant, n_dir, n_corr = gains[active_term].shape

        invert_gains(gains, inverse_gains, corr_mode)

        complex_dtype = gains[active_term].dtype
        real_dtype = gains[active_term].real.dtype

        # Initialise arrays for storing intemediary results. TODO: Custom
        # kernels might make the leading time dimension superfluous.

        intemediary_shape = (1, n_fint, n_ant, n_dir, n_corr)

        jhj = np.empty(intemediary_shape, dtype=complex_dtype)
        jhr = np.empty(intemediary_shape, dtype=complex_dtype)
        g_update = np.empty(intemediary_shape, dtype=complex_dtype)
        p_update = np.empty(intemediary_shape, dtype=real_dtype)

        # Q and P have the same dimensions as the gain (or we represent them as
        # though they do).

        Q = np.zeros_like(gains[active_term], dtype=real_dtype)
        P = np.ones_like(gains[active_term], dtype=real_dtype)
        P[0] += Q[0]  # Add Q to P for for first iteration.

        n_epoch = 2

        for _ in range(n_epoch):

            # ----------------------------FILTER-------------------------------

            for i in range(n_tint):

                # TODO: This is a dirty hack - implement special jhj and jhr
                # code. This DOES NOT WORK for BDA data, and will need to be
                # fixed.
                sel = np.where(t_map_arr[:, active_term] == i)[0]

                # Compute the (unweighted) residual values at the current time.
                residual = \
                    compute_residual(data[sel, :, :],
                                     model[sel, :, :, :],
                                     [gains[active_term][i:i+1, :, :, :, :]],
                                     a1[sel],
                                     a2[sel],
                                     np.zeros_like(t_map_arr[sel]),
                                     f_map_arr,
                                     d_map_arr,
                                     row_map,
                                     row_weights,
                                     corr_mode)

                # Compute the entries of JHWr and the diagonal entries of JHWJ.
                compute_jhj_jhr(jhj,
                                jhr,
                                model[sel, :, :, :],
                                [gains[active_term][i:i+1, :, :, :, :]],
                                inverse_gains,
                                residual,
                                a1[sel],
                                a2[sel],
                                weights[sel, :, :],
                                np.zeros_like(t_map_arr[sel]),
                                f_map_arr,
                                d_map_arr,
                                row_map,
                                row_weights,
                                active_term,
                                corr_mode)

                # Compute the updates - this is where the inverse happens.
                compute_update(g_update,
                               p_update,
                               jhj,
                               jhr,
                               P[i:i+1, :, :, :, :],
                               corr_mode)

                # Update the gains and covariances (P).
                gains[active_term][i:i+1, :, :, :, :] += 0.5*g_update
                P[i:i+1, :, :, :, :] -= 0.5*p_update

                # On all except the last iteration, we need to add Q to P and
                # set the gain at t+1 equal to the gain at the current t.
                if i < n_tint - 1:
                    gains[active_term][i+1, :, :, :, :] = \
                        gains[active_term][i, :, :, :, :]
                    P[i+1, :, :, :, :] = P[i, :, :, :, :] + Q[i, :, :, :, :]

            # ---------------------------SMOOTHER------------------------------

            # Set up the arrays to store the smoothed versions of G and P. Gs
            # are needed to update Q.

            smooth_gains = np.zeros_like(gains[active_term])
            smooth_P = np.zeros_like(P)
            Gs = np.zeros_like(P)

            # The filter and smoother are set to agree at the final time step.
            smooth_gains[-1] = gains[active_term][-1]
            smooth_P[-1] = P[-1]

            for i in range(n_tint-2, -1, -1):

                # TODO: This shouldn't be returning G - this was just PoC.
                Gs[i:i+1] = \
                    compute_smoother_update(
                        g_update,
                        p_update,
                        gains[active_term][i:i+1, :, :, :, :],
                        smooth_gains[i+1:i+2, :, :, :, :],
                        P[i:i+1, :, :, :, :],
                        smooth_P[i+1:i+2, :, :, :, :],
                        Q[i:i+1, :, :, :, :],
                        corr_mode)

                smooth_gains[i:i+1, :, :, :, :] = \
                    gains[active_term][i:i+1] + 0.5*g_update
                smooth_P[i:i+1, :, :, :, :] = P[i:i+1] + 0.5*p_update

            # TODO: Q really shouldn't have a time axis - I was just being
            # lazy.
            Q[:] = update_q(smooth_gains, smooth_P, Gs)

            # Add Q here for the first step of the next filter run.
            g0_diff = smooth_gains[0] - gains[active_term][0]
            P[0] = (smooth_P[0] + g0_diff * g0_diff.conjugate()).real + Q[0]

            # Gains are set to the last set of smoothed gains.
            gains[active_term][:] = smooth_gains[:]

        return term_conv_info(0, 0)  # This has no meaning for this solver.

    return impl


@register_jitable
def update_diag(g_update, p_update, jhj, jhr, P, corr_mode):

    n_tint, n_fint, n_ant, n_dir, n_corr = jhj.shape

    for t in range(n_tint):
        for f in range(n_fint):
            for a in range(n_ant):
                for d in range(n_dir):

                    jhj00 = jhj[t, f, a, d, 0]
                    jhj11 = jhj[t, f, a, d, 1]

                    P00 = P[t, f, a, d, 0]
                    P11 = P[t, f, a, d, 1]

                    det = (jhj00*jhj11)

                    if det.real < 1e-6:
                        jhjinv00 = 0
                        jhjinv11 = 0

                    else:
                        P00inv = 1/P00
                        P11inv = 1/P11

                        jhjinv00 = 1/(P00inv + jhj00)
                        jhjinv11 = 1/(P11inv + jhj11)

                    jhr00 = jhr[t, f, a, d, 0]
                    jhr11 = jhr[t, f, a, d, 1]

                    # Component of Woodbury Matrix Identity.
                    wb00 = P00*(1-jhj00*jhjinv00)
                    wb11 = P11*(1-jhj11*jhjinv11)

                    g_update[t, f, a, d, 0] = wb00*jhr00
                    g_update[t, f, a, d, 1] = wb11*jhr11

                    p_update[t, f, a, d, 0] = (wb00*jhj00*P00).real
                    p_update[t, f, a, d, 1] = (wb11*jhj11*P11).real

    return


@register_jitable
def update_full(g_update, p_update, jhj, jhr, P, corr_mode):

    n_tint, n_fint, n_ant, n_dir, n_corr = jhj.shape

    for t in range(n_tint):
        for f in range(n_fint):
            for a in range(n_ant):
                for d in range(n_dir):

                    P0 = P[t, f, a, d, 0]
                    P1 = P[t, f, a, d, 1]
                    P2 = P[t, f, a, d, 2]
                    P3 = P[t, f, a, d, 3]

                    Pinv0 = 1/P0
                    Pinv1 = 1/P1
                    Pinv2 = 1/P2
                    Pinv3 = 1/P3

                    WJHJ = np.kron(jhj[t, f, a, d, :].reshape(2, 2).T,
                                   np.eye(2))

                    PinvWJHJ = WJHJ.copy()

                    PinvWJHJ[0, 0] += Pinv0
                    PinvWJHJ[1, 1] += Pinv1
                    PinvWJHJ[2, 2] += Pinv2
                    PinvWJHJ[3, 3] += Pinv3

                    wb = np.eye(4) - WJHJ.dot(np.linalg.inv(PinvWJHJ))

                    wb[0, :] *= P0
                    wb[1, :] *= P1
                    wb[2, :] *= P2
                    wb[3, :] *= P3

                    g_update[t, f, a, d, :] = wb.dot(jhr[t, f, a, d, :])

                    tmp = np.eye(4).astype(np.complex128)

                    tmp[0, 0] = P0
                    tmp[1, 1] = P1
                    tmp[2, 2] = P2
                    tmp[3, 3] = P3

                    WJHJP = WJHJ.dot(tmp)

                    pup = wb.dot(WJHJP)

                    p_update[t, f, a, d, 0] = pup[0, 0].real
                    p_update[t, f, a, d, 1] = pup[1, 1].real
                    p_update[t, f, a, d, 2] = pup[2, 2].real
                    p_update[t, f, a, d, 3] = pup[3, 3].real

    return


@register_jitable
def smoother_update_diag(g_update, p_update, gains, smooth_gains, P, smooth_P,
                         Q, corr_mode):

    n_tint, n_fint, n_ant, n_dir, n_corr = g_update.shape

    Gs = np.zeros(g_update.shape, dtype=p_update.dtype)

    for t in range(n_tint):
        for f in range(n_fint):
            for a in range(n_ant):
                for d in range(n_dir):

                    P00 = P[t, f, a, d, 0]
                    P11 = P[t, f, a, d, 1]

                    Q00 = Q[t, f, a, d, 0]
                    Q11 = Q[t, f, a, d, 1]

                    Pp00 = P00 + Q00
                    Pp11 = P11 + Q11

                    G00 = P00/Pp00
                    G11 = P11/Pp11

                    g00 = gains[t, f, a, d, 0]
                    g11 = gains[t, f, a, d, 1]

                    gs00 = smooth_gains[t, f, a, d, 0]
                    gs11 = smooth_gains[t, f, a, d, 1]

                    g_update[t, f, a, d, 0] = G00*(gs00 - g00)
                    g_update[t, f, a, d, 1] = G11*(gs11 - g11)

                    Ps00 = smooth_P[t, f, a, d, 0]
                    Ps11 = smooth_P[t, f, a, d, 1]

                    p_update[t, f, a, d, 0] = G00*(Ps00 - Pp00)*G00
                    p_update[t, f, a, d, 1] = G11*(Ps11 - Pp11)*G11

                    Gs[t, f, a, d, 0] = G00
                    Gs[t, f, a, d, 1] = G11

    return Gs


@register_jitable
def smoother_update_full(g_update, p_update, gains, smooth_gains, P, smooth_P,
                         Q, corr_mode):

    n_tint, n_fint, n_ant, n_dir, n_corr = g_update.shape

    Gs = np.zeros(g_update.shape, dtype=p_update.dtype)

    for t in range(n_tint):
        for f in range(n_fint):
            for a in range(n_ant):
                for d in range(n_dir):

                    P0 = P[t, f, a, d, 0]
                    P1 = P[t, f, a, d, 1]
                    P2 = P[t, f, a, d, 2]
                    P3 = P[t, f, a, d, 3]

                    Q0 = Q[t, f, a, d, 0]
                    Q1 = Q[t, f, a, d, 1]
                    Q2 = Q[t, f, a, d, 2]
                    Q3 = Q[t, f, a, d, 3]

                    Pp0 = P0 + Q0
                    Pp1 = P1 + Q1
                    Pp2 = P2 + Q2
                    Pp3 = P3 + Q3

                    G0 = P0/Pp0
                    G1 = P1/Pp1
                    G2 = P2/Pp2
                    G3 = P3/Pp3

                    g0 = gains[t, f, a, d, 0]
                    g1 = gains[t, f, a, d, 1]
                    g2 = gains[t, f, a, d, 2]
                    g3 = gains[t, f, a, d, 3]

                    gs0 = smooth_gains[t, f, a, d, 0]
                    gs1 = smooth_gains[t, f, a, d, 1]
                    gs2 = smooth_gains[t, f, a, d, 2]
                    gs3 = smooth_gains[t, f, a, d, 3]

                    g_update[t, f, a, d, 0] = G0*(gs0 - g0)
                    g_update[t, f, a, d, 1] = G1*(gs1 - g1)
                    g_update[t, f, a, d, 2] = G2*(gs2 - g2)
                    g_update[t, f, a, d, 3] = G3*(gs3 - g3)

                    Ps0 = smooth_P[t, f, a, d, 0]
                    Ps1 = smooth_P[t, f, a, d, 1]
                    Ps2 = smooth_P[t, f, a, d, 2]
                    Ps3 = smooth_P[t, f, a, d, 3]

                    p_update[t, f, a, d, 0] = G0*(Ps0 - Pp0)*G0
                    p_update[t, f, a, d, 1] = G1*(Ps1 - Pp1)*G1
                    p_update[t, f, a, d, 2] = G2*(Ps2 - Pp2)*G2
                    p_update[t, f, a, d, 3] = G3*(Ps3 - Pp3)*G3

                    Gs[t, f, a, d, 0] = G0
                    Gs[t, f, a, d, 1] = G1
                    Gs[t, f, a, d, 2] = G2
                    Gs[t, f, a, d, 3] = G3

    return Gs


@register_jitable
def update_q(ms, Ps, G):
    """
    Here we compute the optimal prior parameters resulting from estimation for
    a linear state-space model as defined by eq. 12.44-12.47 of the filtering
    and smoothing textbook. Note that we can only compute parameters of the
    prior since our H matrix implicitly depends on time.
    """

    n_tint, n_fint, n_ant, n_dir, n_corr = ms.shape

    Sigma = np.zeros((n_fint, n_ant, n_dir, n_corr), dtype=np.complex128)
    Phi = np.zeros((n_fint, n_ant, n_dir, n_corr), dtype=np.complex128)
    C = np.zeros((n_fint, n_ant, n_dir, n_corr), dtype=np.complex128)

    # This is doing unecessary referencing at the moment. TODO: Properly
    # unpack the various elements onto variables. Will also need a
    # non-diagonal version of this eventually.

    for t in range(n_tint):
        for f in range(n_fint):
            for a in range(n_ant):
                for d in range(n_dir):
                    Sigma[f, a, d, 0] += \
                        Ps[t, f, a, d, 0] + np.abs(ms[t, f, a, d, 0])**2
                    Sigma[f, a, d, 1] += \
                        Ps[t, f, a, d, 1] + np.abs(ms[t, f, a, d, 1])**2
                    # assuming the equations are not defined when t=0 for
                    # these two
                    if t > 0:
                        Phi[f, a, d, 0] += (Ps[t-1, f, a, d, 0] +
                                            np.abs(ms[t-1, f, a, d, 0])**2)
                        Phi[f, a, d, 1] += (Ps[t-1, f, a, d, 1] +
                                            np.abs(ms[t-1, f, a, d, 1])**2)
                        C[f, a, d, 0] += \
                            Ps[t, f, a, d, 0] * G[t-1, f, a, d, 0] + \
                            ms[t, f, a, d, 0] * ms[t-1, f, a, d, 0].conjugate()
                        C[f, a, d, 1] += \
                            Ps[t, f, a, d, 1] * G[t-1, f, a, d, 1] + \
                            ms[t, f, a, d, 1] * ms[t-1, f, a, d, 1].conjugate()
    # normalise
    Sigma /= n_tint
    Phi /= n_tint
    C /= n_tint

    # Return real diagonal part
    return (Sigma - C - C.conjugate() + Phi).real
