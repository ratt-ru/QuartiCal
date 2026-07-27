# -*- coding: utf-8 -*-
"""Forward-model null-V crosshand-phase solver.

This kernel solves for a single array-wide crosshand phase under the prior
that the calibrator has no circular polarisation (V = 0), WITHOUT requiring a
polarised model - but, unlike its data-correcting predecessor, it does so in
the same forward-model form as every other QuartiCal term: gains are only
ever applied to a model, never to the data.

**The model completion.** A fixed unpolarised model cannot constrain the
crosshand phase (the corrupted model's cross-hands are identically zero, so
the residual has no phase derivative). The solver therefore augments the
model with the calibrator's unknown linear polarisation as a nuisance
parameter: per solution interval, the sky cross-hands are taken to be
``XY = z`` and ``YX = conj(z)`` for a complex nuisance ``z`` - exactly the
V = 0 manifold, since ``z = U * exp(i * dphi)`` is the (U >= 0, dphi) pair in
polar form. The supplied model's own cross-hands are IGNORED (zeroed at
source): the completion absorbs all linear polarisation, which is what "no
polarised model" means. The model's parallel hands (total intensity and Q)
are used as given - they predict the leakage mixing of I into the
cross-hands through any sky-side full-Jones term, which is precisely what
the data-correcting estimator could not do (its iterated update walks O(1)
from the V-null with a leakage-bearing term sky-side of the active term).

**The per-iteration update.** With the chain held at its current estimates
(active term at phi), the corrupted visibility is linear in the nuisance::

    V(z) = V_m(phi) + z * P1(phi) + conj(z) * P2(phi),

where ``V_m`` is the corrupted (cross-hand-zeroed) model, and ``P1``/``P2``
are the sky-frame unit cross-hands ``e01``/``e10`` pushed through the chain:
``P = W_p X_p (F_p e F_q^H) X_q^H W_q^H`` with ``F`` the sky-side flank,
``W`` the data-side flank and ``X = diag(exp(i*phi), 1)`` the active term.
Writing ``z = x + i*y``, the least-squares fit of (x, y) to the residual
``R = D - V_m`` is a real 2x2 linear solve per solution interval - the exact
joint minimiser over the V = 0 nuisance at the current phi. The update is
then the pure reparameterisation ``phi += arg(z)`` (move the recovered phase
into the gain, leave the non-negative amplitude ``U = |z|`` in the sky):

* For chains whose data-side flank is diagonal (amplitude/phase/delay data
  side of the active term - the production configuration), one step lands on
  the exact solution from ANY starting point: there is no watershed at
  +/-pi/2, no repeller, and no attenuation at low SNR (the fit is linear, so
  noise enters the estimate, not the step size).
* With an exact sky-side leakage estimate, the truth is a fixed point (the
  residual at the truth is exactly representable in the (P1, P2) basis), and
  the iteration converges to it; leakage error sets the usual floor.
* The branch is deterministic: U >= 0 by construction, so the recovered
  phase equals the truth when the calibrator's Stokes U > 0 and sits exactly
  pi away when U < 0 - the irreducible sign ambiguity, now pinned to the
  sign of U rather than to an arbitrary principal branch.

**Bookkeeping.** ``solve_per="array"`` sums the per-antenna normal equations
over the array, which is exact for a common phi. ``solve_per="antenna"``
retains per-antenna accumulators; contributions enter each antenna's slots
with the common-phi linearisation, so per-antenna solving is Jacobi-flavoured
(supported, inexact - "array" is the production configuration). The exported
``jhj`` carries ``|z|^2 * M22`` - a Gauss-Newton curvature proxy for phi at
the solution - and ``jhr`` carries the V-nulling component ``y``; the
``max_iter=0`` (jhj-only) path fills both without touching the parameters.
Direction-dependent solving and scalar mode raise: the nuisance completion
is built from the direction-summed residual and a single sky completion.
"""
import numpy as np
from numba import njit, prange
from numba.extending import overload
from quartical.utils.numba import (coerce_literal,
                                   JIT_OPTIONS,
                                   PARALLEL_JIT_OPTIONS)
from quartical.gains.general.generics import native_intermediaries
from quartical.gains.general.flagging import (flag_intermediaries,
                                              update_gain_flags,
                                              finalize_gain_flags,
                                              apply_gain_flags_to_flag_col,
                                              update_param_flags)
from quartical.gains.general.convenience import get_extents, get_row
import quartical.gains.general.factories as factories


def get_identity_params(corr_mode):

    if corr_mode.literal_value == 4:
        return np.zeros((1,), dtype=np.float64)
    else:
        raise ValueError("Unsupported number of correlations.")


@njit(**JIT_OPTIONS)
def null_v_crosshand_phase_solver(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    return null_v_crosshand_phase_solver_impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        corr_mode
    )


def null_v_crosshand_phase_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    raise NotImplementedError


@overload(null_v_crosshand_phase_solver_impl, jit_options=JIT_OPTIONS)
def nb_null_v_crosshand_phase_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):

    coerce_literal(nb_null_v_crosshand_phase_solver_impl, ["corr_mode"])

    identity_params = get_identity_params(corr_mode)

    def impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        corr_mode
    ):

        gains = chain_inputs.gains
        gain_flags = chain_inputs.gain_flags

        active_term = meta_inputs.active_term
        max_iter = meta_inputs.iters
        solve_per = meta_inputs.solve_per
        scalar = meta_inputs.scalar
        dd_term = meta_inputs.dd_term

        if scalar:
            raise ValueError(
                "Scalar mode not supported for crosshand phase terms."
            )

        if dd_term:
            raise ValueError(
                "Direction-dependent solving is not supported for "
                "crosshand_phase_null_v - the V = 0 completion is built "
                "from the direction-summed residual."
            )

        active_gain = gains[active_term]
        active_gain_flags = gain_flags[active_term]
        active_params = chain_inputs.params[active_term]

        # Set up some intemediaries used for flagging.
        km1_gain = active_gain.copy()
        km1_abs2_diffs = np.zeros_like(active_gain_flags, dtype=np.float64)
        abs2_diffs_trend = np.zeros_like(active_gain_flags, dtype=np.float64)
        flag_imdry = \
            flag_intermediaries(km1_gain, km1_abs2_diffs, abs2_diffs_trend)

        # Set up some intemediaries used for solving.
        real_dtype = active_gain.real.dtype
        param_shape = active_params.shape

        active_t_map_g = mapping_inputs.time_maps[active_term]
        active_f_map_g = mapping_inputs.freq_maps[active_term]

        # Determine the starts and stops of the rows and channels associated
        # with each solution interval.
        extents = get_extents(active_t_map_g, active_f_map_g)

        n_tint, n_fint, n_ant, n_dir, n_param = param_shape

        # Per-(interval, antenna) normal equations for the complex nuisance
        # z: (m11, m12, m22, r1, r2) with basis b1 = P1 + P2 (the real part
        # of z) and b2 = i*(P1 - P2) (the imaginary part).
        acc = np.zeros((n_tint, n_fint, n_ant, 5), dtype=real_dtype)

        # jhj/jhr are exported diagnostics here (the update is the closed
        # form above, not a jhj/jhr inversion) - see the module docstring.
        jhj = np.zeros(param_shape + (param_shape[-1],), dtype=real_dtype)
        jhr = np.zeros(param_shape, dtype=real_dtype)
        update = np.zeros(param_shape, dtype=real_dtype)
        native_imdry = native_intermediaries(jhj, jhr, update)

        for loop_idx in range(max_iter or 1):

            compute_z_normal_eqs(
                ms_inputs,
                mapping_inputs,
                chain_inputs,
                meta_inputs,
                acc,
                extents,
                corr_mode
            )

            if solve_per == "array":
                per_array_normal_eqs(acc)

            finalize_update(
                chain_inputs,
                meta_inputs,
                native_imdry,
                acc,
                max_iter > 0,
                corr_mode
            )

            if not max_iter:  # Non-solvable term, we just want jhj.
                conv_perc = 0.  # Didn't converge.
                loop_idx = -1  # Did zero iterations.
                break

            # Check for gain convergence. Produced as a side effect of
            # flagging. The converged percentage is based on unflagged
            # intervals.
            conv_perc = update_gain_flags(
                chain_inputs,
                meta_inputs,
                flag_imdry,
                loop_idx,
                corr_mode,
                numbness=1e9
            )

            # Propagate gain flags to parameter flags.
            update_param_flags(
                mapping_inputs,
                chain_inputs,
                meta_inputs,
                identity_params
            )

            if conv_perc >= meta_inputs.stop_frac:
                break

        # NOTE: Removes soft flags and flags points which have bad trends.
        finalize_gain_flags(
            chain_inputs,
            meta_inputs,
            flag_imdry,
            corr_mode
        )

        # Call this one last time to ensure points flagged by finialize are
        # propagated (in the DI case).
        if not dd_term:
            apply_gain_flags_to_flag_col(
                ms_inputs,
                mapping_inputs,
                chain_inputs,
                meta_inputs
            )

        return native_imdry.jhj, loop_idx + 1, conv_perc

    return impl


def compute_z_normal_eqs(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    acc,
    extents,
    corr_mode
):
    return NotImplementedError


@overload(compute_z_normal_eqs, jit_options=PARALLEL_JIT_OPTIONS)
def nb_compute_z_normal_eqs(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    acc,
    extents,
    corr_mode
):

    coerce_literal(nb_compute_z_normal_eqs, ["corr_mode"])

    # We want to dispatch based on this field so we need its type.
    row_weights_idx = ms_inputs.fields.index('ROW_WEIGHTS')
    row_weights_type = ms_inputs[row_weights_idx]

    tuple_unpack = factories.tuple_unpack_factory(corr_mode)
    tuple_unpack_rweight = factories.tuple_unpack_rweight_factory(
        corr_mode, row_weights_type
    )
    v1_mul_v2 = factories.v1_mul_v2_factory(corr_mode)
    v1_mul_v2ct = factories.v1_mul_v2ct_factory(corr_mode)
    corrupt_pieces = corrupt_pieces_factory(corr_mode)
    nuisance_base_01 = nuisance_base_01_factory(corr_mode)
    nuisance_base_10 = nuisance_base_10_factory(corr_mode)
    accumulate_normal_eqs = accumulate_normal_eqs_factory(corr_mode)

    def impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        acc,
        extents,
        corr_mode
    ):

        data = ms_inputs.DATA
        model = ms_inputs.MODEL_DATA
        weights = ms_inputs.WEIGHT
        flags = ms_inputs.FLAG
        antenna1 = ms_inputs.ANTENNA1
        antenna2 = ms_inputs.ANTENNA2
        row_map = ms_inputs.ROW_MAP
        row_weights = ms_inputs.ROW_WEIGHTS

        time_maps = mapping_inputs.time_maps
        freq_maps = mapping_inputs.freq_maps
        dir_maps = mapping_inputs.dir_maps

        gains = chain_inputs.gains
        active_term = meta_inputs.active_term

        n_dir = model.shape[2]
        n_gains = len(gains)

        acc[:] = 0

        n_tint, n_fint = acc.shape[:2]
        n_int = n_tint*n_fint

        row_starts = extents.row_starts
        row_stops = extents.row_stops
        chan_starts = extents.chan_starts
        chan_stops = extents.chan_stops

        active_t_map = time_maps[active_term]
        active_f_map = freq_maps[active_term]

        # A zero in the compute (gain) dtype, used to promote lower
        # precision inputs.
        zc = gains[active_term][0, 0, 0, 0, 0]*0

        # Parallel over all solution intervals.
        for i in prange(n_int):

            ti = i//n_fint
            fi = i - ti*n_fint

            rs = row_starts[ti]
            re = row_stops[ti]
            fs = chan_starts[fi]
            fe = chan_stops[fi]

            acc_tifi = acc[ti, fi]

            for row_ind in range(rs, re):

                row = get_row(row_ind, row_map)
                a1_m, a2_m = antenna1[row], antenna2[row]

                # Per-row register-resident accumulator - the antennas are
                # fixed for the duration of a row, so flush once per row.
                row_acc = (zc.real, zc.real, zc.real, zc.real, zc.real)

                for f in range(fs, fe):

                    if flags[row, f]:  # Skip flagged data points.
                        continue

                    # Apply row weights in the BDA case, else a no-op.
                    w = tuple_unpack_rweight(
                        weights[row, f], row_weights, row_ind
                    )

                    # Data-side flanks W_p/W_q: the product of all terms
                    # with a lower chain index than the active term (term 0
                    # outermost). Direction-dependent terms data-side of the
                    # active term make no sense for this term - the
                    # broadcast (index 0) direction map is used.
                    wp = (zc + 1, zc, zc, zc + 1)
                    wq = (zc + 1, zc, zc, zc + 1)

                    for gi in range(active_term):

                        d_m = dir_maps[gi][0]
                        t_m = time_maps[gi][row_ind]
                        f_m = freq_maps[gi][f]

                        gain = gains[gi][t_m, f_m]

                        wp = v1_mul_v2(wp, tuple_unpack(gain[a1_m, d_m]))
                        wq = v1_mul_v2(wq, tuple_unpack(gain[a2_m, d_m]))

                    # The active-term gains: gp0/gq0 are the e^{i phi}
                    # elements (the [1, 1] elements are unity by
                    # construction).
                    active_gain_tifi = gains[active_term][
                        active_t_map[row_ind], active_f_map[f]
                    ]
                    gp0 = active_gain_tifi[a1_m, 0, 0]
                    gq0 = active_gain_tifi[a2_m, 0, 0]

                    # Corrupted (cross-hand-zeroed) model, summed over
                    # directions, plus the nuisance bases P1/P2 from the
                    # direction 0 sky flank.
                    vm = (zc, zc, zc, zc)
                    p1 = (zc, zc, zc, zc)
                    p2 = (zc, zc, zc, zc)

                    for d in range(n_dir):

                        # Sky-side flanks F_p/F_q: the product of all terms
                        # with a higher chain index than the active term.
                        fp = (zc + 1, zc, zc, zc + 1)
                        fq = (zc + 1, zc, zc, zc + 1)

                        for gi in range(active_term + 1, n_gains):

                            d_m = dir_maps[gi][d]
                            t_m = time_maps[gi][row_ind]
                            f_m = freq_maps[gi][f]

                            gain = gains[gi][t_m, f_m]

                            fp = v1_mul_v2(fp, tuple_unpack(gain[a1_m, d_m]))
                            fq = v1_mul_v2(fq, tuple_unpack(gain[a2_m, d_m]))

                        # The supplied model's cross-hands are IGNORED - the
                        # V = 0 completion absorbs all linear polarisation
                        # (see the module docstring).
                        m = model[row, f, d]
                        md = (m[0] + zc, zc, zc, m[3] + zc)

                        # N_m = F_p diag(model) F_q^H, then through the
                        # active term and the data-side flanks.
                        nm = v1_mul_v2(fp, md)
                        nm = v1_mul_v2ct(nm, fq)
                        vmd = corrupt_pieces(nm, wp, wq, gp0, gq0)
                        vm = (
                            vm[0] + vmd[0],
                            vm[1] + vmd[1],
                            vm[2] + vmd[2],
                            vm[3] + vmd[3],
                        )

                        if d == 0:
                            ne1 = nuisance_base_01(fp, fq)
                            ne2 = nuisance_base_10(fp, fq)
                            p1 = corrupt_pieces(ne1, wp, wq, gp0, gq0)
                            p2 = corrupt_pieces(ne2, wp, wq, gp0, gq0)

                    rd = tuple_unpack(data[row, f])
                    res = (
                        rd[0] - vm[0],
                        rd[1] - vm[1],
                        rd[2] - vm[2],
                        rd[3] - vm[3],
                    )

                    row_acc = accumulate_normal_eqs(w, p1, p2, res, row_acc)

                # Both antennas of the baseline receive the same (common
                # phi) contributions: exact for solve_per="array" (the
                # ratio-invariant factor of two cancels), Jacobi-flavoured
                # for solve_per="antenna".
                acc_tifi[a1_m, 0] += row_acc[0]
                acc_tifi[a1_m, 1] += row_acc[1]
                acc_tifi[a1_m, 2] += row_acc[2]
                acc_tifi[a1_m, 3] += row_acc[3]
                acc_tifi[a1_m, 4] += row_acc[4]

                acc_tifi[a2_m, 0] += row_acc[0]
                acc_tifi[a2_m, 1] += row_acc[1]
                acc_tifi[a2_m, 2] += row_acc[2]
                acc_tifi[a2_m, 3] += row_acc[3]
                acc_tifi[a2_m, 4] += row_acc[4]

        return

    return impl


def corrupt_pieces_factory(corr_mode):
    """Push a sky-frame coherency through the active term and data flanks.

    Computes ``W_p X_p N X_q^H W_q^H`` for a sky-frame 2x2 ``N`` (a flat
    4-tuple), data-side flank products ``wp``/``wq`` (flat 4-tuples) and the
    active-term [0, 0] gain elements ``gp0``/``gq0`` (the [1, 1] elements
    are unity for a crosshand phase term). Expanding over the index pairs of
    N, entry (a, b) of the result is::

        sum_ij c_ij * wp[2a + i] * conj(wq[2b + j]),

    with c_00 = gp0*conj(gq0)*n00, c_01 = gp0*n01, c_10 = conj(gq0)*n10 and
    c_11 = n11 - i.e. only the cross slots of N carry the active phase for a
    common phi, which is what makes the nuisance fit exact.
    """

    if corr_mode.literal_value == 4:
        def impl(n, wp, wq, gp0, gq0):

            c00 = gp0*np.conjugate(gq0)*n[0]
            c01 = gp0*n[1]
            c10 = np.conjugate(gq0)*n[2]
            c11 = n[3]

            wq0_c = np.conjugate(wq[0])
            wq1_c = np.conjugate(wq[1])
            wq2_c = np.conjugate(wq[2])
            wq3_c = np.conjugate(wq[3])

            v0 = c00*wp[0]*wq0_c + c01*wp[0]*wq1_c + \
                c10*wp[1]*wq0_c + c11*wp[1]*wq1_c
            v1 = c00*wp[0]*wq2_c + c01*wp[0]*wq3_c + \
                c10*wp[1]*wq2_c + c11*wp[1]*wq3_c
            v2 = c00*wp[2]*wq0_c + c01*wp[2]*wq1_c + \
                c10*wp[3]*wq0_c + c11*wp[3]*wq1_c
            v3 = c00*wp[2]*wq2_c + c01*wp[2]*wq3_c + \
                c10*wp[3]*wq2_c + c11*wp[3]*wq3_c

            return v0, v1, v2, v3
    else:
        raise ValueError("Crosshand phase can only be solved for with four "
                         "correlation data.")

    return factories.qcjit(impl)


def nuisance_base_01_factory(corr_mode):
    """Sky-frame unit cross-hand e01 through the sky flanks.

    ``(F_p e01 F_q^H)_ij = Fp[i, 0] * conj(Fq[j, 1])`` - the outer product
    of F_p's first column with the conjugate of F_q's second column.
    """

    if corr_mode.literal_value == 4:
        def impl(fp, fq):

            fq1_c = np.conjugate(fq[1])
            fq3_c = np.conjugate(fq[3])

            return fp[0]*fq1_c, fp[0]*fq3_c, fp[2]*fq1_c, fp[2]*fq3_c
    else:
        raise ValueError("Crosshand phase can only be solved for with four "
                         "correlation data.")

    return factories.qcjit(impl)


def nuisance_base_10_factory(corr_mode):
    """Sky-frame unit cross-hand e10 through the sky flanks.

    ``(F_p e10 F_q^H)_ij = Fp[i, 1] * conj(Fq[j, 0])`` - the outer product
    of F_p's second column with the conjugate of F_q's first column.
    """

    if corr_mode.literal_value == 4:
        def impl(fp, fq):

            fq0_c = np.conjugate(fq[0])
            fq2_c = np.conjugate(fq[2])

            return fp[1]*fq0_c, fp[1]*fq2_c, fp[3]*fq0_c, fp[3]*fq2_c
    else:
        raise ValueError("Crosshand phase can only be solved for with four "
                         "correlation data.")

    return factories.qcjit(impl)


def accumulate_normal_eqs_factory(corr_mode):
    """Accumulate one visibility into the nuisance normal equations.

    The nuisance bases are ``b1 = P1 + P2`` (real part of z) and
    ``b2 = i*(P1 - P2)`` (imaginary part of z). The accumulator is the flat
    real tuple ``(m11, m12, m22, r1, r2)``: the weighted 2x2 normal matrix
    and right-hand side of the linear fit of (Re z, Im z) to the residual.
    """

    if corr_mode.literal_value == 4:
        def impl(w, p1, p2, res, acc5):

            m11, m12, m22, r1, r2 = acc5

            b1_0 = p1[0] + p2[0]
            b1_1 = p1[1] + p2[1]
            b1_2 = p1[2] + p2[2]
            b1_3 = p1[3] + p2[3]

            b2_0 = 1j*(p1[0] - p2[0])
            b2_1 = 1j*(p1[1] - p2[1])
            b2_2 = 1j*(p1[2] - p2[2])
            b2_3 = 1j*(p1[3] - p2[3])

            m11 += w[0]*(b1_0.real**2 + b1_0.imag**2) + \
                w[1]*(b1_1.real**2 + b1_1.imag**2) + \
                w[2]*(b1_2.real**2 + b1_2.imag**2) + \
                w[3]*(b1_3.real**2 + b1_3.imag**2)

            m22 += w[0]*(b2_0.real**2 + b2_0.imag**2) + \
                w[1]*(b2_1.real**2 + b2_1.imag**2) + \
                w[2]*(b2_2.real**2 + b2_2.imag**2) + \
                w[3]*(b2_3.real**2 + b2_3.imag**2)

            m12 += w[0]*(b1_0.real*b2_0.real + b1_0.imag*b2_0.imag) + \
                w[1]*(b1_1.real*b2_1.real + b1_1.imag*b2_1.imag) + \
                w[2]*(b1_2.real*b2_2.real + b1_2.imag*b2_2.imag) + \
                w[3]*(b1_3.real*b2_3.real + b1_3.imag*b2_3.imag)

            r1 += w[0]*(b1_0.real*res[0].real + b1_0.imag*res[0].imag) + \
                w[1]*(b1_1.real*res[1].real + b1_1.imag*res[1].imag) + \
                w[2]*(b1_2.real*res[2].real + b1_2.imag*res[2].imag) + \
                w[3]*(b1_3.real*res[3].real + b1_3.imag*res[3].imag)

            r2 += w[0]*(b2_0.real*res[0].real + b2_0.imag*res[0].imag) + \
                w[1]*(b2_1.real*res[1].real + b2_1.imag*res[1].imag) + \
                w[2]*(b2_2.real*res[2].real + b2_2.imag*res[2].imag) + \
                w[3]*(b2_3.real*res[3].real + b2_3.imag*res[3].imag)

            return m11, m12, m22, r1, r2
    else:
        raise ValueError("Crosshand phase can only be solved for with four "
                         "correlation data.")

    return factories.qcjit(impl)


@njit(**JIT_OPTIONS)
def per_array_normal_eqs(acc):
    """Sum the per-antenna normal equations over the array and broadcast."""

    n_tint, n_fint, n_ant, n_val = acc.shape

    for t in range(n_tint):
        for f in range(n_fint):
            for a in range(1, n_ant):
                for v in range(n_val):
                    acc[t, f, 0, v] += acc[t, f, a, v]

            for a in range(1, n_ant):
                for v in range(n_val):
                    acc[t, f, a, v] = acc[t, f, 0, v]


def finalize_update(
    chain_inputs,
    meta_inputs,
    native_imdry,
    acc,
    apply_update,
    corr_mode
):
    raise NotImplementedError


@overload(finalize_update, jit_options=JIT_OPTIONS)
def nb_finalize_update(
    chain_inputs,
    meta_inputs,
    native_imdry,
    acc,
    apply_update,
    corr_mode
):

    coerce_literal(nb_finalize_update, ["corr_mode"])

    set_identity = factories.set_identity_factory(corr_mode)
    param_to_gain = param_to_gain_factory(corr_mode)

    def impl(
        chain_inputs,
        meta_inputs,
        native_imdry,
        acc,
        apply_update,
        corr_mode
    ):

        active_term = meta_inputs.active_term

        gain = chain_inputs.gains[active_term]
        gain_flags = chain_inputs.gain_flags[active_term]
        params = chain_inputs.params[active_term]

        jhj = native_imdry.jhj
        jhr = native_imdry.jhr

        n_tint, n_fint, n_ant, n_dir, n_corr = gain.shape

        for ti in range(n_tint):
            for fi in range(n_fint):
                for a in range(n_ant):

                    m11 = acc[ti, fi, a, 0]
                    m12 = acc[ti, fi, a, 1]
                    m22 = acc[ti, fi, a, 2]
                    r1 = acc[ti, fi, a, 3]
                    r2 = acc[ti, fi, a, 4]

                    # Solve the 2x2 normal equations for z = x + i*y. A
                    # degenerate system (no unflagged data, or collinear
                    # bases) yields z = 0, i.e. no update.
                    det = m11*m22 - m12*m12

                    if det > 0:
                        x = (m22*r1 - m12*r2)/det
                        y = (m11*r2 - m12*r1)/det
                    else:
                        x = 0.
                        y = 0.

                    # atan2(0, 0) = 0: intervals with no signal are a no-op.
                    dphi = np.arctan2(y, x)
                    umag2 = x*x + y*y

                    # Exported diagnostics: a phi-curvature proxy (jhj) and
                    # the V-nulling component of the nuisance fit (jhr).
                    jhj[ti, fi, a, 0, 0, 0] = umag2*m22
                    jhr[ti, fi, a, 0, 0] = y

                    p = params[ti, fi, a, 0]
                    g = gain[ti, fi, a, 0]
                    fl = gain_flags[ti, fi, a, 0]

                    if fl == 1:
                        p[:] = 0
                        set_identity(g)
                    elif apply_update:
                        p[0] += dphi
                        param_to_gain(p, g)

    return impl


def param_to_gain_factory(corr_mode):

    if corr_mode.literal_value == 4:
        def impl(params, gain):
            gain[0] = np.exp(1j*params[0])
    else:
        raise ValueError("Crosshand phase can only be solved for with four "
                         "correlation data.")

    return factories.qcjit(impl)
