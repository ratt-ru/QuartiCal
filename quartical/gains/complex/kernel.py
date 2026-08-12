# -*- coding: utf-8 -*-
from numba import njit
from numba.extending import overload
from quartical.utils.numba import (coerce_literal,
                                   JIT_OPTIONS,
                                   PARALLEL_JIT_OPTIONS)
import quartical.gains.general.factories as factories
from quartical.gains.general.solver_components import build_jhj_jhr_impl
from quartical.gains.general.solver_loop import build_gain_solver_impl
from quartical.gains.general.residuals import standard_residual_factory


@njit(**JIT_OPTIONS)
def complex_solver(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    return complex_solver_impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        corr_mode
    )


def complex_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    raise NotImplementedError


@overload(complex_solver_impl, jit_options=JIT_OPTIONS)
def nb_complex_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):

    coerce_literal(nb_complex_solver_impl, ["corr_mode"])

    # The outer solver loop is shared between non-parameterised kernels - only
    # the hooks below are specific to complex terms. Complex has full-block jhj
    # dims, does not support scalar mode, and needs no referencing stage.
    # Inlined into the trampoline below for a private cache namespace.
    shared_impl = build_gain_solver_impl(
        get_jhj_dims=get_jhj_dims_factory(corr_mode),
        compute_jhj_jhr=compute_jhj_jhr,
        collapse_to_scalar_jhj_jhr=None,
        scalar_error_message="Scalar mode not supported for complex terms.",
        finalize_update=finalize_update,
        reference_gains=None,
    )

    def impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        corr_mode
    ):
        return shared_impl(
            ms_inputs,
            mapping_inputs,
            chain_inputs,
            meta_inputs,
            corr_mode
        )

    return impl


def compute_jhj_jhr(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    upsampled_imdry,
    extents,
    corr_mode
):
    return NotImplementedError


@overload(compute_jhj_jhr, jit_options=PARALLEL_JIT_OPTIONS)
def nb_compute_jhj_jhr(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    upsampled_imdry,
    extents,
    corr_mode
):

    coerce_literal(nb_compute_jhj_jhr, ["corr_mode"])

    # We want to dispatch based on this field so we need its type.
    row_weights_idx = ms_inputs.fields.index('ROW_WEIGHTS')
    row_weights_type = ms_inputs[row_weights_idx]

    # The accumulation loop itself is shared between kernels - only the
    # hooks below (the per-term maths) are specific to complex terms. The
    # complex residual has no per-channel coefficients, so there is no
    # channel-coefficient hook (the accumulate hook receives an empty
    # channel_coeffs tuple).
    # Inlined into the trampoline below for a private cache namespace.
    shared_impl = build_jhj_jhr_impl(
        corr_mode=corr_mode,
        row_weights_type=row_weights_type,
        accumulate_jhj_jhr_factory=accumulate_jhj_jhr_factory,
        zero_jhj_jhr_factory=zero_jhj_jhr_factory,
        flush_jhj_jhr_factory=flush_jhj_jhr_factory,
        compute_residual_factory=standard_residual_factory,
        compute_channel_coeffs_factory=None,
        mirror_jhj_factory=mirror_jhj_factory,
    )

    def impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        upsampled_imdry,
        extents,
        corr_mode
    ):
        return shared_impl(
            ms_inputs,
            mapping_inputs,
            chain_inputs,
            meta_inputs,
            upsampled_imdry,
            extents,
            corr_mode
        )

    return impl


def finalize_update(
    chain_inputs,
    meta_inputs,
    native_imdry,
    loop_idx,
    corr_mode
):
    raise NotImplementedError


@overload(finalize_update, jit_options=JIT_OPTIONS)
def nb_finalize_update(
    chain_inputs,
    meta_inputs,
    native_imdry,
    loop_idx,
    corr_mode
):

    coerce_literal(nb_finalize_update, ["corr_mode"])

    set_identity = factories.set_identity_factory(corr_mode)

    def impl(
        chain_inputs,
        meta_inputs,
        native_imdry,
        loop_idx,
        corr_mode
    ):

        dd_term = meta_inputs.dd_term
        active_term = meta_inputs.active_term
        pinned_directions = meta_inputs.pinned_directions

        gain = chain_inputs.gains[active_term]
        gain_flags = chain_inputs.gain_flags[active_term]

        update = native_imdry.update

        n_tint, n_fint, n_ant, n_dir, n_corr = gain.shape

        if dd_term:
            dir_loop = [d for d in range(n_dir) if d not in pinned_directions]
        else:
            dir_loop = [d for d in range(n_dir)]

        for ti in range(n_tint):
            for fi in range(n_fint):
                for a in range(n_ant):
                    for d in dir_loop:

                        g = gain[ti, fi, a, d]
                        fl = gain_flags[ti, fi, a, d]
                        upd = update[ti, fi, a, d]

                        if fl == 1:
                            set_identity(g)
                        elif dd_term or (loop_idx % 2 == 0):
                            upd /= 2
                            g += upd
                        else:
                            g += upd

    return impl


def zero_jhj_jhr_factory(corr_mode):
    """Produce the zero jhwj/jhwr accumulator tuple for a given corr mode.

    The accumulator is a single flat tuple holding the (upper triangle,
    row-major, in the 4 correlation case) jhwj element followed by the jhwr
    element. A flat tuple is used deliberately - returning nested tuples
    from inlined functions inside a prange trips a numba parfor array
    analysis bug. In the diagonal cases the jhwj entries are real, as
    w|rop|^2 has no imaginary part, so those leading slots hold real zeros
    while the trailing jhwr slots hold complex zeros.
    """

    if corr_mode.literal_value == 4:
        def impl(invec):
            z = invec[0]*0
            return z, z, z, z, z, z, z, z, z, z, z, z, z, z
    elif corr_mode.literal_value == 2:
        def impl(invec):
            z = invec[0]*0
            zr = invec[0].real*0
            return zr, zr, z, z
    elif corr_mode.literal_value == 1:
        def impl(invec):
            return invec[0].real*0, invec[0]*0
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def flush_jhj_jhr_factory(corr_mode):
    """Add a register-accumulated jhwr/jhwj accumulator into the arrays.

    In the 4 correlation case the jhj part of the accumulator holds the
    upper triangle of the (4, 4) jhj element in row-major order - the lower
    triangle is filled in by mirror_jhj once per solution interval.
    """

    if corr_mode.literal_value == 4:
        def impl(jhj, jhr, jhj_jhr):

            jhj[0, 0] += jhj_jhr[0]
            jhj[0, 1] += jhj_jhr[1]
            jhj[0, 2] += jhj_jhr[2]
            jhj[0, 3] += jhj_jhr[3]
            jhj[1, 1] += jhj_jhr[4]
            jhj[1, 2] += jhj_jhr[5]
            jhj[1, 3] += jhj_jhr[6]
            jhj[2, 2] += jhj_jhr[7]
            jhj[2, 3] += jhj_jhr[8]
            jhj[3, 3] += jhj_jhr[9]

            jhr[0] += jhj_jhr[10]
            jhr[1] += jhj_jhr[11]
            jhr[2] += jhj_jhr[12]
            jhr[3] += jhj_jhr[13]
    elif corr_mode.literal_value == 2:
        def impl(jhj, jhr, jhj_jhr):

            jhj[0] += jhj_jhr[0]
            jhj[1] += jhj_jhr[1]

            jhr[0] += jhj_jhr[2]
            jhr[1] += jhj_jhr[3]
    elif corr_mode.literal_value == 1:
        def impl(jhj, jhr, jhj_jhr):

            jhj[0] += jhj_jhr[0]

            jhr[0] += jhj_jhr[1]
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def accumulate_jhj_jhr_factory(corr_mode):
    """Accumulate a jhwr/jhwj element into a register-resident accumulator.

    All inputs and the returned accumulator are tuples (register-resident
    values) - the accumulator is only flushed to memory by flush_jhj_jhr.
    The accumulator is a single flat tuple (jhwj followed by jhwr - see
    zero_jhj_jhr_factory). In the 4 correlation case only the upper
    triangle of the (4, 4) jhj element is accumulated (in row-major order) -
    the lower triangle is filled in once per solution interval by
    mirror_jhj.

    The signature follows the unified accumulate_jhj_jhr contract of the
    shared accumulation loop (see solver_components.py). Complex terms have no
    chain rule beyond the operators themselves, so the gain and channel_coeffs
    arguments are unused - the compiler eliminates them entirely after
    inlining.
    """

    tuple_v1_mul_v2 = factories.tuple_v1_mul_v2_factory(corr_mode)

    if corr_mode.literal_value == 4:
        def impl(lop, rop, w, gain, channel_coeffs, wres, jhj_jhr):

            # Accumulate an element of jhwr.
            upd = tuple_v1_mul_v2(lop, wres)
            upd = tuple_v1_mul_v2(upd, rop)

            # Accumulate an element of jhwj.

            # WARNING: In this instance we are using the row-major
            # version of the kronecker product identity. This is because the
            # MS stores the correlations in row-major order (XX, XY, YX, YY),
            # whereas the standard maths assumes column-major ordering
            # (XX, YX, XY, YY). This subtle change means we can use the MS
            # data directly without worrying about swapping elements around.
            #
            # Row (2i + j), column (2p + q) of J^H is a_ip*b_jq, so an
            # element of J^HWJ factorises as
            #   jhj[2i+j, 2k+l] = sum_pq w_pq (a_ip b_jq) conj(a_kp b_lq)
            #                   = sum_p a_ip conj(a_kp) C_jl^p,
            #   C_jl^p = w_p0 B_jl^0 + w_p1 B_jl^1,  B_jl^q = b_jq conj(b_lq).
            # Exploiting this (and hermitian symmetry, i.e. only computing
            # the upper triangle - see mirror_jhj) roughly halves the number
            # of multiplications relative to forming the kronecker product
            # rows explicitly.
            a00, a01, a10, a11 = lop[0], lop[1], lop[2], lop[3]
            b00, b10, b01, b11 = rop[0], rop[1], rop[2], rop[3]  # Transpose.

            w_0, w_1, w_2, w_3 = w[0], w[1], w[2], w[3]  # NOTE: XX XY YX YY

            # A_ik^p = a_ip conj(a_kp); A_10^p = conj(A_01^p) is not needed
            # as only the upper triangle is accumulated. The diagonal (in ik)
            # entries are |a_ip|^2 i.e. real - keeping them as real scalars
            # (likewise for B and C below) roughly halves the flops relative
            # to treating every factor as complex.
            a00_0 = a00.real*a00.real + a00.imag*a00.imag
            a00_1 = a01.real*a01.real + a01.imag*a01.imag
            a01_0, a01_1 = a00*a10.conjugate(), a01*a11.conjugate()
            a11_0 = a10.real*a10.real + a10.imag*a10.imag
            a11_1 = a11.real*a11.real + a11.imag*a11.imag

            # B_jl^q = b_jq conj(b_lq).
            b00_0 = b00.real*b00.real + b00.imag*b00.imag
            b00_1 = b01.real*b01.real + b01.imag*b01.imag
            b01_0, b01_1 = b00*b10.conjugate(), b01*b11.conjugate()
            b11_0 = b10.real*b10.real + b10.imag*b10.imag
            b11_1 = b11.real*b11.real + b11.imag*b11.imag

            # C_jl^p; the weights are real so C_10^p = conj(C_01^p) and the
            # diagonal (in jl) entries remain real.
            c00_0, c00_1 = w_0*b00_0 + w_1*b00_1, w_2*b00_0 + w_3*b00_1
            c01_0, c01_1 = w_0*b01_0 + w_1*b01_1, w_2*b01_0 + w_3*b01_1
            c10_0, c10_1 = c01_0.conjugate(), c01_1.conjugate()
            c11_0, c11_1 = w_0*b11_0 + w_1*b11_1, w_2*b11_0 + w_3*b11_1

            return (
                jhj_jhr[0] + (a00_0*c00_0 + a00_1*c00_1),
                jhj_jhr[1] + (a00_0*c01_0 + a00_1*c01_1),
                jhj_jhr[2] + (a01_0*c00_0 + a01_1*c00_1),
                jhj_jhr[3] + (a01_0*c01_0 + a01_1*c01_1),
                jhj_jhr[4] + (a00_0*c11_0 + a00_1*c11_1),
                jhj_jhr[5] + (a01_0*c10_0 + a01_1*c10_1),
                jhj_jhr[6] + (a01_0*c11_0 + a01_1*c11_1),
                jhj_jhr[7] + (a11_0*c00_0 + a11_1*c00_1),
                jhj_jhr[8] + (a11_0*c01_0 + a11_1*c01_1),
                jhj_jhr[9] + (a11_0*c11_0 + a11_1*c11_1),
                jhj_jhr[10] + upd[0],
                jhj_jhr[11] + upd[1],
                jhj_jhr[12] + upd[2],
                jhj_jhr[13] + upd[3],
            )

    elif corr_mode.literal_value == 2:
        def impl(lop, rop, w, gain, channel_coeffs, wres, jhj_jhr):

            # Accumulate an element of jhwr.
            upd = tuple_v1_mul_v2(wres, rop)

            # Accumulate an element of jhwj: w|rop|^2, which is real.
            jh_00, jh_11 = rop[0], rop[1]

            return (
                jhj_jhr[0] + w[0]*(jh_00.real*jh_00.real +
                                   jh_00.imag*jh_00.imag),
                jhj_jhr[1] + w[1]*(jh_11.real*jh_11.real +
                                   jh_11.imag*jh_11.imag),
                jhj_jhr[2] + upd[0],
                jhj_jhr[3] + upd[1],
            )
    elif corr_mode.literal_value == 1:
        def impl(lop, rop, w, gain, channel_coeffs, wres, jhj_jhr):

            # Accumulate an element of jhwr.
            upd = tuple_v1_mul_v2(wres, rop)

            # Accumulate an element of jhwj: w|rop|^2, which is real.
            jh_00 = rop[0]

            return (
                jhj_jhr[0] + w[0]*(jh_00.real*jh_00.real +
                                   jh_00.imag*jh_00.imag),
                jhj_jhr[1] + upd[0],
            )
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def mirror_jhj_factory(corr_mode):
    """Fill in the lower triangle of the per-interval jhj elements.

    In the 4 correlation case, accumulation in accumulate_jhj_jhr only
    writes the upper triangle of each (4, 4) jhj element. As jhj is Hermitian,
    the lower triangle is the conjugate of the upper triangle and can be
    filled in once per solution interval rather than once per visibility.
    This is a no-op in the 1/2 correlation (diagonal) cases.
    """

    if corr_mode.literal_value == 4:
        def impl(jhj_tifi):
            n_ant, n_gdir = jhj_tifi.shape[:2]
            for a in range(n_ant):
                for d in range(n_gdir):
                    jhj_ad = jhj_tifi[a, d]
                    for i in range(1, 4):
                        for j in range(i):
                            jhj_ad[i, j] = jhj_ad[j, i].conjugate()
    else:
        def impl(jhj_tifi):
            pass

    return factories.qcjit(impl)


def get_jhj_dims_factory(corr_mode):

    if corr_mode.literal_value == 4:
        def impl(shape):
            return shape[:4] + (4, 4)
    elif corr_mode.literal_value in (1, 2):
        def impl(shape):
            return shape
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)
