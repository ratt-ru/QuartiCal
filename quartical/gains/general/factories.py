# -*- coding: utf-8 -*-
from numba import jit, types
import numpy as np

# Handy alias for functions that need to be jitted in this way.
qcjit = jit(nogil=True,
            nopython=True,
            fastmath=True,
            cache=True,
            inline="always")


def imul_rweight_factory(mode, weight):

    if isinstance(weight, types.NoneType):
        if mode.literal_value == 4:
            def impl(invec, outvec, weight, ind):
                outvec[0] = invec[0]
                outvec[1] = invec[1]
                outvec[2] = invec[2]
                outvec[3] = invec[3]
        elif mode.literal_value == 2:
            def impl(invec, outvec, weight, ind):
                outvec[0] = invec[0]
                outvec[1] = invec[1]
        elif mode.literal_value == 1:
            def impl(invec, outvec, weight, ind):
                outvec[0] = invec[0]
        else:
            raise ValueError("Unsupported number of correlations.")
    else:

        unpack = unpack_factory(mode)

        if mode.literal_value == 4:
            def impl(invec, outvec, weight, ind):
                v00, v01, v10, v11 = unpack(invec)
                w = weight[ind]
                outvec[0] = w*v00
                outvec[1] = w*v01
                outvec[2] = w*v10
                outvec[3] = w*v11
        elif mode.literal_value == 2:
            def impl(invec, outvec, weight, ind):
                v00, v11 = unpack(invec)
                w = weight[ind]
                outvec[0] = w*v00
                outvec[1] = w*v11
        elif mode.literal_value == 1:
            def impl(invec, outvec, weight, ind):
                v00 = unpack(invec)
                w = weight[ind]
                outvec[0] = w*v00
        else:
            raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def v1_mul_v2_factory(mode):

    unpack = unpack_factory(mode)

    if mode.literal_value == 4:
        def impl(v1, v2):
            v1_00, v1_01, v1_10, v1_11 = unpack(v1)
            v2_00, v2_01, v2_10, v2_11 = unpack(v2)

            v3_00 = (v1_00*v2_00 + v1_01*v2_10)
            v3_01 = (v1_00*v2_01 + v1_01*v2_11)
            v3_10 = (v1_10*v2_00 + v1_11*v2_10)
            v3_11 = (v1_10*v2_01 + v1_11*v2_11)

            return v3_00, v3_01, v3_10, v3_11
    elif mode.literal_value == 2:
        def impl(v1, v2):
            v1_00, v1_11 = unpack(v1)
            v2_00, v2_11 = unpack(v2)

            v3_00 = v1_00*v2_00
            v3_11 = v1_11*v2_11

            return v3_00, v3_11
    elif mode.literal_value == 1:
        def impl(v1, v2):
            v1_00 = unpack(v1)
            v2_00 = unpack(v2)

            v3_00 = v1_00*v2_00

            return v3_00
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def v1_imul_v2_factory(mode):

    unpack = unpack_factory(mode)

    if mode.literal_value == 4:
        def impl(v1, v2, o1):
            v1_00, v1_01, v1_10, v1_11 = unpack(v1)
            v2_00, v2_01, v2_10, v2_11 = unpack(v2)

            o1[0] = (v1_00*v2_00 + v1_01*v2_10)
            o1[1] = (v1_00*v2_01 + v1_01*v2_11)
            o1[2] = (v1_10*v2_00 + v1_11*v2_10)
            o1[3] = (v1_10*v2_01 + v1_11*v2_11)
    elif mode.literal_value == 2:
        def impl(v1, v2, o1):
            v1_00, v1_11 = unpack(v1)
            v2_00, v2_11 = unpack(v2)

            o1[0] = v1_00*v2_00
            o1[1] = v1_11*v2_11
    elif mode.literal_value == 1:
        def impl(v1, v2, o1):
            v1_00 = unpack(v1)
            v2_00 = unpack(v2)

            o1[0] = v1_00*v2_00
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def v1_mul_v2ct_factory(mode):

    unpack = unpack_factory(mode)
    unpackct = unpackct_factory(mode)

    if mode.literal_value == 4:
        def impl(v1, v2):
            v1_00, v1_01, v1_10, v1_11 = unpack(v1)
            v2_00, v2_01, v2_10, v2_11 = unpackct(v2)

            v3_00 = (v1_00*v2_00 + v1_01*v2_10)
            v3_01 = (v1_00*v2_01 + v1_01*v2_11)
            v3_10 = (v1_10*v2_00 + v1_11*v2_10)
            v3_11 = (v1_10*v2_01 + v1_11*v2_11)

            return v3_00, v3_01, v3_10, v3_11
    elif mode.literal_value == 2:
        def impl(v1, v2):
            v1_00, v1_11 = unpack(v1)
            v2_00, v2_11 = unpackct(v2)

            v3_00 = v1_00*v2_00
            v3_11 = v1_11*v2_11

            return v3_00, v3_11
    elif mode.literal_value == 1:
        def impl(v1, v2):
            v1_00 = unpack(v1)
            v2_00 = unpackct(v2)

            v3_00 = v1_00*v2_00

            return v3_00
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def v1_imul_v2ct_factory(mode):

    unpack = unpack_factory(mode)
    unpackct = unpackct_factory(mode)

    if mode.literal_value == 4:
        def impl(v1, v2, o1):
            v1_00, v1_01, v1_10, v1_11 = unpack(v1)
            v2_00, v2_01, v2_10, v2_11 = unpackct(v2)

            o1[0] = (v1_00*v2_00 + v1_01*v2_10)
            o1[1] = (v1_00*v2_01 + v1_01*v2_11)
            o1[2] = (v1_10*v2_00 + v1_11*v2_10)
            o1[3] = (v1_10*v2_01 + v1_11*v2_11)
    elif mode.literal_value == 2:
        def impl(v1, v2, o1):
            v1_00, v1_11 = unpack(v1)
            v2_00, v2_11 = unpackct(v2)

            o1[0] = v1_00*v2_00
            o1[1] = v1_11*v2_11
    elif mode.literal_value == 1:
        def impl(v1, v2, o1):
            v1_00 = unpack(v1)
            v2_00 = unpackct(v2)

            o1[0] = v1_00*v2_00
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def v1ct_mul_v2_factory(mode):

    unpack = unpack_factory(mode)
    unpackct = unpackct_factory(mode)

    if mode.literal_value == 4:
        def impl(v1, v2):
            v1_00, v1_01, v1_10, v1_11 = unpackct(v1)
            v2_00, v2_01, v2_10, v2_11 = unpack(v2)

            v3_00 = (v1_00*v2_00 + v1_01*v2_10)
            v3_01 = (v1_00*v2_01 + v1_01*v2_11)
            v3_10 = (v1_10*v2_00 + v1_11*v2_10)
            v3_11 = (v1_10*v2_01 + v1_11*v2_11)

            return v3_00, v3_01, v3_10, v3_11
    elif mode.literal_value == 2:
        def impl(v1, v2):
            v1_00, v1_11 = unpackct(v1)
            v2_00, v2_11 = unpack(v2)

            v3_00 = v1_00*v2_00
            v3_11 = v1_11*v2_11

            return v3_00, v3_11
    elif mode.literal_value == 1:
        def impl(v1, v2):
            v1_00 = unpackct(v1)
            v2_00 = unpack(v2)

            v3_00 = v1_00*v2_00

            return v3_00
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def v1ct_imul_v2_factory(mode):

    unpack = unpack_factory(mode)
    unpackct = unpackct_factory(mode)

    if mode.literal_value == 4:
        def impl(v1, v2, o1):
            v1_00, v1_01, v1_10, v1_11 = unpackct(v1)
            v2_00, v2_01, v2_10, v2_11 = unpack(v2)

            o1[0] = (v1_00*v2_00 + v1_01*v2_10)
            o1[1] = (v1_00*v2_01 + v1_01*v2_11)
            o1[2] = (v1_10*v2_00 + v1_11*v2_10)
            o1[3] = (v1_10*v2_01 + v1_11*v2_11)
    elif mode.literal_value == 2:
        def impl(v1, v2, o1):
            v1_00, v1_11 = unpackct(v1)
            v2_00, v2_11 = unpack(v2)

            o1[0] = v1_00*v2_00
            o1[1] = v1_11*v2_11
    elif mode.literal_value == 1:
        def impl(v1, v2, o1):
            v1_00 = unpackct(v1)
            v2_00 = unpack(v2)

            o1[0] = v1_00*v2_00
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def iwmul_factory(mode):

    unpack = unpack_factory(mode)

    if mode.literal_value == 4:
        def impl(v1, w1):
            w1_00, w1_01, w1_10, w1_11 = unpack(w1)

            v1[0] *= w1_00
            v1[1] *= w1_00
            v1[2] *= w1_11
            v1[3] *= w1_11
    elif mode.literal_value == 2:
        def impl(v1, w1):
            w1_00, w1_11 = unpack(w1)

            v1[0] *= w1_00
            v1[1] *= w1_11
    elif mode.literal_value == 1:
        def impl(v1, w1):
            w1_00 = unpack(w1)

            v1[0] *= w1_00
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def v1_wmul_v2ct_factory(mode):

    unpack = unpack_factory(mode)
    unpackct = unpackct_factory(mode)

    if mode.literal_value == 4:
        def impl(v1, v2, w1):
            v1_00, v1_01, v1_10, v1_11 = unpack(v1)
            v2_00, v2_01, v2_10, v2_11 = unpackct(v2)
            w1_00, w1_01, w1_10, w1_11 = unpack(w1)

            v3_00 = (v1_00*w1_00*v2_00 + v1_01*w1_11*v2_10)
            v3_01 = (v1_00*w1_00*v2_01 + v1_01*w1_11*v2_11)
            v3_10 = (v1_10*w1_00*v2_00 + v1_11*w1_11*v2_10)
            v3_11 = (v1_10*w1_00*v2_01 + v1_11*w1_11*v2_11)

            return v3_00, v3_01, v3_10, v3_11
    elif mode.literal_value == 2:
        def impl(v1, v2, w1):
            v1_00, v1_11 = unpack(v1)
            v2_00, v2_11 = unpackct(v2)
            w1_00, w1_11 = unpack(w1)

            v3_00 = v1_00*w1_00*v2_00
            v3_11 = v1_11*w1_11*v2_11

            return v3_00, v3_11
    elif mode.literal_value == 1:
        def impl(v1, v2, w1):
            v1_00 = unpack(v1)
            v2_00 = unpackct(v2)
            w1_00 = unpack(w1)

            v3_00 = v1_00*w1_00*v2_00

            return (v3_00,)  # Ensure that this doesn't come out as a scalar.
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def v1ct_wmul_v2_factory(mode):

    unpack = unpack_factory(mode)
    unpackct = unpackct_factory(mode)

    if mode.literal_value == 4:
        def impl(v1, v2, w1):
            v1_00, v1_01, v1_10, v1_11 = unpackct(v1)
            v2_00, v2_01, v2_10, v2_11 = unpack(v2)
            w1_00, w1_01, w1_10, w1_11 = unpack(w1)

            v3_00 = (v1_00*w1_00*v2_00 + v1_01*w1_11*v2_10)
            v3_01 = (v1_00*w1_00*v2_01 + v1_01*w1_11*v2_11)
            v3_10 = (v1_10*w1_00*v2_00 + v1_11*w1_11*v2_10)
            v3_11 = (v1_10*w1_00*v2_01 + v1_11*w1_11*v2_11)

            return v3_00, v3_01, v3_10, v3_11
    elif mode.literal_value == 2:
        def impl(v1, v2, w1):
            v1_00, v1_11 = unpackct(v1)
            v2_00, v2_11 = unpack(v2)
            w1_00, w1_11 = unpack(w1)

            v3_00 = v1_00*w1_00*v2_00
            v3_11 = v1_11*w1_11*v2_11

            return v3_00, v3_11
    elif mode.literal_value == 1:
        def impl(v1, v2, w1):
            v1_00 = unpackct(v1)
            v2_00 = unpack(v2)
            w1_00 = unpack(w1)

            v3_00 = v1_00*w1_00*v2_00

            return (v3_00,)  # Ensure that this doesn't come out as a scalar.
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def unpack_factory(mode):

    if mode.literal_value == 4:
        def impl(invec):
            return invec[0], invec[1], invec[2], invec[3]
    elif mode.literal_value == 2:
        def impl(invec):
            return invec[0], invec[1]
    elif mode.literal_value == 1:
        def impl(invec):
            return invec[0]
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def unpackc_factory(mode):

    if mode.literal_value == 4:
        def impl(invec):
            return invec[0].conjugate(), \
                   invec[1].conjugate(), \
                   invec[2].conjugate(), \
                   invec[3].conjugate()
    elif mode.literal_value == 2:
        def impl(invec):
            return invec[0].conjugate(), invec[1].conjugate()
    elif mode.literal_value == 1:
        def impl(invec):
            return invec[0].conjugate()
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def unpackct_factory(mode):

    if mode.literal_value == 4:
        def impl(invec):
            return np.conjugate(invec[0]), \
                   np.conjugate(invec[2]), \
                   np.conjugate(invec[1]), \
                   np.conjugate(invec[3])
    elif mode.literal_value == 2:
        def impl(invec):
            return np.conjugate(invec[0]), \
                   np.conjugate(invec[1])
    elif mode.literal_value == 1:
        def impl(invec):
            return np.conjugate(invec[0])
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def iunpack_factory(mode):

    if mode.literal_value == 4:
        def impl(outvec, invec):
            outvec[0] = invec[0]
            outvec[1] = invec[1]
            outvec[2] = invec[2]
            outvec[3] = invec[3]
    elif mode.literal_value == 2:
        def impl(outvec, invec):
            outvec[0] = invec[0]
            outvec[1] = invec[1]
    elif mode.literal_value == 1:
        def impl(outvec, invec):
            outvec[0] = invec[0]
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def iunpackct_factory(mode):

    if mode.literal_value == 4:
        def impl(outvec, invec):
            outvec[0] = np.conjugate(invec[0])
            outvec[1] = np.conjugate(invec[2])
            outvec[2] = np.conjugate(invec[1])
            outvec[3] = np.conjugate(invec[3])
    elif mode.literal_value == 2:
        def impl(outvec, invec):
            outvec[0] = np.conjugate(invec[0])
            outvec[1] = np.conjugate(invec[1])
    elif mode.literal_value == 1:
        def impl(outvec, invec):
            outvec[0] = np.conjugate(invec[0])
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def iadd_factory(mode):

    if mode.literal_value == 4:
        def impl(outvec, invec):
            outvec[0] += invec[0]
            outvec[1] += invec[1]
            outvec[2] += invec[2]
            outvec[3] += invec[3]
    elif mode.literal_value == 2:
        def impl(outvec, invec):
            outvec[0] += invec[0]
            outvec[1] += invec[1]
    elif mode.literal_value == 1:
        def impl(outvec, invec):
            outvec[0] += invec[0]
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def isub_factory(mode):

    if mode.literal_value == 4:
        def impl(outvec, invec):
            outvec[0] -= invec[0]
            outvec[1] -= invec[1]
            outvec[2] -= invec[2]
            outvec[3] -= invec[3]
    elif mode.literal_value == 2:
        def impl(outvec, invec):
            outvec[0] -= invec[0]
            outvec[1] -= invec[1]
    elif mode.literal_value == 1:
        def impl(outvec, invec):
            outvec[0] -= invec[0]
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def imul_factory(mode):

    if mode.literal_value == 4:
        def impl(outvec, invec):
            outvec[0] *= invec[0]
            outvec[1] *= invec[1]
            outvec[2] *= invec[2]
            outvec[3] *= invec[3]
    elif mode.literal_value == 2:
        def impl(outvec, invec):
            outvec[0] *= invec[0]
            outvec[1] *= invec[1]
    elif mode.literal_value == 1:
        def impl(outvec, invec):
            outvec[0] *= invec[0]
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def valloc_factory(mode):
    if mode.literal_value == 4:
        def impl(dtype, leading_dims=()):
            return np.empty((*leading_dims, 4), dtype=dtype)
    elif mode.literal_value == 2:
        def impl(dtype, leading_dims=()):
            return np.empty((*leading_dims, 2), dtype=dtype)
    elif mode.literal_value == 1:
        def impl(dtype, leading_dims=()):
            return np.empty((*leading_dims, 1), dtype=dtype)
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def loop_var_factory(mode):
    if mode.literal_value == 4:
        def impl(n_gains, active_term):
            all_terms = np.arange(n_gains - 1, -1, -1)
            gt_active = np.arange(n_gains - 1, active_term, -1)
            lt_active = np.arange(active_term)
            return all_terms, gt_active, lt_active
    else:  # True for both scalar and diagonal gains.
        def impl(n_gains, active_term):
            all_terms = np.arange(n_gains - 1, -1, -1)
            gt_active = np.where(np.arange(n_gains) != active_term)[0]
            lt_active = np.arange(0)
            return all_terms, gt_active, lt_active

    return qcjit(impl)


def compute_det_factory(mode):
    if mode.literal_value == 4:
        def impl(v1):
            return v1[0]*v1[3] - v1[1]*v1[2]
    elif mode.literal_value == 2:
        def impl(v1):
            return v1[0]*v1[1]
    elif mode.literal_value == 1:
        def impl(v1):
            return v1[0]
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def iinverse_factory(mode):

    unpack = unpack_factory(mode)

    if mode.literal_value == 4:
        def impl(v1, det, o1):
            v1_00, v1_01, v1_10, v1_11 = unpack(v1)

            o1[0] = v1_11/det
            o1[1] = -v1_01/det
            o1[2] = -v1_10/det
            o1[3] = v1_00/det
    elif mode.literal_value == 2:
        def impl(v1, det, o1):
            v1_00, v1_11 = unpack(v1)

            o1[0] = v1_11/det
            o1[1] = v1_00/det
    elif mode.literal_value == 1:  # TODO: Is this correct?
        def impl(v1, det, o1):
            v1_00 = unpack(v1)

            o1[0] = 1.0/v1_00
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def set_identity_factory(mode):

    if mode.literal_value == 4:
        def impl(v1):
            v1[0] = 1
            v1[1] = 0
            v1[2] = 0
            v1[3] = 1
    elif mode.literal_value == 2:
        def impl(v1):
            v1[0] = 1
            v1[1] = 1
    elif mode.literal_value == 1:
        def impl(v1):
            v1[0] = 1
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def iabs_factory(mode):

    if mode.literal_value == 4:
        def impl(v1):
            v1[0] = np.abs(v1[0])
            v1[1] = np.abs(v1[1])
            v1[2] = np.abs(v1[2])
            v1[3] = np.abs(v1[3])
    elif mode.literal_value == 2:
        def impl(v1):
            v1[0] = np.abs(v1[0])
            v1[1] = np.abs(v1[1])
    elif mode.literal_value == 1:
        def impl(v1):
            v1[0] = np.abs(v1[0])
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def iabsdiv_factory(mode):

    if mode.literal_value == 4:
        def impl(v1):
            v1[0] = 0 if v1[0] == 0 else 1/np.abs(v1[0])
            v1[1] = 0 if v1[1] == 0 else 1/np.abs(v1[1])
            v1[2] = 0 if v1[2] == 0 else 1/np.abs(v1[2])
            v1[3] = 0 if v1[3] == 0 else 1/np.abs(v1[3])
    elif mode.literal_value == 2:
        def impl(v1):
            v1[0] = 0 if v1[0] == 0 else 1/np.abs(v1[0])
            v1[1] = 0 if v1[1] == 0 else 1/np.abs(v1[1])
    elif mode.literal_value == 1:
        def impl(v1):
            v1[0] = 0 if v1[0] == 0 else 1/np.abs(v1[0])
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def iabsdivsq_factory(mode):

    unpack = unpack_factory(mode)

    if mode.literal_value == 4:
        def impl(v1):
            v1_0, v1_1, v1_2, v1_3 = unpack(v1)

            v1[0] = 0 if v1_0 == 0 else 1/(v1_0.real**2 + v1_0.imag**2)
            v1[1] = 0 if v1_1 == 0 else 1/(v1_1.real**2 + v1_1.imag**2)
            v1[2] = 0 if v1_2 == 0 else 1/(v1_2.real**2 + v1_2.imag**2)
            v1[3] = 0 if v1_3 == 0 else 1/(v1_3.real**2 + v1_3.imag**2)
    elif mode.literal_value == 2:
        def impl(v1):
            v1_0, v1_1 = unpack(v1)

            v1[0] = 0 if v1_0 == 0 else 1/(v1_0.real**2 + v1_0.imag**2)
            v1[1] = 0 if v1_1 == 0 else 1/(v1_1.real**2 + v1_1.imag**2)
    elif mode.literal_value == 1:
        def impl(v1):
            v1_0 = unpack(v1)

            v1[0] = 0 if v1_0 == 0 else 1/(v1_0.real**2 + v1_0.imag**2)
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def v1_idiv_absv2_factory(mode):

    if mode.literal_value == 4:
        def impl(v1, v2, o1):
            o1[0] = 0 if v2[0] == 0 else v1[0]/np.abs(v2[0])
            o1[1] = 0 if v2[1] == 0 else v1[1]/np.abs(v2[1])
            o1[2] = 0 if v2[2] == 0 else v1[2]/np.abs(v2[2])
            o1[3] = 0 if v2[3] == 0 else v1[3]/np.abs(v2[3])
    elif mode.literal_value == 2:
        def impl(v1, v2, o1):
            o1[0] = 0 if v2[0] == 0 else v1[0]/np.abs(v2[0])
            o1[1] = 0 if v2[1] == 0 else v1[1]/np.abs(v2[1])
    elif mode.literal_value == 1:
        def impl(v1, v2, o1):
            o1[0] = 0 if v2[0] == 0 else v1[0]/np.abs(v2[0])
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def absv1_idiv_absv2_factory(mode):

    if mode.literal_value == 4:
        def impl(v1, v2, o1):
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
    elif mode.literal_value == 2:
        def impl(v1, v2, o1):
            o1[0] = 0 if v2[0] == 0 else np.sqrt(
                (v1[0].real**2 + v1[0].imag**2)/(v2[0].real**2 + v2[0].imag**2)
            )
            o1[1] = 0 if v2[1] == 0 else np.sqrt(
                (v1[1].real**2 + v1[1].imag**2)/(v2[1].real**2 + v2[1].imag**2)
            )
    elif mode.literal_value == 1:
        def impl(v1, v2, o1):
            o1[0] = 0 if v2[0] == 0 else np.sqrt(
                (v1[0].real**2 + v1[0].imag**2)/(v2[0].real**2 + v2[0].imag**2)
            )
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


def a_kron_bt_factory(corr_mode):

    unpack = unpack_factory(corr_mode)

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

    return qcjit(impl)
