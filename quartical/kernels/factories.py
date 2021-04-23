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
        if mode.literal_value == "full" or mode.literal_value == "mixed":
            def impl(invec, outvec, weight, ind):
                outvec[0] = invec[0]
                outvec[1] = invec[1]
                outvec[2] = invec[2]
                outvec[3] = invec[3]
        else:
            def impl(invec, outvec, weight, ind):
                outvec[0] = invec[0]
                outvec[1] = invec[1]
    else:

        unpack = unpack_factory(mode)

        if mode.literal_value == "full" or mode.literal_value == "mixed":
            def impl(invec, outvec, weight, ind):
                v00, v01, v10, v11 = unpack(invec)
                w = weight[ind]
                outvec[0] = w*v00
                outvec[1] = w*v01
                outvec[2] = w*v10
                outvec[3] = w*v11
        else:
            def impl(invec, outvec, weight, ind):
                v00, v11 = unpack(invec)
                w = weight[ind]
                outvec[0] = w*v00
                outvec[1] = w*v11
    return qcjit(impl)


def v1_mul_v2_factory(mode):

    unpack = unpack_factory(mode)

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(v1, v2):
            v1_00, v1_01, v1_10, v1_11 = unpack(v1)
            v2_00, v2_01, v2_10, v2_11 = unpack(v2)

            v3_00 = (v1_00*v2_00 + v1_01*v2_10)
            v3_01 = (v1_00*v2_01 + v1_01*v2_11)
            v3_10 = (v1_10*v2_00 + v1_11*v2_10)
            v3_11 = (v1_10*v2_01 + v1_11*v2_11)

            return v3_00, v3_01, v3_10, v3_11
    else:
        def impl(v1, v2):
            v1_00, v1_11 = unpack(v1)
            v2_00, v2_11 = unpack(v2)

            v3_00 = v1_00*v2_00
            v3_11 = v1_11*v2_11

            return v3_00, v3_11
    return qcjit(impl)


def v1_imul_v2_factory(mode):

    unpack = unpack_factory(mode)

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(v1, v2, o1):
            v1_00, v1_01, v1_10, v1_11 = unpack(v1)
            v2_00, v2_01, v2_10, v2_11 = unpack(v2)

            o1[0] = (v1_00*v2_00 + v1_01*v2_10)
            o1[1] = (v1_00*v2_01 + v1_01*v2_11)
            o1[2] = (v1_10*v2_00 + v1_11*v2_10)
            o1[3] = (v1_10*v2_01 + v1_11*v2_11)
    else:
        def impl(v1, v2, o1):
            v1_00, v1_11 = unpack(v1)
            v2_00, v2_11 = unpack(v2)

            o1[0] = v1_00*v2_00
            o1[1] = v1_11*v2_11
    return qcjit(impl)


def v1_mul_v2ct_factory(mode):

    unpack = unpack_factory(mode)
    unpackct = unpackct_factory(mode)

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(v1, v2):
            v1_00, v1_01, v1_10, v1_11 = unpack(v1)
            v2_00, v2_01, v2_10, v2_11 = unpackct(v2)

            v3_00 = (v1_00*v2_00 + v1_01*v2_10)
            v3_01 = (v1_00*v2_01 + v1_01*v2_11)
            v3_10 = (v1_10*v2_00 + v1_11*v2_10)
            v3_11 = (v1_10*v2_01 + v1_11*v2_11)

            return v3_00, v3_01, v3_10, v3_11
    else:
        def impl(v1, v2):
            v1_00, v1_11 = unpack(v1)
            v2_00, v2_11 = unpackct(v2)

            v3_00 = v1_00*v2_00
            v3_11 = v1_11*v2_11

            return v3_00, v3_11
    return qcjit(impl)


def v1_imul_v2ct_factory(mode):

    unpack = unpack_factory(mode)
    unpackct = unpackct_factory(mode)

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(v1, v2, o1):
            v1_00, v1_01, v1_10, v1_11 = unpack(v1)
            v2_00, v2_01, v2_10, v2_11 = unpackct(v2)

            o1[0] = (v1_00*v2_00 + v1_01*v2_10)
            o1[1] = (v1_00*v2_01 + v1_01*v2_11)
            o1[2] = (v1_10*v2_00 + v1_11*v2_10)
            o1[3] = (v1_10*v2_01 + v1_11*v2_11)
    else:
        def impl(v1, v2, o1):
            v1_00, v1_11 = unpack(v1)
            v2_00, v2_11 = unpackct(v2)

            o1[0] = v1_00*v2_00
            o1[1] = v1_11*v2_11
    return qcjit(impl)


def v1ct_mul_v2_factory(mode):

    unpack = unpack_factory(mode)
    unpackct = unpackct_factory(mode)

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(v1, v2):
            v1_00, v1_01, v1_10, v1_11 = unpackct(v1)
            v2_00, v2_01, v2_10, v2_11 = unpack(v2)

            v3_00 = (v1_00*v2_00 + v1_01*v2_10)
            v3_01 = (v1_00*v2_01 + v1_01*v2_11)
            v3_10 = (v1_10*v2_00 + v1_11*v2_10)
            v3_11 = (v1_10*v2_01 + v1_11*v2_11)

            return v3_00, v3_01, v3_10, v3_11
    else:
        def impl(v1, v2):
            v1_00, v1_11 = unpackct(v1)
            v2_00, v2_11 = unpack(v2)

            v3_00 = v1_00*v2_00
            v3_11 = v1_11*v2_11

            return v3_00, v3_11
    return qcjit(impl)


def v1ct_imul_v2_factory(mode):

    unpack = unpack_factory(mode)
    unpackct = unpackct_factory(mode)

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(v1, v2, o1):
            v1_00, v1_01, v1_10, v1_11 = unpackct(v1)
            v2_00, v2_01, v2_10, v2_11 = unpack(v2)

            o1[0] = (v1_00*v2_00 + v1_01*v2_10)
            o1[1] = (v1_00*v2_01 + v1_01*v2_11)
            o1[2] = (v1_10*v2_00 + v1_11*v2_10)
            o1[3] = (v1_10*v2_01 + v1_11*v2_11)
    else:
        def impl(v1, v2, o1):
            v1_00, v1_11 = unpackct(v1)
            v2_00, v2_11 = unpack(v2)

            o1[0] = v1_00*v2_00
            o1[1] = v1_11*v2_11
    return qcjit(impl)


def iwmul_factory(mode):

    unpack = unpack_factory(mode)

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(v1, w1):
            w1_00, w1_01, w1_10, w1_11 = unpack(w1)

            v1[0] *= w1_00
            v1[1] *= w1_00
            v1[2] *= w1_11
            v1[3] *= w1_11
    else:
        def impl(v1, w1):
            w1_00, w1_11 = unpack(w1)

            v1[0] *= w1_00
            v1[1] *= w1_11
    return qcjit(impl)


def v1_wmul_v2ct_factory(mode):

    unpack = unpack_factory(mode)
    unpackct = unpackct_factory(mode)

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(v1, v2, w1):
            v1_00, v1_01, v1_10, v1_11 = unpack(v1)
            v2_00, v2_01, v2_10, v2_11 = unpackct(v2)
            w1_00, w1_01, w1_10, w1_11 = unpack(w1)

            v3_00 = (v1_00*w1_00*v2_00 + v1_01*w1_11*v2_10)
            v3_01 = (v1_00*w1_00*v2_01 + v1_01*w1_11*v2_11)
            v3_10 = (v1_10*w1_00*v2_00 + v1_11*w1_11*v2_10)
            v3_11 = (v1_10*w1_00*v2_01 + v1_11*w1_11*v2_11)

            return v3_00, v3_01, v3_10, v3_11
    else:
        def impl(v1, v2, w1):
            v1_00, v1_11 = unpack(v1)
            v2_00, v2_11 = unpackct(v2)
            w1_00, w1_11 = unpack(w1)

            v3_00 = v1_00*w1_00*v2_00
            v3_11 = v1_11*w1_11*v2_11

            return v3_00, v3_11
    return qcjit(impl)


def v1ct_wmul_v2_factory(mode):

    unpack = unpack_factory(mode)
    unpackct = unpackct_factory(mode)

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(v1, v2, w1):
            v1_00, v1_01, v1_10, v1_11 = unpackct(v1)
            v2_00, v2_01, v2_10, v2_11 = unpack(v2)
            w1_00, w1_01, w1_10, w1_11 = unpack(w1)

            v3_00 = (v1_00*w1_00*v2_00 + v1_01*w1_11*v2_10)
            v3_01 = (v1_00*w1_00*v2_01 + v1_01*w1_11*v2_11)
            v3_10 = (v1_10*w1_00*v2_00 + v1_11*w1_11*v2_10)
            v3_11 = (v1_10*w1_00*v2_01 + v1_11*w1_11*v2_11)

            return v3_00, v3_01, v3_10, v3_11
    else:
        def impl(v1, v2, w1):
            v1_00, v1_11 = unpackct(v1)
            v2_00, v2_11 = unpack(v2)
            w1_00, w1_11 = unpack(w1)

            v3_00 = v1_00*w1_00*v2_00
            v3_11 = v1_11*w1_11*v2_11

            return v3_00, v3_11
    return qcjit(impl)


def unpack_factory(mode):

    if mode.literal_value == "full":
        def impl(invec):
            return invec[0], invec[1], invec[2], invec[3]
    elif mode.literal_value == "diag":
        def impl(invec):
            return invec[0], invec[1]
    else:
        def impl(invec):
            if len(invec) == 4:
                return invec[0], invec[1], invec[2], invec[3]
            else:
                return invec[0], 0, 0, invec[1]
    return qcjit(impl)


def unpackct_factory(mode):

    if mode.literal_value == "full":
        def impl(invec):
            return np.conjugate(invec[0]), \
                   np.conjugate(invec[2]), \
                   np.conjugate(invec[1]), \
                   np.conjugate(invec[3])
    elif mode.literal_value == "diag":
        def impl(invec):
            return np.conjugate(invec[0]), \
                   np.conjugate(invec[1])
    else:
        def impl(invec):
            if len(invec) == 4:
                return np.conjugate(invec[0]), \
                       np.conjugate(invec[2]), \
                       np.conjugate(invec[1]), \
                       np.conjugate(invec[3])
            else:
                return np.conjugate(invec[0]), \
                       0, \
                       0, \
                       np.conjugate(invec[1])
    return qcjit(impl)


def iunpack_factory(mode):

    if mode.literal_value == "full":
        def impl(outvec, invec):
            outvec[0] = invec[0]
            outvec[1] = invec[1]
            outvec[2] = invec[2]
            outvec[3] = invec[3]
    elif mode.literal_value == "diag":
        def impl(outvec, invec):
            outvec[0] = invec[0]
            outvec[1] = invec[1]
    else:
        def impl(outvec, invec):
            if len(invec) == 4:
                outvec[0] = invec[0]
                outvec[1] = invec[1]
                outvec[2] = invec[2]
                outvec[3] = invec[3]
            else:
                outvec[0] = invec[0]
                outvec[1] = 0
                outvec[2] = 0
                outvec[3] = invec[1]
    return qcjit(impl)


def iunpackct_factory(mode):

    if mode.literal_value == "full":
        def impl(outvec, invec):
            outvec[0] = np.conjugate(invec[0])
            outvec[1] = np.conjugate(invec[2])
            outvec[2] = np.conjugate(invec[1])
            outvec[3] = np.conjugate(invec[3])
    elif mode.literal_value == "diag":
        def impl(outvec, invec):
            outvec[0] = np.conjugate(invec[0])
            outvec[1] = np.conjugate(invec[1])
    else:
        def impl(outvec, invec):
            if len(invec) == 4:
                outvec[0] = np.conjugate(invec[0])
                outvec[1] = np.conjugate(invec[2])
                outvec[2] = np.conjugate(invec[1])
                outvec[3] = np.conjugate(invec[3])
            else:
                outvec[0] = np.conjugate(invec[0])
                outvec[1] = 0
                outvec[2] = 0
                outvec[3] = np.conjugate(invec[1])
    return qcjit(impl)


def iadd_factory(mode):

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(outvec, invec):
            outvec[0] += invec[0]
            outvec[1] += invec[1]
            outvec[2] += invec[2]
            outvec[3] += invec[3]
    else:
        def impl(outvec, invec):
            outvec[0] += invec[0]
            outvec[1] += invec[1]
    return qcjit(impl)


def valloc_factory(mode):
    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(dtype, leading_dims=()):
            return np.empty((*leading_dims, 4), dtype=dtype)
    else:
        def impl(dtype, leading_dims=()):
            return np.empty((*leading_dims, 2), dtype=dtype)
    return qcjit(impl)


def loop_var_factory(mode):
    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(n_gains, active_term):
            all_terms = np.arange(n_gains - 1, -1, -1)
            gt_active = np.arange(n_gains - 1, active_term, -1)
            lt_active = np.arange(active_term)
            return all_terms, gt_active, lt_active
    else:
        def impl(n_gains, active_term):
            all_terms = np.arange(n_gains - 1, -1, -1)
            gt_active = np.where(np.arange(n_gains) != active_term)[0]
            lt_active = np.arange(0)
            return all_terms, gt_active, lt_active
    return qcjit(impl)


def compute_det_factory(mode):
    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(v1):
            return v1[0]*v1[3] - v1[1]*v1[2]
    else:
        def impl(v1):
            return v1[0]*v1[1]
    return qcjit(impl)


def iinverse_factory(mode):

    unpack = unpack_factory(mode)

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(v1, det, o1):
            v1_00, v1_01, v1_10, v1_11 = unpack(v1)

            o1[0] = v1_11/det
            o1[1] = -v1_01/det
            o1[2] = -v1_10/det
            o1[3] = v1_00/det
    else:
        def impl(v1, det, o1):
            v1_00, v1_11 = unpack(v1)

            o1[0] = v1_11/det
            o1[1] = v1_00/det
    return qcjit(impl)


def set_identity_factory(mode):

    if mode.literal_value == "full" or mode.literal_value == "mixed":
        def impl(v1):
            v1[0] = 1
            v1[1] = 0
            v1[2] = 0
            v1[3] = 1
    else:
        def impl(v1):
            v1[0] = 1
            v1[1] = 1
    return qcjit(impl)
