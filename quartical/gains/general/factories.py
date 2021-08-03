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

def rb_weight_mult(mode):
    unpack = unpack_factory(mode)
    unpackct = unpackct_factory(mode)
    if mode.literal_value == "full":
        def impl(r, ic):
            r00, r01, r10, r11 = unpack(r)
            rc00, rc01, rc10, rc11 = unpackct(r)
            denom = (rc00*ic[0,0] + rc01*ic[1,0] + rc10*ic[2,0] + rc11*ic[3,0])*r00 + \
                            (rc00*ic[0,1] + rc01*ic[1,1] + rc10*ic[2,1] + rc11*ic[3,1])*r01 + \
                            (rc00*ic[0,2] + rc01*ic[1,2] + rc10*ic[2,2] + rc11*ic[3,2])*r10 + \
                            (rc00*ic[0,3] + rc01*ic[1,3] + rc10*ic[2,3] + rc11*ic[3,3])*r11
            return denom.real

    else:
        def impl(r, ic):
            r00, r11 = unpack(r)
            rc00, rc11 = unpackct(r)  
            denom = rc00*ic[0,0]*r00 + rc11*ic[3,3]*r11 
            return denom.real

    return qcjit(impl)

def rb_weight_upd(mode):

    if mode.literal_value == "full":
        def impl(v, denom, o1, robust_thresh, wsum):
            if o1[0].real !=0: 
                w = (v+4)/(v+denom)
                if w<robust_thresh:
                    w = 0
                o1[0] = w
                o1[1] = w
                o1[2] = w
                o1[3] = w
                wsum[0] += w

    else:
        def impl(v, denom, o1, robust_thresh, wsum):
            if o1[0].real !=0:
                w = (v+2)/(v+denom)
                if w < robust_thresh:
                    w = 0
                o1[0] = w
                o1[1] = w
                wsum[0] += w

    return qcjit(impl)

def rb_cov_mult(mode):
    unpack = unpack_factory(mode)
    unpackct = unpackct_factory(mode)

    if mode.literal_value == "full":
        def impl(r, ic, w0):
            r00, r01, r10, r11 = unpack(r)
            rc00, rc01, rc10, rc11 = unpackct(r)
            
            w0r00 = w0*r00
            w0r01 = w0*r01
            w0r10 = w0*r10
            w0r11 = w0*r11

            ic[0,0] += rc00*w0r00
            ic[1,1] += rc01*w0r01
            ic[2,2] += rc10*w0r10
            ic[3,3] += rc11*w0r11
            

    else:
        def impl(r, ic, w0):
            r00, r11 = unpack(r)
            rc00, rc11 = unpackct(r)
            w0r00 = w0*r00
            w0r11 = w0*r11
            ic[0,0] += rc00*w0r00
            ic[3,3] += rc11*w0r11



    return qcjit(impl)