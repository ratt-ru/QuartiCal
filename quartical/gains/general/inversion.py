import numpy as np
import quartical.gains.general.factories as factories
from collections import namedtuple

buffers = namedtuple("buffers", "Ap Ax p r")


def inversion_buffer_factory(generalised=False):

    if generalised:
        def impl(n_param, dtype):

            r = np.zeros(n_param, dtype=dtype)
            p = np.zeros(n_param, dtype=dtype)
            Ap = np.zeros(n_param, dtype=dtype)
            Ax = np.zeros(n_param, dtype=dtype)

            return buffers(Ap, Ax, p, r)
    else:
        def impl(n_param, dtype):
            return

    return factories.qcjit(impl)


def invert_factory(corr_mode, generalised=False):

    if generalised:

        mat_mul_vec = mat_mul_vec_factory(corr_mode)
        vecct_mul_vec = vecct_mul_vec_factory(corr_mode)
        vec_iadd_svec = vec_iadd_svec_factory(corr_mode)
        vec_isub_svec = vec_isub_svec_factory(corr_mode)

        def impl(A, b, x, buffers):

            Ap, Ax, p, r = buffers

            mat_mul_vec(A, x, Ax)
            r[:] = b
            r -= Ax
            p[:] = r
            r_k = vecct_mul_vec(r, r)

            # If the resdiual is exactly zero, x is exact or missing.
            if r_k == 0:
                return

            for _ in range(x.size):
                mat_mul_vec(A, p, Ap)
                alpha_denom = vecct_mul_vec(p, Ap)
                alpha = r_k / alpha_denom
                vec_iadd_svec(x, alpha, p)
                vec_isub_svec(r, alpha, Ap)
                r_kplus1 = vecct_mul_vec(r, r)
                if r_kplus1.real < 1e-30:
                    break
                p *= (r_kplus1 / r_k)
                p += r
                r_k = r_kplus1

    else:

        v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
        compute_det = factories.compute_det_factory(corr_mode)
        iinverse = factories.iinverse_factory(corr_mode)

        def impl(A, b, x, buffers):

            det = compute_det(A)

            if det.real < 1e-6:
                x[:] = 0
            else:
                iinverse(A, det, x)

            v1_imul_v2(b, x, x)

    return factories.qcjit(impl)


def mat_mul_vec_factory(corr_mode):

    def impl(mat, vec, out):

        n_row, n_col = mat.shape

        out[:] = 0

        for i in range(n_row):
            for j in range(n_col):
                out[i] += mat[i, j] * vec[j]

    return factories.qcjit(impl)


def vecct_mul_mat_factory(corr_mode):

    def impl(vec, mat, out):

        n_row, n_col = mat.shape

        out[:] = 0

        for i in range(n_col):
            for j in range(n_row):
                out[i] += vec[i].conjugate() * mat[i, j]

    return factories.qcjit(impl)


def vecct_mul_vec_factory(corr_mode):

    def impl(vec1, vec2):

        n_ele = vec1.size

        out = 0

        for i in range(n_ele):
            out += vec1[i].conjugate() * vec2[i]

        return out

    return factories.qcjit(impl)


def diag_add_factory(corr_mode):

    def impl(mat, scalar, out):

        n_ele, _ = mat.shape

        out[:] = mat

        for i in range(n_ele):
            out[i, i] += scalar

    return factories.qcjit(impl)


def vec_iadd_svec_factory(corr_mode):

    def impl(vec1, scalar, vec2):

        n_ele = vec1.size

        for i in range(n_ele):
            vec1[i] += scalar * vec2[i]

    return factories.qcjit(impl)


def vec_isub_svec_factory(corr_mode):

    def impl(vec1, scalar, vec2):

        n_ele = vec1.size

        for i in range(n_ele):
            vec1[i] -= scalar * vec2[i]

    return factories.qcjit(impl)
