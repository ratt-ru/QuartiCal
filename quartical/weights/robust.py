import numpy as np
from numba import generated_jit, jit
import quartical.gains.general.factories as factories


qcgjit = generated_jit(nopython=True,
                       fastmath=True,
                       cache=True,
                       nogil=True)

qcjit = jit(nopython=True,
            fastmath=True,
            cache=True,
            nogil=True)


@qcgjit
def compute_inv_covariance(residuals, weights, flags, mode):

    compute_covariance_inner = compute_covariance_inner_factory(mode)

    def impl(residuals, weights, flags, mode):

        n_row, n_chan, n_corr = residuals.shape

        # NOTE: We don't bother with the off diagonal entries.
        covariance = np.zeros(n_corr, dtype=np.float64)
        inv_covariance = np.zeros_like(covariance)

        n_unflagged = n_row * n_chan

        for r in range(n_row):
            for f in range(n_chan):
                if flags[r, f]:
                    n_unflagged -= 1
                    continue
                else:
                    compute_covariance_inner(residuals[r, f],
                                             weights[r, f],
                                             covariance)

        if n_unflagged:
            covariance /= n_unflagged
            inv_covariance[:] = 1/covariance

        return inv_covariance

    return impl


def compute_covariance_inner_factory(mode):

    unpack = factories.unpack_factory(mode)
    unpackc = factories.unpackc_factory(mode)

    if mode.literal_value == 4:
        def impl(res, w, cov):
            r0, r1, r2, r3 = unpack(res)
            r0c, r1c, r2c, r3c = unpackc(res)
            w0, w1, w2, w3 = unpack(w)

            cov[0] += (r0 * w0 * r0c).real
            cov[1] += (r1 * w1 * r1c).real
            cov[2] += (r2 * w2 * r2c).real
            cov[3] += (r3 * w3 * r3c).real
    elif mode.literal_value == 2:
        def impl(res, w, cov):
            r0, r1 = unpack(res)
            r0c, r1c = unpackc(res)
            w0, w1 = unpack(w)

            cov[0] += (r0 * w0 * r0c).real
            cov[1] += (r1 * w1 * r1c).real
    elif mode.literal_value == 1:
        def impl(res, w, cov):
            r0 = unpack(res)
            r0c = unpackc(res)
            w0 = unpack(w)

            cov[0] += (r0 * w0 * r0c).real
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


@qcgjit
def update_weights(residuals, weights, flags, inv_covariance, dof, mode):

    update_weights_inner = update_weights_inner_factory(mode)

    def impl(residuals, weights, flags, inv_covariance, dof, mode):

        n_row, n_chan, n_corr = residuals.shape

        for r in range(n_row):
            for f in range(n_chan):
                if flags[r, f]:
                    continue
                else:
                    numerator = dof + n_corr  # Not correct for diagonal terms.
                    denominator = dof + update_weights_inner(residuals[r, f],
                                                             inv_covariance)
                    weights[r, f] = numerator/denominator

    return impl


def update_weights_inner_factory(mode):

    unpack = factories.unpack_factory(mode)
    unpackc = factories.unpackc_factory(mode)

    if mode.literal_value == 4:
        def impl(res, inv_cov):
            r0, r1, r2, r3 = unpack(res)
            r0c, r1c, r2c, r3c = unpackc(res)
            ic0, ic1, ic2, ic3 = unpack(inv_cov)

            return (r0 * ic0 * r0c).real + \
                   (r1 * ic1 * r1c).real + \
                   (r2 * ic2 * r2c).real + \
                   (r3 * ic3 * r3c).real
    elif mode.literal_value == 2:
        def impl(res, inv_cov):
            r0, r1 = unpack(res)
            r0c, r1c = unpackc(res)
            ic0, ic1 = unpack(inv_cov)

            return (r0 * ic0 * r0c).real + (r1 * ic1 * r1c).real
    elif mode.literal_value == 1:
        def impl(res, inv_cov):
            r0 = unpack(res)
            r0c = unpackc(res)
            ic0 = unpack(inv_cov)

            return (r0 * ic0 * r0c).real
    else:
        raise ValueError("Unsupported number of correlations.")

    return qcjit(impl)


@qcjit
def digamma(x):

    result = 0
    while x <= 6:
        result -= 1/x  # Recurrence relation, gamma(x) = gamma(x+1) - 1/x
        x += 1
    result += np.log(x) - 1/(2*x)

    coeffs = (-1/12, 1/120, -1/252, 1/240, -1/132, 391/32760, -1/12)
    ix = 1/(x*x)
    for c in coeffs:
        result += c*ix
        ix *= ix

    return result


@qcjit
def dof_variable(dof):
    return np.log(dof) - digamma(dof)


@qcjit
def dof_constant(weights, flags, dof):

    n_row, n_chan, n_corr = weights.shape

    n_unflagged = n_row * n_chan

    constant = 0
    for r in range(n_row):
        for f in range(n_chan):
            if flags[r, f]:
                n_unflagged -= 1
                continue
            else:
                w = weights[r, f, 0]  # We have the same weight on all corrs.
                constant += np.log(w) - w

    if n_unflagged:
        constant /= n_unflagged

    return constant + 1 + digamma(dof + n_corr) - np.log(dof + n_corr)


@qcjit
def compute_dof(weights, flags, dof):

    constant = dof_constant(weights, flags, dof)

    left = 1
    right = 30

    maxiter = 50
    tol = 1e-8
    for _ in range(maxiter):
        mid = (left + right)/2
        mid_value = dof_variable(mid) + constant
        if mid_value == 0 or ((right - left) < tol):
            break
        left_value = dof_variable(left) + constant
        if np.sign(mid_value) != np.sign(left_value):
            right = mid
        else:
            left = mid

    return mid
