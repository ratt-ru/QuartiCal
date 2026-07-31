# -*- coding: utf-8 -*-
"""The stock compute_residual hooks of the shared accumulation loop.

``build_jhj_jhr_impl`` (``solver_components.py``) asks each term for a
``compute_residual(r, v)`` hook, where ``r`` is the residual carried by the
rest of the chain and ``v`` is the active term's model contribution. Three
implementations cover every gain type:

* :func:`standard_residual_factory` - ``r - v``, for terms which constrain a
  full complex gain or a real rotation angle.
* :func:`phase_only_residual_factory` - rescales ``r`` to the amplitude of
  ``v``, so only the phase discrepancy reaches the normal equations.
* :func:`amplitude_only_residual_factory` - rescales ``v`` to the amplitude of
  ``r``, so only the amplitude discrepancy reaches them.

These are per-visibility hooks inlined into the accumulation loop, distinct
from ``generics.compute_residual``, which forms the residual of an entire
chain over a whole chunk.
"""
import quartical.gains.general.factories as factories


def standard_residual_factory(corr_mode):
    """Produce the unmodified residual tuple.

    Args:
        corr_mode: A numba literal holding the number of correlations.

    Returns:
        A qcjit ``compute_residual(r, v)`` returning the per-correlation
        ``r - v`` tuple.
    """

    tuple_sub = factories.tuple_sub_factory(corr_mode)

    def impl(r, v):
        return tuple_sub(r, v)

    return factories.qcjit(impl)


def phase_only_residual_factory(corr_mode):
    """Produce the residual tuple with the residual amplitude normalised out.

    The per-correlation factor is normf_i = |v_i| / |r_i| (zero where r_i is
    zero, matching absv1_idiv_absv2), giving r_i*normf_i - v_i: the phase
    discrepancy carried at the model amplitude.

    Args:
        corr_mode: A numba literal holding the number of correlations.

    Returns:
        A qcjit ``compute_residual(r, v)`` returning the per-correlation
        residual tuple.

    Raises:
        ValueError: If corr_mode is not 1, 2 or 4.
    """

    tuple_normf = factories.tuple_normf_factory(corr_mode)

    if corr_mode.literal_value == 4:
        def impl(r, v):
            f0, f1, f2, f3 = tuple_normf(v, r)
            return (
                r[0]*f0 - v[0],
                r[1]*f1 - v[1],
                r[2]*f2 - v[2],
                r[3]*f3 - v[3],
            )
    elif corr_mode.literal_value == 2:
        def impl(r, v):
            f0, f1 = tuple_normf(v, r)
            return (
                r[0]*f0 - v[0],
                r[1]*f1 - v[1],
            )
    elif corr_mode.literal_value == 1:
        def impl(r, v):
            f0, = tuple_normf(v, r)
            return (
                r[0]*f0 - v[0],
            )
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)


def amplitude_only_residual_factory(corr_mode):
    """Produce the residual tuple with the residual phase normalised out.

    The per-correlation factor is normf_i = |r_i| / |v_i| (zero where v_i is
    zero, matching absv1_idiv_absv2), giving normf_i*v_i - v_i: the amplitude
    discrepancy carried along the model phase.

    Args:
        corr_mode: A numba literal holding the number of correlations.

    Returns:
        A qcjit ``compute_residual(r, v)`` returning the per-correlation
        residual tuple.

    Raises:
        ValueError: If corr_mode is not 1, 2 or 4.
    """

    tuple_normf = factories.tuple_normf_factory(corr_mode)

    if corr_mode.literal_value == 4:
        def impl(r, v):
            f0, f1, f2, f3 = tuple_normf(r, v)
            return (
                f0*v[0] - v[0],
                f1*v[1] - v[1],
                f2*v[2] - v[2],
                f3*v[3] - v[3],
            )
    elif corr_mode.literal_value == 2:
        def impl(r, v):
            f0, f1 = tuple_normf(r, v)
            return (
                f0*v[0] - v[0],
                f1*v[1] - v[1],
            )
    elif corr_mode.literal_value == 1:
        def impl(r, v):
            f0, = tuple_normf(r, v)
            return (
                f0*v[0] - v[0],
            )
    else:
        raise ValueError("Unsupported number of correlations.")

    return factories.qcjit(impl)
