# -*- coding: utf-8 -*-
from numba import njit
from numba.extending import overload
from quartical.utils.numba import coerce_literal, JIT_OPTIONS
import quartical.gains.general.factories as factories
from quartical.gains.general.solver_loop import build_gain_solver_impl
from quartical.gains.complex.kernel import (get_jhj_dims_factory,
                                            compute_jhj_jhr)


@njit(**JIT_OPTIONS)
def leakage_solver(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    return leakage_solver_impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        corr_mode
    )


def leakage_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):
    raise NotImplementedError


@overload(leakage_solver_impl, jit_options=JIT_OPTIONS)
def nb_leakage_solver_impl(
    ms_inputs,
    mapping_inputs,
    chain_inputs,
    meta_inputs,
    corr_mode
):

    coerce_literal(nb_leakage_solver_impl, ["corr_mode"])

    # Leakage reuses the (full) complex accumulation via the complex kernel's
    # compute_jhj_jhr, and shares the non-parameterised outer solver loop -
    # only the finalize_update hook (which zeroes the diagonal of the update)
    # and the scalar error string are specific to leakage terms. The shared
    # loop is inlined into the module-local trampoline below rather than
    # returned directly. This gives leakage a private on-disk cache namespace -
    # see the cache correctness constraint in solver_components.py.
    shared_impl = build_gain_solver_impl(
        get_jhj_dims=get_jhj_dims_factory(corr_mode),
        compute_jhj_jhr=compute_jhj_jhr,
        collapse_to_scalar_jhj_jhr=None,
        scalar_error_message="Scalar mode not supported for leakage terms.",
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

                        upd[0] = 0
                        upd[3] = 0

                        if fl == 1:
                            set_identity(g)
                        elif dd_term or (loop_idx % 2 == 0):
                            upd /= 2
                            g += upd
                        else:
                            g += upd

    return impl
