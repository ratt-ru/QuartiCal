# -*- coding: utf-8 -*-
import numpy as np
import quartical.gains.general.factories as factories
from quartical.gains.general.generics import (
    native_intermediaries,
    upsampled_intermediaries,
    per_array_jhj_jhr,
    resample_solints,
    downsample_jhj_jhr,
    scalar_jhj_jhr,
)
from quartical.gains.general.flagging import (
    flag_intermediaries,
    update_gain_flags,
    finalize_gain_flags,
    apply_gain_flags_to_flag_col,
    update_param_flags,
)
from quartical.gains.general.convenience import get_extents
from quartical.gains.general.solver_components import compute_update


# CACHE CORRECTNESS CONSTRAINT: the builders below return the solver-loop body
# as a factories.qcjit (inline="always") function. Every kernel presents this
# body with an identical argument signature, so if it were lowered as its own
# cache unit all kernels would silently share a single on-disk cache file
# (numba keys the cache on source location plus argument types, and the only
# discriminator - the captured hook closures - enters the key merely as an
# unstable cloudpickle hash). Returning an inline="always" function prevents
# the body from ever being lowered standalone: each kernel's
# nb_<term>_solver_impl MUST return a module-local trampoline
# (def impl(...): return shared(...)) that inlines it, giving every kernel a
# private cache namespace. This exact bug shipped once; the canonical
# explanation lives in the docstring of build_jhj_jhr_impl in
# solver_components.py - read it before touching this file.


@factories.qcjit
def identity_dims(shape):
    """Return the upsampled solver shape unchanged.

    A diagonal term stores jhj with the gain shape (a flat correlation vector
    rather than a (corr, corr) block), so its ``get_jhj_dims`` hook is the
    identity.
    """
    return shape


def build_gain_solver_impl(
    *,
    get_jhj_dims,
    compute_jhj_jhr,
    collapse_to_scalar_jhj_jhr,
    scalar_error_message,
    finalize_update,
    reference_gains,
):
    """Return the shared solver-loop impl closure for a non-parameterised term.

    This is the outer solver loop used by the non-parameterised terms: it sets
    up the flagging/solving intermediaries, resamples the solution intervals
    for parallelism, then iterates compute_jhj_jhr -> compute_update ->
    finalize_update -> convergence check until convergence or the iteration
    limit. The per-term maths lives entirely in the hooks passed here; the
    loop itself is identical across the complex, leakage and diagonal-complex
    terms.

    All ``None`` hooks are resolved to build-time no-ops (or, for the scalar
    stage, to one of two prebuilt step closures) so that the compiled body
    never carries a runtime branch for an absent hook - mirroring how
    solver_components.py substitutes its optional
    compute_channel_coeffs/mirror_jhj hooks.

    See the module-level CACHE CORRECTNESS CONSTRAINT: the returned body is an
    inline="always" function and MUST be inlined into a per-kernel trampoline.

    Args:
        get_jhj_dims: A qcjit closure ``get_jhj_dims(upsample_shape) -> dims``
            giving the jhj allocation shape. Full terms pass the closure
            returned by their per-corr ``get_jhj_dims_factory``; diagonal terms
            pass the module-level ``identity_dims`` closure directly, as their
            jhj is gain-shaped.
        compute_jhj_jhr: The kernel module's @overload-ed compute_jhj_jhr;
            called at the top of each iteration to accumulate jhj and jhr.
        collapse_to_scalar_jhj_jhr: Optional hook
            ``collapse_to_scalar_jhj_jhr(native_imdry)`` collapsing jhj/jhr to
            a scalar solve. Note that this is the term's own single-argument
            routine, not the two-argument ``scalar_jhj_jhr`` imported from
            generics for the parameterised builder. ``None`` means scalar mode
            is unsupported for this term, in which case
            ``scalar_error_message`` is raised.
        scalar_error_message: The constant message raised when scalar mode is
            requested and ``collapse_to_scalar_jhj_jhr`` is ``None``.
        finalize_update: The kernel module's @overload-ed finalize_update;
            called with the non-param 5-arg form after compute_update.
        reference_gains: Optional hook
            ``reference_gains(chain_inputs, meta_inputs, corr_mode)`` run once
            after finalize_gain_flags, and only when ``meta_inputs.referenced``
            is set. ``None`` yields a build-time no-op, so a term with no
            referencing stage discards that option silently.

    Returns:
        The ``impl`` closure, wrapped as an inline="always" function.
    """

    # Scalar stage: select one of two prebuilt step closures at build time so
    # that the compiled body carries no branch on whether the term supports
    # scalar mode. When collapse_to_scalar_jhj_jhr is None the term raises
    # regardless of corr_mode - the complex and leakage terms have no scalar
    # solve; otherwise it collapses jhj/jhr to a scalar solve, except in the
    # already-scalar single-corr case. The message is the only thing the user
    # sees when they ask an unsupporting term for a scalar solve, so a term
    # which omits it is caught here rather than raising ValueError(None) from
    # inside a compiled kernel.
    if collapse_to_scalar_jhj_jhr is None and scalar_error_message is None:
        raise ValueError(
            "A term with no scalar collapse must state a message."
        )

    if collapse_to_scalar_jhj_jhr is None:
        def scalar_step(native_imdry, scalar, corr_mode):
            if scalar:
                raise ValueError(scalar_error_message)
    else:
        def scalar_step(native_imdry, scalar, corr_mode):
            if scalar and corr_mode != 1:
                collapse_to_scalar_jhj_jhr(native_imdry)
    scalar_step = factories.qcjit(scalar_step)

    # Optional reference-gains stage: a build-time no-op for a term with no
    # referencing routine, else that routine behind the per-term referenced
    # switch. A term without the stage discards the option silently.
    if reference_gains is None:
        def reference_gains_step(chain_inputs, meta_inputs, corr_mode):
            pass
    else:
        def reference_gains_step(chain_inputs, meta_inputs, corr_mode):
            if meta_inputs.referenced:
                reference_gains(chain_inputs, meta_inputs, corr_mode)
    reference_gains_step = factories.qcjit(reference_gains_step)

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
        n_thread = meta_inputs.threads

        active_gain = gains[active_term]
        active_gain_flags = gain_flags[active_term]

        # Set up some intermediaries used for flagging. TODO: Move?
        km1_gain = active_gain.copy()
        km1_abs2_diffs = np.zeros_like(active_gain_flags, dtype=np.float64)
        abs2_diffs_trend = np.zeros_like(active_gain_flags, dtype=np.float64)
        flag_imdry = flag_intermediaries(
            km1_gain, km1_abs2_diffs, abs2_diffs_trend
        )

        # Set up some intermediaries used for solving.
        complex_dtype = active_gain.dtype
        gain_shape = active_gain.shape

        active_t_map_g = mapping_inputs.time_maps[active_term]
        active_f_map_g = mapping_inputs.freq_maps[active_term]

        # Create more work to do in parallel when needed, else no-op.
        resampler = resample_solints(active_t_map_g, gain_shape, n_thread)

        # Determine the starts and stops of the rows and channels associated
        # with each solution interval.
        extents = get_extents(resampler.upsample_t_map, active_f_map_g)

        upsample_shape = resampler.upsample_shape
        upsampled_jhj = np.empty(get_jhj_dims(upsample_shape),
                                 dtype=complex_dtype)
        upsampled_jhr = np.empty(upsample_shape, dtype=complex_dtype)
        jhj = upsampled_jhj[:gain_shape[0]]
        jhr = upsampled_jhr[:gain_shape[0]]
        update = np.zeros(gain_shape, dtype=complex_dtype)

        upsampled_imdry = upsampled_intermediaries(
            upsampled_jhj, upsampled_jhr
        )
        native_imdry = native_intermediaries(jhj, jhr, update)

        for loop_idx in range(max_iter or 1):

            compute_jhj_jhr(
                ms_inputs,
                mapping_inputs,
                chain_inputs,
                meta_inputs,
                upsampled_imdry,
                extents,
                corr_mode
            )

            if resampler.active:
                downsample_jhj_jhr(upsampled_imdry, resampler.downsample_t_map)

            if solve_per == "array":
                per_array_jhj_jhr(native_imdry)

            # Scalar collapse or unsupported-mode raise (build-time selected).
            scalar_step(native_imdry, scalar, corr_mode)

            if not max_iter:  # Non-solvable term, we just want jhj.
                conv_perc = 0  # Didn't converge.
                loop_idx = -1  # Did zero iterations.
                break

            compute_update(native_imdry, corr_mode)

            finalize_update(
                chain_inputs,
                meta_inputs,
                native_imdry,
                loop_idx,
                corr_mode
            )

            # Check for gain convergence. Produced as a side effect of
            # flagging. The converged percentage is based on unflagged
            # intervals.
            conv_perc = update_gain_flags(
                chain_inputs,
                meta_inputs,
                flag_imdry,
                loop_idx,
                corr_mode
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

        # Optional referencing stage (build-time no-op when absent).
        reference_gains_step(chain_inputs, meta_inputs, corr_mode)

        # Call this one last time to ensure points flagged by finalize are
        # propagated (in the DI case).
        if not dd_term:
            apply_gain_flags_to_flag_col(
                ms_inputs,
                mapping_inputs,
                chain_inputs,
                meta_inputs
            )

        return native_imdry.jhj, loop_idx + 1, conv_perc

    # Return the loop as an inline="always" function so it is never lowered
    # as a standalone (separately disk-cached) unit - see the module-level
    # cache correctness constraint. Each kernel inlines this into a
    # module-local trampoline, giving it a private cache namespace.
    return factories.qcjit(impl)


def build_param_solver_impl(
    *,
    pre_solve,
    compute_jhj_jhr,
    params_per_corr,
    scalar_error_message,
    finalize_update,
    numbness,
    identity_params,
    reference_params,
    post_solve,
):
    """Return the shared solver-loop impl closure for a parameterised term.

    This mirrors build_gain_solver_impl but carries the extra plumbing every
    parameterised term needs: it solves on the parameter grid, allocates a real
    (n_param, n_param) jhj, forwards a numbness value and per-corr identity to
    the flag machinery, propagates gain flags to parameter flags each
    iteration, and exposes optional pre/post-solve stages for terms that enter
    and leave a scaled solver basis (as delay does when it rescales its
    parameters).

    As with the non-param builder, all ``None`` hooks resolve to build-time
    no-ops and the scalar stage is one of two prebuilt step closures, so the
    compiled body carries no runtime branch for an absent hook.

    See the module-level CACHE CORRECTNESS CONSTRAINT: the returned body is an
    inline="always" function and MUST be inlined into a per-kernel trampoline.

    Args:
        pre_solve: Optional hook
            ``pre_solve(ms_inputs, chain_inputs, meta_inputs)`` run once
            between intermediary setup and the loop (e.g. entering a scaled
            solver basis by mutating params in place). ``None`` yields a
            build-time no-op.
        compute_jhj_jhr: The kernel module's @overload-ed compute_jhj_jhr;
            called at the top of each iteration.
        params_per_corr: The number of parameters per correlation, forwarded as
            the second argument to the generic scalar_jhj_jhr. ``None`` means
            scalar mode is unsupported, in which case ``scalar_error_message``
            is raised.
        scalar_error_message: The constant message raised when scalar mode is
            requested and ``params_per_corr`` is ``None``.
        finalize_update: The kernel module's @overload-ed finalize_update;
            called with the standardised 7-arg form after compute_update.
        numbness: The numbness value forwarded to update_gain_flags. Amplitude
            passes 1e-6 (the update_gain_flags default); every other
            parameterised term passes 1e9, which is large enough that the
            divergence test can never trip, so points are never soft-flagged
            for diverging. NB: for the nine terms whose accumulate hook
            consumes the gain, 1e9 also suppresses mid-solve hard flagging,
            which those hooks rely on - see the linearisation-point note in
            solver_components.py before lowering it.
        identity_params: The identity parameter array forwarded to
            update_param_flags, built by ``parameters.get_identity_params``.
        reference_params: Optional @overload-ed referencing routine
            ``reference_params(ms_inputs, mapping_inputs, chain_inputs,
            meta_inputs)`` run once after finalize_gain_flags, and only when
            ``meta_inputs.referenced`` is set. ``None`` yields a build-time
            no-op, so a term with no referencing stage discards that option
            silently.
        post_solve: Optional hook
            ``post_solve(ms_inputs, chain_inputs, meta_inputs, native_imdry)``
            run last, just before the return (e.g. exiting a scaled solver
            basis by unscaling params and jhj in place). ``None`` yields a
            build-time no-op.

            A term solving in the basis p' = Sp has jhj = S jhj' S, so the
            hooks unscale the diagonal blocks of the rescaled parameters and
            leave the blocks coupling those to unrescaled parameters in the
            solver basis. Only the jhj diagonal is consumed - see the reduction
            in calibration/solver.py - so those blocks never reach a caller.

    Returns:
        The ``impl`` closure, wrapped as an inline="always" function.
    """

    # Optional pre-solve stage: a build-time no-op when absent.
    if pre_solve is None:
        def pre_solve_step(ms_inputs, chain_inputs, meta_inputs):
            pass
        pre_solve_step = factories.qcjit(pre_solve_step)
    else:
        pre_solve_step = pre_solve

    # Scalar stage: select one of two prebuilt step closures at build time (see
    # the non-param builder for the rationale). When params_per_corr is None
    # the term raises regardless of corr_mode; otherwise it collapses jhj/jhr
    # to a scalar solve, except in the already-scalar single-corr case.
    if params_per_corr is None and scalar_error_message is None:
        raise ValueError(
            "A term with no scalar collapse must state a message."
        )

    if params_per_corr is None:
        def scalar_step(native_imdry, scalar, corr_mode):
            if scalar:
                raise ValueError(scalar_error_message)
    else:
        def scalar_step(native_imdry, scalar, corr_mode):
            if scalar and corr_mode != 1:
                scalar_jhj_jhr(native_imdry, params_per_corr)
    scalar_step = factories.qcjit(scalar_step)

    # Optional referencing stage: a build-time no-op for a term with no
    # referencing routine, else that routine behind the per-term referenced
    # switch. A term without the stage discards the option silently.
    if reference_params is None:
        def reference_params_step(
            ms_inputs, mapping_inputs, chain_inputs, meta_inputs
        ):
            pass
    else:
        def reference_params_step(
            ms_inputs, mapping_inputs, chain_inputs, meta_inputs
        ):
            if meta_inputs.referenced:
                reference_params(
                    ms_inputs, mapping_inputs, chain_inputs, meta_inputs
                )
    reference_params_step = factories.qcjit(reference_params_step)

    # Optional post-solve stage: a build-time no-op when absent.
    if post_solve is None:
        def post_solve_step(
            ms_inputs, chain_inputs, meta_inputs, native_imdry
        ):
            pass
        post_solve_step = factories.qcjit(post_solve_step)
    else:
        post_solve_step = post_solve

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
        n_thread = meta_inputs.threads

        active_gain = gains[active_term]
        active_gain_flags = gain_flags[active_term]
        active_params = chain_inputs.params[active_term]

        # Set up some intermediaries used for flagging.
        km1_gain = active_gain.copy()
        km1_abs2_diffs = np.zeros_like(active_gain_flags, dtype=np.float64)
        abs2_diffs_trend = np.zeros_like(active_gain_flags, dtype=np.float64)
        flag_imdry = \
            flag_intermediaries(km1_gain, km1_abs2_diffs, abs2_diffs_trend)

        # Set up some intermediaries used for solving.
        real_dtype = active_gain.real.dtype
        param_shape = active_params.shape

        active_t_map_g = mapping_inputs.time_maps[active_term]

        # Parameterised terms always solve on the parameter grid. The parameter
        # frequency map is the base binned map
        # (ParameterizedGain._make_param_freq_map delegates to
        # Gain._make_freq_map), whereas the gain frequency map may be
        # overridden per term - e.g. delay/tec/rotation_measure solve in every
        # channel. jhj/jhr/update are allocated on the parameter shape, so the
        # extents must come from the parameter grid; the gain grid is only ever
        # equal to it (phase, amplitude, ...) or inconsistent with it
        # (delay, ...). See docs/wiki/solver-architecture.md.
        active_f_map_p = mapping_inputs.param_freq_maps[active_term]

        # Create more work to do in parallel when needed, else no-op.
        resampler = resample_solints(active_t_map_g, param_shape, n_thread)

        # Determine the starts and stops of the rows and channels associated
        # with each solution interval.
        extents = get_extents(resampler.upsample_t_map, active_f_map_p)

        upsample_shape = resampler.upsample_shape
        upsampled_jhj = np.empty(upsample_shape + (upsample_shape[-1],),
                                 dtype=real_dtype)
        upsampled_jhr = np.empty(upsample_shape, dtype=real_dtype)
        jhj = upsampled_jhj[:param_shape[0]]
        jhr = upsampled_jhr[:param_shape[0]]
        update = np.zeros(param_shape, dtype=real_dtype)

        upsampled_imdry = upsampled_intermediaries(
            upsampled_jhj, upsampled_jhr
        )
        native_imdry = native_intermediaries(jhj, jhr, update)

        # Optional pre-solve stage, e.g. entering a scaled solver basis by
        # rescaling the parameters in place (build-time no-op when absent).
        pre_solve_step(ms_inputs, chain_inputs, meta_inputs)

        for loop_idx in range(max_iter or 1):

            compute_jhj_jhr(
                ms_inputs,
                mapping_inputs,
                chain_inputs,
                meta_inputs,
                upsampled_imdry,
                extents,
                corr_mode
            )

            if resampler.active:
                downsample_jhj_jhr(upsampled_imdry, resampler.downsample_t_map)

            if solve_per == "array":
                per_array_jhj_jhr(native_imdry)

            # Scalar collapse or unsupported-mode raise (build-time selected).
            scalar_step(native_imdry, scalar, corr_mode)

            if not max_iter:  # Non-solvable term, we just want jhj.
                conv_perc = 0  # Didn't converge.
                loop_idx = -1  # Did zero iterations.
                break

            compute_update(native_imdry, corr_mode)

            finalize_update(
                ms_inputs,
                mapping_inputs,
                chain_inputs,
                meta_inputs,
                native_imdry,
                loop_idx,
                corr_mode
            )

            # Check for gain convergence. Produced as a side effect of
            # flagging. The converged percentage is based on unflagged
            # intervals.
            conv_perc = update_gain_flags(
                chain_inputs,
                meta_inputs,
                flag_imdry,
                loop_idx,
                corr_mode,
                numbness=numbness
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

        # Optional referencing stage (build-time no-op when absent).
        reference_params_step(
            ms_inputs,
            mapping_inputs,
            chain_inputs,
            meta_inputs,
        )

        # Call this one last time to ensure points flagged by finalize are
        # propagated (in the DI case).
        if not dd_term:
            apply_gain_flags_to_flag_col(
                ms_inputs,
                mapping_inputs,
                chain_inputs,
                meta_inputs
            )

        # Optional post-solve stage, e.g. exiting the scaled solver basis by
        # unscaling params and jhj in place (build-time no-op when absent).
        post_solve_step(ms_inputs, chain_inputs, meta_inputs, native_imdry)

        return native_imdry.jhj, loop_idx + 1, conv_perc

    # Return the loop as an inline="always" function so it is never lowered
    # as a standalone (separately disk-cached) unit - see the module-level
    # cache correctness constraint. Each kernel inlines this into a
    # module-local trampoline, giving it a private cache namespace.
    return factories.qcjit(impl)
