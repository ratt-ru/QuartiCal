# -*- coding: utf-8 -*-
"""The two numerical primitives of a single solver iteration.

The outer solver loop (``solver_loop.py``) drives a Gauss-Newton style
optimisation. Each iteration is built from exactly two operations, both of
which live in this module:

* :func:`build_jhj_jhr_impl` **forms the normal equations** - it builds the
  numba loop that accumulates the weighted Jacobian products JHJ and JHr over
  all solution intervals.
* :func:`compute_update` **solves the normal equations** - it inverts JHJ and
  applies JHr to produce the parameter update.

Everything above these (the iteration, convergence, flagging) lives in
``solver_loop.py``; the shared, hook-parameterised machinery they build on
lives in ``factories.py``, ``inversion.py`` and ``convenience.py``.
"""
from numba import prange
from numba.extending import overload
import quartical.gains.general.factories as factories
from quartical.utils.numba import coerce_literal, PARALLEL_JIT_OPTIONS
from quartical.gains.general.convenience import get_row
from quartical.gains.general.inversion import (invert_factory,
                                               inversion_buffer_factory)


def build_jhj_jhr_impl(
    *,
    corr_mode,
    row_weights_type,
    accumulate_jhj_jhr_factory,
    zero_jhj_jhr_factory,
    flush_jhj_jhr_factory,
    compute_residual_factory,
    compute_channel_coeffs_factory,
    mirror_jhj_factory,
):
    """Return the shared compute_jhj_jhr impl closure, specialised per term.

    This is the register-resident, tuple-based accumulation loop first written
    for the complex kernel (see the c5c1d2a rewrite). It is parameterised by a
    small set of per-term hooks so that every gain solver can share a single
    copy of the (otherwise near-identical) prange over solution intervals, the
    chain-product operator construction, the single-direction fast path, and
    the general multi-direction path. The per-term maths lives entirely in the
    hook closures.

    All hook factories are plain-Python compile-time compositions (the same way
    the complex kernel composes its factories): they take ``corr_mode`` and
    return a ``qcjit``-wrapped closure. Numba only ever sees the final
    specialised closures, so the indirection is free after inlining.

    CACHE CORRECTNESS CONSTRAINT: the built loop is returned as a
    ``factories.qcjit`` (``inline="always"``) function and MUST be inlined into
    a per-kernel-module trampoline (see any kernel's ``nb_compute_jhj_jhr``).
    Numba keys its on-disk cache on the source location plus the argument type
    signature of each separately-lowered function; the hook closures captured
    here only enter the key as a cloudpickle hash which numba itself documents
    as unstable across processes, so it cannot reliably distinguish kernels.
    Every kernel presents this loop with an identical argument signature, so if
    the loop were lowered as its own cache unit (a bare closure returned from
    the overload) all kernels would share a single cache file - a process
    compiling kernel B could then load kernel A's machine code written by an
    earlier session, silently running the wrong maths. Returning an
    ``inline="always"`` function prevents this: it is never lowered as a
    standalone cache unit, and the trampoline that inlines it lives in the
    kernel's own module, giving each kernel a private cache namespace.

    The residual hook returns the per-correlation residual tuple. The
    ``channel_coeffs`` tuple passed to the accumulate hook is exactly the
    per-channel coefficient tuple produced by ``compute_channel_coeffs`` (an
    empty tuple for terms with no channel-coefficient hook).

    LINEARISATION POINT AND FLAGGED GAINS: nine parameterised accumulate hooks
    consume the ``gain`` argument, because their chain rule differentiates the
    active-term gain itself - the delay/tec families and crosshand_phase via
    ``-1j*conj(g)``, rotation and rotation_measure by reading cos/sin out of
    the rotation matrix. Those hooks therefore linearise about the gain rather
    than about the parameters, and the two part company at hard-flagged
    elements: ``set_identity`` forces a hard-flagged gain to the identity (each
    kernel's finalize_update, plus ``update_gain_flags`` and
    ``apply_gain_flags_to_gains`` in general/flagging.py), while
    ``update_param_flags`` only resets a parameter interval whose contributing
    gain intervals are ALL flagged. Where the two grids coincide (phase,
    crosshand_phase, rotation) that reset always fires, so the linearisation
    points agree. The six terms which evaluate their gain in every channel but
    solve on a coarser frequency grid (delay, delay_and_offset, delay_and_tec,
    delay_tec_and_offset, tec_and_offset, rotation_measure) have no such
    protection: one hard-flagged channel would contribute a derivative
    linearised at the identity against a non-zero parameter.

    Nothing reaches that state, but the guard is not local. Initial gain flags
    are the AND of the flag column over every contributing visibility
    (``init_flags``), so a hard-flagged interval has no unflagged data left for
    the row/channel loop below to visit; and mid-solve hard flagging is gated
    by ``numbness``, which every gain-consuming term sets to 1e9. Lowering
    ``numbness`` for any of the six terms above would expose the disagreement.

    INPUT FIELDS READ: ``ms_inputs.{DATA, MODEL_DATA, WEIGHT, FLAG, ANTENNA1,
    ANTENNA2, ROW_MAP, ROW_WEIGHTS}``, ``mapping_inputs.{time, freq,
    dir}_maps``, ``chain_inputs.gains`` and ``meta_inputs.active_term``. This
    set is not private: ``crosshand_phase/null_v_kernel.py`` reuses this loop
    by forging the input namedtuples, substituting the visibilities and
    reversing the chain, so anything read here has to be something it supplies
    correctly. It reverses every per-term field, not only those read today.

    Args:
        corr_mode: Numba literal carrying ``corr_mode.literal_value`` (1/2/4).
        row_weights_type: Numba type of the ``ROW_WEIGHTS`` ms_inputs field,
            used to dispatch the (BDA) row-weight application.
        accumulate_jhj_jhr_factory: ``accumulate_jhj_jhr_factory(corr_mode) ->
            accumulate_jhj_jhr(lop, rop, w, gain, channel_coeffs, wres,
            jhj_jhr) -> jhj_jhr``. Accumulates one weighted jhj/jhr element
            into the register-resident flat accumulator tuple, which packs the
            jhj block first and the jhr block after it.
        zero_jhj_jhr_factory: ``zero_jhj_jhr_factory(corr_mode) ->
            zero_jhj_jhr(ref_elem) -> flat zero tuple``. Produces the zero
            accumulator tuple in the (promoted) dtype of the reference element.
        flush_jhj_jhr_factory: ``flush_jhj_jhr_factory(corr_mode) ->
            flush_jhj_jhr(jhj_el, jhr_el, jhj_jhr) -> None``. Adds a completed
            accumulator into the jhj/jhr array slices.
        compute_residual_factory: ``compute_residual_factory(corr_mode) ->
            compute_residual(r, v) -> tuple`` of the per-correlation residual
            values. Every term uses one of the three implementations in
            ``residuals.py``.
        compute_channel_coeffs_factory: Optional
            ``compute_channel_coeffs_factory(corr_mode) ->
            compute_channel_coeffs(ms_inputs, meta_inputs, f) -> flat coeff
            tuple`` computing the per-channel coefficients that form the
            ``channel_coeffs`` tuple passed to the accumulate hook. ``None``
            yields an empty coefficient tuple.
        mirror_jhj_factory: Optional ``mirror_jhj_factory(corr_mode) ->
            mirror_jhj(jhj_tifi) -> None`` filling the lower triangle of the
            per-interval jhj elements. ``None`` yields a no-op.

    Returns:
        The ``impl`` closure with the standard compute_jhj_jhr runtime
        signature.
    """

    tuple_unpack = factories.tuple_unpack_factory(corr_mode)
    tuple_unpackct = factories.tuple_unpackct_factory(corr_mode)
    tuple_unpack_rweight = factories.tuple_unpack_rweight_factory(
        corr_mode, row_weights_type
    )
    tuple_zeros = factories.tuple_zeros_factory(corr_mode)
    tuple_identity = factories.tuple_identity_factory(corr_mode)
    tuple_add = factories.tuple_add_factory(corr_mode)
    tuple_wmul = factories.tuple_wmul_factory(corr_mode)
    tuple_v1_mul_v2 = factories.tuple_v1_mul_v2_factory(corr_mode)
    tuple_v1_mul_v2ct = factories.tuple_v1_mul_v2ct_factory(corr_mode)
    tuple_v1ct_mul_v2 = factories.tuple_v1ct_mul_v2_factory(corr_mode)
    iunpack = factories.iunpack_factory(corr_mode)
    iadd = factories.iadd_factory(corr_mode)
    valloc = factories.valloc_factory(corr_mode)
    make_loop_vars = factories.loop_var_factory(corr_mode)

    accumulate_jhj_jhr = accumulate_jhj_jhr_factory(corr_mode)
    flush_jhj_jhr = flush_jhj_jhr_factory(corr_mode)
    zero_jhj_jhr = zero_jhj_jhr_factory(corr_mode)
    compute_residual = compute_residual_factory(corr_mode)

    if mirror_jhj_factory is None:
        def mirror_jhj(jhj_tifi):
            pass
        mirror_jhj = factories.qcjit(mirror_jhj)
    else:
        mirror_jhj = mirror_jhj_factory(corr_mode)

    if compute_channel_coeffs_factory is None:
        def compute_channel_coeffs(ms_inputs, meta_inputs, f):
            return ()
        compute_channel_coeffs = factories.qcjit(compute_channel_coeffs)
    else:
        compute_channel_coeffs = compute_channel_coeffs_factory(corr_mode)

    def impl(
        ms_inputs,
        mapping_inputs,
        chain_inputs,
        meta_inputs,
        upsampled_imdry,
        extents,
        corr_mode
    ):

        active_term = meta_inputs.active_term

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

        jhj = upsampled_imdry.jhj
        jhr = upsampled_imdry.jhr

        n_row, n_chan, n_dir, n_corr = model.shape

        jhj[:] = 0
        jhr[:] = 0

        n_tint, n_fint, n_ant, n_gdir, n_corr = jhr.shape
        n_int = n_tint*n_fint

        # In the (very common) single direction case the per-direction
        # accumulators are unnecessary - the loop-invariant flag below lets
        # the compiler produce a fast path which skips them entirely.
        single_dir = (n_dir == 1) and (n_gdir == 1)

        complex_dtype = gains[active_term].dtype

        n_gains = len(gains)

        row_starts = extents.row_starts
        row_stops = extents.row_stops
        chan_starts = extents.chan_starts
        chan_stops = extents.chan_stops

        # Determine loop variables based on where we are in the chain.
        # gt means greater than (n>j) and lt means less than (n<j).
        all_terms, gt_active, lt_active = make_loop_vars(n_gains, active_term)

        active_t_map = time_maps[active_term]
        active_f_map = freq_maps[active_term]
        active_d_map = dir_maps[active_term]

        # Parallel over all solution intervals.
        for i in prange(n_int):

            ti = i//n_fint
            fi = i - ti*n_fint

            rs = row_starts[ti]
            re = row_stops[ti]
            fs = chan_starts[fi]
            fe = chan_stops[fi]

            # Per-direction accumulators - these are the only per-visibility
            # intermediaries which need to live in memory (a direction loop
            # may accumulate several model directions into one gain
            # direction). Everything else is a tuple i.e. register-resident.
            lop_pq_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
            rop_pq_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
            lop_qp_arr = valloc(complex_dtype, leading_dims=(n_gdir,))
            rop_qp_arr = valloc(complex_dtype, leading_dims=(n_gdir,))

            jhr_tifi = jhr[ti, fi]
            jhj_tifi = jhj[ti, fi]

            # Zero/identity tuples in the compute (gain) dtype. Adding the
            # zero tuple is also used to promote lower precision inputs.
            zero_vec = tuple_zeros(jhr_tifi[0, 0])
            identity_vec = tuple_identity(jhr_tifi[0, 0])
            jhj_jhr_zero = zero_jhj_jhr(jhr_tifi[0, 0])

            for row_ind in range(rs, re):

                row = get_row(row_ind, row_map)
                a1_m, a2_m = antenna1[row], antenna2[row]

                if single_dir:

                    # Fast path: a single direction accumulates into a single
                    # jhr/jhj element per antenna. As the antennas are fixed
                    # for the duration of a row, the accumulation can be done
                    # in registers and flushed to memory once per row.
                    jhj_jhr_p = jhj_jhr_zero
                    jhj_jhr_q = jhj_jhr_zero

                    for f in range(fs, fe):

                        if flags[row, f]:  # Skip flagged data points.
                            continue

                        # Per-channel coefficients (staged terms only, else an
                        # empty tuple - the compiler drops it entirely). This
                        # is the channel_coeffs tuple passed to the accumulate
                        # hook.
                        channel_coeffs = compute_channel_coeffs(
                            ms_inputs, meta_inputs, f
                        )

                        # Apply row weights in the BDA case, else a no-op.
                        w = tuple_unpack_rweight(
                            weights[row, f], row_weights, row_ind
                        )
                        r_pq = tuple_unpack(data[row, f])

                        # NOTE: This block is duplicated in the general path
                        # below - a shared helper returning the four operator
                        # tuples trips a numba parfor array analysis bug
                        # (nested tuple returns are misread as array shapes).
                        lop_pq = identity_vec
                        lop_qp = identity_vec

                        # Promote the model to the compute dtype - the zero
                        # add is free but ensures type stability below.
                        rop_qp = tuple_add(model[row, f, 0], zero_vec)
                        rop_pq = tuple_unpackct(rop_qp)

                        for gi in all_terms:

                            d_m = dir_maps[gi][0]  # Broadcast dir.
                            t_m = time_maps[gi][row_ind]
                            f_m = freq_maps[gi][f]

                            gain = gains[gi][t_m, f_m]

                            g_q = tuple_unpack(gain[a2_m, d_m])
                            rop_pq = tuple_v1_mul_v2(g_q, rop_pq)

                            g_p = tuple_unpack(gain[a1_m, d_m])
                            rop_qp = tuple_v1_mul_v2(g_p, rop_qp)

                        for gi in gt_active:

                            d_m = dir_maps[gi][0]
                            t_m = time_maps[gi][row_ind]
                            f_m = freq_maps[gi][f]

                            gain = gains[gi][t_m, f_m]

                            g_p = tuple_unpack(gain[a1_m, d_m])
                            rop_pq = tuple_v1_mul_v2ct(rop_pq, g_p)

                            g_q = tuple_unpack(gain[a2_m, d_m])
                            rop_qp = tuple_v1_mul_v2ct(rop_qp, g_q)

                        for gi in lt_active:

                            d_m = dir_maps[gi][0]
                            t_m = time_maps[gi][row_ind]
                            f_m = freq_maps[gi][f]

                            gain = gains[gi][t_m, f_m]

                            g_p = tuple_unpack(gain[a1_m, d_m])
                            lop_pq = tuple_v1ct_mul_v2(g_p, lop_pq)

                            g_q = tuple_unpack(gain[a2_m, d_m])
                            lop_qp = tuple_v1ct_mul_v2(g_q, lop_qp)

                        # The active-term gain for each antenna. The p-side
                        # gain also builds the model visibility below; both are
                        # passed to the respective accumulate call
                        # (parameterised terms need them for the chain rule -
                        # the complex accumulate hook ignores its gain argument
                        # and the fetch is elided).
                        active_gain_tifi = gains[active_term][
                            active_t_map[row_ind], active_f_map[f]
                        ]
                        g_active_p = tuple_unpack(active_gain_tifi[a1_m, 0])
                        g_active_q = tuple_unpack(active_gain_tifi[a2_m, 0])

                        v_pq = tuple_v1ct_mul_v2(lop_pq, g_active_p)
                        v_pq = tuple_v1_mul_v2ct(v_pq, rop_pq)

                        # Residual for this visibility.
                        r_pq = compute_residual(r_pq, v_pq)

                        wr_pq = tuple_wmul(r_pq, w)
                        wr_qp = tuple_unpackct(wr_pq)

                        jhj_jhr_p = accumulate_jhj_jhr(
                            lop_pq, rop_pq, w, g_active_p, channel_coeffs,
                            wr_pq, jhj_jhr_p
                        )
                        jhj_jhr_q = accumulate_jhj_jhr(
                            lop_qp, rop_qp, w, g_active_q, channel_coeffs,
                            wr_qp, jhj_jhr_q
                        )

                    flush_jhj_jhr(
                        jhj_tifi[a1_m, 0], jhr_tifi[a1_m, 0], jhj_jhr_p
                    )
                    flush_jhj_jhr(
                        jhj_tifi[a2_m, 0], jhr_tifi[a2_m, 0], jhj_jhr_q
                    )

                    continue

                # General path: multiple directions require per-direction
                # accumulation through memory.
                for f in range(fs, fe):

                    if flags[row, f]:  # Skip flagged data points.
                        continue

                    # Per-channel coefficients (staged terms only, else empty).
                    # This is the channel_coeffs tuple passed to the accumulate
                    # hook.
                    channel_coeffs = compute_channel_coeffs(
                        ms_inputs, meta_inputs, f
                    )

                    # Apply row weights in the BDA case, otherwise a no-op.
                    w = tuple_unpack_rweight(
                        weights[row, f], row_weights, row_ind
                    )
                    r_pq = tuple_unpack(data[row, f])

                    active_gain_tifi = gains[active_term][
                        active_t_map[row_ind], active_f_map[f]
                    ]

                    lop_pq_arr[:] = 0
                    rop_pq_arr[:] = 0
                    lop_qp_arr[:] = 0
                    rop_qp_arr[:] = 0
                    v_pq = zero_vec

                    for d in range(n_dir):

                        lop_pq = identity_vec
                        lop_qp = identity_vec

                        # Promote the model to the compute dtype - the zero
                        # add is free but ensures type stability below.
                        rop_qp = tuple_add(model[row, f, d], zero_vec)
                        rop_pq = tuple_unpackct(rop_qp)

                        for gi in all_terms:

                            d_m = dir_maps[gi][d]  # Broadcast dir.
                            t_m = time_maps[gi][row_ind]
                            f_m = freq_maps[gi][f]

                            gain = gains[gi][t_m, f_m]

                            g_q = tuple_unpack(gain[a2_m, d_m])
                            rop_pq = tuple_v1_mul_v2(g_q, rop_pq)

                            g_p = tuple_unpack(gain[a1_m, d_m])
                            rop_qp = tuple_v1_mul_v2(g_p, rop_qp)

                        for gi in gt_active:

                            d_m = dir_maps[gi][d]
                            t_m = time_maps[gi][row_ind]
                            f_m = freq_maps[gi][f]

                            gain = gains[gi][t_m, f_m]

                            g_p = tuple_unpack(gain[a1_m, d_m])
                            rop_pq = tuple_v1_mul_v2ct(rop_pq, g_p)

                            g_q = tuple_unpack(gain[a2_m, d_m])
                            rop_qp = tuple_v1_mul_v2ct(rop_qp, g_q)

                        for gi in lt_active:

                            d_m = dir_maps[gi][d]
                            t_m = time_maps[gi][row_ind]
                            f_m = freq_maps[gi][f]

                            gain = gains[gi][t_m, f_m]

                            g_p = tuple_unpack(gain[a1_m, d_m])
                            lop_pq = tuple_v1ct_mul_v2(g_p, lop_pq)

                            g_q = tuple_unpack(gain[a2_m, d_m])
                            lop_qp = tuple_v1ct_mul_v2(g_q, lop_qp)

                        out_d = active_d_map[d]

                        iunpack(lop_pq_arr[out_d], lop_pq)
                        iadd(rop_pq_arr[out_d], rop_pq)

                        iunpack(lop_qp_arr[out_d], lop_qp)
                        iadd(rop_qp_arr[out_d], rop_qp)

                        g_active = tuple_unpack(active_gain_tifi[a1_m, out_d])
                        v_pqd = tuple_v1ct_mul_v2(lop_pq, g_active)
                        v_pqd = tuple_v1_mul_v2ct(v_pqd, rop_pq)
                        v_pq = tuple_add(v_pq, v_pqd)

                    # Residual for this visibility.
                    r_pq = compute_residual(r_pq, v_pq)

                    # Weighted residual and its conjugate transpose. These
                    # are direction independent and can be computed once.
                    wr_pq = tuple_wmul(r_pq, w)
                    wr_qp = tuple_unpackct(wr_pq)

                    for d in range(n_gdir):

                        g_active_p = tuple_unpack(active_gain_tifi[a1_m, d])
                        g_active_q = tuple_unpack(active_gain_tifi[a2_m, d])

                        lop_pq_d = tuple_unpack(lop_pq_arr[d])
                        rop_pq_d = tuple_unpack(rop_pq_arr[d])

                        jhj_jhr = accumulate_jhj_jhr(
                            lop_pq_d, rop_pq_d, w, g_active_p, channel_coeffs,
                            wr_pq, jhj_jhr_zero
                        )
                        flush_jhj_jhr(
                            jhj_tifi[a1_m, d], jhr_tifi[a1_m, d], jhj_jhr
                        )

                        lop_qp_d = tuple_unpack(lop_qp_arr[d])
                        rop_qp_d = tuple_unpack(rop_qp_arr[d])

                        jhj_jhr = accumulate_jhj_jhr(
                            lop_qp_d, rop_qp_d, w, g_active_q, channel_coeffs,
                            wr_qp, jhj_jhr_zero
                        )
                        flush_jhj_jhr(
                            jhj_tifi[a2_m, d], jhr_tifi[a2_m, d], jhj_jhr
                        )

            # Accumulation only touches the upper triangle of each jhj
            # element (4 correlation case) - fill in the lower triangle.
            mirror_jhj(jhj_tifi)
        return

    # Return the loop as an inline="always" function so that it is never
    # lowered as a standalone (and separately disk-cached) unit - see the cache
    # correctness constraint in the docstring above. Each kernel inlines this
    # into a module-local trampoline, giving it a private cache namespace.
    return factories.qcjit(impl)


def compute_update(native_imdry, corr_mode):
    raise NotImplementedError


@overload(compute_update, jit_options=PARALLEL_JIT_OPTIONS)
def nb_compute_update(native_imdry, corr_mode):

    coerce_literal(nb_compute_update, ["corr_mode"])

    # We want to dispatch based on this field so we need its type.
    jhj = native_imdry[native_imdry.fields.index('jhj')]

    generalised = jhj.ndim == 6
    inversion_buffer = inversion_buffer_factory(generalised=generalised)
    invert = invert_factory(corr_mode, generalised=generalised)

    def impl(native_imdry, corr_mode):

        jhj = native_imdry.jhj
        jhr = native_imdry.jhr
        update = native_imdry.update

        n_tint, n_fint, n_ant, n_dir, n_param = jhr.shape

        n_int = n_tint * n_fint

        result_dtype = jhr.dtype

        for i in prange(n_int):

            t = i // n_fint
            f = i - t * n_fint

            buffers = inversion_buffer(n_param, result_dtype)

            for a in range(n_ant):
                for d in range(n_dir):

                    invert(jhj[t, f, a, d],
                           jhr[t, f, a, d],
                           update[t, f, a, d],
                           buffers)

    return impl
