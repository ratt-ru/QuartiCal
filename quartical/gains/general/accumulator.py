# -*- coding: utf-8 -*-
"""The accumulator-layout hooks of the shared accumulation loop.

``build_jhj_jhr_impl`` (``solver_components.py``) accumulates each JHJ/JHr
element in a flat, register-resident tuple and asks the term for three hooks
describing that tuple's layout: ``zero_jhj_jhr`` creates it, ``flush_jhj_jhr``
adds a completed one into the JHJ/JHr arrays, and ``mirror_jhj`` completes the
JHJ elements once a solution interval is finished.

For a parameterised term all three follow from a single number: ``n_param``,
how many real parameters the term solves per element, which
``parameters.get_n_param`` derives from the term's ``params_per_corr`` and the
correlation mode. JHJ is then a real symmetric ``(n_param, n_param)`` matrix
whose upper triangle is packed row-major ahead of the ``n_param`` JHr entries,
so :func:`triangular_accumulator_factories` builds the trio from that alone.

The two hooks which touch the accumulator tuple are built as numba intrinsics,
so that the layout can be walked by an ordinary loop here at compile time while
the emitted code still names every slot by a literal index. Naming a slot by a
computed index instead makes numba lower the access through memory, which costs
the accumulation loop its register-resident accumulator - and that cost lands
on every visibility, not on the flush. ``mirror_jhj`` only ever indexes arrays,
so it stays an ordinary jitted loop.

Two things deliberately live elsewhere. The per-term maths which *fills* the
accumulator is the separate ``accumulate_jhj_jhr`` hook and stays in each
kernel. And the complex terms are not covered here at all: their JHJ is a
correlation-space block with a mix of real and complex slots and a Hermitian
(conjugating) mirror, so ``complex`` and ``diag_complex`` keep their own trio.
"""
from typing import Callable, NamedTuple
from llvmlite import ir
from numba.core import cgutils, types
from numba.core.errors import TypingError
from numba.core.extending import intrinsic
import quartical.gains.general.factories as factories
from quartical.gains.general.parameters import get_n_param


class AccumulatorFactories(NamedTuple):
    """The three accumulator-layout hook factories of a term.

    Attributes:
        zero: The ``zero_jhj_jhr_factory`` hook factory.
        flush: The ``flush_jhj_jhr_factory`` hook factory.
        mirror: The ``mirror_jhj_factory`` hook factory. A term whose jhj is
            (1, 1) in every correlation mode - i.e. one with no parameter set
            per correlation - has no triangle to mirror and passes ``None`` to
            the builder instead. That is not the only way to reach a (1, 1)
            jhj, so the factory handles ``n_param == 1`` as well: a term with
            one parameter per correlation hits it in the single-correlation
            case, and it selects the hook once per term, not per mode.
    """

    zero: Callable
    flush: Callable
    mirror: Callable


def triangular_accumulator_factories(params_per_corr):
    """Build the accumulator-layout hooks for a real, triangular jhj.

    Args:
        params_per_corr: The number of parameters the term solves per diagonal
            correlation. See :func:`parameters.get_n_param`.

    Returns:
        An :class:`AccumulatorFactories` triple. Each member is an ordinary
        ``factory(corr_mode) -> hook`` as expected by ``build_jhj_jhr_impl``.
    """

    def zero_jhj_jhr_factory(corr_mode):
        """Produce the zero jhj/jhr accumulator tuple for a given corr mode.

        The reference element is a jhr slice, whose dtype is real, so every
        slot of a parameterised term's accumulator holds a real zero.
        """

        n_param = get_n_param(corr_mode, params_per_corr)

        # The jhj upper triangle, then one jhr entry per parameter.
        n_accumulator_slots = n_param * (n_param + 1) // 2 + n_param

        @intrinsic
        def build_zero(typingctx, value):
            """Emit a tuple of ``value``, one entry per accumulator slot."""

            tuple_type = types.UniTuple(value, n_accumulator_slots)

            def codegen(context, builder, signature, args):
                # NB: make_tuple increfs nothing, which is safe only because
                # every accumulator slot is a scalar.
                return context.make_tuple(
                    builder, tuple_type, [args[0]]*n_accumulator_slots
                )

            return tuple_type(value), codegen

        def impl(invec):
            return build_zero(invec[0]*0)

        return factories.qcjit(impl)

    def flush_jhj_jhr_factory(corr_mode):
        """Add a register-accumulated jhj/jhr accumulator into the arrays."""

        n_param = get_n_param(corr_mode, params_per_corr)

        # Also the slot at which the jhr entries start.
        n_triu = n_param * (n_param + 1) // 2

        @intrinsic
        def add_accumulator(typingctx, jhj, jhr, jhj_jhr):
            """Emit one ``array[i, j] += jhj_jhr[slot]`` per accumulator slot.

            The checks below are what makes the layout self-policing: a term
            whose accumulate hook returns the wrong number of slots, or slots
            of the wrong kind, fails to compile here instead of silently
            writing the wrong entries.
            """

            if not isinstance(jhj_jhr, types.UniTuple):
                raise TypingError(
                    "The jhj/jhr accumulator must be a homogeneous tuple."
                )
            if jhj_jhr.count != n_triu + n_param:
                raise TypingError(
                    f"The jhj/jhr accumulator holds {jhj_jhr.count} slots, "
                    f"but a ({n_param}, {n_param}) jhj element needs "
                    f"{n_triu + n_param}."
                )
            for name, operand in (("jhj", jhj), ("jhr", jhr),
                                  ("accumulator", jhj_jhr)):
                dtype = getattr(operand, "dtype", None)
                if not isinstance(dtype, types.Float):
                    raise TypingError(
                        f"A triangular accumulator needs a real {name}, got "
                        f"{dtype}."
                    )

            def codegen(context, builder, signature, args):
                jhj_type, jhr_type, _ = signature.args
                jhj_array = context.make_array(jhj_type)(
                    context, builder, args[0]
                )
                jhr_array = context.make_array(jhr_type)(
                    context, builder, args[1]
                )
                index_type = context.get_value_type(types.intp)

                def add_slot(array_type, array, indices, slot):
                    """Emit ``array[indices] += jhj_jhr[slot]``."""

                    pointer = cgutils.get_item_pointer(
                        context, builder, array_type, array,
                        [ir.Constant(index_type, i) for i in indices],
                    )
                    value = builder.extract_value(args[2], slot)
                    builder.store(
                        builder.fadd(builder.load(pointer), value), pointer
                    )

                # Row-major over the upper triangle, then the jhr entries.
                slot = 0
                for i in range(n_param):
                    for j in range(i, n_param):
                        add_slot(jhj_type, jhj_array, (i, j), slot)
                        slot += 1
                for i in range(n_param):
                    add_slot(jhr_type, jhr_array, (i,), n_triu + i)

                return context.get_dummy_value()

            return types.void(jhj, jhr, jhj_jhr), codegen

        def impl(jhj, jhr, jhj_jhr):
            add_accumulator(jhj, jhr, jhj_jhr)

        return factories.qcjit(impl)

    def mirror_jhj_factory(corr_mode):
        """Fill in the lower triangle of the per-interval jhj elements.

        The accumulate hook only writes the upper triangle of each real,
        symmetric jhj element, so the lower triangle is a straight copy done
        once per solution interval rather than once per visibility. In the 2
        correlation case the entries coupling the two correlation blocks are
        structurally zero, so copying them is harmless.
        """

        n_param = get_n_param(corr_mode, params_per_corr)

        if n_param == 1:
            def impl(jhj_tifi):
                pass
        else:
            def impl(jhj_tifi):
                n_ant, n_gdir = jhj_tifi.shape[:2]
                for a in range(n_ant):
                    for d in range(n_gdir):
                        jhj_ad = jhj_tifi[a, d]
                        for i in range(1, n_param):
                            for j in range(i):
                                jhj_ad[i, j] = jhj_ad[j, i]

        return factories.qcjit(impl)

    return AccumulatorFactories(
        zero_jhj_jhr_factory, flush_jhj_jhr_factory, mirror_jhj_factory
    )
