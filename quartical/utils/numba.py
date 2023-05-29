from numba.core.extending import SentryLiteralArgs
import inspect

JIT_OPTIONS = {
    "nogil": True,
    "fastmath": True,
    "cache": True
}

PARALLEL_JIT_OPTIONS = {
    **JIT_OPTIONS,
    "parallel": True
}


def coerce_literal(func, literals):
    func_locals = inspect.currentframe().f_back.f_locals  # One frame up.
    arg_types = [func_locals[k] for k in inspect.signature(func).parameters]
    SentryLiteralArgs(literals).for_function(func).bind(*arg_types)
