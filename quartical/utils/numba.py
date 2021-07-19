from numba.core.extending import SentryLiteralArgs
import inspect


def coerce_literal(func, literals):
    func_locals = inspect.stack()[1].frame.f_locals  # One frame up.
    arg_types = [func_locals[k] for k in inspect.signature(func).parameters]
    SentryLiteralArgs(literals).for_function(func).bind(*arg_types)
