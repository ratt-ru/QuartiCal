# -*- coding: utf-8 -*-
"""Table-backend selection for QuartiCal.

QuartiCal reads and writes Measurement Sets through dask-ms, which reaches its
backend as ``lazy_import("casacore.tables")``. casacure is a Python package
mirroring the ``casacore`` surface, so selecting it is a matter of aliasing
``casacore.{tables,measures,quanta}`` onto the matching ``casacure`` submodules
before any of those imports are resolved.

Importing :mod:`quartical` performs that aliasing, so the rest of the package -
and dask-ms - talks to casacure without knowing it.

The alias is skipped when casacore is already bound: a table handle and a
measures server from different backends must not be mixed.
"""

import sys
import warnings

#: The table backend in use: ``"casacure"`` or ``"casacore"``.
BACKEND_NAME = "casacore"

_SUBMODULES = ("tables", "measures", "quanta")


def activate_backend():
    """Route ``casacore`` imports to casacure.

    Idempotent. Returns the name of the backend in effect.
    """

    global BACKEND_NAME

    if "casacore.tables" in sys.modules:
        # Something already resolved the table backend. Rebinding now would
        # leave open table handles on one backend and measures on the other.
        warnings.warn(
            "casacore.tables was imported before quartical; keeping the "
            "existing table backend. Import quartical first to use casacure.",
            RuntimeWarning,
            stacklevel=2,
        )
        return BACKEND_NAME

    try:
        import casacure
    except ImportError:
        warnings.warn(
            "casacure is not installed; falling back to python-casacore. "
            "Install casacure to drop the casacore dependency.",
            RuntimeWarning,
            stacklevel=2,
        )
        return BACKEND_NAME

    submodules = {}
    for name in _SUBMODULES:
        __import__("casacure.%s" % name)
        submodules[name] = sys.modules["casacure.%s" % name]

    sys.modules["casacore"] = casacure
    for name, module in submodules.items():
        sys.modules["casacore.%s" % name] = module
        # ``import casacore; casacore.measures...`` does not go through
        # sys.modules, so bind the names on the module object too.
        setattr(casacure, name, module)

    BACKEND_NAME = "casacure"

    return BACKEND_NAME
