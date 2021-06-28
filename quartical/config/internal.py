from dataclasses import make_dataclass
from quartical.config.external import Gain


def convert_gain_config(opts):

    terms = opts.solver.terms

    Gains = make_dataclass(
        "Gains",
        [(t, Gain, Gain()) for t in terms]
    )

    gains = Gains()

    for t in terms:
        setattr(gains, t, getattr(opts, t))

    return gains


def yield_from(obj, fld):
    for k in obj.__dataclass_fields__.keys():
        yield (k, getattr(getattr(obj, k), fld))
