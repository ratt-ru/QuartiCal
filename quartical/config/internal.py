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
