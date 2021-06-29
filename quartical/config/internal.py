from dataclasses import make_dataclass
from quartical.config.external import Gain


def gains_to_chain(opts):

    terms = opts.solver.terms

    Chain = make_dataclass(
        "Chain",
        [(t, Gain, Gain()) for t in terms]
    )

    chain = Chain()

    for t in terms:
        setattr(chain, t, getattr(opts, t))

    return chain


def yield_from(obj, flds=None, name=True):

    flds = (flds,) if isinstance(flds, str) else flds

    for k in obj.__dataclass_fields__.keys():
        if flds is None:
            yield k
        elif name:
            yield (k, *(getattr(getattr(obj, k), fld) for fld in flds))
        else:
            yield (*(getattr(getattr(obj, k), fld) for fld in flds),)
