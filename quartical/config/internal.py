from dataclasses import make_dataclass
from pathlib import Path
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


def additional_validation(config):

    chain = gains_to_chain(config)

    root_path = Path(config.output.directory).absolute()

    if root_path.exists() and not config.output.overwrite:
        raise FileExistsError(f"{root_path} already exists. Specify "
                              f"output.overwrite=1 to suppress this "
                              f"error and overwrite *.qc files/folders.")

    gain_path = root_path / Path("gains.qc")
    load_dirs = [Path(lf).absolute().parent
                 for _, lf in yield_from(chain, "load_from") if lf]

    msg = (
        f"Output directory {str(gain_path)} contains terms which will be "
        f"loaded/interpolated. This is not supported. Please specify a "
        f"different output directory."
    )

    assert all(gain_path != ld for ld in load_dirs), msg

    return
