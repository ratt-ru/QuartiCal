from dataclasses import make_dataclass
from quartical.config.external import Gain
from daskms.fsspec_store import DaskMSStore


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

    store = DaskMSStore(config.output.gain_directory)

    if store.exists() and not config.output.overwrite:
        raise FileExistsError(f"{store.url} already exists. Specify "
                              f"output.overwrite=1 to suppress this "
                              f"error and overwrite files/folders.")
    elif store.exists():
        store.rm(recursive=True)

    load_stores = \
        [DaskMSStore(lf) for _, lf in yield_from(chain, "load_from") if lf]

    msg = (
        f"Output directory {str(store.url)} contains terms which will be "
        f"loaded/interpolated. This is not supported. Please specify a "
        f"different output directory."
    )

    assert not any(store.full_path in ll.full_path for ll in load_stores), msg

    if config.mad_flags.whitening == "robust":
        assert config.solver.robust, (
            "mad_flags.whitening specified as robust but solver.robust is "
            "not enabled."
        )

    return
