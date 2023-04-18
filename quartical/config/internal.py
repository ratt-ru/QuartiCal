from daskms.fsspec_store import DaskMSStore
from quartical.gains import TERM_TYPES


def gains_to_chain(opts):

    terms = opts.solver.terms

    # NOTE: Currently a simple list, but we could also implement an object.
    chain = [
        TERM_TYPES[getattr(opts, t).type](t, getattr(opts, t)) for t in terms
    ]

    return chain


def additional_validation(config):

    chain = gains_to_chain(config)

    store = DaskMSStore(config.output.gain_directory)

    if store.exists() and not config.output.overwrite:
        raise FileExistsError(f"{store.url} already exists. Specify "
                              f"output.overwrite=1 to suppress this "
                              f"error and overwrite files/folders.")
    elif store.exists():
        store.rm(recursive=True)

    load_stores = [DaskMSStore(t.load_from) for t in chain if t.load_from]

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
