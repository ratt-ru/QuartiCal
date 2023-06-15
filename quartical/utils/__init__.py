from daskms.fsspec_store import DaskMSStore


def remove_store(path):
    store = DaskMSStore(path)
    if store.exists():
        store.rm(recursive=True)
