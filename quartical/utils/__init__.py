from daskms.fsspec_store import DaskMSStore


def remove_store(directory):
    store = DaskMSStore(directory)
    if store.exists():
        store.rm(recursive=True)
