from daskms.fsspec_store import DaskMSStore


def remove_directory(directory):
    store = DaskMSStore(directory)
    if store.exists():
        store.rm(recursive=True)
