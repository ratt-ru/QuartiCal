import dask
import dask.array as da
from daskms.constants import DASKMS_PARTITION_KEY
from dask.highlevelgraph import HighLevelGraph
import xarray


def annotate_dask_collection(collection, annotations):
    if not isinstance(collection, da.Array):
        raise NotImplementedError(f"annotation of {type(collection)} "
                                  f"not yet supported")

    hlg = collection.__dask_graph__()

    if type(hlg) is not HighLevelGraph:
        raise TypeError(f"hlg is not a HighLevelGraph: {type(hlg)}")

    top_layer = hlg.layers[collection.name]

    # Integrate array data with existing annotations
    annotations = {
        "__dask_array__": {
            "dtype": collection.dtype.name,
            "chunks": collection.chunks,
            **annotations
        }
    }

    # Try ensure the number of dims match the number of chunks (ndim)
    try:
        dims = annotations["dims"]
    except KeyError:
        pass
    else:
        assert len(annotations["chunks"]) == len(dims)

    if top_layer.annotations:
        # Annotations already exist, do some sanity checks
        try:
            a = top_layer.annotations["__dask_array__"]
        except KeyError:
            # Destructive update
            top_layer.annotations = annotations
        else:
            if "row" in a["dims"] and top_layer.annotations != annotations:
                raise ValueError(f"Trying to annotate a row array with "
                                 f"differing annotations\n."
                                 f"original: {top_layer.annotations}"
                                 f"new: {annotations}")
            else:
                # TODO(sjperkins) Improve this
                # We hit this case when trying to annotate a single array
                # that has been assigned to multiple datasets.
                # i.e. POSITION: ((ant,), position) from the ANTENNA subtable
                # This causes issues because partitioning information differs
                # between datasets, but we don't really care about annotations
                # for this type of array
                pass
    else:
        top_layer.annotations = annotations


def dataset_partition(ds):
    # Add any dataset partitioning information to annotations
    try:
        partition = getattr(ds, DASKMS_PARTITION_KEY)
    except AttributeError:
        raise ValueError(f"{ds} has no {DASKMS_PARTITION_KEY} attribute")
    else:
        return tuple((p, getattr(ds, p)) for p, _ in partition)


def annotate(obj, **annotations):
    if isinstance(obj, (tuple, list)):
        for o in obj:
            annotate(o, **annotations)
    elif isinstance(obj, xarray.Dataset):
        partition = dataset_partition(obj)

        if partition:
            annotations = {**annotations, "partition": partition}

        for var in obj.data_vars.values():
            annotate(var, **annotations)

    elif isinstance(obj, xarray.DataArray):
        # Annotate any wrapped dask collections
        # with dimension information
        if not dask.is_dask_collection(obj.data):
            return  # Ignore numpy arrays etc.

        annotate_dask_collection(obj.data, {**annotations, "dims": obj.dims})
    elif dask.is_dask_collection(obj):
        annotate_dask_collection(obj, annotations)
    else:
        raise TypeError(f"obj must be a dask collection, "
                        f"xarray.Dataset, xarray.DataArray, "
                        f"or list of the previous types. "
                        f"Got a: {type(obj)}")