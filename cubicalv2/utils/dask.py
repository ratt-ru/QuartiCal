import dask.array as da
import numpy as np
from dask.base import tokenize
from dask.array.core import HighLevelGraph
from operator import getitem


def blockwise_unique(arr, chunks=None, return_index=False,
                     return_inverse=False, return_counts=False, axis=None):
    """Given an chunked dask.array, applies numpy.unique on a per chunk basis.

    Applies numpy.unique to a chunked input array to produce unique values
    (and optionally the indices, inverse and counts) per chunk. While this
    can be accomplished using a combination of blockwise and map_blocks, this
    function is cleaner and avoids ambiguous intemediary results. numpy.unique
    produces chunks with unknown dimensions. If those dimensions are known in
    advance they can be specified via the chunks argument. Chunks are assumed
    consistent for values, indices and counts. The inverse will always have
    the same chunks as the input.

    Inputs:
        arr: A chunked dask.array of input values.
        chunks: A tuple of tuples specifying chunk dimensions.
        return_index: See numpy.unique.
        return_inverse: See numpy.unique.
        return_counts: See numpy.unique.
        axis: See numpy.unique. Not currently supported.

    Returns:
        outputs: With no optional arguments, a dask.array of unique values.
            With optional arguments, a tuple of dask.arrays containing the
            unique values and selected optional outputs.
    """

    if axis is not None:
        raise ValueError("Axis argument not yet supported.")

    if arr.ndim > 1:
        raise ValueError("Multi-dimensional arrays not yet supported.")

    arg_list = [return_index, return_inverse, return_counts]

    token = tokenize(arr, *arg_list)  # Append to names to make unique.

    unique_name = "unique-" + token

    # Create layer for the numpy.unique call.

    unique_dsk = {(unique_name, i): (np.unique, input_key, *arg_list)
                  for i, input_key in enumerate(arr.__dask_keys__())}

    n_return = 1 + sum(arg_list)
    consumed = 0

    input_name = arr.name
    input_graph = arr.__dask_graph__()
    input_deps = input_graph.dependencies
    out_chunks = chunks if chunks else ((np.nan,)*arr.npartitions,)

    # Without kwargs, no additional work is necessary and we can simply add
    # the layer to the graph and express the result as an array.

    if n_return == 1:

        graph = input_graph.from_collections(unique_name,
                                             unique_dsk,
                                             arr)

        return da.Array(graph,
                        name=unique_name,
                        chunks=out_chunks,
                        dtype=arr.dtype)

    # With kwargs, we need to add layers to split the tuple output of the
    # numpy.unique calls into separate layers. We ensure that these layers
    # all share the numpy.unique call as a dependency to avoid recomputation.

    outputs = ()
    layers = {input_name: input_graph, unique_name: unique_dsk}
    deps = {input_name: input_deps, unique_name: {input_name}}

    def getitem_layer(name, dsk, i):
        return {(name, j): (getitem, k, i) for j, k in enumerate(dsk.keys())}

    # Construct the unique value layer.

    uval_name = "unique-values-" + token

    values_dsk = getitem_layer(uval_name, unique_dsk, consumed)

    layers.update({uval_name: values_dsk})
    deps.update({uval_name: {unique_name}})
    consumed += 1

    # Construct the indices layer.

    if return_index:

        uind_name = "unique-index-" + token

        index_dsk = getitem_layer(uind_name, unique_dsk, consumed)

        layers.update({uind_name: index_dsk})
        deps.update({uind_name: {unique_name}})
        consumed += 1

    # Construct the inverse layer.

    if return_inverse:

        uinv_name = "unique-inverse-" + token

        inverse_dsk = getitem_layer(uinv_name, unique_dsk, consumed)

        layers.update({uinv_name: inverse_dsk})
        deps.update({uinv_name: {unique_name}})
        consumed += 1

    # Construct the counts layer.

    if return_counts:

        ucnt_name = "unique-counts-" + token

        counts_dsk = getitem_layer(ucnt_name, unique_dsk, consumed)

        layers.update({ucnt_name: counts_dsk})
        deps.update({ucnt_name: {unique_name}})
        consumed += 1

    # Construct the HighLevelGraph containing the relevant layers.

    graph = HighLevelGraph(layers, deps)

    # Turn each getitem layer into a dask.array which will contain the result.

    outputs += (da.Array(graph,
                         name=uval_name,
                         chunks=out_chunks,
                         dtype=arr.dtype),)

    if return_index:
        outputs += (da.Array(graph,
                             name=uind_name,
                             chunks=out_chunks,
                             dtype=np.int64),)

    if return_inverse:
        outputs += (da.Array(graph,
                             name=uinv_name,
                             chunks=arr.chunks,  # This is always the case.
                             dtype=np.int64),)

    if return_counts:
        outputs += (da.Array(graph,
                             name=ucnt_name,
                             chunks=out_chunks,
                             dtype=np.int64),)

    return outputs


if __name__ == "__main__":

    test_data = da.random.randint(0, 10, size=(20,), chunks=(10,))

    # Add in an extra step to make sure dependecies are handled correctly.
    test_data = \
        test_data.map_blocks(lambda x: np.random.randint(0, 5, x.shape))

    just_unq = blockwise_unique(test_data)

    import time
    t0 = time.time()
    print(da.compute(just_unq, num_workers=12))
    print(time.time() - t0)

    unq, ind, inv, cnt = blockwise_unique(test_data,
                                          return_index=True,
                                          return_inverse=True,
                                          return_counts=True)

    import time
    t0 = time.time()
    print(da.compute(unq, ind, inv, cnt, num_workers=12))
    print(time.time() - t0)

    import dask
    dask.visualize(unq, ind, inv, cnt,
                   filename='custom.pdf',
                   optimize_graph=False)
