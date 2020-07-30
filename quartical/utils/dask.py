import dask.array as da
import numpy as np
from dask.base import tokenize
from dask.array.core import HighLevelGraph
from operator import getitem
from dask.utils import apply, funcname
from collections import namedtuple
from itertools import product


class Blocker:
    """Constructor object for custom blockwise dask operations.

    A blocker object provides an interface for constructing complicated
    one-to-many blockwise operations. This is more flexible but less complete
    than the implementation of blockwise in dask. Importantly the called
    function is expected to return a dictionary of outputs. This restriction
    is to avoid error-prone reliance on output index.
    """

    # Convenient named tuples for holding input and output specs.
    _input = namedtuple("input", "name value index_list")
    _output = namedtuple("output", "name index_list chunks dtype")

    def __init__(self, func, index_string):
        """Basic initialisation of the blocker object.

        Args:
            func: The function which is to be applied per block.
            index_string: A string describing the chunked axes of the output
                of func. Note that this is only used to construct an
                intemediary graph layer - outputs will be created in
                accordance with the ourput specs.
        """

        self.func = func
        self.func_axes = list(index_string)
        self.input_axes = {}
        self.dask_inputs = []

        self.inputs = []
        self.outputs = []

    def add_input(self, name, value, index_string=None):
        """Adds an input to the blocker object.

        Args:
            name: Name string of variable. Should match argument name if
                it is a positional argument of func.
            value: The value of this input.
            index_string: If dask.array or mapping, describes the axes which
                this input possesses. If None, input is scalar (unblocked).
        """

        index_list = list(index_string) if index_string else None

        if isinstance(value, da.Array):
            blockdims = {k: v for k, v in zip(index_list, value.numblocks)}
            self._merge_dict(self.input_axes, blockdims)
            self.dask_inputs.append(value)

        self.inputs.append(self._input(name, value, index_list))

    def add_output(self, name, index_string, chunks, dtype):
        """Adds an output to the blocker object.

        Args:
            name: Name string for output dict.
            index_string: Describes the axes which this output possesses.
            chunks: Tuple describing chunking of this output.
            dtype: Type of this output.
        """

        index_list = list(index_string)

        self.outputs.append(self._output(name, index_list, chunks, dtype))

    def get_output_arrays(self):
        """Given the current state of the blocker, create a graph.

        Returns:
            output_dict: A dictionary of outputs containing dask.Arrays.
        """

        token = tokenize(*[inp.value for inp in self.inputs])

        layer_name = "-".join([funcname(self.func), token])

        graph = {}

        layers = {inp.name: inp.__dask_graph__() for inp in self.dask_inputs}

        deps = {inp.name: inp.__dask_graph__().dependencies
                for inp in self.dask_inputs}

        axes_lengths = list([self.input_axes[k] for k in self.func_axes])

        # Set up the solver graph. Loop over the chunked axes.
        for i, k in enumerate(product(*map(range, axes_lengths))):

            # Set up the per-chunk functions calls. Note how keys are
            # assosciated. We will pass everything in as a kwarg - this
            # means we don't need to worry about maintaining argument order.
            # This will make adding additional arguments simple.

            block_ids = dict(zip(self.func_axes, k))

            kwargs = [[inp.name, self._get_key_or_value(inp, i, block_ids)]
                      for inp in self.inputs]

            graph[(layer_name, *k)] = (apply, self.func, [], (dict, kwargs))

        layers.update({layer_name: graph})
        deps.update({layer_name: {k for k in deps.keys()}})

        # At this point we have a dictionary which describes the chunkwise
        # application of func.

        input_keys = list(graph.keys())

        for o in self.outputs:

            output_layer_name = "-".join([o.name, token])

            axes_lengths = \
                list([self.input_axes.get(k, 1) for k in o.index_list])

            output_graph = {}

            for i, k in enumerate(product(*map(range, axes_lengths))):

                output_graph[(output_layer_name, *k)] = \
                    (getitem, input_keys[i], o.name)

            layers.update({output_layer_name: output_graph})
            deps.update({output_layer_name: {layer_name}})

        hlg = HighLevelGraph(layers, deps)

        output_dict = {}

        for o in self.outputs:

            output_layer_name = "-".join([o.name, token])

            output_dict[o.name] = da.Array(hlg,
                                           name=output_layer_name,
                                           chunks=o.chunks,
                                           dtype=o.dtype)

        return output_dict

    def _get_key_or_value(self, inp, idx, block_ids):
        """Given an input, returns appropriate key/value based on indices.

        Args:
            inp: An _input named tuple descriing an input.
            idx: The flat index of the current block.
            block_ids: The block indices of the current block.
        """

        if inp.index_list is None:
            return inp.value
        elif isinstance(inp.value, da.Array):
            name = inp.value.name
            block_idxs = list(map(lambda ax: block_ids.get(ax, 0),
                              inp.index_list))
            return (name, *block_idxs)
        elif isinstance(inp.value, list):
            return inp.value[idx]
        else:
            raise ValueError("Cannot generate graph input for {}".format(inp))

    def _merge_dict(self, dict0, dict1):
        """Merge two dicts, raising an error when values do not agree."""

        for k, v in dict1.items():
            if k not in dict0:
                dict0.update({k: v})
            elif k in dict0 and dict0[k] != v:
                raise ValueError("Block dimensions are not in agreement.")


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
