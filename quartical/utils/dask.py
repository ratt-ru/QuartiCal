import dask.array as da
import numpy as np
from dask.distributed import performance_report
from dask.base import tokenize
from dask.array.core import HighLevelGraph
from dask.highlevelgraph import MaterializedLayer
from operator import getitem
from dask.utils import apply, funcname
from collections import namedtuple
from itertools import product
from daskms.fsspec_store import DaskMSStore
from contextlib import nullcontext


def compute_context(dask_opts, output_opts, time_str):
    if dask_opts.scheduler == "distributed":
        store = DaskMSStore(output_opts.log_directory)
        report_path = store.join([store.full_path, f"{time_str}.html.qc"])
        return performance_report(filename=str(report_path))
    else:
        return nullcontext()


def get_block_id_arr(arr):

    def _get_block_ids(arr, block_id=None):

        assert arr.ndim != 0, "Array must have one or more dimensions"

        sel = (np.newaxis,) * (arr.ndim - 1) + (slice(None),)

        return np.array(block_id)[sel]

    block_id_arr = arr.map_blocks(
        _get_block_ids,
        meta=np.array((0,)*arr.ndim, dtype=np.int64),
        chunks=(1,) * (arr.ndim - 1) + (arr.ndim,)
    )

    return block_id_arr


class as_dict:
    """Decorator which turns function outputs into dictionary entries."""
    def __init__(self, func, *keys):
        self.func = func
        self.keys = keys

    def __call__(self, *args, **kwargs):
        values = self.func(*args, **kwargs)
        keys = self.keys

        if len(keys) == 1:
            values = (values,)
        elif len(keys) != len(values):
            raise ValueError(f"{len(keys)} keys provided for "
                             f"{len(values)} output values.")

        return dict(zip(keys, values))


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
            self._check_axis(blockdims)
            self._merge_dict(self.input_axes, blockdims)
            self.dask_inputs.append(value)
        elif isinstance(value, list) and index_list:
            lengths = self._len_at_depth(value)
            blockdims = {k: v for k, v in zip(index_list, lengths)}
            self._merge_dict(self.input_axes, blockdims)

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

    def get_dask_outputs(self):
        """Given the current state of the blocker, create a graph.

        Returns:
            output_dict: A dictionary of outputs containing dask.Arrays.
        """

        token = tokenize(*[inp.value for inp in self.inputs])

        layer_name = "-".join([funcname(self.func) + "~blocker", token])

        graph = {}

        layers = dict()
        deps = dict()

        for inp in self.dask_inputs:
            layers.update(inp.__dask_graph__().layers)
            deps.update(inp.__dask_graph__().dependencies)

        try:
            axes_lengths = list([self.input_axes[k] for k in self.func_axes])
        except KeyError as e:
            raise KeyError("Output key not present in input keys. "
                           "This behaviour is not yet supported.") from e

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

        layers.update({layer_name: MaterializedLayer(graph)})
        deps.update({layer_name: {k.name for k in self.dask_inputs}})

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

            layers.update({output_layer_name: MaterializedLayer(output_graph)})
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
            block_idxs = list(map(lambda ax: block_ids.get(ax, 0),
                              inp.index_list))
            return self._get_from_lol(inp.value, block_idxs)
        else:
            raise ValueError(f"Cannot generate graph input for {inp}.")

    def _check_axis(self, blockdims):
        """Check that contraction axes are unchunked."""

        for k, v in blockdims.items():
            if k not in self.func_axes and v != 1:
                raise ValueError(f"Contraction axis {k} has multiple chunks. "
                                 f"This is not yet supported by Blocker.")

    def _merge_dict(self, dict0, dict1):
        """Merge two dicts, raising an error when values do not agree."""

        for k, v in dict1.items():
            if k not in dict0:
                dict0.update({k: v})
            elif k in dict0 and dict0[k] != v:
                raise ValueError("Block dimensions are not in agreement. "
                                 "Check input array indices and chunking.")

    def _len_at_depth(self, lol):
        """Determines length at depth of a list of lists."""

        lengths = [len(lol)]

        l_at_d = lol

        while True:
            l_item = getitem(l_at_d, 0)
            is_list = isinstance(l_item, list)
            if is_list:
                l_at_d = l_item
                lengths.append(len(l_at_d))
            else:
                break

        return lengths

    def _get_from_lol(self, lol, ind):
        """Get an item from a list of lists based on ind."""

        result = lol

        for i in ind:
            result = getitem(result, i)

        return result


def blockwise_unique(arr, chunks=None, return_index=False,
                     return_inverse=False, return_counts=False, axis=None):
    """Given a chunked dask.array, applies numpy.unique on a per chunk basis.

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

    output_names = ["values"]

    if return_index:
        output_names.append("indices")
    if return_inverse:
        output_names.append("inverse")
    if return_counts:
        output_names.append("counts")

    B = Blocker(as_dict(np.unique, *output_names), "r")

    B.add_input("ar", arr, "r")
    B.add_input("return_index", return_index)
    B.add_input("return_inverse", return_inverse)
    B.add_input("return_counts", return_counts)

    chunks = chunks or ((np.nan,)*arr.npartitions,)

    B.add_output("values", "r", chunks, arr.dtype)
    if return_index:
        B.add_output("indices", "r", chunks, np.int64)
    if return_inverse:
        B.add_output("inverse", "r", arr.chunks, np.int64)
    if return_counts:
        B.add_output("counts", "r", chunks, np.int64)

    output_dict = B.get_dask_outputs()
    output_tuple = tuple(output_dict.get(k) for k in output_names)

    if len(output_tuple) > 1:
        return output_tuple
    else:
        return output_tuple[0]
