import numpy as np
import dask.array as da
from dask.array.core import HighLevelGraph
from operator import getitem
from dask.base import tokenize
from quartical.calibration.solver import solver_wrapper
from collections import namedtuple
from itertools import product
from dask.utils import apply

term_spec_tup = namedtuple("term_spec", "name type shape")


def construct_solver(model_col,
                     data_col,
                     ant1_col,
                     ant2_col,
                     weight_col,
                     t_map_arr,
                     f_map_arr,
                     d_map_arr,
                     corr_mode,
                     gain_xds_list,
                     opts):
    """Constructs the dask graph for the solver layer.

    This constructs a custom dask graph for the solver layer given the slew
    of solver inputs. This is arguably the most important function in V2 and
    should not be tampered with without a certain level of expertise with dask.

    Args:
        model_col: dask.Array containing the model column.
        data_col: dask.Array containing the data column.
        ant1_col: dask.Array containing the first antenna column.
        ant2_col: dask.Array containing the second antenna column.
        wegith_col: dask.Array containing the weight column.
        t_map_arr: dask.Array containing time mappings.
        f_map_arr: dask.Array containing frequency mappings.
        d_map_arr: dask.Array containing direction mappings.
        corr_mode: A string indicating the correlation mode.
        gain_xds_list: A list of xarray.Dataset objects describing gain terms.
        opts: A Namespace object containing global options.

    Returns:
        gain_list: A list of dask.Arrays containing the gains.
        conv_perc_list: A list of dask.Arrays containing the converged
            percentages.
        conv_iter_list: A list of dask.Arrays containing the iterations taken
            to reach convergence.
    """

    # Grab the number of input chunks - doing this on the data should be safe.
    n_t_chunks, n_f_chunks, _ = data_col.numblocks

    # Take the compact chunking info on the gain xdss and expand it.
    spec_list = expand_specs(gain_xds_list)

    C = Constructor(solver_wrapper, "rf")

    C.add_input("model", model_col, "rfdc")
    C.add_input("data", data_col, "rfc")
    C.add_input("a1", ant1_col, "r")
    C.add_input("a2", ant2_col, "r")
    C.add_input("weights", weight_col, "rfc")
    C.add_input("t_map_arr", t_map_arr, "rj")
    C.add_input("f_map_arr", f_map_arr, "rj")
    C.add_input("d_map_arr", d_map_arr)
    C.add_input("corr_mode", corr_mode)
    C.add_input("term_spec_list", spec_list, "rf")

    for gn in opts.solver_gain_terms:
        C.add_output("{}-gain".format(gn), "rfadc")
        C.add_output("{}-conviter".format(gn), "rf")
        C.add_output("{}-convperc".format(gn), "rf")

    graph, token = C.make_graph()

    # The following constructs the getitem layer which is necessary to handle
    # returning several results from the solver. TODO: This is improving but
    # can I add convenience functions to streamline this process?

    gain_keys = \
        ["{}-gain-{}".format(gn, token) for gn in opts.solver_gain_terms]

    conv_iter_keys = \
        ["{}-conviter-{}".format(gn, token) for gn in opts.solver_gain_terms]

    conv_perc_keys = \
        ["{}-convperc-{}".format(gn, token) for gn in opts.solver_gain_terms]

    # Now that we have the graph, we need to construct arrays from the results.
    # We loop over the outputs and construct appropriate dask arrays. These are
    # then assigned to the relevant gain xarray.Dataset object.

    solved_xds_list = []

    for gi, gain_xds in enumerate(gain_xds_list):

        gain = da.Array(graph,
                        name=gain_keys[gi],
                        chunks=gain_xds.CHUNK_SPEC,
                        dtype=np.complex64)

        conv_perc = da.Array(graph,
                             name=conv_perc_keys[gi],
                             chunks=((1,)*n_t_chunks,
                                     (1,)*n_f_chunks),
                             dtype=np.float32)

        conv_iter = da.Array(graph,
                             name=conv_iter_keys[gi],
                             chunks=((1,)*n_t_chunks,
                                     (1,)*n_f_chunks),
                             dtype=np.float32)

        solved_xds = gain_xds.assign(
            {"gains": (("time_int", "freq_int", "ant", "dir", "corr"), gain),
             "conv_perc": (("t_chunk", "f_chunk"), conv_perc),
             "conv_iter": (("t_chunk", "f_chunk"), conv_iter)})

        solved_xds_list.append(solved_xds)

    return solved_xds_list


def expand_specs(gain_xds_list):
    """Convert compact spec to a per-term list per-chunk."""

    spec_lists = []

    for gxds in gain_xds_list:

        term_name = gxds.NAME
        term_type = gxds.TYPE
        chunk_spec = gxds.CHUNK_SPEC

        ac = chunk_spec.achunk[0]  # No chunking along antenna axis.
        dc = chunk_spec.dchunk[0]  # No chunking along direction axis.
        cc = chunk_spec.cchunk[0]  # No chunking along correlation axis.

        shapes = [(tc, fc, ac, dc, cc)
                  for tc, fc in product(chunk_spec.tchunk, chunk_spec.fchunk)]

        term_spec_list = [term_spec_tup(term_name, term_type, shape)
                          for shape in shapes]

        spec_lists.append(term_spec_list)

    return list(zip(*spec_lists))


class Constructor:

    _input = namedtuple("input", "name value index_list")
    _output = namedtuple("output", "name index_list")

    def __init__(self, func, index_string):

        self.func = func
        self.func_axes = list(index_string)
        self.input_axes = {}
        self.dask_inputs = []

        self.inputs = []
        self.outputs = []

    def add_input(self, name, value, index_string=None):

        index_list = list(index_string) if index_string else None

        if isinstance(value, da.Array):
            blockdims = {k: v for k, v in zip(index_list, value.numblocks)}
            self._merge_dict(self.input_axes, blockdims)
            self.dask_inputs.append(value)

        self.inputs.append(self._input(name, value, index_list))

    def add_output(self, name, index_string):

        index_list = list(index_string)

        self.outputs.append(self._output(name, index_list))

    def make_graph(self):
        """Given the current state of the Constructor, create a graph."""

        token = tokenize(*[inp.value for inp in self.inputs])

        layer_name = "-".join([self.func.__name__, token])

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
        # application of func. TODO: Add the getitem layer.

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

        return HighLevelGraph(layers, deps), token

    def _get_key_or_value(self, inp, idx, block_ids):
        """Given an input, returns appropriate key/value based on indices."""

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
