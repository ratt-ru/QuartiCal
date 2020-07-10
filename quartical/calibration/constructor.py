import numpy as np
import dask.array as da
from dask.array.core import HighLevelGraph
from operator import getitem
from dask.base import tokenize
from quartical.calibration.solver import solver_wrapper
from dask.core import flatten
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

    C = Constructor(n_t_chunks, n_f_chunks)

    C.add_input("model", model_col, "tf")
    C.add_input("data", data_col, "tf")
    C.add_input("a1", ant1_col, "t")
    C.add_input("a2", ant2_col, "t")
    C.add_input("weights", weight_col, "tf")
    C.add_input("t_map_arr", t_map_arr, "t")
    C.add_input("f_map_arr", f_map_arr, "t")
    C.add_input("d_map_arr", d_map_arr)
    C.add_input("corr_mode", corr_mode)
    C.add_input("term_spec_list", spec_list, "tf")

    solver_dsk, layers, deps, token = C.make_graph()

    solver_name = "solver-" + token  # TODO: This is a hack for now.

    # The following constructs the getitem layer which is necessary to handle
    # returning several results from the solver. TODO: This is improving but
    # can I add convenience functions to streamline this process?

    gain_keys = \
        ["gain-{}-{}".format(gn, token) for gn in opts.solver_gain_terms]

    conv_iter_keys = \
        ["conviter-{}-{}".format(gn, token) for gn in opts.solver_gain_terms]

    conv_perc_keys = \
        ["convperc-{}-{}".format(gn, token) for gn in opts.solver_gain_terms]

    for gi, gn in enumerate(opts.solver_gain_terms):

        get_gain = {(gain_keys[gi], k[1], k[2], 0, 0, 0):
                    (getitem, (getitem, k, gn), "gain")
                    for i, k in enumerate(solver_dsk.keys())}

        layers.update({gain_keys[gi]: get_gain})
        deps.update({gain_keys[gi]: {solver_name}})

        get_conv_iters = \
            {(conv_iter_keys[gi], k[1], k[2]):
             (np.atleast_2d, (getitem, (getitem, k, gn), "conv_iter"))
             for i, k in enumerate(solver_dsk.keys())}

        layers.update({conv_iter_keys[gi]: get_conv_iters})
        deps.update({conv_iter_keys[gi]: {solver_name}})

        get_conv_perc = \
            {(conv_perc_keys[gi], k[1], k[2]):
             (np.atleast_2d, (getitem, (getitem, k, gn), "conv_perc"))
             for i, k in enumerate(solver_dsk.keys())}

        layers.update({conv_perc_keys[gi]: get_conv_perc})
        deps.update({conv_perc_keys[gi]: {solver_name}})

    # Turn the layers and dependencies into a high level graph.
    graph = HighLevelGraph(layers, deps)

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

    _input = namedtuple("input", "name value index_string")

    def __init__(self, n_t_chunks, n_f_chunks):

        self.n_t_chunks = n_t_chunks
        self.n_f_chunks = n_f_chunks

        self.inputs = []

    def add_input(self, name, value, index_string=None):

        self.inputs.append(self._input(name, value, index_string))

    def make_graph(self):
        """Given the current state of the Constructor, create a graph."""

        token = tokenize(*[inp.value for inp in self.inputs])

        solver_name = "solver-" + token

        graph = {}

        layers = {inp.value.name: inp.value.__dask_graph__()
                  for inp in self.inputs
                  if hasattr(inp.value, "__dask_graph__")}

        deps = {inp.value.name: inp.value.__dask_graph__().dependencies
                for inp in self.inputs
                if hasattr(inp.value, "__dask_graph__")}

        # Set up the solver graph. Loop over the chunked axes.
        for t_ind in range(self.n_t_chunks):
            for f_ind in range(self.n_f_chunks):

                # Set up the per-chunk solves. Note how keys are assosciated.
                # We will pass everything in as a kwarg - this means we don't
                # need to worry about maintaining argument order. This will
                # make adding additional arguments simple.

                kwargs = [[inp.name, self._get_key_or_value(inp, t_ind, f_ind)]
                          for inp in self. inputs]

                graph[(solver_name, t_ind, f_ind)] = \
                    (apply, solver_wrapper, [], (dict, kwargs))

        layers.update({solver_name: graph})
        deps.update({solver_name: {k for k in deps.keys()}})

        return graph, layers, deps, token

    def _get_key_or_value(self, inp, t_ind, f_ind):
        """Given an input, returns appropriate key/value based on indices."""

        if isinstance(inp.value, da.Array):
            chunked_values = list(flatten(inp.value.__dask_keys__()))
        else:
            chunked_values = inp.value

        if inp.index_string is None:
            return inp.value
        elif inp.index_string == "tf":
            ind = np.ravel_multi_index((t_ind, f_ind),
                                       (self.n_t_chunks, self.n_f_chunks))
            return chunked_values[ind]
        elif inp.index_string == "t":
            return chunked_values[t_ind]
        elif inp.index_string == "f":
            return chunked_values[f_ind]


# def rec_list_get(lst, ind):
#     if len(ind) > 1:
#         return rec_list_get(lst[ind[0]], ind[1:])
#     else:
#         return lst[ind[0]]
