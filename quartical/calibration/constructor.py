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

    # Grab and flatten the keys of all the inputs we intend to pass to the
    # solver.

    model_col_keys = list(flatten(model_col.__dask_keys__()))
    data_col_keys = list(flatten(data_col.__dask_keys__()))
    ant1_col_keys = list(flatten(ant1_col.__dask_keys__()))
    ant2_col_keys = list(flatten(ant2_col.__dask_keys__()))
    weight_col_keys = list(flatten(weight_col.__dask_keys__()))
    t_map_arr_keys = list(flatten(t_map_arr.__dask_keys__()))
    f_map_arr_keys = list(flatten(f_map_arr.__dask_keys__()))

    # Tuple of all dasky inputs over which we will loop for establishing
    # dependencies.
    dask_inputs = (model_col,
                   data_col,
                   ant1_col,
                   ant2_col,
                   weight_col,
                   t_map_arr,
                   f_map_arr)

    # These should be guarateed to have the correct dimension in each chunking
    # axis. This is important for key matching.
    n_t_chunks = len(t_map_arr_keys)
    n_f_chunks = len(f_map_arr_keys)

    # Based on the inputs, generate a unique hash which can be used to uniquely
    # identify nodes in the graph.
    token = tokenize(*dask_inputs)

    # Layers are the high level graph structure. Initialise with the dasky
    # inputs. Similarly for their dependencies.
    layers = {inp.name: inp.__dask_graph__() for inp in dask_inputs}
    deps = {inp.name: inp.__dask_graph__().dependencies for inp in dask_inputs}

    solver_name = "solver-" + token

    solver_dsk = {}

    # Take the compact chunking info on the gain xdss and expand it.
    spec_list = expand_specs(gain_xds_list)

    # Set up the solver graph. Loop over the chunked axes.
    for t in range(n_t_chunks):
        for f in range(n_f_chunks):

            # Convert time and freq chunk indices to a flattened index.
            ind = t*n_f_chunks + f

            # Set up the per-chunk solves. Note how keys are assosciated.
            # We will pass everything in as a kwarg - this means we don't need
            # to worry about maintaining argument order. This will make
            # adding additional arguments simple.

            args = []

            # This is a dasky description of dict creation.
            kwargs = (dict, [["model", model_col_keys[ind]],
                             ["data", data_col_keys[ind]],
                             ["a1", ant1_col_keys[t]],
                             ["a2", ant2_col_keys[t]],
                             ["weights", weight_col_keys[ind]],
                             ["t_map_arr", t_map_arr_keys[t]],
                             ["f_map_arr", f_map_arr_keys[f]],
                             ["d_map_arr", d_map_arr],
                             ["corr_mode", corr_mode],
                             ["term_spec_list", spec_list[ind]]])

            solver_dsk[(solver_name, t, f,)] = \
                (apply, solver_wrapper, args, kwargs)

    # Add the solver layer and its dependencies.
    layers.update({solver_name: solver_dsk})
    deps.update({solver_name: {inp.name for inp in dask_inputs}})

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
