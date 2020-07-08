import numpy as np
import dask.array as da
from dask.array.core import HighLevelGraph
from operator import getitem
from dask.base import tokenize
from quartical.calibration.solver import solver_wrapper
from dask.core import flatten
from collections import namedtuple

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

    term_type_list = [getattr(opts, "{}_type".format(t))
                      for t in opts.solver_gain_terms]

    # Based on the inputs, generate a unique hash which can be used to uniquely
    # identify nodes in the graph.
    token = tokenize(*dask_inputs)

    # Layers are the high level graph structure. Initialise with the dasky
    # inputs. Similarly for their dependencies.
    layers = {inp.name: inp.__dask_graph__() for inp in dask_inputs}
    deps = {inp.name: inp.__dask_graph__().dependencies for inp in dask_inputs}

    solver_name = "solver-" + token

    solver_dsk = {}

    # Set up the solver graph. Loop over the chunked axes.
    for t in range(n_t_chunks):
        for f in range(n_f_chunks):

            # Convert time and freq chunk indices to a flattened index.
            ind = t*n_f_chunks + f

            # Each term may differ in both type and shape - we build a spec
            # per term, per chunk. TODO: This can possibly be refined.
            term_spec_list = []

            spec_iterator = zip(opts.solver_gain_terms,
                                term_type_list,
                                [gxds.CHUNK_SPEC for gxds in gain_xds_list])

            for tn, tt, cs in spec_iterator:

                shape = (cs.tchunk[t],
                         cs.fchunk[f],
                         cs.achunk[0],
                         cs.dchunk[0],
                         cs.cchunk[0])

                term_spec_list.append(term_spec_tup(tn, tt, shape))

            # Set up the per-chunk solves. Note how keys are assosciated.
            # TODO: Figure out how to pass ancilliary information into the
            # solver wrapper. Possibly via a dict constructed above.

            args = [model_col_keys[ind],
                    data_col_keys[ind],
                    ant1_col_keys[t],
                    ant2_col_keys[t],
                    weight_col_keys[ind],
                    t_map_arr_keys[t],
                    f_map_arr_keys[f],
                    d_map_arr,
                    corr_mode,
                    term_spec_list]

            kwargs = (dict, [["foo", model_col_keys[ind]]])

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
