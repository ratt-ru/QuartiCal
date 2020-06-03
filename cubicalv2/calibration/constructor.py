import numpy as np
import dask.array as da
from dask.array.core import HighLevelGraph
from operator import getitem
from dask.base import tokenize
from cubicalv2.calibration.solver import solver_wrapper
from dask.core import flatten
from collections import namedtuple


term_spec_tup = namedtuple("term_spec", "type shape")


def construct_solver(model_col,
                     data_col,
                     ant1_col,
                     ant2_col,
                     weight_col,
                     t_map_arr,
                     f_map_arr,
                     d_map_arr,
                     corr_mode,
                     term_shape_list,
                     term_chunk_list,
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
        gain_list: A list of dask.Arrays corresponding to the gain terms.
        flag_list: A list of dask.Arrays corresponding to the gain flags.
        param_list: A list of dask.Arrays corresponding to parameterisations.
        opts: A Namespace object containing global options.

    Returns:
        output_gain_list: A list of dask.Arrays containing the gains.
        info_array: An dask.Array containing convergence info.
    """

    # This is slightly ugly but works for now. Grab and flatten the keys of all
    # the inputs we intend to pass to the solver.

    model_col_keys = list(flatten(model_col.__dask_keys__()))
    data_col_keys = list(flatten(data_col.__dask_keys__()))
    ant1_col_keys = list(flatten(ant1_col.__dask_keys__()))
    ant2_col_keys = list(flatten(ant2_col.__dask_keys__()))
    weight_col_keys = list(flatten(weight_col.__dask_keys__()))
    t_map_arr_keys = list(flatten(t_map_arr.__dask_keys__()))
    f_map_arr_keys = list(flatten(f_map_arr.__dask_keys__()))

    # Tuple of all dasky inputs over which we will loop for establishing
    # dependencies.
    args = (model_col, data_col, ant1_col, ant2_col, weight_col,
            t_map_arr, f_map_arr)

    # These should be guarateed to have the correct dimension in each chunking
    # axis. This is important for key matching.
    n_t_chunks = len(t_map_arr_keys)
    n_f_chunks = len(f_map_arr_keys)
    n_term = len(opts.solver_gain_terms)

    term_type_list = [getattr(opts, "{}_type".format(t))
                      for t in opts.solver_gain_terms]

    # Based on the inputs, generate a unique hash which can be used to uniquely
    # identify nodes in the graph.
    token = tokenize(*args)

    # Layers are the high level graph structure. Initialise with the dasky
    # inputs. Similarly for the their dependencies.
    layers = {inp.name: inp.__dask_graph__() for inp in args}
    deps = {inp.name: inp.__dask_graph__().dependencies for inp in args}

    solver_name = "solver-" + token

    solver_dsk = {}

    # Set up the solver graph. Loop over the chunked axes.
    for t in range(n_t_chunks):
        for f in range(n_f_chunks):

            ind = t*n_f_chunks + f

            term_spec_list = []
            for tt, cs in zip(term_type_list, term_chunk_list):

                shape = (cs.tchunk[t],
                         cs.fchunk[f],
                         cs.achunk[0],
                         cs.dchunk[0],
                         cs.cchunk[0])

                term_spec_list.append(term_spec_tup(tt, shape))

            # Set up the per-chunk solves. Note how keys are assosciated.
            solver_dsk[(solver_name, t, f,)] = \
                (solver_wrapper,
                 model_col_keys[ind],
                 data_col_keys[ind],
                 ant1_col_keys[t],
                 ant2_col_keys[t],
                 weight_col_keys[ind],
                 t_map_arr_keys[t],
                 f_map_arr_keys[f],
                 d_map_arr,
                 corr_mode,
                 term_spec_list)

    # Add the solver layer and its dependencies.
    layers.update({solver_name: solver_dsk})
    deps.update({solver_name: {inp.name for inp in args}})

    # The following constructs the getitem layer which is necessary to handle
    # returning several results from the solver.
    gain_names = \
        ["gain-{}-{}".format(i, token) for i in range(n_term)]

    for gi, gn in enumerate(gain_names):
        get_gain = {(gn, k[1], k[2], 0, 0, 0): (getitem, (getitem, k, 0), gi)
                    for i, k in enumerate(solver_dsk.keys())}

        layers.update({gn: get_gain})
        deps.update({gn: {solver_name}})

    # We also want to get the named tuple of convergance info - note that this
    # may no longer be needed as we have a way to pull out values from the
    # solver in a more transparent fashion than before. TODO: Change this.
    info_name = "info-" + token

    get_info = \
        {(info_name, k[1], k[2]): (getitem, k, 1)
         for i, k in enumerate(solver_dsk.keys())}

    layers.update({info_name: get_info})
    deps.update({info_name: {solver_name}})

    # Turn the layers and dependencies into a high level graph.
    graph = HighLevelGraph(layers, deps)

    # Now that we have the graph, we need to construct arrays from the results.
    # We loop over the output gains and pack them into a list of dask arrays.
    output_gain_list = []
    for gi, gn in enumerate(gain_names):
        output_gain_list.append(da.Array(graph,
                                         name=gn,
                                         chunks=term_chunk_list[gi],
                                         dtype=np.complex64))

    # We also partially unpack the info tuple, but see earlier comment. This
    # can probably be refined.
    info_array = da.Array(graph,
                          name=info_name,
                          chunks=((1,)*n_t_chunks,
                                  (1,)*n_f_chunks),
                          dtype=np.float32)

    return output_gain_list, info_array
