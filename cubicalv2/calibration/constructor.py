import numpy as np
import dask.array as da
from itertools import chain
from dask.array.core import HighLevelGraph
from operator import getitem
from dask.base import tokenize


def flatten(l):
    for el in l:
        if isinstance(el, list):
            yield from flatten(el)
        else:
            yield el


def construct_solver(chain_solver,
                     model_col,
                     data_col,
                     ant1_col,
                     ant2_col,
                     weight_col,
                     t_map_arr,
                     f_map_arr,
                     d_map_arr,
                     compute_jhj_and_jhr,
                     compute_update,
                     gain_list,
                     gain_flag_list):

    model_col_keys = list(flatten(model_col.__dask_keys__()))
    data_col_keys = list(flatten(data_col.__dask_keys__()))
    ant1_col_keys = list(flatten(ant1_col.__dask_keys__()))
    ant2_col_keys = list(flatten(ant2_col.__dask_keys__()))
    weight_col_keys = list(flatten(weight_col.__dask_keys__()))
    t_map_arr_keys = list(flatten(t_map_arr.__dask_keys__()))
    f_map_arr_keys = list(flatten(f_map_arr.__dask_keys__()))
    gain_keys = [list(flatten(gain.__dask_keys__())) for gain in gain_list]
    gain_flag_keys = \
        [list(flatten(flag.__dask_keys__())) for flag in gain_flag_list]

    args = [model_col, data_col, ant1_col, ant2_col, weight_col,
            t_map_arr, f_map_arr, *gain_list, *gain_flag_list]

    n_t_chunks = len(t_map_arr_keys)
    n_f_chunks = len(f_map_arr_keys)

    solver_dsk = {}

    token = tokenize(*args)

    solver_name = "solver-" + token

    # Set up the solver layer. This is the first step - the next is to add the
    # getitem layers to pull values out.
    for t in range(n_t_chunks):
        for f in range(n_f_chunks):
            ind = t*n_f_chunks + f

            gain_inputs = \
                list(chain.from_iterable(zip(gain_keys, gain_flag_keys)))

            solver_dsk[(solver_name, t, f,)] = \
                (chain_solver,
                 model_col_keys[ind],
                 data_col_keys[ind],
                 ant1_col_keys[t],
                 ant2_col_keys[t],
                 weight_col_keys[ind],
                 t_map_arr_keys[t],
                 f_map_arr_keys[f],
                 d_map_arr,
                 compute_jhj_and_jhr,
                 compute_update,
                 *[gi[ind] for gi in gain_inputs])

    layers = {inp.name: inp.__dask_graph__() for inp in args}
    deps = {inp.name: inp.__dask_graph__().dependencies for inp in args}

    layers.update({solver_name: solver_dsk})
    deps.update({solver_name: {inp.name for inp in args}})

    gain_names = \
        ["gain-{}-{}".format(i, token) for i in range(len(gain_list))]

    for gi, gn in enumerate(gain_names):
        get_gain = \
            {(gn, k[1], k[2], 0, 0, 0): (getitem, (getitem, k, 0), gi)
             for i, k in enumerate(solver_dsk.keys())}

        layers.update({gn: get_gain})
        deps.update({gn: {solver_name}})

    info_name = "info-" + token

    get_info = \
        {(info_name, k[1], k[2]): (getitem, k, 1)
         for i, k in enumerate(solver_dsk.keys())}

    layers.update({info_name: get_info})
    deps.update({info_name: {solver_name}})

    graph = HighLevelGraph(layers, deps)

    output_gain_list = []
    for gi, gn in enumerate(gain_names):
        output_gain_list.append(da.Array(graph,
                                         name=gn,
                                         chunks=gain_list[gi].chunks,
                                         dtype=np.complex64))

    info_array = da.Array(graph,
                          name=info_name,
                          chunks=((1,)*n_t_chunks,
                                  (1,)*n_f_chunks),
                          dtype=np.float32)

    return output_gain_list, info_array
