import numpy as np 
import dask
import dask.array as da
from itertools import chain, repeat
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
                     *gain_list):

    model_col_keys = list(flatten(model_col.__dask_keys__())) 
    data_col_keys = list(flatten(data_col.__dask_keys__())) 
    ant1_col_keys = list(flatten(ant1_col.__dask_keys__())) 
    ant2_col_keys = list(flatten(ant2_col.__dask_keys__()))
    weight_col_keys = list(flatten(weight_col.__dask_keys__()))
    t_map_arr_keys = list(flatten(t_map_arr.__dask_keys__()))
    f_map_arr_keys = list(flatten(f_map_arr.__dask_keys__()))

    gain = gain_list[0]
    gain_keys = list(flatten(gain.__dask_keys__()))

    args = [model_col, data_col, ant1_col, ant2_col, weight_col,
            t_map_arr, f_map_arr, gain]

    n_t_chunks = len(t_map_arr_keys)
    n_f_chunks = len(f_map_arr_keys)

    solver_dsk = {}

    token = tokenize(*args)

    solver_name = "solver" + token

    # Set up the solver layer. This is the first step - the next is to add the 
    # getitem layers to pull values out. 
    for t in range(n_t_chunks):
        for f in range(n_f_chunks):
            ind = t*n_f_chunks + f

            solver_dsk[(solver_name, t, f,)] = (chain_solver,
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
                                                gain_keys[ind])

    layers = {inp.name: inp.__dask_graph__() for inp in args}
    deps = {inp.name: inp.__dask_graph__().dependencies for inp in args}

    layers.update({solver_name: solver_dsk})
    deps.update({solver_name: {inp.name for inp in args}})

    gain_name = "gain" + token

    get_gain = \
        {(gain_name, k[1], k[2], 0, 0, 0): (getitem, (getitem, k, 0), 0)
         for i, k in enumerate(solver_dsk.keys())}

    info_name = "info" + token

    get_info = \
        {(info_name, k[1], k[2]): (getitem, k, 1)
         for i, k in enumerate(solver_dsk.keys())}

    layers.update({gain_name: get_gain})
    deps.update({gain_name: {solver_name}})
    layers.update({info_name: get_info})
    deps.update({info_name: {solver_name}})

    graph = HighLevelGraph(layers, deps)

    out1 = da.Array(graph,
                    name=gain_name,
                    chunks=gain_list[0].chunks,
                    dtype=np.complex64)

    out2 = da.Array(graph,
                    name=info_name,
                    chunks=((1,)*n_t_chunks,
                            (1,)*n_f_chunks),
                    dtype=np.float32)

    return [out1], out2

    # print(out1.compute())
    # print(out2.compute())