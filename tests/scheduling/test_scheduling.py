# from argparse import Namespace
# from contextlib import ExitStack

# import dask
# import dask.array as da
# from daskms.constants import DASKMS_PARTITION_KEY as PARTITION_KEY
# from distributed import Client, LocalCluster
# import numpy as np
# import pytest
# import xarray as xr

# from africanus.rime.dask import phase_delay
# from quartical.data_handling.ms_handler import read_xds_list
# from quartical.scheduling import install_plugin, annotate, dataset_partition


# def test_array_annotation():
#     A = da.ones((10, 10), chunks=(3, 4), dtype=np.complex64)
#     assert A.__dask_graph__().layers[A.name].annotations is None
#     annotate(A)

#     expected = {"__dask_array__": {
#         "chunks": ((3, 3, 3, 1), (4, 4, 2)),
#         "dtype": "complex64"}
#     }

#     assert A.__dask_graph__().layers[A.name].annotations == expected


# def test_xarray_datarray_annotation():
#     A = da.ones((10, 10), chunks=(3, 4), dtype=np.complex64)
#     xA = xr.DataArray(A, dims=("x", "y"))
#     assert A.__dask_graph__().layers[A.name].annotations is None
#     annotate(xA)

#     expected = {"__dask_array__": {
#         "dims": ("x", "y"),
#         "chunks": ((3, 3, 3, 1), (4, 4, 2)),
#         "dtype": "complex64"}
#     }

#     assert A.__dask_graph__().layers[A.name].annotations == expected


# def test_xarray_dataset_annotation():
#     A = da.ones((10, 10), chunks=(3, 4), dtype=np.complex64)
#     B = da.ones((10,), chunks=3, dtype=np.float32)

#     partition_schema = (("SCAN", "int32"), ("DDID", "int32"))

#     ds = xr.Dataset({"A": (("x", "y"), A), "B": (("x",), B)},
#                     attrs={
#                         # Schema must be present to extract
#                         # partition values
#                         PARTITION_KEY: partition_schema,
#                         "SCAN": 1,
#                         "DDID": 2
#                     })

#     assert A.__dask_graph__().layers[A.name].annotations is None
#     assert B.__dask_graph__().layers[B.name].annotations is None

#     annotate(ds)

#     expected = {"__dask_array__": {
#         "dims": ("x", "y"),
#         "chunks": ((3, 3, 3, 1), (4, 4, 2)),
#         "partition": (("SCAN", 1), ("DDID", 2)),
#         "dtype": "complex64"}
#     }

#     assert A.__dask_graph__().layers[A.name].annotations == expected

#     expected = {"__dask_array__": {
#         "dims": ("x",),
#         "chunks": ((3, 3, 3, 1),),
#         "partition": (("SCAN", 1), ("DDID", 2)),
#         "dtype": "float32"}
#     }

#     assert B.__dask_graph__().layers[B.name].annotations == expected


# def test_distributed(base_opts):
#     opts = Namespace(**vars(base_opts))

#     with ExitStack() as stack:
#         cluster = \
#             stack.enter_context(LocalCluster(processes=False, n_workers=4))
#         client = stack.enter_context(Client(cluster))
#         scheduler = cluster.scheduler

#         opts.input_ms_time_chunk = 4
#         opts.parallel = Namespace(scheduler='distributed',
#                                   address=scheduler.address)

#         opts._model_columns = ["MODEL_DATA"]

#         datasets, _ = read_xds_list(opts)
#         assert len(datasets) == 2
#         assert len(datasets[0].chunks["row"]) == 29
#         assert len(datasets[1].chunks["row"]) == 27

#         client.run_on_scheduler(install_plugin)
#         assert len(scheduler.plugins) > 1

#         chan_freq = da.linspace(.856e9, 2*.856e9, 16, chunks=4)
#         lm = da.random.random((4, 2), chunks=(2, 2))

#         new_datasets = []

#         for ds in datasets:
#             K = phase_delay(lm, ds.UVW.data, chan_freq)
#             partition = dataset_partition(ds)
#             annotate(K, dims=("source", "row", "chan"), partition=partition)
#             vis = K.sum(axis=0)[:, :, None]
#             nds = xr.Dataset({"DATA": (("row", "chan", "corr"), vis)})
#             nds.attrs.update(ds.attrs)
#             new_datasets.append(nds)

#         annotate(new_datasets)

#         with dask.config.set(optimization__fuse__active=False):
#             dask.compute(new_datasets)
