# -*- coding: utf-8 -*-
# Sets up logger - hereafter import logger from Loguru.

from contextlib import ExitStack

import quartical.logging.init_logger  # noqa
from loguru import logger
from quartical.parser import parser, preprocess
from quartical.data_handling.ms_handler import (read_xds_list,
                                                write_xds_list,
                                                preprocess_xds_list)
from quartical.data_handling.model_handler import add_model_graph
from quartical.calibration.calibrate import add_calibration_graph
from quartical.flagging.flagging import finalise_flags, add_mad_graph
import time
from dask.diagnostics import ProgressBar
import dask
from dask.distributed import Client, LocalCluster
import zarr


@logger.catch
def execute():
    with ExitStack() as stack:
        _execute(stack)


def _execute(exitstack):
    """Runs the application."""

    opts = parser.parse_inputs()

    # TODO: This check needs to be fleshed out substantially.

    preprocess.check_opts(opts)
    preprocess.interpret_model(opts)

    if opts.parallel_scheduler == "distributed" and opts.parallel_address:
        logger.info("Initializing distributed client.")
        client = Client(opts.parallel_address)
        exitstack.enter_context(client)
        logger.info("Distributed client sucessfully initialized.")
    elif opts.parallel_scheduler == "distributed":
        logger.info("Initializing distributed client using LocalCluster.")
        cluster = LocalCluster(processes=opts.parallel_nworker > 1,
                               n_workers=opts.parallel_nworker,
                               threads_per_worker=opts.parallel_nthread,
                               memory_limit=0)
        cluster = exitstack.enter_context(cluster)
        exitstack.enter_context(Client(cluster))
        logger.info("Distributed client sucessfully initialized.")

    t0 = time.time()

    # Reads the measurement set using the relavant configuration from opts.
    data_xds_list, ref_xds_list, col_kwrds = read_xds_list(opts)

    from glob import glob
    import xarray
    logger.info("Reading data from zms.")
    data_xds_list = [xarray.open_zarr(pth)
                     for pth in glob("3C147_bandpass.zms/*")]

    # import zarr
    # store = zarr.DirectoryStore("3C147_bandpass.zms/")

    # zarr_writes = []
    # for xds in data_xds_list:
    #     zarr_writes.append(
    #         xds.to_zarr(store,
    #                     mode="w",
    #                     group=f"F{xds.FIELD_ID:0>4}"
    #                           f"D{xds.DATA_DESC_ID:0>4}",
    #                     compute=False))

    # dask.compute(zarr_writes)
    # return

    # data_xds_list = data_xds_list[:2]
    # ref_xds_list = ref_xds_list[:16]

    # Preprocess the xds_list - initialise some values and fix bad data.
    data_xds_list = preprocess_xds_list(data_xds_list, col_kwrds, opts)

    # Model xds is a list of xdss onto which appropriate model data has been
    # assigned.
    data_xds_list = add_model_graph(data_xds_list, opts)

    # Adds the dask graph describing the calibration of the data.
    gains_per_xds, data_xds_list = \
        add_calibration_graph(data_xds_list, col_kwrds, opts)

    if opts.flags_mad_enable:
        data_xds_list = add_mad_graph(data_xds_list, opts)

    writable_xds = finalise_flags(data_xds_list, col_kwrds, opts)

    writes = write_xds_list(writable_xds, ref_xds_list, col_kwrds, opts)

    # This shouldn't be here. TODO: Move to separate function. In fact, this
    # entire write construction needs some tidying.
    store = zarr.DirectoryStore("qcal_gains")

    gain_writes = []

    for xds_ind, gain_terms in enumerate(gains_per_xds):
        term_writes = []
        for term_ind, term in enumerate(gain_terms):
            term_write = term.chunk({"time_int": -1}).to_zarr(
                store,
                mode="w",
                group=f"{term.NAME}{xds_ind}",
                compute=False)
            term_writes.append(term_write)
        gain_writes.append(term_writes)

    writes = [writes] if not isinstance(writes, list) else writes

    # import pdb; pdb.set_trace()

    stride = len(writes)//len(gain_writes)

    # Match up column and gain writes - avoids recompute, and necessary for
    # handling BDA data.
    outputs = []
    for ind in range(len(gain_writes)):

        ms_writes = writes[ind*stride: (ind + 1)*stride]

        outputs.append(dask.delayed(tuple)([*gain_writes[ind]]))

    logger.success("{:.2f} seconds taken to build graph.", time.time() - t0)

    t0 = time.time()

    with ProgressBar():
        dask.compute(outputs,
                     num_workers=opts.parallel_nthread,
                     optimize_graph=True,
                     scheduler=opts.parallel_scheduler)

    logger.success("{:.2f} seconds taken to execute graph.", time.time() - t0)

    # dask.visualize(outputs,
    #                color='order', cmap='autumn',
    #                filename='order.pdf', node_attr={'penwidth': '10'})

    # dask.visualize(outputs,
    #                filename='graph.pdf',
    #                optimize_graph=True)

    # dask.visualize(*gains_per_xds["G"],
    #                filename='gain_graph',
    #                format="pdf",
    #                optimize_graph=True,
    #                collapse_outputs=True,
    #                node_attr={'penwidth': '4',
    #                           'fontsize': '18',
    #                           'fontname': 'helvetica'},
    #                edge_attr={'penwidth': '2', })
