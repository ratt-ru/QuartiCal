# -*- coding: utf-8 -*-
import dask.array as da
import numpy as np
from cubicalv2.data_handling.predict import predict
from loguru import logger


def make_model(ms_xds, opts):

    return predict(ms_xds, opts) if opts._predict else ms_xds

    # model_xds = []

    # for xds in ms_xds:

    # model_data = da.stack(model_data, axis=2).rechunk({2: len(sky_model)})

    # # Assign visibilities to MODEL_DATA array on the dataset
    # xds = xds.assign({"MODEL_DATA":
    #                  (("row", "chan", "dir", "corr"),model_data)})

    # model_xds.append(xds)