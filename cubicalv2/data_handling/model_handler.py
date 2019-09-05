# -*- coding: utf-8 -*-
import dask.array as da
import numpy as np
from cubicalv2.data_handling.predict import predict
from loguru import logger


def make_model(ms_xds, opts):

    xds_predicts = predict(ms_xds, opts) if opts._predict else ms_xds

    for xds, xds_predict in zip(ms_xds, xds_predicts):

        for idx, recipe in opts._internal_recipe.items():

            ingredients = recipe[::2]
            operations = recipe[1::2]

            print(ingredients, operations)


        model_data = da.stack(model_data, axis=2).rechunk({2: len(sky_model)})

    model_xds = xds_predicts

    return model_xds

    # model_xds = []

    # for xds in ms_xds:

    # model_data = da.stack(model_data, axis=2).rechunk({2: len(sky_model)})

    # # Assign visibilities to MODEL_DATA array on the dataset
    # xds = xds.assign({"MODEL_DATA":
    #                  (("row", "chan", "dir", "corr"),model_data)})

    # model_xds.append(xds)