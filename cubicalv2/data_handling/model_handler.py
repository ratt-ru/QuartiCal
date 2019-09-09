# -*- coding: utf-8 -*-
import dask.array as da
import numpy as np
from cubicalv2.data_handling.predict import predict
from loguru import logger
from xarray import DataArray


def make_model(ms_xds, opts):

    xds_predicts = predict(ms_xds, opts) if opts._predict else [{}]*len(ms_xds)

    model_xds = []

    for xds, xds_predict in zip(ms_xds, xds_predicts):

        recipe_model = []

        for idx, recipe in opts._internal_recipe.items():

            ingredients = recipe[::2]
            operations = recipe[1::2]

            if not operations:

                if ingredients[0] in xds_predict.keys():
                    result = xds_predict.get(ingredients[0])
                else:
                    result = [xds.get(ingredients[0]).data]

                recipe_model.extend(result)

                continue

            for op_idx, op in enumerate(operations):

                if op_idx == 0:
                    if ingredients[op_idx] in xds_predict.keys():
                        result = xds_predict.get(ingredients[op_idx])
                    else:
                        result = [xds.get(ingredients[op_idx]).data]

                if ingredients[op_idx + 1] in xds_predict.keys():
                    in_b = xds_predict.get(ingredients[op_idx + 1])
                else:
                    in_b = [xds.get(ingredients[op_idx + 1]).data]

                if len(in_b) > 1 and len(result) > 1:
                    raise(ValueError("Model recipes do not support add or "
                                     "subtract operations between two "
                                     "direction-dependent inputs."))
                elif len(result) > len(in_b):
                    op(result[0], in_b[0], out=result[0])
                else:
                    result, in_b = in_b, result
                    op(result[0], in_b[0], out=result[0])

            recipe_model.extend(result)

        n_dir = len(recipe_model)
        model_data = da.stack(recipe_model, axis=2).rechunk({2: n_dir})

        xds = xds.assign({"MODEL_DATA":
                         (("row", "chan", "dir", "corr"), model_data)})

        model_xds.append(xds)

    return model_xds
