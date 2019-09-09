# -*- coding: utf-8 -*-
import dask.array as da
import numpy as np
from cubicalv2.data_handling.predict import predict
from loguru import logger
import xarray


def make_model(ms_xds, opts):

    xds_predicts = predict(ms_xds, opts) if opts._predict else ms_xds

    model_xds = []

    for xds, xds_predict in zip(ms_xds, xds_predicts):

        recipe_model = []

        for idx, recipe in opts._internal_recipe.items():

            ingredients = recipe[::2]
            operations = recipe[1::2]

            if not operations:
                # try:
                #     result  = getattr(xds_predict, ingredients[0])
                # except AttributeError:
                #     result = getattr(xds, ingredients[0])
                # else:
                #     if isinstance(result, array.DataArray)
                #         result = [result.data]
                #     elif not isinstance(result, list):
                #         result = [result]
                #     else:
                #         raise TypeError("blah")

                result = xds_predict.get(ingredients[0],
                                         xds.get(ingredients[0]))
                result = result.data if isinstance(result, xarray.DataArray) else result
                result = [result] if not isinstance(result, list) else result
                recipe_model.extend(result)

                continue

            for op_idx, op in enumerate(operations):

                if op_idx == 0:
                    result = xds_predict.get(ingredients[op_idx],
                                             [xds.get(ingredients[op_idx],
                                                      None)])

                in_b = xds_predict.get(ingredients[op_idx + 1],
                                       [xds.get(ingredients[op_idx + 1],
                                                None)])

                # TODO: Currently this assumes that the first ingredient can
                # be a list. This needs to be changed to work even if a
                # column/untagged lsm is the first input.

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
