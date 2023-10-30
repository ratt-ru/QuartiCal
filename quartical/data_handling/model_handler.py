# -*- coding: utf-8 -*-
import dask.array as da
import numpy as np
from quartical.data_handling.predict import predict
from quartical.data_handling.angles import apply_parangles
from quartical.config.preprocess import IdentityRecipe, Ingredients
from quartical.utils.array import flat_ident_like
from loguru import logger  # noqa


def add_model_graph(
    data_xds_list,
    parangle_xds_list,
    model_vis_recipe,
    ms_path,
    model_opts
):
    """Creates the graph necessary to produce a model per xds.

    Given a list of input xarray data sets and the options, constructs a graph
    in accordance with the internal model recipe. This can produce
    direction-dependent models using the recipe syntax.

    Args:
        data_xds_list: A list of xarray datasets generated from an MS.
        parangle_xds_list: A list of xarray datasets containing parallactic
            angle information.
        model_vis_recipe: A Recipe object.
        ms_path: Path to the input measurement set.
        model_opts: A ModelInputs configuration object.

    Returns:
        model_xds_list: A list of xarray datasets containing the model data.
    """

    # Generates a predicition scheme (graph) per-xds. If no predict is
    # required, it is a list of empty dictionaries.

    # TODO: Add handling for mds inputs. This will need to read the model
    # and figure out the relevant steps to take.

    predict_required = bool(model_vis_recipe.ingredients.sky_models)
    degrid_required = bool(model_vis_recipe.ingredients.degrid_models)

    # TODO: Ensure that things work correctly when we have a mixture of the
    # below.

    predict_schemes = [{}]*len(data_xds_list)

    if predict_required:
        rime_schemes = predict(
            data_xds_list,
            model_vis_recipe,
            ms_path,
            model_opts
        )
        predict_schemes = [
            {**ps, **rs} for ps, rs in zip(predict_schemes, rime_schemes)
        ]

    if degrid_required:

        try:
            from quartical.data_handling.degridder import degrid
        except ImportError:
            raise ImportError(
                "QuartiCal was unable to import the degrid module. This may "
                "indicate that QuartiCal was installed without the necessary "
                "extras. Please try 'pip install quartical[degrid]'. If the "
                "error persists, please raise an issue."
            )

        degrid_schemes = degrid(
            data_xds_list,
            model_vis_recipe,
            ms_path,
            model_opts
        )
        predict_schemes = [
            {**ps, **ds} for ps, ds in zip(predict_schemes, degrid_schemes)
        ]

    # Special case: in the event that we have an IdentityRecipe, modify the
    # datasets and model appropriately.
    if isinstance(model_vis_recipe, IdentityRecipe):
        data_xds_list, model_vis_recipe = assign_identity_model(data_xds_list)

    model_columns = model_vis_recipe.ingredients.model_columns

    # NOTE: At this point we are ready to construct the model array. First,
    # however, we need to apply parallactic angle corrections to model columns
    # which require them. P Jones is applied to predicted components
    # internally, so we only need to consider model columns for now.

    n_corr = {xds.dims["corr"] for xds in data_xds_list}.pop()

    if model_opts.apply_p_jones:
        # NOTE: Applying parallactic angle when there are fewer than four
        # correlations is problematic for linear feeds as it amounts to
        # rotating information to/from correlations which are not present i.e.
        # it is not reversible. We support it for input models but warn the
        # user that it is not a good idea.
        if n_corr != 4:
            logger.warning(
                "input_model.apply_p_jones is not recommended for data with "
                "less than four correlations. Proceed with caution."
            )
        data_xds_list = apply_parangles(data_xds_list,
                                        parangle_xds_list,
                                        model_columns)

    # Initialise a list to contain the xdss after the model data has been
    # assigned.

    model_xds_list = []

    # Loops over the xdss and prediciton schemes.

    for xds, prediction in zip(data_xds_list, predict_schemes):

        model = []  # A list to contain the model generated by the recipe.

        for instruction in model_vis_recipe.instructions.values():

            ingredients = instruction[::2]  # Columns/sky models.
            operations = instruction[1::2]  # Add or subtract operations.

            if not operations:  # Handle recipe components without operations.
                if ingredients[0] in prediction.keys():
                    result = prediction.get(ingredients[0])
                else:
                    result = [xds.get(ingredients[0]).data]  # Must be a list.

                model.extend(result)

                continue

            # If we have operations, we loop over them to construct the model.

            for op_idx, op in enumerate(operations):

                # The first operation will require two inputs. Thereafter,
                # the result of the previous operation will be one of the
                # inputs. If the first ingredient is an empty string, we must
                # have a leading operation, usually a negative. For simplicity,
                # a leading negative is implemented by subtracting the first
                # ingredient from zero.

                if op_idx == 0:
                    if ingredients[op_idx] == "":
                        in_a = [0]
                    elif ingredients[op_idx] in prediction.keys():
                        in_a = prediction.get(ingredients[op_idx])
                    else:
                        in_a = [xds.get(ingredients[op_idx]).data]

                if ingredients[op_idx + 1] in prediction.keys():
                    in_b = prediction.get(ingredients[op_idx + 1])
                else:
                    in_b = [xds.get(ingredients[op_idx + 1]).data]

                # Adding and subtracting direction dependent models is not
                # supported. If we have an operation with a single
                # direction-dependent term, it is assumed to apply to the first
                # direction only.

                if len(in_a) > 1 and len(in_b) > 1:
                    raise ValueError(
                        "Model recipes do not support add or subtract "
                        "operations between two direction-dependent inputs."
                    )
                elif len(in_a) > len(in_b):
                    result = [op(in_a[0], in_b[0]), *in_a[1:]]
                elif len(in_a) < len(in_b):
                    result = [op(in_a[0], in_b[0]), *in_b[1:]]
                else:
                    result = [op(in_a[0], in_b[0])]
                in_a = result

            model.extend(result)  # Add terms generated by the operations.

        n_dir = len(model)  # The number of terms is the number of directions.

        # This creates the direction axis by stacking the model terms. The
        # rechunking is necessary to ensure the solver gets appropriate blocks.
        model = da.stack(model, axis=2).rechunk({2: n_dir})

        # Get rid of model columns which are not used after this point.
        modified_xds = xds.drop_vars(model_columns)

        modified_xds = modified_xds.assign(
            {"MODEL_DATA": (("row", "chan", "dir", "corr"), model)}
        )

        model_xds_list.append(modified_xds)

    return model_xds_list


def assign_identity_model(data_xds_list):
    """Given dataset list, creates recipe and assigns an identity model.

    This is a special case where we have no input model and simply want to use
    the identity. This is common when constraining phase solutions on a
    calibrator.

    Args:
        data_xds_list: A list of xarray.Datasets objects containing MS data.

    Returns:
        data_xds_list: A list of xarray.Datasets with new model assigned.
        recipe: A modified Recipe object consistent with this case.
    """

    ingredients = Ingredients({"__IDENT__"}, set(), set())
    instructions = {0: ["__IDENT__"]}

    recipe = IdentityRecipe(ingredients, instructions)

    model_dims = [
        (
            xds.dims['row'],
            xds.dims['chan'],
            xds.dims['corr']
        )
        for xds in data_xds_list
    ]

    model_chunks = [
        (
            xds.chunks['row'],
            xds.chunks['chan'],
            xds.chunks['corr']
        )
        for xds in data_xds_list
    ]

    data_xds_list = [
        xds.assign(
            {
                "__IDENT__": (
                    ("row", "chan", "corr"),
                    flat_ident_like(
                        da.empty(dims, chunks=chunks, dtype=np.complex64)
                    )
                )
            }
        )
        for xds, dims, chunks in zip(
            data_xds_list, model_dims, model_chunks
        )
    ]

    return data_xds_list, recipe
