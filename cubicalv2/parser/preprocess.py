# -*- coding: utf-8 -*-
from loguru import logger
import re
import dask.array as da


def interpret_model(opts):
    """Preprocesses the namespace/dictionary given by opts.

    Given a namespace/dictionary of options, this should verify that that
    the options can be understood. Some options specified as strings need
    further processing which may include the raising of certain flags.

    Args:
        opts: A namepsace/dictionary of options.

    Returns:
        Namespace: An updated namespace object.
    """

    model_columns = set()
    sky_models = set()
    internal_recipe = {}

    input_recipes = opts.input_model_recipe.replace(" ", "").split(":")

    if input_recipes == ['']:
        raise ValueError("No model recipe was specified. Please set/check "
                         "--input-model-recipe.")

    # TODO: Repeated .lsm files overwrite dictionary contents. This needs to
    # be fixed.

    for recipe_index, recipe in enumerate(input_recipes):

        internal_recipe[recipe_index] = []

        # A raw string is required to avoid insane escape characters. Splits
        # on understood operators, ~ for subtract, + for add.

        ingredients = re.split(r'([\+~])', recipe)

        # Behaviour of re.split guarantees every second term is either a column
        # or .lsm. This may lead to the first element being an empty string.

        # Split the ingredients into operations and model sources. We preserve
        # empty strings in the recipe to avoid more complicated code elsewhere.

        for ingredient in ingredients:

            if ingredient in "~+" and ingredient != "":

                operation = da.add if ingredient == "+" else da.subtract
                internal_recipe[recipe_index].append(operation)

            elif ".lsm.html" in ingredient:

                filename, _, tags = ingredient.partition("@")
                tags = tags.split(",")
                sky_models.add(filename)

                internal_recipe[recipe_index].append((filename, tags))

            elif ingredient != "":
                model_columns.add(ingredient)
                internal_recipe[recipe_index].append(ingredient)

            else:
                internal_recipe[recipe_index].append(ingredient)

    logger.info("The following model sources were obtained from "
                "--input-model-recipe: \n"
                "   Columns: {} \n"
                "   Sky Models: {}",
                model_columns or 'None',
                sky_models or 'None')

    # Add processed recipe components to opts Namespace.

    opts._model_columns = model_columns
    opts._sky_models = sky_models
    opts._internal_recipe = internal_recipe
    opts._predict = True if sky_models else False

    if opts._predict:
        logger.info("Recipe contains sky models - enabling prediction step.")
