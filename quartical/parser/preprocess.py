# -*- coding: utf-8 -*-
from loguru import logger
import re
import dask.array as da
from collections import namedtuple
import os.path


sm_tup = namedtuple("sky_model", ("name", "tags"))


def interpret_model(opts):
    """Interpret the model recipe string.

    Given a namespace/dictionary of options, read and interpret the model
    recipe. Results are added to the options namepsace with leading
    leading underscores to indicate that they are set internally.

    Args:
        opts: A namepsace/dictionary of options.

    Returns:
        Namespace: An updated namespace object.
    """

    model_columns = set()
    sky_models = set()
    internal_recipe = {}

    # Strip accidental whitepsace from input recipe and splits on ":".
    input_recipes = opts.input_model.recipe.replace(" ", "").split(":")

    if input_recipes == ['']:
        raise ValueError("No model recipe was specified. Please set/check "
                         "--input-model-recipe.")

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
                tags = tuple(tags.split(",")) if tags else ()

                if not os.path.isfile(filename):
                    raise FileNotFoundError("{} not found.".format(filename))

                sky_model = sm_tup(filename, tags)
                sky_models.add(sky_model)
                internal_recipe[recipe_index].append(sky_model)

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
                {sm.name for sm in sky_models} or 'None')

    # Add processed recipe components to opts Namespace.

    opts._model_columns = model_columns
    opts._sky_models = sky_models
    opts._internal_recipe = internal_recipe
    opts._predict = True if sky_models else False

    if opts._predict:
        logger.info("Recipe contains sky models - enabling prediction step.")


def check_opts(opts):
    """Check that there is nothing untoward in the options namespace.

    Given a namespace/dictionary of options, check options which may trip
    users up or cause failures. Log warnings about experimental modes.

    Args:
        opts: A namepsace/dictionary of options.
    """

    # TODO: Add this functionality - should check opts for problems in addition
    # to interpreting weird options. Can also raise flags for different modes
    # of operation. The idea is that all our configuration state lives in this
    # options dictionary. Down with OOP!

    if opts.input_ms.is_bda:
        logger.warning("BDA data is only partially supported. Please report "
                       "problems via the issue tracker.")

    return
