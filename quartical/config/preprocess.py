# -*- coding: utf-8 -*-
from loguru import logger
import re
import dask.array as da
from collections import namedtuple
import os.path
from dataclasses import dataclass
from typing import List, Dict, Set, Any


sky_model_nt = namedtuple("sky_model_nt", ("name", "tags"))


@dataclass
class Ingredients:
    model_columns: Set[Any]
    sky_models: Set[sky_model_nt]


@dataclass
class Recipe:
    ingredients: Ingredients
    instructions: Dict[int, List[Any]]


def transcribe_recipe(user_recipe):
    """Interpret the model recipe string.

    Given the config object, create an internal recipe implementing the user
    specified recipe.

    Args:
        model_opts: An ModelInputs configuration object.

    Returns:
        model_Recipe: A Recipe object.
    """

    model_columns = set()
    sky_models = set()
    instructions = {}

    # Strip accidental whitepsace from input recipe and splits on ":".
    input_recipes = user_recipe.replace(" ", "").split(":")

    if input_recipes == ['']:
        raise ValueError("No model recipe was specified. Please set/check "
                         "--input-model-recipe.")

    for recipe_index, recipe in enumerate(input_recipes):

        instructions[recipe_index] = []

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
                instructions[recipe_index].append(operation)

            elif ".lsm.html" in ingredient:

                filename, _, tags = ingredient.partition("@")
                tags = tuple(tags.split(",")) if tags else ()

                if not os.path.isfile(filename):
                    raise FileNotFoundError("{} not found.".format(filename))

                sky_model = sky_model_nt(filename, tags)
                sky_models.add(sky_model)
                instructions[recipe_index].append(sky_model)

            elif ingredient != "":
                model_columns.add(ingredient)
                instructions[recipe_index].append(ingredient)

            else:
                instructions[recipe_index].append(ingredient)

    logger.info("The following model sources were obtained from "
                "--input-model-recipe: \n"
                "   Columns: {} \n"
                "   Sky Models: {}",
                model_columns or 'None',
                {sm.name for sm in sky_models} or 'None')

    # Generate a named tuple containing all the information required to
    # build the model visibilities.

    model_recipe = Recipe(Ingredients(model_columns, sky_models), instructions)

    if model_recipe.ingredients.sky_models:
        logger.info("Recipe contains sky models - enabling prediction step.")

    return model_recipe
