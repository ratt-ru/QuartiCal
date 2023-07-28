# -*- coding: utf-8 -*-
from loguru import logger
import re
import dask.array as da
from collections import namedtuple
import os.path
from dataclasses import dataclass
from typing import List, Dict, Set, Any
from ast import literal_eval


sky_model_nt = namedtuple("sky_model_nt", ("name", "tags"))
degrid_model_nt = namedtuple(
    "degrid_model_nt",
    (
        "name",
        "nxo",
        "nyo",
        "cellxo",
        "cellyo",
        "x0o",
        "y0o",
        "ipi",
        "cpi"
    )
)


@dataclass
class Ingredients:
    model_columns: Set[Any]
    sky_models: Set[sky_model_nt]
    degrid_models: Set[degrid_model_nt]


@dataclass
class Recipe:
    ingredients: Ingredients
    instructions: Dict[int, List[Any]]


@dataclass
class IdentityRecipe(Recipe):
    pass


def transcribe_recipe(user_recipe):
    """Interpret the model recipe string.

    Given the config object, create an internal recipe implementing the user
    specified recipe.

    Args:
        model_opts: An ModelInputs configuration object.

    Returns:
        model_Recipe: A Recipe object.
    """

    if user_recipe is None:
        logger.warning(
            "input_model.recipe was not supplied. Assuming identity model."
        )
        return IdentityRecipe(Ingredients(set(), set()), dict())

    model_columns = set()
    sky_models = set()
    degrid_models = set()

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

            elif ".mds" in ingredient:

                filename, _, options = ingredient.partition("@")
                options = literal_eval(options)  # Add fail on missing option.

                if not os.path.exists(filename):
                    raise FileNotFoundError("{} not found.".format(filename))

                degrid_model = degrid_model_nt(filename, *options)
                degrid_models.add(degrid_model)
                instructions[recipe_index].append(degrid_model)

            elif ingredient != "":
                model_columns.add(ingredient)
                instructions[recipe_index].append(ingredient)

            else:
                instructions[recipe_index].append(ingredient)

    # TODO: Add message to log.
    logger.info("The following model sources were obtained from "
                "--input-model-recipe: \n"
                "   Columns: {} \n"
                "   Sky Models: {} \n"
                "   Degrid Models: {}",
                model_columns or 'None',
                {sm.name for sm in sky_models} or 'None',
                {dm.name for dm in degrid_models} or 'None')

    # Generate a named tuple containing all the information required to
    # build the model visibilities.

    model_recipe = Recipe(
        Ingredients(
            model_columns,
            sky_models,
            degrid_models
        ),
        instructions
    )

    if model_recipe.ingredients.sky_models:
        logger.info("Recipe contains sky models - enabling prediction step.")

    if model_recipe.ingredients.degrid_models:
        logger.info("Recipe contains degrid models - enabling degridding.")

    return model_recipe
