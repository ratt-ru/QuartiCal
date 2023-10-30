import re
from dataclasses import make_dataclass
from omegaconf import OmegaConf as oc
from typing import Dict, Any
from scabha.cargo import Parameter
from quartical.config import Gain, ModelComponent, BaseConfig, gain_schema


def finalize_structure(additional_config):

    terms = None

    # Get last specified version of solver.terms.
    for cfg in additional_config[::-1]:
        terms = oc.select(cfg, "solver.terms")
        if terms is not None:
            break

    # Use the default terms if no alternative is specified.
    terms = terms or BaseConfig.solver.terms

    recipe = None
    models = []  # No components by default.

    # Get last specified version of input_model.recipe.
    for cfg in additional_config[::-1]:
        advanced_recipe = oc.select(cfg, "input_model.advanced_recipe")
        recipe = oc.select(cfg, "input_model.recipe")
        if recipe is not None and advanced_recipe:
            ingredients = re.split(r'([\+~:])', recipe)
            ingredients = [
                i for i in ingredients if not bool(re.search(r'([\+~:])', i))
            ]
            models = list(dict.fromkeys(i.split("@")[0] for i in ingredients))
            break

    FinalConfig = make_dataclass(
        "FinalConfig",
        [
            *[(m, ModelComponent, ModelComponent()) for m in models],
            *[(t, Gain, Gain()) for t in terms]
        ],
        bases=(BaseConfig,)
    )

    return FinalConfig


def make_stimela_schema(
    params: Dict[str, Any],
    inputs: Dict[str, Parameter],
    outputs: Dict[str, Parameter]
):
    """Augments a schema for stimela based on solver.terms."""

    inputs = inputs.copy()

    if 'solver' in params:  # Is this ever the case?
        terms = params['solver'].get('terms', None)
    else:
        terms = params.get('solver.terms', None)
    if terms is None:
        terms = BaseConfig.solver.terms  # Fall back to default.

    # For each term, add the relevant entries to the inputs.
    for jones in terms:
        for key, value in gain_schema.items():
            inputs[f"{jones}.{key}"] = value

    return inputs, outputs
