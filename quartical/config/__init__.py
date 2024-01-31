import os.path
from dataclasses import dataclass
from omegaconf import OmegaConf as oc
from typing import Dict
from scabha import schema_utils
from scabha.cargo import Parameter
from quartical.config.config_classes import BaseConfigSection, POST_INIT_MAP

base_schema = gain_schema = None

# This should only run on the first import of this module. Sets up schemas.
if base_schema is None:
    dirname = os.path.dirname(__file__)

    # Make dataclass based on the argument schema.
    base_schema = oc.load(f"{dirname}/argument_schema.yaml")

    # Create the base config class.
    BaseConfig = schema_utils.nested_schema_to_dataclass(
        base_schema,
        "BaseConfig",
        section_bases=(BaseConfigSection,),
        post_init_map=POST_INIT_MAP
    )

    @dataclass
    class _GainSchema(object):
        gain: Dict[str, Parameter]

    # The gain and model sections are loaded explicitly, since we need to form
    # up multiple instances.
    gain_schema = oc.merge(
        oc.structured(_GainSchema),
        oc.load(f"{dirname}/gain_schema.yaml")
    )
    gain_schema = gain_schema.gain

    # Create gain dataclass.
    Gain = schema_utils.schema_to_dataclass(
        gain_schema,
        "Gain",
        bases=(BaseConfigSection,),
        post_init=POST_INIT_MAP['gain']
    )

    @dataclass
    class _ModelComponentSchema(object):
        model_component: Dict[str, Parameter]

    model_component_schema = oc.merge(
        oc.structured(_ModelComponentSchema),
        oc.load(f"{dirname}/model_component_schema.yaml")
    )
    model_component_schema = model_component_schema.model_component

    # Create model dataclass.
    ModelComponent = schema_utils.schema_to_dataclass(
        model_component_schema,
        "ModelComponent",
        bases=(BaseConfigSection,),
        post_init=POST_INIT_MAP['model_component']
    )
