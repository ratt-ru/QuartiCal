import os.path
from dataclasses import dataclass
from omegaconf import OmegaConf as oc
from typing import Dict
from scabha import schema_utils
from scabha.cargo import Parameter
from quartical.config.config_classes import BaseConfigSection, POST_INIT_MAP

_schema = _gain_schema = None

# This should only run on the first import of this module. Sets up schemas.
if _schema is None:
    dirname = os.path.dirname(__file__)

    # Make dataclass based on the argument schema.
    _schema = oc.load(f"{dirname}/argument_schema.yaml")

    # Create the base config class.
    BaseConfig = schema_utils.nested_schema_to_dataclass(
        _schema,
        "BaseConfig",
        section_bases=(BaseConfigSection,),
        post_init_map=POST_INIT_MAP
    )

    @dataclass
    class _GainSchema(object):
        gain: Dict[str, Parameter]

    # The gain section is loaded explicitly, since we need to form up multiple
    # instances.
    _gain_schema = oc.merge(
        oc.structured(_GainSchema),
        oc.load(f"{dirname}/gain_schema.yaml")
    )
    _gain_schema = _gain_schema.gain

    # Create gain dataclass.
    Gain = schema_utils.schema_to_dataclass(
        _gain_schema,
        "Gain",
        bases=(BaseConfigSection,),
        post_init=POST_INIT_MAP['gain']
    )
