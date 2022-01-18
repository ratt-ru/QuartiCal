from lib2to3.pytree import Base
import os.path
from collections import OrderedDict
from dataclasses import dataclass, field, make_dataclass, fields
from urllib.request import OpenerDirector
from omegaconf import OmegaConf as oc
from typing import List, Optional, Dict, Any
from quartical.config.converters import as_time, as_freq
from scabha import schema_utils, configuratt
from scabha.cargo import Parameter


class BaseConfigSection:
    """Base class for config dataclasses. Also implements specific post-init methods for them"""
    def __post_init__ (self):
        self.validate_choice_fields()

    def validate_choice_fields(self):
        choice_fields = {f.name: f.metadata["choices"]
                         for f in fields(self) if "choices" in f.metadata}
        for field_name, field_choices in choice_fields.items():
            args = getattr(self, field_name)
            if args is None:  # An optional choices field might be None.
                continue
            elif isinstance(args, List):
                assert all(arg in field_choices for arg in args), \
                       f"Invalid input in {field_name}. " \
                       f"User specified '{args}'. " \
                       f"Valid choices are {field_choices}."
            else:
                assert args in field_choices, \
                       f"Invalid input in {field_name}. " \
                       f"User specified '{args}'. " \
                       f"Valid choices are {field_choices}."

    def __msinput_post_init__(self):
        """__post_init__ method attached to MSInput dataclass when it is dynamically created"""
        self.validate_choice_fields()
        self.time_chunk = as_time(self.time_chunk)
        self.freq_chunk = as_freq(self.freq_chunk)

        assert len(self.select_uv_range) == 2, \
            "input_ms.select_uv_range expects a two-element list."

        assert not (self.sigma_column and self.weight_column), \
            "sigma_column and weight_column are mutually exclusive."


    def __output_post_init__(self):
        self.validate_choice_fields()
        assert not(bool(self.products) ^ bool(self.columns)), \
            "Neither or both of products and columns must be specified."
        if self.products:
            assert len(self.products) == len(self.columns), \
                    "Number of products not equal to number of columns."


    def __solver_post_init__(self):
        self.validate_choice_fields()
        assert len(self.iter_recipe) >= len(self.terms), \
                "User has specified solver.iter_recipe with too few elements."

        assert self.convergence_criteria >= 1e-8, \
                "User has specified solver.convergence_criteria below 1e-8."

    def __gain_post_init__(self):
        self.validate_choice_fields()
        self.time_interval = as_time(self.time_interval)
        self.freq_interval = as_freq(self.freq_interval)


@dataclass
class _GainSchema(object):
    gain: Dict[str, Parameter]

# load schema files
_schema = _gain_schema = None

if _schema is None:
    dirname = os.path.dirname(__file__)

    # make dataclass based on argument schema
    _schema = oc.load(f"{dirname}/argument_schema.yaml")

    # make map of post_init methods from BaseConfigSection class
    post_init_map = {}
    for section in _schema.keys():
        post_init = getattr(BaseConfigSection, f"_{section}_post_init__", None)
        if post_init:
            post_init_map[section] = post_init

    # create the base config class
    BaseConfig = schema_utils.nested_schema_to_dataclass(_schema,
                   "BaseConfig", section_bases=(BaseConfigSection,), post_init_map=post_init_map)

    # make map of section classes
    _config_dataclasses = OrderedDict((f.name, f.type) for f in fields(BaseConfig))
            
    # the gain section is loaded explicitly, since we need to form up multiple instances
    _gain_schema = oc.merge(oc.structured(_GainSchema), oc.load(f"{dirname}/gain_schema.yaml"))
    _gain_schema = _gain_schema.gain

    # create gain dataclass
    Gain = schema_utils.schema_to_dataclass(_gain_schema, "Gain", bases=(BaseConfigSection,), 
                                            post_init=BaseConfigSection.__gain_post_init__)


def get_config_sections():
    """returns list of known config sections
    """
    return list(_config_dataclasses.keys()) + ["gain"]


def finalize_structure(additional_config):

    terms = None

    for cfg in additional_config[::-1]:
        terms = oc.select(cfg, "solver.terms")
        if terms is not None:
            break

    # Use the default terms if no alternative is specified.
    terms = terms or _config_dataclasses["solver"].terms

    FinalConfig = make_dataclass(
        "FinalConfig",
        [(t, Gain, Gain()) for t in terms],
        bases=(BaseConfig,)
    )

    return FinalConfig


def make_stimela_schema(params: Dict[str, Any], inputs: Dict[str, Parameter], outputs: Dict[str, Parameter]):
    """Augments a schema for stimela based on solver.terms"""
    inputs = inputs.copy()

    terms = params.get('solver.terms', None)
    if terms is None:
        terms  = _config_dataclasses["solver"].terms

    for jones in terms:
        for key, value in _gain_schema.items():
            inputs[f"{jones}.{key}"] = value

    return inputs, outputs
