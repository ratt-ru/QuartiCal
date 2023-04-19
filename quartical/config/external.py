import os.path
from dataclasses import dataclass, make_dataclass, fields, _MISSING_TYPE
from omegaconf import OmegaConf as oc
from typing import List, Dict, Any
from scabha import schema_utils
from scabha.cargo import Parameter
from quartical.config.postinits import POST_INIT_MAP


class BaseConfigSection:
    """Base class for dynamically generated config dataclasses."""

    def __validate_choices__(self):

        for fld in fields(self):
            name = fld.name
            value = getattr(self, name)
            meta = fld.metadata

            if value is None or "choices" not in meta:
                continue
            else:
                choices = meta.get("choices")
                assert value in choices, (
                    f"Invalid input in {name}. User specified '{value}'. "
                    f"Valid choices are {choices}."
                )

    def __validate_element_choices__(self):

        for fld in fields(self):
            name = fld.name
            value = getattr(self, name)
            meta = fld.metadata

            if value is None or "element_choices" not in meta:
                continue
            else:
                element_choices = meta.get("element_choices")
                if isinstance(value, List):
                    elements = value
                elif isinstance(value, Dict):
                    elements = value.values()
                else:
                    raise ValueError(
                        f"Paramter {name} of type {type(value)} has element "
                        f"choices. This is not supported."
                    )
                invalid_elements = set(elements) - set(element_choices)
                assert not invalid_elements, (
                    f"Invalid input in {name}. User specified "
                    f"{elements}. Valid choices: {element_choices}."
                )

    def __helpstr__(self):

        helpstrings = {}

        for fld in fields(self):

            meta = fld.metadata

            help_str = meta.get('help')

            if meta.get("choices", None):
                choice_str = f"Choices: {meta.get('choices')} "
            else:
                choice_str = ""

            if meta.get("element_choices", None):
                element_choice_str = f"Choices: {meta.get('element_choices')} "
            else:
                element_choice_str = ""

            if isinstance(fld.default, _MISSING_TYPE):
                default = fld.default_factory()
            else:
                default = fld.default

            if fld.metadata.get('required', False):
                default_str = "REQUIRED"
            else:
                default_str = f"Default: {default}"

            helpstrings[fld.name] = (
                f"{help_str} {choice_str}{element_choice_str}{default_str}"
            )

        return helpstrings


@dataclass
class _GainSchema(object):
    gain: Dict[str, Parameter]


# load schema files
_schema = _gain_schema = None

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

    # Create a mapping from section name to section dataclass.
    _config_dataclasses = {f.name: f.type for f in fields(BaseConfig)}

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


def finalize_structure(additional_config):

    terms = None

    for cfg in additional_config[::-1]:
        terms = oc.select(cfg, "solver.terms")
        if terms is not None:
            break

    # Use the default terms if no alternative is specified.
    terms = terms or _config_dataclasses["solver"]().terms

    FinalConfig = make_dataclass(
        "FinalConfig",
        [(t, Gain, Gain()) for t in terms],
        bases=(BaseConfig,)
    )

    return FinalConfig


def make_stimela_schema(params: Dict[str, Any],
                        inputs: Dict[str, Parameter],
                        outputs: Dict[str, Parameter]):
    """Augments a schema for stimela based on solver.terms."""

    inputs = inputs.copy()

    if 'solver' in params:
        terms = params['solver'].get('terms', None)
    else:
        terms = params.get('solver.terms', None)
    if terms is None:
        terms = _config_dataclasses["solver"]().terms

    for jones in terms:
        for key, value in _gain_schema.items():
            inputs[f"{jones}.{key}"] = value

    return inputs, outputs
