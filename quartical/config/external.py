import os.path
from dataclasses import dataclass, make_dataclass, fields
from omegaconf import OmegaConf as oc
from typing import List, Dict, Any
from quartical.config.converters import as_time, as_freq
from scabha import schema_utils
from scabha.cargo import Parameter


class BaseConfigSection:
    """Base class for dynamically generated config dataclasses.

    Also implements specific post-init methods for them.
    """

    def validate_choice_fields(self):
        for fld in fields(self):
            name = fld.name
            value = getattr(self, name)
            meta = fld.metadata

            # Optional field values can be None, validated at construction.
            if value is None:
                continue
            elif "choices" in meta:  # Check for choices.
                choices = meta.get("choices")
                assert value in choices, \
                    f"Invalid input in {fld.name}. " \
                    f"User specified '{value}'. " \
                    f"Valid choices are {choices}."
            elif "element_choices" in meta:  # Check for element choices.
                element_choices = meta.get("element_choices")
                if isinstance(value, List):
                    args = value
                elif isinstance(value, Dict):
                    args = value.values()
                else:
                    raise ValueError(f"Paramter {name} of type {type(value)}"
                                     f"has element choices. Not understood.")
                invalid = set(args) - set(element_choices)
                assert not invalid, \
                    f"Invalid input in {fld.name}. " \
                    f"User specified '{','.join(map(str, invalid))}'. " \
                    f"Valid choices are {','.join(map(str,element_choices))}."

    def __input_ms_post_init__(self):

        self.validate_choice_fields()
        self.time_chunk = as_time(self.time_chunk)
        self.freq_chunk = as_freq(self.freq_chunk)

        assert len(self.select_uv_range) == 2, \
            "input_ms.select_uv_range expects a two-element list."

        assert not (self.sigma_column and self.weight_column), \
            "sigma_column and weight_column are mutually exclusive."

        if self.is_bda:
            assert self.time_chunk == 0, \
                ("input_ms.is_bda does not support chunking in time. Please "
                 "set input_ms.time_chunk to 0.")
            assert self.freq_chunk == 0, \
                ("input_ms.is_bda does not support chunking in freq. Please "
                 "set input_ms.freq_chunk to 0.")

    def __input_model_post_init__(self):
        pass

    def __output_post_init__(self):

        self.validate_choice_fields()
        assert not (bool(self.products) ^ bool(self.columns)), \
            "Neither or both of products and columns must be specified."
        if self.products:
            assert len(self.products) == len(self.columns), \
                    "Number of products not equal to number of columns."

        if self.net_gains:
            nested = any(isinstance(i, list) for i in self.net_gains)
            if nested:
                assert all(isinstance(i, list) for i in self.net_gains), \
                    ("Contents of outputs.net_gains not understood. "
                     "Must be strictly a list or list of lists.")
            else:
                assert all(isinstance(i, str) for i in self.net_gains), \
                    ("Contents of outputs.net_gains not understood. "
                     "Must be strictly a list or list of lists.")
                # In the non-nested case, introduce outer list (consistent).
                self.net_gains = [self.net_gains]

    def __mad_flags_post_init__(self):
        pass

    def __solver_post_init__(self):

        self.validate_choice_fields()
        assert len(self.iter_recipe) >= len(self.terms), \
            "User has specified solver.iter_recipe with too few elements."

        assert self.convergence_criteria >= 1e-8, \
            "User has specified solver.convergence_criteria below 1e-8."

    def __dask_post_init__(self):
        self.validate_choice_fields()
        if self.address:
            msg = (
                "Scheduler address supplied but dask.scheduler has not "
                "been set to distributed."
            )
            assert self.scheduler == "distributed", msg

    def __gain_post_init__(self):
        self.validate_choice_fields()
        self.time_interval = as_time(self.time_interval)
        self.freq_interval = as_freq(self.freq_interval)
        if self.type == "crosshand_phase" and self.solve_per != "array":
            raise ValueError("Crosshand phase can only be solved as a per "
                             "array term. Please set the appropriate "
                             "term.solve_per to 'array'.")


@dataclass
class _GainSchema(object):
    gain: Dict[str, Parameter]


# load schema files
_schema = _gain_schema = None

if _schema is None:
    dirname = os.path.dirname(__file__)

    # Make dataclass based on the argument schema.
    _schema = oc.load(f"{dirname}/argument_schema.yaml")

    # Map __post_init__ methods to sections.
    post_init_map = {s: getattr(BaseConfigSection, f"__{s}_post_init__")
                     for s in _schema.keys()}

    # Create the base config class.
    BaseConfig = schema_utils.nested_schema_to_dataclass(
        _schema,
        "BaseConfig",
        section_bases=(BaseConfigSection,),
        post_init_map=post_init_map
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
        post_init=BaseConfigSection.__gain_post_init__
    )


def get_config_sections():
    """Returns list of known config sections."""
    return list(_config_dataclasses.keys()) + ["gain"]


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

    terms = params.get('solver.terms', None)
    if terms is None:
        terms = _config_dataclasses["solver"]().terms

    for jones in terms:
        for key, value in _gain_schema.items():
            inputs[f"{jones}.{key}"] = value

    return inputs, outputs
