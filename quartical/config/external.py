from dataclasses import dataclass, field, make_dataclass, fields
from omegaconf import OmegaConf as oc
from typing import List, Optional
from quartical.config.converters import as_time, as_freq


class Input:

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


@dataclass
class MSInputs(Input):
    path: str = "???"
    data_column: str = "DATA"
    weight_column: Optional[str] = "???"
    time_chunk: str = "0"
    freq_chunk: str = "0"
    is_bda: bool = False
    group_by: Optional[List[str]] = field(
        default_factory=lambda: ["SCAN_NUMBER", "FIELD_ID", "DATA_DESC_ID"]
    )
    select_corr: Optional[List[int]] = field(
        default=None,
        metadata=dict(choices=[0, 1, 2, 3])
    )
    select_fields: List[int] = field(
        default_factory=lambda: []
    )
    select_ddids: List[int] = field(
        default_factory=lambda: []
    )

    def __post_init__(self):
        self.validate_choice_fields()
        self.time_chunk = as_time(self.time_chunk)
        self.freq_chunk = as_freq(self.freq_chunk)


@dataclass
class ModelInputs(Input):
    recipe: str = "???"
    beam: Optional[str] = None
    beam_l_axis: str = field(
        default="X",
        metadata=dict(choices=["X", "~X", "Y", "~Y", "L", "~L", "M", "~M"])
    )
    beam_m_axis: str = field(
        default="Y",
        metadata=dict(choices=["X", "~X", "Y", "~Y", "L", "~L", "M", "~M"])
    )
    invert_uvw: bool = True
    source_chunks: int = 10
    apply_p_jones: bool = True

    def __post_init__(self):
        self.validate_choice_fields()


@dataclass
class Outputs(Input):
    gain_dir: str = "gains.qc"
    products: Optional[List[str]] = field(
        default=None,
        metadata=dict(choices=["corrected_data",
                               "corrected_residual",
                               "residual"])
    )
    columns: Optional[List[str]] = None
    net_gain: bool = False

    def __post_init__(self):
        self.validate_choice_fields()
        assert not(bool(self.products) ^ bool(self.columns)), \
            "Neither or both of products and columns must be specified."
        if self.products:
            assert len(self.products) == len(self.columns), \
                   "Number of products not equal to number of columns."


@dataclass
class MadFlags(Input):
    enable: bool = False
    threshold_bl: int = 10
    threshold_global: int = 12

    def __post_init__(self):
        self.validate_choice_fields()


@dataclass
class Solver(Input):
    terms: List[str] = field(default_factory=lambda: ["G"])
    iter_recipe: List[int] = field(default_factory=lambda: [25])
    threads: int = 1

    def __post_init__(self):
        self.validate_choice_fields()
        assert len(self.iter_recipe) >= len(self.terms), \
               "User has specified solver.iter_recipe with too few elements."


@dataclass
class Dask(Input):
    threads: int = 0
    workers: int = 1
    address: Optional[str] = None
    scheduler: str = field(
        default="threads",
        metadata=dict(choices=["threads",
                               "single-threaded",
                               "distributed"])
    )

    def __post_init__(self):
        self.validate_choice_fields()


@dataclass
class Gain(Input):
    type: str = field(
        default="complex",
        metadata=dict(choices=["complex",
                               "delay",
                               "phase",
                               "slow_complex",
                               "slow_delay"])
    )
    direction_dependent: bool = False
    time_interval: str = "1"
    freq_interval: str = "1"
    load_from: Optional[str] = None
    interp_mode: str = field(
        default="reim",
        metadata=dict(choices=["reim",
                               "ampphase"])
    )
    interp_method: str = field(
        default="2dlinear",
        metadata=dict(choices=["2dlinear",
                               "2dspline"])
    )

    def __post_init__(self):
        self.validate_choice_fields()
        self.time_interval = as_time(self.time_interval)
        self.freq_interval = as_freq(self.freq_interval)


@dataclass
class BaseConfig:
    input_ms: MSInputs = MSInputs()
    input_model: ModelInputs = ModelInputs()
    solver: Solver = Solver()
    output: Outputs = Outputs()
    mad_flags: MadFlags = MadFlags()
    dask: Dask = Dask()


def finalize_structure(additional_config):

    terms = None

    for cfg in additional_config[::-1]:
        terms = oc.select(cfg, "solver.terms")
        if terms is not None:
            break

    # Use the default terms if no alternative is specified.
    terms = terms or Solver().terms

    FinalConfig = make_dataclass(
        "FinalConfig",
        [(t, Gain, Gain()) for t in terms],
        bases=(BaseConfig,)
    )

    return FinalConfig
