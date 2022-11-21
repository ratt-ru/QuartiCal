from dataclasses import dataclass, field, make_dataclass, fields
from omegaconf import OmegaConf as oc
from typing import List, Optional, Any
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
    sigma_column: Optional[str] = None
    weight_column: Optional[str] = None
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
    select_uv_range: List[float] = field(
        default_factory=lambda: [0, 0]
    )

    def __post_init__(self):
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
    source_chunks: int = 500
    apply_p_jones: bool = False

    def __post_init__(self):
        self.validate_choice_fields()


@dataclass
class Outputs(Input):
    gain_directory: str = "gains.qc"
    log_directory: str = "logs.qc"
    log_to_terminal: bool = True
    overwrite: bool = False
    products: Optional[List[str]] = field(
        default=None,
        metadata=dict(choices=["corrected_data",
                               "corrected_residual",
                               "residual",
                               "weight",
                               "corrected_weight",
                               "model_data"])
    )
    columns: Optional[List[str]] = None
    flags: bool = True
    apply_p_jones_inv: bool = False
    subtract_directions: Optional[List[int]] = None
    net_gains: Optional[List[Any]] = None

    def __post_init__(self):
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


@dataclass
class MadFlags(Input):
    enable: bool = False
    whitening: str = field(
        default="disabled",
        metadata=dict(choices=["disabled", "native", "robust"])
    )
    threshold_bl: float = 5
    threshold_global: float = 10
    max_deviation: float = 10

    def __post_init__(self):
        self.validate_choice_fields()


@dataclass
class Solver(Input):
    terms: List[str] = field(default_factory=lambda: ["G"])
    iter_recipe: List[int] = field(default_factory=lambda: [25])
    propagate_flags: bool = True
    robust: bool = False
    threads: int = 1
    convergence_fraction: float = 0.99
    convergence_criteria: float = 1e-6
    reference_antenna: int = 0

    def __post_init__(self):
        self.validate_choice_fields()
        assert len(self.iter_recipe) >= len(self.terms), \
               "User has specified solver.iter_recipe with too few elements."

        assert self.convergence_criteria >= 1e-8, \
               "User has specified solver.convergence_criteria below 1e-8."


@dataclass
class Dask(Input):
    threads: Optional[int] = None
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

        if self.address:
            msg = (
                "Scheduler address supplied but dask.scheduler has not "
                "been set to distributed."
            )
            assert self.scheduler == "distributed", msg


@dataclass
class Gain(Input):
    type: str = field(
        default="complex",
        metadata=dict(choices=["complex",
                               "diag_complex",
                               "amplitude",
                               "delay",
                               "phase",
                               "tec",
                               "rotation_measure",
                               "crosshand_phase"])
    )
    solve_per: str = field(
        default="antenna",
        metadata=dict(choices=["antenna",
                               "array"])
    )
    direction_dependent: bool = False
    time_interval: str = "1"
    freq_interval: str = "1"
    respect_scan_boundaries: bool = True
    initial_estimate: bool = True
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

        if self.type == "crosshand_phase" and self.solve_per != "array":
            raise ValueError("Crosshand phase can only be solved as a per "
                             "array term. Please set the appropriate "
                             "term.solve_per to 'array'.")


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
