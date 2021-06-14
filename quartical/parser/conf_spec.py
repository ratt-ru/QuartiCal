from dataclasses import dataclass, field, fields
from pathlib import Path
from omegaconf import OmegaConf as oc
from typing import List, Dict, Any, Optional
from quartical.parser.converters import as_time, as_freq
import sys
import os
import textwrap
from loguru import logger


path_to_help_str = Path(__file__).parent.joinpath("helpstrings.yaml")
help_str = oc.load(path_to_help_str)


@dataclass
class MSInputs:
    path: str = "???"
    column: str = "DATA"
    weight_column: str = "WEIGHT_SPECTRUM"
    time_chunk: str = "0"
    freq_chunk: int = 0
    is_bda: bool = False
    group_by: Optional[List[str]] = field(
        default_factory=lambda: ["SCAN_NUMBER", "FIELD_ID", "DATA_DESC_ID"]
    )
    select_corr: Optional[List[int]] = None

    def __post_init__(self):
        self.time_chunk = as_time(self.time_chunk)


@dataclass
class ModelInputs:
    recipe: str = "???"
    beam: Optional[str] = None
    beam_l_axis: str = "X"
    beam_m_axis: str = "Y"
    invert_uvw: bool = True
    source_chunks: int = 10
    apply_p_jones: bool = True

    def __post_init__(self):
        if self.beam:
            choices = ["X", "~X", "Y", "~Y", "L", "~L", "M", "~M"]
            assert self.beam_l_axis in choices, "Unknown beam axis."
            assert self.beam_m_axis in choices, "Unknown beam axis."


@dataclass
class Outputs:
    gain_dir: str = "gains.qc"
    products: Optional[List[str]] = None
    columns: Optional[List[str]] = None

    def __post_init__(self):
        assert not(bool(self.products) ^ bool(self.columns)), \
            "Neither or both of products and columns must be specified."
        if self.products:
            choices = ["corrected_data", "corrected_residual", "residual"]
            assert all((i in choices) for i in self.products), \
                   "Invalid visibility product."
            assert len(self.products) == len(self.columns), \
                   "Number of products not equal to number of columns."


@dataclass
class MadFlags:
    enable: bool = False
    threshold_bl: int = 10
    threshold_global: int = 12


@dataclass
class Solver:
    gain_terms: List[str] = "???"


@dataclass
class Parallel:
    n_thread: int = 0
    n_worker: int = 1
    address: Optional[str] = None
    scheduler: str = "threads"

    def __post_init__(self):
        choices = ["threads", "single-threaded", "distributed"]
        assert self.scheduler in choices, "Unknown scheduler."


@dataclass
class __gain__:
    type: str = "complex"
    direction_dependent: bool = False
    time_interval: str = 1
    freq_interval: str = 1
    load_from: Optional[str] = None
    interp_mode: str = "reim"
    interp_method: str = "2dlinear"

    def __post_init__(self):
        self.time_interval = as_time(self.time_interval)
        self.freq_interval = as_freq(self.freq_interval)
        choices = ["complex", "delay", "phase"]
        assert self.type in choices, "Unknown gain type."
        choices = ["reim", "ampphase"]
        assert self.interp_mode in choices, "Unknown interpolation mode."
        choices = ["2dlinear", "2dspline", "smoothingspline"]
        assert self.interp_method in choices, "Unknown interpolation method."


@dataclass
class QCConfig:
    input_ms: MSInputs = MSInputs()
    input_model: ModelInputs = ModelInputs()
    solver: Solver = Solver()
    output: Outputs = Outputs()
    mad_flags: MadFlags = MadFlags()
    parallel: Parallel = Parallel()


def populate_help(help_obj, help_str):

    for k, v in help_obj.items():
        if isinstance(v, dict):
            populate_help(v, help_str[k])
        else:
            help_obj[k] = str(help_str[k]) + f" Default: {v}."

    return help_obj


def print_help(help_obj, selection=None):
    """Logs the final state of the args Namespace.

    Given the overlapping nature of the various configuration options, this
    produces a pretty log message describing the final configuration state
    of the args Namespace.

    Args:
        args: A Namespace object.
    """

    printable = {}

    selection = (selection,) if selection else help_obj.keys()

    for g in selection:
        v = help_obj[g]
        printable.update({".".join([g, kk]): v[kk] for kk in v.keys()})

    # This guards against attempting to get the terminal size when the output
    # is being piped/redirected.
    if sys.stdout.isatty():
        columns, _ = os.get_terminal_size(0)
    else:
        columns = 80  # Fall over to some sensible default.

    log_message = ""

    section = None

    for key, value in printable.items():
        current_section = key.split('.')[0]

        if current_section != section:
            log_message += "" if section is None \
                else "<blue>{0:-^{1}}</blue>\n".format("", columns)
            log_message += "\n<blue>{0:-^{1}}</blue>\n".format(
                current_section, columns)
            section = current_section

        log_message += f"<red>{key:<}</red>\n"
        txt = textwrap.fill(value,
                            width=columns,
                            initial_indent="    ",
                            subsequent_indent="    ")
        log_message += f"{txt:<{columns}}\n"

    log_message += "<blue>{0:-^{1}}</blue>".format("", columns)

    logger.opt(ansi=True).info(log_message)



if __name__ == "__main__":

    qcconf = oc.structured(QCConfig)
    help_obj = oc.to_container(qcconf)
    populate_help(help_obj, help_str)
    print_help(help_obj)
    blah = oc.merge(qcconf,
                    oc.from_dotlist(["input_ms.path=foo",
                                     "input_model.recipe=bar",
                                     "solver.gain_terms=[G]"]))
    oconf = oc.to_object(blah)
