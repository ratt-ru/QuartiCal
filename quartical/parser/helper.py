import sys
import os
import textwrap
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf as oc


path_to_helpstrings = Path(__file__).parent.joinpath("helpstrings.yaml")
help_str = oc.load(path_to_helpstrings)


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
