import sys
import os
import textwrap
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf as oc
from quartical.parser.configuration import finalize_structure


path_to_helpstrings = Path(__file__).parent.joinpath("helpstrings.yaml")
HELPSTRINGS = oc.load(path_to_helpstrings)


def populate(help_obj, help_str):
    for k, v in help_obj.items():
        if isinstance(v, dict):
            populate(v, help_str[k])
        else:
            help_obj[k] = str(help_str[k]) + f" Default: {v}."


def make_help_obj():

    # We add this so that the help object incudes a generic gain field.
    additional_config = [oc.from_dotlist(["solver.gain_terms=['gain']"])]

    config = finalize_structure(additional_config)
    config = oc.merge(config, *additional_config)

    help_obj = oc.to_container(config)

    populate(help_obj, HELPSTRINGS)

    return help_obj


def help():
    """Prints the help."""

    help_args = [arg for arg in sys.argv if arg.startswith('help')]

    # Always take the last specified help request.
    help_arg = help_args.pop() if help_args else help_args

    if len(sys.argv) == 1 or help_arg == "help":
        help_obj = make_help_obj()
        selection = help_obj.keys()
    elif help_arg:
        help_obj = make_help_obj()
        selection = help_arg.split("=")[-1].strip("[]").split(",")
    else:
        return

    # This guards against attempting to get the terminal size when the output
    # is being piped/redirected.
    if sys.stdout.isatty():
        columns, _ = os.get_terminal_size(0)
    else:
        columns = 80  # Fall over to some sensible default.

    log_message = ""

    current_section = None

    for section, options in help_obj.items():

        if section not in selection:
            continue

        if current_section != section:
            log_message += "" if current_section is None \
                else "<blue>{0:-^{1}}</blue>\n".format("", columns)
            log_message += "\n<blue>{0:-^{1}}</blue>\n".format(
                section, columns)
            current_section = section

        for key, value in options.items():
            option = f"{section}.{key}"
            log_message += f"<red>{option:<}</red>\n"
            txt = textwrap.fill(value,
                                width=columns,
                                initial_indent=" "*4,
                                subsequent_indent=" "*4)
            log_message += f"{txt:<{columns}}\n"

    log_message += "<blue>{0:-^{1}}</blue>".format("", columns)

    logger.opt(ansi=True).info(log_message)

    sys.exit()
