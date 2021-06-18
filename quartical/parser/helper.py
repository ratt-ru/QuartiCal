import sys
import os
import textwrap
import re
from colorama import Fore, Style
from pathlib import Path
from omegaconf import OmegaConf as oc
from dataclasses import fields, MISSING, is_dataclass
from quartical.parser.configuration import finalize_structure


path_to_helpstrings = Path(__file__).parent.joinpath("helpstrings.yaml")
HELPSTRINGS = oc.load(path_to_helpstrings)

GAIN_MSG = "Gains make use of a special configuration " \
           "mechanism. Use 'solver.gain_terms' to specify the name of " \
           "each gain term, e.g. 'solver.gain_terms=[G,B]'. Each gain " \
           "can then be configured using its name and any gain option, " \
           "e.g. 'G.type=complex' or 'B.direction_dependent=True'."

HELP_MSG = f"For full help, use 'goquartical help'. For help with a " \
           f"specific section, use e.g. 'goquartical help='[section1," \
           f"section2]''. Help is available for " \
           f"[{', '.join(HELPSTRINGS.keys())}]."


def populate(typ, help_str, help_dict=None):

    if help_dict is None:
        help_dict = {}

    if not is_dataclass(typ):
        return False

    flds = fields(typ)
    field_types = {f.name: f.type for f in flds}
    field_defaults = {f.name: f.default for f in flds}
    field_choices = {f.name: f.metadata.get("choices", "") for f in flds}
    field_factories = {f.name: f.default_factory for f in flds}

    for k, v in field_types.items():
        help_dict[k] = {}
        again = populate(v, help_str[k], help_dict[k])
        if not again:
            msg = f"{help_str[k]} "
            if isinstance(field_defaults[k], type(MISSING)):
                field_defaults[k] = field_factories[k]()
            if field_defaults[k] == "???":
                msg += f"{Fore.RED}MANDATORY. "
            else:
                msg += f"Default: {field_defaults[k]}. "
            if field_choices[k]:
                msg += f"Choices: {field_choices[k]}"
            help_dict[k] = msg

    return help_dict


def make_help_dict():

    # We add this so that the help object incudes a generic gain field.
    additional_config = [oc.from_dotlist(["solver.gain_terms=['gain']"])]

    FinalConfig = finalize_structure(additional_config)

    help_dict = populate(FinalConfig, HELPSTRINGS)

    return help_dict


def print_help(help_dict, selection):

    # This guards against attempting to get the terminal size when the output
    # is being piped/redirected.
    if sys.stdout.isatty():
        columns, _ = os.get_terminal_size(0)
    else:
        columns = 80  # Fall over to some sensible default.

    help_message = f"{Style.BRIGHT}"

    current_section = None

    for section, options in help_dict.items():

        if section not in selection:
            continue

        if current_section != section:
            help_message += "" if current_section is None \
                else f"{Fore.CYAN}{'':-^{columns}}\n"
            help_message += f"\n{Fore.CYAN}{section:-^{columns}}\n"
            current_section = section

        if section == "gain":
            txt = textwrap.fill(GAIN_MSG, width=columns)
            help_message += f"{Fore.GREEN}{txt:-^{columns}}\n"

        for key, value in options.items():
            option = f"{section}.{key}"
            help_message += f"{Fore.MAGENTA}{option:<}\n"
            txt = textwrap.fill(value,
                                width=columns,
                                initial_indent=" "*4,
                                subsequent_indent=" "*4)
            help_message += f"{Fore.WHITE}{txt:<{columns}}\n"

    help_message += f"{Fore.CYAN}{'':-^{columns}}"

    txt = textwrap.fill(HELP_MSG, width=columns)

    help_message += f"{Fore.GREEN}{txt:-^{columns}}\n"

    help_message += f"{Fore.CYAN}{'':-^{columns}}{Style.RESET_ALL}"

    print(help_message)


def help():
    """Prints the help."""

    help_args = [arg for arg in sys.argv if arg.startswith('help')]

    # Always take the last specified help request.
    help_arg = help_args.pop() if help_args else help_args

    if len(sys.argv) == 1 or help_arg == "help":
        help_dict = make_help_dict()
        selection = help_dict.keys()
    elif help_arg:
        help_dict = make_help_dict()
        selection = help_arg.split("=")[-1]
        selection = re.sub('[\[\] ]', "", selection)  # noqa
        selection = selection.split(",")
    else:
        return

    print_help(help_dict, selection)

    sys.exit()
