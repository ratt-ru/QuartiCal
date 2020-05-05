# -*- coding: utf-8 -*-
import argparse
import ruamel.yaml
import sys
import os
from pathlib import Path
from collections import abc
import builtins
from loguru import logger
import cubicalv2.parser.custom_types as custom_types

path_to_default = Path(__file__).parent.joinpath("default_config.yaml")


def to_type(type_str):
    """Converts type string to a type."""

    if type_str is None:
        return None
    elif type_str in custom_types.custom_types.keys():
        return custom_types.custom_types[type_str]
    else:
        return getattr(builtins, type_str)


def build_args(parser, argdict, base_name="-", depth=0):
    """Recursively traverses a nested dictionary and configures the parser.

    Traverses a nested dictionary (such as the output of a .yaml file) in a
    recursive fashion and adds discovered argument groups and arguments to the
    parser object.

    Args:
        parser: An ArgumentParser object.
        argdict: A (nested) dictionary of arguments.
        base_name: An optional string prefix to which the current argument
            name can be appended.
        depth: The current recursion depth.
    """

    depth += 1

    group = parser

    for name, contents in argdict.items():
        if depth == 1:
            group = parser.add_argument_group(name,
                                              contents.get("_description", ""))

        if isinstance(contents, abc.Mapping):

            build_args(group, contents, base_name + "-" + name, depth)

        elif not name.startswith("_"):

            # Removing action and const for now - for the most part I think
            # actions are more likely to cause problems than solve them, as
            # they require certain arguments to receive special treatment.

            kwargs = {}

            kwargs["nargs"] = argdict.get("nargs", "?")
            kwargs["default"] = argdict.get("default", None)
            kwargs["default"] = \
                None if kwargs["default"] == "None" else kwargs["default"]
            kwargs["type"] = to_type(argdict.get("type", None))
            kwargs["choices"] = argdict.get("choices", None)
            kwargs["required"] = argdict.get("required", False)
            kwargs["help"] = argdict.get("help", "Undocumented option.")
            kwargs["metavar"] = argdict.get("metavar", None)

            group.add_argument(base_name, **kwargs)

            return


def create_command_line_parser():
    """Instantiates and populates a parser from default_config.yaml."""

    parser = argparse.ArgumentParser(
        usage="""gocubical [<args>]

        Performs calibration in accordance with args. If the first argument is
        positional, it is assumed to be a user-defined config file in .yaml
        format. See user_config.yaml for an example.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    with open(path_to_default, 'r') as stream:
        defaults_dict = ruamel.yaml.safe_load(stream)

    build_args(parser, defaults_dict)

    return parser


def create_and_merge_gain_parsers(args, remaining_args):
    """Dynamically creates and merges gain parsers into an existing namespace.

    Given an existing namespace and a list of sys.argv-like arguments which
    were not understood by the main parser, dynamically creates additional
    parsers by inspecting the contents of args.solver_gain_terms. These are
    the popluated with any remaining, understood arguments and merged into
    the args namespace. This function also removes the base (gain) options
    which only serve as a template.

    Args:
        opts: A namespace object.
    """

    with open(path_to_default, 'r') as stream:
        gain_defaults = ruamel.yaml.safe_load(stream)["(gain)"]

    argdict = vars(args)

    for key in list(argdict.keys()):
        if key.startswith("(gain)"):
            del argdict[key]

    for term in args.solver_gain_terms:
        gain_parser = argparse.ArgumentParser()
        build_args(gain_parser, {term: gain_defaults})
        gain_args, remaining_opts = \
            gain_parser.parse_known_args(remaining_args)
        argdict.update(vars(gain_args))


def strip_dict(argdict):
    """Recursively traverses and strips a nested dictionary.

    Traverses a nested dictionary (such as the output of a .yaml file) in a
    recursive fashion and replaces the per-argument dicitonaries with the
    contents of the associated default field. This is used to produce a basic
    user config .yaml file.

    Args:
        argdict: A (nested) dictionary of arguments.
    """

    for name, contents in argdict.items():

        if isinstance(contents, abc.Mapping):

            max_depth_reached = strip_dict(contents)

            if max_depth_reached:
                argdict[name] = contents.get("default", None)
                argdict[name] = \
                    None if argdict[name] == "None" else argdict[name]
                max_depth_reached = False

        elif not name.startswith("_"):

            return True


def create_user_config():
    """Creates a blank .yaml file with up-to-date field names and defaults."""

    if not sys.argv[-1].endswith("gocubical-config"):
        config_file_path = sys.argv[-1]
    else:
        config_file_path = "user_config.yaml"

    logger.info("Output config file path: {}", config_file_path)

    with open(path_to_default, 'r') as stream:
        defaults_dict = ruamel.yaml.safe_load(stream)

    strip_dict(defaults_dict)

    with open(config_file_path, 'w') as outfile:
        ruamel.yaml.round_trip_dump(defaults_dict,
                                    outfile,
                                    default_flow_style=False,
                                    width=60,
                                    indent=4)

    logger.success("{} successfully generated. Go forth and calibrate!",
                   config_file_path)

    return


def argdict_to_arglist(argdict, arglist=[], base_name=""):
    """Converts a nested dictionary to a list of option and value strings.

    Given a nested dictionary of options, recusively builds a list of names
    and values in the style of sys.argv. This list can be consumed by
    parse_args, triggering the relevant type checks.

    Args:
        argdict: A (nested) dictionary of arguments.
        arglist: A list of strings which can consumed by parse_args.
        base_name: A string prefix to which the argument name can be appended.

    Returns:
        arglist: A list of strings which can consumed by parse_args.
    """

    for name, contents in argdict.items():
        new_key = base_name + "-" + name if base_name else "--" + name
        if name.startswith("_"):
            pass
        elif isinstance(contents, abc.Mapping):
            argdict_to_arglist(contents, arglist, new_key)
        elif isinstance(contents, list):
            arglist.append(new_key)

            for arg in contents:
                arglist.append(str(arg))

        else:
            arglist.append(new_key)
            arglist.append(str(contents))

    return arglist


def log_final_config(args):
    """Logs the final state of the args Namespace.

    Given the overlapping nature of the various configuration options, this
    produces a pretty log message describing the final configuration state
    of the args Namespace.

    Args:
        args: A Namespace object.
    """

    # This guards against attempting to get the terminal size when the output
    # is being piped/redirected.
    if sys.stdout.isatty():
        columns, _ = os.get_terminal_size(0)
    else:
        columns = 80  # Fall over to some sensible default.
    left_column = columns//2
    right_column = columns - left_column

    log_message = "Final configuration state:"

    section = None

    for key, value in vars(args).items():
        current_section = key.split('_')[0]

        if current_section != section:
            log_message += "" if section is None \
                else "<blue>{0:-^{1}}</blue>\n".format("", columns)
            log_message += "\n<blue>{0:-^{1}}</blue>\n".format(
                current_section.upper(), columns)
            section = current_section

        log_message += \
            "{0:<{1}}".format("--" + key.replace('_', '-'), left_column)
        log_message += "{0:>{1}}".format(str(value), right_column) + "\n"

    log_message += "<blue>{0:-^{1}}</blue>".format("", columns)

    logger.opt(ansi=True).info(log_message)


def parse_inputs(bypass_sysargv=None):
    """Combines command line and config files to produce a Namespace."""

    # Firstly we generate our argparse from the defaults.

    cl_parser = create_command_line_parser()

    # Determine if we have a user defined config file. This is assumed to be
    # positional. Arguments to the left of the config file name will be
    # ignored.

    config_file_name = None

    # We use sys.argv unless the bypass is set - this is needed for testing.

    for arg_ind, arg in enumerate(bypass_sysargv or sys.argv):
        if arg.endswith('.yaml'):
            config_file_name = arg
            remaining_args = sys.argv[arg_ind + 1:]
            logger.info("User defined config file: {}", config_file_name)
            break

    # If a config file is given, we load its contents into a list of options
    # and values. We then add the remaining command line options. Finally,
    # we submit the resulting list to the parser. This handles validation
    # for us. In the absence of a positional argument, we assume all
    # arguments are specified via the command line.

    if config_file_name:

        with open(config_file_name, 'r') as stream:
            cf_args = argdict_to_arglist(ruamel.yaml.safe_load(stream))

        cf_args.extend(remaining_args)

        args, remaining_args = cl_parser.parse_known_args(cf_args)

    else:
        args, remaining_args = cl_parser.parse_known_args()

    # This is a piece of dark magic which creates new parsers on the fly. This
    # is necessary to support arbitrary gain specifications.

    create_and_merge_gain_parsers(args, remaining_args)

    # Finally, due to the merging of various argument sources, we can end up
    # with "None" strings. We convert these all to None types here so we can
    # safely check for None in the code. Lists require special treatment.

    for key, value in vars(args).items():
        if isinstance(value, list):
            vars(args)[key] = [v if v != "None" else None for v in value]
        elif value == "None":
            vars(args)[key] = None

    # Log the final state of the Namespace object so that users are aware
    # of what the ultimate configuration was.

    log_final_config(args)

    return args


if __name__ == "__main__":
    print(parse_inputs())
