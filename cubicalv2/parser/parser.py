# -*- coding: utf-8 -*-
import argparse
import ruamel.yaml
import sys
from pathlib import Path
from collections import abc
import builtins
from loguru import logger
import cubicalv2.parser.custom_types as custom_types


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
            gain_label = contents.get("_label", None)
            name = gain_label if gain_label is not None else name
            group = parser.add_argument_group(name,
                                              contents.get("_description", ""))

        if isinstance(contents, abc.Mapping):

            build_args(group, contents, base_name + "-" + name, depth)

        elif not name.startswith("_"):

            # Removing these for now - for the most part I think actions are
            # more likely to cause problems than solve them, as they require
            # certain arguments to receive special treatment.

            kwargs = {  # "action": argdict.get("action", "store"),
                      "nargs": argdict.get("nargs", "?"),
                        # "const": argdict.get("const", None),
                      "default": argdict.get("default", None),
                      "type": to_type(argdict.get("type", None)),
                      "choices": argdict.get("choices", None),
                      "required": argdict.get("required", False),
                      "help": argdict.get("help", "Undocumented option."),
                      "metavar": argdict.get("metavar", None)}

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

    path_to_default = Path(__file__).parent.joinpath("default_config.yaml")

    with open(path_to_default, 'r') as stream:
        defaults_dict = ruamel.yaml.safe_load(stream)

    build_args(parser, defaults_dict)

    return parser


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
                max_depth_reached = False

        elif not name.startswith("_"):

            return True


def create_user_config():
    """Creates a blank .yaml file with up-to-date field names and defaults."""

    path_to_default = Path(__file__).parent.joinpath("default_config.yaml")

    with open(path_to_default, 'r') as stream:
        defaults_dict = ruamel.yaml.safe_load(stream)

    strip_dict(defaults_dict)

    with open("user_config.yaml", 'w') as outfile:
        ruamel.yaml.round_trip_dump(defaults_dict,
                                    outfile,
                                    default_flow_style=False,
                                    width=60,
                                    indent=4)

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


def parse_inputs():

    # Firstly we generate our argparse from the defaults.

    cl_parser = create_command_line_parser()

    # This generates a default user config file in the current directory -
    # add this as a script to the the package.

    # create_user_config()

    # Determine if we have a user defined config file. This is assumed to be
    # positional. Arguments to the left of the config file name will be
    # ignored.

    config_file_name = None

    for arg_ind, arg in enumerate(sys.argv):
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

    # TODO: Move this to a function. This is a piece of dark magic which
    # creates new parsers on the fly. This is necessary to support arbitrary
    # gain specifications.

    path_to_default = Path(__file__).parent.joinpath("default_config.yaml")

    with open(path_to_default, 'r') as stream:
        gain_defaults = ruamel.yaml.safe_load(stream)["(gain)"]

    for term in args.solver_gain_terms:
        gain_parser = argparse.ArgumentParser()
        build_args(gain_parser, {term: gain_defaults})
        gain_args, remaining_args = \
            gain_parser.parse_known_args(remaining_args)
        vars(args).update(vars(gain_args))

    return args


if __name__ == "__main__":
    print(parse_inputs())
