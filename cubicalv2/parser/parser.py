import argparse
import ruamel.yaml
import sys
from pathlib import Path
from collections import abc

def get_builtin(type_str):
    """ Converts type string to a type. """
        
    return None if type_str is None else getattr(__builtins__, type_str)

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
            group = parser.add_argument_group(name)        

        if isinstance(contents, abc.Mapping):
            build_args(group, contents, base_name + "-" + name, depth)
        else:

            kwargs = {"action"   : argdict.get("action", "store"),
                      "nargs"    : argdict.get("nargs", "?"),
                      "const"    : argdict.get("const", None),
                      "default"  : argdict.get("default", None),
                      "type"     : get_builtin(argdict.get("type", None)),
                      "choices"  : argdict.get("choices", None),
                      "required" : argdict.get("required", False),
                      "help"     : argdict.get("help", "Undocumented option.")}

            group.add_argument(base_name + "-" + name, **kwargs)
            
            return


def generate_command_line_parser():
    """Instantiates and populates a parser from default_config.yaml."""

    parser = argparse.ArgumentParser(
        description="Suite of calibration routines.",
        usage="""gocubical [<args>]
        
        Performs calibration in accordance with args. If the first argument is 
        positional, it is assumed to be a user-defined config file in .yaml 
        format. See user_config.yaml for an example. 
        """
    )

    path_to_default = Path(__file__).parent.joinpath("default_config.yaml")

    with open(path_to_default, 'r') as stream:
        defaults_dict = ruamel.yaml.safe_load(stream)

    build_args(parser, defaults_dict)

    return parser

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
        if isinstance(contents, abc.Mapping):
            argdict_to_arglist(contents, arglist, new_key)
        elif isinstance(contents, list):
            arglist.append(new_key)

            for arg in contents:
                arglist.append(arg)

        else:
            arglist.append(new_key)
            arglist.append(contents)

    return arglist

if __name__=="__main__":

    # Firstly we generate our argparse from the defaults.

    cl_parser = generate_command_line_parser()

    # Determine if we have positional arguments. May be possible to replace
    # this with a try except.

    has_args = len(sys.argv) > 1
    first_arg = sys.argv[1] if has_args else '-'
    has_positional_arg = not first_arg.startswith('-')  

    # If we have a positional argument, we assume that it is a config file.
    # We consume this file name and load its contents into a list of options
    # and values. We then add the remaining command line options. Finally,
    # we submit the resulting list to the parser. This handles validation 
    # for us. In the absence of a positional argument, we assume all
    # arguments are specified via the command line.

    if has_positional_arg:

        remaining_args = sys.argv[2:] 
             
        with open(sys.argv[1], 'r') as stream:
            cf_args = argdict_to_arglist(ruamel.yaml.safe_load(stream))

        cf_args.extend(remaining_args)

        cl_args, _ = cl_parser.parse_known_args(cf_args)

    else:
        cl_args, _ = cl_parser.parse_known_args() 

    print(cl_args)