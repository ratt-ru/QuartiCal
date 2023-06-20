# -*- coding: utf-8 -*-
import sys
import os
from loguru import logger
from ruamel.yaml import round_trip_dump
from omegaconf import OmegaConf as oc
from quartical.config.external import finalize_structure
from quartical.config.internal import additional_validation
from omegaconf.errors import ConfigKeyError, ValidationError


def create_user_config():
    """Creates a blank .yaml file with up-to-date field names and defaults."""

    config_file_path = (
        "user_config.yaml" if len(sys.argv) == 1 else sys.argv[-1]
    )

    logger.info("Output config file path: {}", config_file_path)

    # We add this so that the user config comes out with a gain field.
    additional_config = [oc.from_dotlist(["solver.terms=['G']"])]

    FinalConfig = finalize_structure(additional_config)
    config = oc.structured(FinalConfig)
    config = oc.merge(config, *additional_config)

    with open(config_file_path, 'w') as outfile:
        round_trip_dump(
            oc.to_container(config),
            outfile,
            default_flow_style=False,
            width=60,
            indent=2
        )

    logger.success(
        f"{config_file_path} successfully generated. Go forth and calibrate!"
    )

    return


def log_final_config(config, config_files=[]):
    """Logs the final state of the configuration object.

    Given the overlapping nature of the various configuration options, this
    produces a pretty log message describing the final configuration state.

    Args:
        config: A FinalConfig object.
        config_files: A list of filenames.
    """

    config = oc.structured(config)

    # This guards against attempting to get the terminal size when the output
    # is being piped/redirected.
    if sys.stdout.isatty():
        columns, _ = os.get_terminal_size(0)
    else:
        columns = 80  # Fall over to some sensible default.

    for cf in config_files:
        logger.info(f"Using user-defined config file: {cf}")

    log_message = "Final configuration state:\n\n"

    for section, options in config.items():

        log_message += f"<blue>{section:-^{columns}}</blue>\n"

        maxlen = max(map(len, [f"{section}.{key}" for key in options.keys()]))

        for key, value in options.items():
            option = f"{section}.{key}"
            msg = f"{option:<{maxlen + 1}}{str(value):>{columns - maxlen - 1}}"

            if len(msg) > columns:

                split = [
                    msg[i:i + columns] for i in range(0, len(msg), columns)
                ]

                msg = "\n".join(
                    (split[0], *[f"{s:>{columns}}" for s in split[1:]])
                )

            log_message += msg + "\n"

        log_message += f"<blue>{'':-^{columns}}</blue>\n\n"

    logger.opt(ansi=True).info(log_message)


def parse_inputs(bypass_sysargv=None):
    """Combines command line and config files to produce a config object."""

    # Determine if we have a user defined config files. Scan sys.argv for
    # appropriate extensions.

    config_files = []

    # We use sys.argv unless the bypass is set - this is needed for testing.
    for arg in (bypass_sysargv or sys.argv):
        if arg.endswith(('.yaml', '.yml')):
            if arg in config_files:
                raise ValueError(f"Config file {arg} is duplicated.")
            config_files.append(arg)

    for file in config_files:
        (bypass_sysargv or sys.argv).remove(file)  # Remove config files.

    # Get all specified configuration - multiple yaml files and cli.
    yml_config = [oc.load(file) for file in config_files]
    cli_config = [] if bypass_sysargv else [oc.from_cli()]
    additional_config = [*yml_config, *cli_config]

    # Merge all configuration - priority is file1 < file2 < ... < cli.
    FinalConfig = finalize_structure(additional_config)
    config = oc.structured(FinalConfig)

    try:
        config = oc.merge(config, *additional_config)
    except ConfigKeyError as error:
        raise ValueError(
            f"User has specified an unrecognised parameter: {error.full_key}. "
            f"This often indicates a simple typo or the use of a deprecated "
            f"parameter. Please use 'goquartical help' to confirm that the "
            f"parameter exists."
        )
    except ValidationError as error:
        raise ValueError(
            f"The value specified for {error.full_key} was not understood. "
            f"This often means that the type of the argument was incorrect. "
            f"Please use 'goquartical help' to check for the expected type "
            f"and pay particular attention to parameters which expect lists."
        )

    # Log the final state of the configuration object so that users are aware
    # of what the ultimate configuration was.

    config_obj = oc.to_object(config)  # Ensures post_init methods are run.

    additional_validation(config_obj)

    return config_obj, config_files
