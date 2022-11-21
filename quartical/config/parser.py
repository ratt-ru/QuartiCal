# -*- coding: utf-8 -*-
import sys
import os
from loguru import logger
from ruamel.yaml import round_trip_dump
from omegaconf import OmegaConf as oc
from quartical.config.external import finalize_structure
from quartical.config.internal import additional_validation


def create_user_config():
    """Creates a blank .yaml file with up-to-date field names and defaults."""

    if not sys.argv[-1].endswith("goquartical-config"):
        config_file_path = sys.argv[-1]
    else:
        config_file_path = "user_config.yaml"

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

    logger.success("{} successfully generated. Go forth and calibrate!",
                   config_file_path)

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

    log_message = "Final configuration state:"

    current_section = None

    for section, options in config.items():

        if current_section != section:
            log_message += "" if current_section is None \
                else "<blue>{0:-^{1}}</blue>\n".format("", columns)
            log_message += "\n<blue>{0:-^{1}}</blue>\n".format(
                section, columns)
            current_section = section

        maxlen = max(map(len, [f"{section}.{key}" for key in options.keys()]))

        for key, value in options.items():
            option = f"{section}.{key}"
            msg = f"{option:<{maxlen + 1}}{str(value):>{columns - maxlen - 1}}"

            if len(msg) > columns:
                split = [msg[i:i+columns] for i in range(0, len(msg), columns)]

                msg = "\n".join((split[0],
                                *[f"{s:>{columns}}" for s in split[1:]]))

            log_message += msg + "\n"

    log_message += "<blue>{0:-^{1}}</blue>".format("", columns)

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
    config = oc.merge(config, *additional_config)

    # Log the final state of the configuration object so that users are aware
    # of what the ultimate configuration was.

    config_obj = oc.to_object(config)  # Ensures post_init methods are run.

    additional_validation(config_obj)

    return config_obj, config_files
