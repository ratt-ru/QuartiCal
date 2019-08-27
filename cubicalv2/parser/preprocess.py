# -*- coding: utf-8 -*-
from loguru import logger


def preprocess_opts(opts):
    """Preprocesses the namespace/dictionary given by opts.

    Given a namespace/dictionary of options, this should verify that that
    the options can be understood. Some options specified as strings need
    further processing which may include the raising of certain flags.

    Args:
        opts: A namepsace/dictionary of options.

    Returns:
        Namespace: An updated namespace object.
    """

    opts._model_columns = []
    opts._sky_models = []
    opts._predict = False

    for model in opts.input_model_recipe:
        if ".lsm.html" in model:
            opts._sky_models.append(model)
            opts._predict = True
        else:
            opts._model_columns.append(model)

    logger.info("The following model sources were obtained from "
                "--input-model-recipe: \n"
                "   Columns: {} \n"
                "   Sky Models: {}",
                opts._model_columns,
                opts._sky_models)

    if opts._predict:
        logger.info("Enabling prediction step.")
