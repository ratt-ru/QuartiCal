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
    opts._sky_models = {}
    opts._predict = False

    models = opts.input_model_recipe.split(":")

    # TODO: Consider how to implement operations on model sources. This will
    # require a stacking and rechunking approach in addition to adding and
    # subtracting visibilities.

    for model in models:
        if ".lsm.html" in model:
            filename, _, tags = model.partition("@")
            opts._sky_models[filename] = tags
            opts._predict = True
        else:
            opts._model_columns.append(model)

    logger.info("The following model sources were obtained from "
                "--input-model-recipe: \n"
                "   Columns: {} \n"
                "   Sky Models: {}",
                opts._model_columns,
                list(opts._sky_models.keys()))

    if opts._predict:
        logger.info("Enabling prediction step.")
