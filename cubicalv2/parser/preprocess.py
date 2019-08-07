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

    if opts.input_ms_time_chunk.isnumeric():
        opts.input_ms_time_chunk = int(opts.input_ms_time_chunk)
    elif opts.input_ms_time_chunk.endswith('s'):
        opts.input_ms_time_chunk = float(opts.input_ms_time_chunk.rstrip('s'))
    else:
        raise ValueError("--input-ms-time-chunk must be either an integer \
                          number of intergrations or a duration in seconds.")

    if opts.input_ms_time_chunk == 0:
        opts.input_ms_time_chunk = int(1e99)
        logger.warning("--input-ms-time-chunk is zero: not chunking on rows.")
