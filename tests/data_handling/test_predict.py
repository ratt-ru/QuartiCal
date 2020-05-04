import pytest
from cubicalv2.data_handling.ms_handler import read_ms
from cubicalv2.parser.preprocess import interpret_model
from cubicalv2.data_handling.predict import predict
from argparse import Namespace
import numpy as np


@pytest.fixture(scope="module")
def opts(base_opts, freq_chunk, time_chunk, correlation_mode, model_recipe):

    # Don't overwrite base config - instead create a new Namespace and update.

    options = Namespace(**vars(base_opts))

    options.input_ms_freq_chunk = freq_chunk
    options.input_ms_time_chunk = time_chunk
    options.input_ms_correlation_mode = correlation_mode
    options.input_model_recipe = model_recipe

    return options


@pytest.mark.slow
@pytest.mark.data_handling
@pytest.mark.usefixtures("requires_data")
def test_predict(opts):

    interpret_model(opts)

    ms_xds_list, col_kwrds = read_ms(opts)

    predict_schemes = predict(ms_xds_list, opts)