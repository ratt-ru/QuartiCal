import pytest
from cubicalv2.data_handling.ms_handler import read_ms
from cubicalv2.parser.preprocess import interpret_model
from cubicalv2.data_handling.predict import predict, parse_sky_model
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

    interpret_model(options)

    return options


@pytest.fixture(scope="module")
def _predict(opts):

    ms_xds_list, col_kwrds = read_ms(opts)

    return predict(ms_xds_list, opts)


def test_parse_sky_model_di(lsm_name):

    options = Namespace(**{"input_model_recipe": lsm_name,
                           "input_model_source_chunks": 10})

    interpret_model(options)

    sky_model_dict = parse_sky_model(options)

    assert 1==1


def test_parse_sky_model_dd(lsm_name):

    options = Namespace(**{"input_model_recipe": lsm_name + "@dE",
                           "input_model_source_chunks": 10})

    interpret_model(options)

    sky_model_dict = parse_sky_model(options)

    assert 1==1


# def test_predict2(_predict):

#     print(id(_predict))

#     assert 1==1


# def test_predict3(_predict):

#     print(id(_predict))

#     assert 1==1
