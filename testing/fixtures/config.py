import pytest
from quartical.config.preprocess import transcribe_recipe
from quartical.config.internal import gains_to_chain


@pytest.fixture(scope="module")
def model_opts(opts):
    return opts.input_model


@pytest.fixture(scope="module")
def ms_opts(opts):
    return opts.input_ms


@pytest.fixture(scope="module")
def solver_opts(opts):
    return opts.solver


@pytest.fixture(scope="module")
def output_opts(opts):
    return opts.output


@pytest.fixture(scope="module")
def mad_flag_opts(opts):
    return opts.mad_flags


@pytest.fixture(scope="module")
def dask_opts(opts):
    return opts.dask


@pytest.fixture(scope="module")
def chain_opts(opts):
    return gains_to_chain(opts)


@pytest.fixture(scope="module")
def recipe(model_opts):
    return transcribe_recipe(model_opts.recipe)
