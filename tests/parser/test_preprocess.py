import pytest
from cubicalv2.parser.preprocess import interpret_model, sm_tup
import dask.array as da
from argparse import Namespace
import os.path


# A dictionary mapping recipe inputs to expected outputs.

test_recipes = {
    "COL1":
        {0: ["COL1"]},
    "~COL1":
        {0: ['', da.subtract, 'COL1']},
    "MODEL.lsm.html":
        {0: [sm_tup('MODEL.lsm.html', ())]},
    "~MODEL.lsm.html":
        {0: ['', da.subtract, sm_tup('MODEL.lsm.html', ())]},
    "MODEL.lsm.html@dE":
        {0: [sm_tup('MODEL.lsm.html', ('dE',))]},
    "MODEL.lsm.html@dE,dG":
        {0: [sm_tup('MODEL.lsm.html', ('dE', 'dG'))]},
    "COL1:COL2":
        {0: ['COL1'], 1: ['COL2']},
    "COL1:MODEL.lsm.html":
        {0: ['COL1'], 1: [sm_tup('MODEL.lsm.html', ())]},
    "COL1:MODEL.lsm.html@dE":
        {0: ['COL1'], 1: [sm_tup('MODEL.lsm.html', ('dE',))]},
    "MODEL.lsm.html:MODEL.lsm.html@dE":
        {0: [sm_tup('MODEL.lsm.html', ())],
         1: [sm_tup('MODEL.lsm.html', ('dE',))]},
    "COL1~COL2":
        {0: ['COL1', da.subtract, 'COL2']},
    "COL1+COL2":
        {0: ['COL1', da.add, 'COL2']},
    "COL1~MODEL.lsm.html":
        {0: ['COL1', da.subtract, sm_tup('MODEL.lsm.html', ())]},
    "COL1+MODEL.lsm.html":
        {0: ['COL1', da.add, sm_tup('MODEL.lsm.html', ())]},
    "COL1~MODEL.lsm.html:COL2":
        {0: ['COL1', da.subtract, sm_tup('MODEL.lsm.html', ())], 1: ['COL2']},
    "COL1+MODEL.lsm.html:COL2":
        {0: ['COL1', da.add, sm_tup('MODEL.lsm.html', ())], 1: ['COL2']}}


@pytest.fixture(params=test_recipes.keys())
def input_recipe(request):
    return request.param


@pytest.fixture
def opts(input_recipe):

    return Namespace(**{"input_model_recipe": input_recipe})


@pytest.mark.preprocess
def test_interpret_model(opts, monkeypatch):

    # Patch isfile functionality to allow use of ficticious files.
    monkeypatch.setattr(os.path, "isfile", lambda filename: True)

    interpret_model(opts)

    # Check that the opts has been updated with the correct internal recipe.
    assert opts._internal_recipe == test_recipes[opts.input_model_recipe]

