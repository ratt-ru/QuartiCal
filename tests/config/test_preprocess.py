import pytest
from quartical.config.preprocess import transcribe_recipe, _sky_model
import dask.array as da
import os.path
from copy import deepcopy


# A dictionary mapping valid recipe inputs to expected outputs.

valid_recipes = {
    "COL1":
        {0: ["COL1"]},
    "~COL1":
        {0: ['', da.subtract, 'COL1']},
    "MODEL.lsm.html":
        {0: [_sky_model('MODEL.lsm.html', ())]},
    "~MODEL.lsm.html":
        {0: ['', da.subtract, _sky_model('MODEL.lsm.html', ())]},
    "MODEL.lsm.html@dE":
        {0: [_sky_model('MODEL.lsm.html', ('dE',))]},
    "MODEL.lsm.html@dE,dG":
        {0: [_sky_model('MODEL.lsm.html', ('dE', 'dG'))]},
    "COL1:COL2":
        {0: ['COL1'],
         1: ['COL2']},
    "COL1:MODEL.lsm.html":
        {0: ['COL1'],
         1: [_sky_model('MODEL.lsm.html', ())]},
    "COL1:MODEL.lsm.html@dE":
        {0: ['COL1'],
         1: [_sky_model('MODEL.lsm.html', ('dE',))]},
    "MODEL.lsm.html:MODEL.lsm.html@dE":
        {0: [_sky_model('MODEL.lsm.html', ())],
         1: [_sky_model('MODEL.lsm.html', ('dE',))]},
    "COL1~COL2":
        {0: ['COL1', da.subtract, 'COL2']},
    "COL1+COL2":
        {0: ['COL1', da.add, 'COL2']},
    "COL1~MODEL.lsm.html":
        {0: ['COL1', da.subtract, _sky_model('MODEL.lsm.html', ())]},
    "COL1+MODEL.lsm.html":
        {0: ['COL1', da.add, _sky_model('MODEL.lsm.html', ())]},
    "COL1~MODEL.lsm.html:COL2":
        {0: ['COL1', da.subtract, _sky_model('MODEL.lsm.html', ())],
         1: ['COL2']},
    "COL1+MODEL.lsm.html:COL2":
        {0: ['COL1', da.add, _sky_model('MODEL.lsm.html', ())],
         1: ['COL2']}
}

# A dictionary mapping invalid inputs to their expected errors. Currently
# we do not attempt to validate column names in the preprocess step.

invalid_recipes = {
    "": ValueError,
    "dummy.lsm.html": FileNotFoundError
}


@pytest.fixture(params=valid_recipes.keys())
def valid_recipe(request):
    return request.param


@pytest.fixture
def opts(base_opts, valid_recipe):

    _opts = deepcopy(base_opts)
    _opts.input_model.recipe = valid_recipe

    return _opts


@pytest.fixture(params=invalid_recipes.keys())
def invalid_recipe(request):
    return request.param


@pytest.fixture
def bad_opts(base_opts, invalid_recipe):

    _opts = deepcopy(base_opts)
    _opts.input_model.recipe = invalid_recipe

    return _opts


@pytest.mark.preprocess
def test_transcribe_recipe_valid(opts, monkeypatch):

    # Patch isfile functionality to allow use of ficticious files.
    monkeypatch.setattr(os.path, "isfile", lambda filename: True)

    recipe = transcribe_recipe(opts)

    # Check that the opts has been updated with the correct internal recipe.
    assert recipe.instructions == valid_recipes[opts.input_model.recipe]


@pytest.mark.preprocess
def test_transcribe_recipe_invalid(bad_opts):

    # This verifies that an appropriate error is raised for obvious bad input.

    with pytest.raises(invalid_recipes[bad_opts.input_model.recipe]):
        transcribe_recipe(bad_opts)
