import pytest
from quartical.config.preprocess import (transcribe_recipe,
                                         sky_model_nt,
                                         prepare_output_directory)
import dask.array as da
from pathlib import Path
import os.path
import os


# A dictionary mapping valid recipe inputs to expected outputs.

valid_recipes = {
    "COL1":
        {0: ["COL1"]},
    "~COL1":
        {0: ['', da.subtract, 'COL1']},
    "MODEL.lsm.html":
        {0: [sky_model_nt('MODEL.lsm.html', ())]},
    "~MODEL.lsm.html":
        {0: ['', da.subtract, sky_model_nt('MODEL.lsm.html', ())]},
    "MODEL.lsm.html@dE":
        {0: [sky_model_nt('MODEL.lsm.html', ('dE',))]},
    "MODEL.lsm.html@dE,dG":
        {0: [sky_model_nt('MODEL.lsm.html', ('dE', 'dG'))]},
    "COL1:COL2":
        {0: ['COL1'],
         1: ['COL2']},
    "COL1:MODEL.lsm.html":
        {0: ['COL1'],
         1: [sky_model_nt('MODEL.lsm.html', ())]},
    "COL1:MODEL.lsm.html@dE":
        {0: ['COL1'],
         1: [sky_model_nt('MODEL.lsm.html', ('dE',))]},
    "MODEL.lsm.html:MODEL.lsm.html@dE":
        {0: [sky_model_nt('MODEL.lsm.html', ())],
         1: [sky_model_nt('MODEL.lsm.html', ('dE',))]},
    "COL1~COL2":
        {0: ['COL1', da.subtract, 'COL2']},
    "COL1+COL2":
        {0: ['COL1', da.add, 'COL2']},
    "COL1~MODEL.lsm.html":
        {0: ['COL1', da.subtract, sky_model_nt('MODEL.lsm.html', ())]},
    "COL1+MODEL.lsm.html":
        {0: ['COL1', da.add, sky_model_nt('MODEL.lsm.html', ())]},
    "COL1~MODEL.lsm.html:COL2":
        {0: ['COL1', da.subtract, sky_model_nt('MODEL.lsm.html', ())],
         1: ['COL2']},
    "COL1+MODEL.lsm.html:COL2":
        {0: ['COL1', da.add, sky_model_nt('MODEL.lsm.html', ())],
         1: ['COL2']}
}

# A dictionary mapping invalid inputs to their expected errors. Currently
# we do not attempt to validate column names in the preprocess step.

invalid_recipes = {
    "": ValueError,
    "dummy.lsm.html": FileNotFoundError
}


@pytest.fixture(params=valid_recipes.items())
def valid_recipe(request):
    return request.param


@pytest.fixture(params=invalid_recipes.items())
def invalid_recipe(request):
    return request.param


@pytest.mark.preprocess
def test_transcribe_recipe_valid(valid_recipe, monkeypatch):

    input_recipe, expected_output = valid_recipe
    # Patch isfile functionality to allow use of ficticious files.
    monkeypatch.setattr(os.path, "isfile", lambda filename: True)

    recipe = transcribe_recipe(input_recipe)

    # Check that the opts has been updated with the correct internal recipe.
    assert recipe.instructions == expected_output


@pytest.mark.preprocess
def test_transcribe_recipe_invalid(invalid_recipe):

    # This verifies that an appropriate error is raised for obvious bad input.

    input_recipe, expected_output = invalid_recipe

    with pytest.raises(expected_output):
        transcribe_recipe(input_recipe)


@pytest.mark.preprocess
def test_prepare_output_directory():

    test_dir_path = Path(__file__).absolute().parent

    tmp_dir_path = test_dir_path / "tmp_dir.qc"

    os.mkdir(tmp_dir_path)

    open(tmp_dir_path / "tmp.qc", "x")

    # This verifies that the .qc files get cleaned out.

    prepare_output_directory(tmp_dir_path)

    assert not list(tmp_dir_path.iterdir()), ".qc files were not removed."

    tmp_dir_path.rmdir()  # Remove temp directory.
