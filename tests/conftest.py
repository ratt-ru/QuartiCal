import sys
import pytest
from pathlib import Path
from cubicalv2.parser.parser import parse_inputs
import requests
import tarfile
from shutil import rmtree


test_root_path = Path(__file__).resolve().parent
test_data_path = Path(test_root_path, "test_data")

_tar_name = "C147_subset.tar.gz"
_ms_name = "C147_subset.MS"
_conf_name = "test_config.yaml"
_lsm_name = "3C147_apparent.lsm.html"

tar_path = Path(test_data_path, _tar_name)
ms_path = Path(test_data_path, _ms_name)
conf_path = Path(test_data_path, _conf_name)
lsm_path = Path(test_data_path, _lsm_name)

dl_link = "https://www.dropbox.com/s/q5fxx44wche046v/C147_subset.tar.gz"


def pytest_sessionstart(session):
    """Called after Session object has been created, before run test loop."""

    if ms_path.exists():
        print("Test data already present - not downloading.")
    else:
        print("Test data not found - downloading...")
        download = requests.get(dl_link, params={"dl": 1})
        with open(tar_path, 'wb') as f:
            f.write(download.content)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=test_data_path)
        tar_path.unlink()
        print("Test data successfully downloaded.")


def pytest_sessionfinish(session, exitstatus):
    """Called after test run finished, before returning exit status."""

    if ms_path.exists():
        print("\nRemoving test data...")
        rmtree(ms_path)
        print("Test data successfully removed.")


@pytest.fixture(scope="session")
def ms_name():
    """Session level fixture for test data path."""

    return str(ms_path)


@pytest.fixture(params=["UNITY", "WEIGHT", "WEIGHT_SPECTRUM"], scope="module")
def weight_column(request):
    return request.param


@pytest.fixture(params=[0, 7], scope="module")
def freq_chunk(request):
    return request.param


@pytest.fixture(params=[0, 58, 291.0], scope="module")
def time_chunk(request):
    return request.param


@pytest.fixture(params=["full", "diag"], scope="module")
def correlation_mode(request):
    return request.param


@pytest.fixture(params=["MODEL_DATA", str(lsm_path)],
                scope="module")
def model_recipe(request):
    return request.param


@pytest.fixture(scope="session")
def base_opts(ms_name):
    """Get basic config from .yaml file."""

    # We use bypass_sysargv to avoid mucking with the CLI.
    options = parse_inputs(bypass_sysargv=['gocubical', str(conf_path)])
    options.input_ms_name = ms_name  # Ensure the ms path is correct.

    return options
