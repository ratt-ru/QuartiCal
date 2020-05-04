import os
import sys
import pytest
from pathlib import Path
from cubicalv2.parser.parser import parse_inputs


data_name = "C147_subset.MS"
conf_name = "test_config.yaml"

test_root_path = Path(__file__).resolve().parent

data_path = Path(test_root_path, data_name)
conf_path = Path(test_root_path, conf_name)

link = "https://www.dropbox.com/s/q5fxx44wche046v/C147_subset.tar.gz?dl=0"


@pytest.fixture(scope="session")
def ms_name():
    """Session level fixture for test data path."""

    return str(data_path)


@pytest.fixture()
def requires_data():
    """Fixture will download the test data if it is not already available."""

    if data_path.exists():
        print("Test data already present - not downloading.")
    else:
        print("Test data not found - downloading.")
        os.system("wget {} --content-disposition".format(link))
        os.system("tar -zxvf C147_subset.tar.gz")
        os.system("rm C147_subset.tar.gz")


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


@pytest.fixture(params=["MODEL_DATA", "3C147_apparent.lsm.html"],
                scope="module")
def model_recipe(request):
    return request.param


@pytest.fixture(scope="session")
def base_opts(request):

    backup_sysargv = sys.argv

    sys.argv = ['gocubical', str(conf_path)]

    options = parse_inputs()

    sys.argv = backup_sysargv

    return options
