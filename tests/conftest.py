import os
import sys
import pytest
from pathlib import Path
from cubicalv2.parser.parser import parse_inputs


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

dl_link = "https://www.dropbox.com/s/q5fxx44wche046v/C147_subset.tar.gz?dl=0"


@pytest.fixture(scope="session")
def ms_name():
    """Session level fixture for test data path."""

    return str(ms_path)


@pytest.fixture()
def requires_data():
    """Fixture will download the test data if it is not already available."""

    if ms_path.exists():
        print("Test data already present - not downloading.")
    else:
        print("Test data not found - downloading.")
        os.system("wget -P {} {} --content-disposition".format(test_data_path,
                                                               dl_link))
        os.system("tar -zxvf {} -C {}".format(tar_path, test_data_path))
        os.system("rm {}".format(tar_path))


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
def base_opts(request):

    backup_sysargv = sys.argv

    sys.argv = ['gocubical', str(conf_path)]

    options = parse_inputs()

    sys.argv = backup_sysargv

    return options
