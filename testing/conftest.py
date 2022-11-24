import pytest
from pathlib import Path
from quartical.config.parser import parse_inputs
import requests
import tarfile
from shutil import rmtree
from testing.fixtures.config import *  # NOQA
from testing.fixtures.data_handling import *  # NOQA
from testing.fixtures.calibration import *  # NOQA
from testing.fixtures.gains import *  # NOQA


test_root_path = Path(__file__).resolve().parent
test_data_path = Path(test_root_path, "data")

_data_tar_name = "C147_subset.tar.gz"
_beam_tar_name = "beams.tar.gz"
_ms_name = "C147_subset.MS"
_beam_name = "beams"
_conf_name = "test_config.yaml"
_lsm_name = "3C147_intrinsic.lsm.html"

data_tar_path = Path(test_data_path, _data_tar_name)
beam_tar_path = Path(test_data_path, _beam_tar_name)
ms_path = Path(test_data_path, _ms_name)
beam_path = Path(test_data_path, _beam_name)
conf_path = Path(test_data_path, _conf_name)
lsm_path = Path(test_data_path, _lsm_name)

data_lnk = "https://www.dropbox.com/s/8e49mfgsh4h6skq/C147_subset.tar.gz"
beam_lnk = "https://www.dropbox.com/s/26bgrolo1qyfy4k/beams.tar.gz"

tar_lnk_list = [data_lnk, beam_lnk]
tar_pth_list = [data_tar_path, beam_tar_path]
dat_pth_list = [ms_path, beam_path]


def pytest_sessionstart(session):
    """Called after Session object has been created, before run test loop."""

    if ms_path.exists() and beam_path.exists():
        print("Test data already present - not downloading.")
    else:
        print("Test data not found - downloading...")
        for lnk, pth in zip(tar_lnk_list, tar_pth_list):
            download = requests.get(lnk, params={"dl": 1})
            with open(pth, 'wb') as f:
                f.write(download.content)
            with tarfile.open(pth, "r:gz") as tar:
                tar.extractall(path=test_data_path)
            pth.unlink()
        print("Test data successfully downloaded.")


# def pytest_sessionfinish(session, exitstatus):
#     """Called after test run finished, before returning exit status."""

#     for pth in dat_pth_list:
#         if pth.exists():
#             print("\nRemoving test data ({}).".format(pth))
#             rmtree(pth)
#             print("Test data successfully removed.")


@pytest.fixture(scope="session")
def ms_name():
    """Session level fixture for test data path."""

    return str(ms_path)


@pytest.fixture(scope="session")
def lsm_name():
    """Session level fixture for lsm path."""

    return str(lsm_path)


@pytest.fixture(scope="session")
def beam_name():
    """Session level fixture for beam path."""

    return str(beam_path)


@pytest.fixture(params=["", "WEIGHT", "WEIGHT_SPECTRUM"], scope="module")
def weight_column(request):
    return request.param


@pytest.fixture(params=[0, 7], scope="module")
def freq_chunk(request):
    return request.param


@pytest.fixture(params=[0, 291.0], scope="module")
def time_chunk(request):
    # Note that 291.0 is equivalent to 58 unique times. This just probes some
    # addiotional functionality by specifying time in seconds.
    return request.param


@pytest.fixture(params=[[0, 1, 2, 3], [0, 3], [0]], scope="module")
def select_corr(request):
    return request.param


@pytest.fixture(params=[0, 1, 20.0], scope="module")
def time_int(request):
    return request.param


@pytest.fixture(params=[0, 1, 20.0], scope="module")
def freq_int(request):
    return request.param


@pytest.fixture(params=[['G'], ['G', 'B']], scope="module")
def gain_terms(request):
    return request.param


@pytest.fixture(params=["MODEL_DATA", str(lsm_path)],
                scope="module")
def model_recipe(request):
    return request.param


@pytest.fixture(scope="module")
def base_opts(ms_name):
    """Get basic config from .yaml file."""

    # We use bypass_sysargv to avoid mucking with the CLI.
    options, _ = parse_inputs(bypass_sysargv=['goquartical', str(conf_path)])
    options.input_ms.path = ms_name  # Ensure the ms path is correct.

    return options
