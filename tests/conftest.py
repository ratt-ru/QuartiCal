import os
import pytest
from pathlib import Path


data_name = "C147_subset.MS"

data_path = Path(Path().absolute(), data_name)

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
