import pathlib

import pytest
import numpy as np

from defdap.file_readers import EBSDDataLoader, DICDataLoader

EXAMPLE_CTF = pathlib.Path("tests/data/testDataEBSD.ctf")
EXAMPLE_CPR = pathlib.Path("tests/data/testDataEBSD.cpr")
EXAMPLE_CRC = pathlib.Path("tests/data/testDataEBSD.crc")
BAD_PATH = pathlib.Path("aaabbb")
EXAMPLE_DIC = "tests/data/testDataDIC.txt"


def ebsd_metadata_tests(data_loader: EBSDDataLoader):
    """This function describes the metadata in the EBSD test files so can be used to check if
    the test metadata is correctly loaded."""
    assert data_loader.loadedMetadata["xDim"] == 25
    assert data_loader.loadedMetadata["yDim"] == 25
    assert data_loader.loadedMetadata["stepSize"] == pytest.approx(0.12)
    assert data_loader.loadedMetadata["numPhases"] == 1
    assert data_loader.loadedMetadata["phaseNames"] == ["Ni-superalloy"]


def ebsd_data_tests(data_loader: EBSDDataLoader):
    """This function describes the data in the EBSD test files so can be used to check if
    the test data is correctly loaded."""
    x_dim = data_loader.loadedMetadata["xDim"]
    y_dim = data_loader.loadedMetadata["yDim"]

    assert data_loader.loadedData['bandContrast'].shape == (y_dim, x_dim)
    assert isinstance(data_loader.loadedData['bandContrast'][0][0], np.uint8)

    assert data_loader.loadedData['phase'].shape == (y_dim, x_dim)
    assert isinstance(data_loader.loadedData['phase'][0][0], np.int8)

    assert data_loader.loadedData['eulerAngle'].shape == (3, y_dim, x_dim)
    assert data_loader.loadedData['eulerAngle'][0][0][3] == pytest.approx(0.067917, rel=1e4)
    assert isinstance(data_loader.loadedData['eulerAngle'][0], np.ndarray)
    assert isinstance(data_loader.loadedData['eulerAngle'][0][0], np.ndarray)
    assert isinstance(data_loader.loadedData['eulerAngle'][0][0][0], np.float64)


class TestEBSDDataLoader:
    """The loader object stores EBSD data and associated metadata."""

    @staticmethod
    def test_init():
        """Test initialisation of the Loader object."""
        data_loader = EBSDDataLoader()
        assert isinstance(data_loader.loadedMetadata, dict)
        assert isinstance(data_loader.loadedData, dict)

    @staticmethod
    def test_checkMetadata_good():
        """The check_metadata method should pass silently if phaseNames
        and numPhases match."""
        data_loader = EBSDDataLoader()
        data_loader.loadedMetadata["phaseNames"] = ["1", "2", "3"]
        data_loader.loadedMetadata["numPhases"] = 3
        assert data_loader.checkMetadata() is None

    @staticmethod
    def test_checkMetadata_bad():
        """The check_metadata method should fail if phaseNames and
        numPhases do not match."""
        data_loader = EBSDDataLoader()
        data_loader.loadedMetadata["phaseNames"] = ["1", "2"]
        data_loader.loadedMetadata["numPhases"] = 3
        with pytest.raises(AssertionError):
            data_loader.checkMetadata()


class TestLoadOxfordBinary:
    """Tests for loading Oxford binary EBSD files. These consist of metadata in a CPR file
    and binary EBSD data in a CRC file."""

    @staticmethod
    def test_load_oxford_cpr_good_file():
        """Load a known good cpr file and check the contents are read correctly."""
        data_loader = EBSDDataLoader()
        data_loader.loadOxfordCPR(EXAMPLE_CPR)

        ebsd_metadata_tests(data_loader)

    @staticmethod
    def test_load_oxford_cpr_bad_file():
        """Check an error is raised on a bad file name."""
        with pytest.raises(FileNotFoundError):
            data_loader = EBSDDataLoader()
            data_loader.loadOxfordCPR(BAD_PATH)

    @staticmethod
    def test_load_oxford_crc_good_file():
        """Load a known good crc file and check the contents are read correctly."""
        data_loader = EBSDDataLoader()
        data_loader.loadOxfordCPR(EXAMPLE_CPR)
        data_loader.loadOxfordCRC(EXAMPLE_CRC)

        ebsd_data_tests(data_loader)

    @staticmethod
    def test_load_oxford_crc_bad():
        """Check an error is raised on a bad file name."""
        with pytest.raises(FileNotFoundError):
            data_loader = EBSDDataLoader()
            data_loader.loadOxfordCRC(BAD_PATH)


class TestLoadCTF:
    """Tests for loading EBSD CTF files. These are text files containing metadata in a
    header and the EBSD data beneath."""

    @staticmethod
    def test_load_oxford_ctf_good():
        """Load a known good ctf file and check the contents are read correctly."""
        data_loader = EBSDDataLoader()
        data_loader.loadOxfordCTF(EXAMPLE_CTF)

        ebsd_metadata_tests(data_loader)
        ebsd_data_tests(data_loader)

    @staticmethod
    def test_load_oxford_ctf_bad():
        """Check an error is raised on a bad file name."""
        with pytest.raises(FileNotFoundError):
            data_loader = EBSDDataLoader()
            data_loader.loadOxfordCTF(BAD_PATH)


class TestLoadDIC:
    """Tests for loading DIC text files. These are text files containing metadata in a
    header and the DIC data beneath."""

    @staticmethod
    def test_load_metadata():
        """Load a known good DIC txt file and check the metadata is read correctly."""
        data_loader = DICDataLoader()
        data_loader.loadDavisMetadata(EXAMPLE_DIC)

        assert data_loader.loadedMetadata["format"] == "DaVis"
        assert data_loader.loadedMetadata["version"] == "8.4.0"
        assert data_loader.loadedMetadata["binning"] == 12
        assert data_loader.loadedMetadata["xDim"] == 17
        assert data_loader.loadedMetadata["yDim"] == 17

    @staticmethod
    def test_load_bad_metadata():
        """Check an error is raised on a bad file name."""
        data_loader = DICDataLoader()
        with pytest.raises(FileNotFoundError):
            data_loader.loadDavisMetadata("bad_file_name")

    @staticmethod
    def test_load_data():
        """Load a known good DIC txt file and check the data are read correctly."""
        data_loader = DICDataLoader()
        data_loader.loadDavisMetadata(EXAMPLE_DIC)
        data_loader.loadDavisData(EXAMPLE_DIC)

        assert data_loader.loadedData["xc"].shape == (289,)
        assert data_loader.loadedData["xc"][0] == 6
        assert data_loader.loadedData["yc"].shape == (289,)
        assert data_loader.loadedData["yc"][0] == 6
        assert data_loader.loadedData["xd"].shape == (289,)
        assert data_loader.loadedData["xd"][0] == 54.1145
        assert data_loader.loadedData["yd"].shape == (289,)
        assert data_loader.loadedData["yd"][0] == 0.0357

    @staticmethod
    def test_load_bad_data():
        """Check an error is raised on a bad file name."""
        data_loader = DICDataLoader()
        with pytest.raises(FileNotFoundError):
            data_loader.loadDavisData("bad_file_name")
