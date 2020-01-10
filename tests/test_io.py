import pathlib

import pytest
import numpy as np

import defdap.file_readers as file_readers

EXAMPLE_CTF = pathlib.Path("tests/data/testDataEBSD.ctf")
EXAMPLE_CPR = pathlib.Path("tests/data/testDataEBSD.cpr")
EXAMPLE_CRC = pathlib.Path("tests/data/testDataEBSD.crc")
BAD_PATH = pathlib.Path("aaabbb")
EXAMPLE_DIC = "tests/data/testDataDIC.txt"


def ebsd_metadata_tests(metadata: file_readers.EBSDMetadata):
    """This function describes the metadata in the EBSD test files so can be used to check if
    the test metadata is correctly loaded."""
    assert metadata.xDim == 25
    assert metadata.yDim == 25
    assert metadata.stepSize == pytest.approx(0.12)
    assert metadata.numPhases == 1
    assert metadata.phaseNames == ["Ni-superalloy"]


def ebsd_data_tests(data: file_readers.EBSDData, metadata: file_readers.EBSDMetadata):
    """This function describes the data in the EBSD test files so can be used to check if
    the test data is correctly loaded."""

    assert data.bandContrast.shape == (metadata.yDim, metadata.xDim)
    assert isinstance(data.bandContrast[0][0], np.uint8)

    assert data.phase.shape == (metadata.yDim, metadata.xDim)
    assert isinstance(data.phase[0][0], np.int8)

    assert data.eulerAngle.shape == (3, metadata.yDim, metadata.xDim)
    assert data.eulerAngle[0][0][3] == pytest.approx(0.067917, rel=1e4)
    assert isinstance(data.eulerAngle[0], np.ndarray)
    assert isinstance(data.eulerAngle[0][0], np.ndarray)
    assert isinstance(data.eulerAngle[0][0][0], np.float32)


class TestEBSDDataLoader:
    """The loader object stores EBSD data and associated metadata."""

    @staticmethod
    def test_checkMetadata_good():
        """The check_metadata method should pass silently if phaseNames
        and numPhases match."""
        metadata = file_readers.EBSDMetadata()
        metadata.phaseNames = ["1", "2", "3"]
        metadata.numPhases = 3
        assert file_readers._checkEBSDMetadata(metadata) is None

    @staticmethod
    def test_checkMetadata_bad():
        """The check_metadata method should fail if phaseNames and
        numPhases do not match."""
        metadata = file_readers.EBSDMetadata()
        metadata.phaseNames = ["1", "2"]
        metadata.numPhases = 3
        with pytest.raises(AssertionError):
            file_readers._checkEBSDMetadata(metadata)


class TestLoadEBSD:
    """Tests for loading EBSD files of various types"""

    @staticmethod
    def test_load_oxford_binary_good_file():
        """Load a known good cpr file and check the contents are read correctly."""
        metadata, data = file_readers.loadEBSDData(EXAMPLE_CPR)

        ebsd_metadata_tests(metadata)
        ebsd_data_tests(data, metadata)

    @staticmethod
    def test_load_oxford_binary_bad_file():
        """Check an error is raised on a non-existent path."""
        with pytest.raises(FileNotFoundError):
            _ = file_readers.loadEBSDData(BAD_PATH)

    @staticmethod
    def test_load_oxford_binary_wrong_file():
        """Check an error is raised on loading an unsupported file type."""
        with pytest.raises(TypeError):
            _ = file_readers.loadEBSDData(EXAMPLE_DIC)

    @staticmethod
    def test_load_oxford_ctf_good():
        """Load a known good ctf file and check the contents are read correctly."""
        metadata, data = file_readers.loadEBSDData(EXAMPLE_CTF)

        ebsd_metadata_tests(metadata)
        ebsd_data_tests(data, metadata)


class TestLoadDIC:
    """Tests for loading DIC text files. These are text files containing metadata in a
    header and the DIC data beneath."""

    @staticmethod
    def test_load_data():
        """Load a known good DIC txt file and check the metadata
        and data are read correctly."""
        metadata, data = file_readers.loadDICData(EXAMPLE_DIC)

        assert metadata.format == "DaVis"
        assert metadata.version == "8.4.0"
        assert metadata.binning == 12
        assert metadata.xDim == 17
        assert metadata.yDim == 17

        assert data.xc.shape == (289,)
        assert data.xc[0] == 6
        assert data.yc.shape == (289,)
        assert data.yc[0] == 6
        assert data.xd.shape == (289,)
        assert data.xd[0] == 54.1145
        assert data.yd.shape == (289,)
        assert data.yd[0] == 0.0357

    @staticmethod
    def test_load_bad_data():
        """Check an error is raised on a bad file name."""
        with pytest.raises(FileNotFoundError):
            file_readers.loadDICData(BAD_PATH)
