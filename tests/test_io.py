import pathlib

import pytest
import numpy as np

import defdap.file_io as file_readers
import defdap.ebsd as ebsd
import defdap.hrdic as hrdic


EXAMPLE_CTF = "tests/data/testDataEBSD.ctf"
EXAMPLE_CPR = "tests/data/testDataEBSD.cpr"
EXAMPLE_CRC = "tests/data/testDataEBSD.crc"
BAD_PATH = "aaabbb"
EXAMPLE_DIC = "tests/data/testDataDIC.txt"


def ebsd_metadata_tests(ebsd_map: ebsd.Map):
    """This function describes the metadata in the EBSD test files so can be used to check if
    the test metadata is correctly loaded."""
    assert ebsd_map.xDim == 25
    assert ebsd_map.yDim == 25
    assert ebsd_map.stepSize == pytest.approx(0.12)
    assert ebsd_map.numPhases == 1
    assert ebsd_map.phaseNames == ["Ni-superalloy"]


def ebsd_data_tests(ebsd_map: ebsd.Map):
    """This function describes the data in the EBSD test files so can be used to check if
    the test data is correctly loaded."""

    assert ebsd_map.bandContrastArray.shape == (ebsd_map.yDim, ebsd_map.xDim)
    assert isinstance(ebsd_map.bandContrastArray[0][0], np.uint8)

    assert ebsd_map.phaseArray.shape == (ebsd_map.yDim, ebsd_map.xDim)
    assert isinstance(ebsd_map.phaseArray[0][0], np.int8)

    assert ebsd_map.eulerAngleArray.shape == (3, ebsd_map.yDim, ebsd_map.xDim)
    assert ebsd_map.eulerAngleArray[0][0][3] == pytest.approx(0.067917, rel=1e4)
    assert isinstance(ebsd_map.eulerAngleArray[0], np.ndarray)
    assert isinstance(ebsd_map.eulerAngleArray[0][0], np.ndarray)
    assert isinstance(ebsd_map.eulerAngleArray[0][0][0], np.float32)


class TestLoadEBSD:
    """Tests for loading EBSD files of various types"""

    @staticmethod
    def test_load_oxford_binary_good_file():
        """Load a known good cpr file and check the contents are read correctly."""
        ebsd_map = ebsd.Map(EXAMPLE_CPR, "cubic")

        ebsd_metadata_tests(ebsd_map)
        ebsd_data_tests(ebsd_map)

    @staticmethod
    def test_load_oxford_binary_bad_file():
        """Check an error is raised on a non-existent path."""
        with pytest.raises(FileNotFoundError):
            _ = ebsd.Map(BAD_PATH, "cubic")

    @staticmethod
    def test_load_oxford_binary_wrong_file():
        """Check an error is raised on loading an unsupported file type."""
        with pytest.raises(TypeError):
            _ = ebsd.Map(EXAMPLE_DIC, "cubic")

    @staticmethod
    def test_load_oxford_ctf_good():
        """Load a known good ctf file and check the contents are read correctly."""
        ebsd_map = ebsd.Map(EXAMPLE_CTF, "cubic")

        ebsd_metadata_tests(ebsd_map)
        ebsd_data_tests(ebsd_map)


class TestLoadDIC:
    """Tests for loading DIC text files. These are text files containing metadata in a
    header and the DIC data beneath."""

    @staticmethod
    def test_load_data():
        """Load a known good DIC txt file and check the metadata
        and data are read correctly."""
        dic_map = hrdic.Map(EXAMPLE_DIC)

        assert dic_map.format == "DaVis"
        assert dic_map.version == "8.4.0"
        assert dic_map.binning == 12
        assert dic_map.xDim == 17
        assert dic_map.yDim == 17

        assert dic_map.xc.shape == (289,)
        assert dic_map.xc[0] == 6
        assert dic_map.yc.shape == (289,)
        assert dic_map.yc[0] == 6
        assert dic_map.xd.shape == (289,)
        assert dic_map.xd[0] == 54.1145
        assert dic_map.yd.shape == (289,)
        assert dic_map.yd[0] == 0.0357

    @staticmethod
    def test_load_bad_data():
        """Check an error is raised on a bad file name."""
        with pytest.raises(FileNotFoundError):
            _ = hrdic.Map(BAD_PATH)
