import pytest
import numpy as np

import defdap.file_readers
from defdap.crystal import crystalStructures, Phase

DATA_DIR = "tests/data/"
EXAMPLE_EBSD = DATA_DIR + "testDataEBSD"
EXAMPLE_DIC = DATA_DIR + "testDataDIC.txt"


class TestEBSDDataLoader:

    @staticmethod
    @pytest.fixture
    def data_loader_oxford_binary():
        return defdap.file_readers.OxfordBinaryLoader()

    @staticmethod
    @pytest.fixture
    def metadata_loaded_oxford_binary(data_loader_oxford_binary):
        data_loader_oxford_binary.loadOxfordCPR(EXAMPLE_EBSD)
        return data_loader_oxford_binary

    @staticmethod
    def test_init(data_loader_oxford_binary):
        assert isinstance(data_loader_oxford_binary.loadedMetadata, dict)
        assert isinstance(data_loader_oxford_binary.loadedData, dict)

    @staticmethod
    def test_check_metadata_good(data_loader_oxford_binary):
        """The check_metadata method should pass silently if each phase
        is of `Phase` type."""
        data_loader_oxford_binary.loadedMetadata["phases"] = [
            Phase("test", 9, ()),
            Phase("tester", 11, ()),
            Phase("testist", 11, ()),
        ]
        assert data_loader_oxford_binary.checkMetadata() is None

    @staticmethod
    def test_check_metadata_bad(data_loader_oxford_binary):
        """The check_metadata method should fail if a phase is is not of
        `Phase` type."""
        data_loader_oxford_binary.loadedMetadata["phases"] = [
            Phase("test", 9, ()),
            "2"
        ]
        with pytest.raises(AssertionError):
            data_loader_oxford_binary.checkMetadata()

    @staticmethod
    def test_load_oxford_cpr_good_file(data_loader_oxford_binary):
        data_loader_oxford_binary.loadOxfordCPR(EXAMPLE_EBSD)
        metadata = data_loader_oxford_binary.loadedMetadata
        assert metadata["xDim"] == 359
        assert metadata["yDim"] == 243
        # Testing for floating point equality so use approx
        assert metadata["stepSize"] == pytest.approx(0.12)
        assert metadata["acquisitionRotation"].quatCoef == \
               pytest.approx((1., 0., 0., 0.))

        assert type(metadata["phases"]) is list
        assert len(metadata["phases"]) == 1
        loaded_phase = metadata["phases"][0]
        assert loaded_phase.name == "Ni-superalloy"
        assert loaded_phase.latticeParams == \
               pytest.approx((3.57, 3.57, 3.57, np.pi/2, np.pi/2, np.pi/2))
        assert loaded_phase.crystalStructure is crystalStructures['cubic']

    @staticmethod
    def test_load_oxford_cpr_bad_file(data_loader_oxford_binary):
        with pytest.raises(FileNotFoundError):
            data_loader_oxford_binary.loadOxfordCPR("badger")

    @staticmethod
    def test_load_oxford_crc_good_file(metadata_loaded_oxford_binary):
        metadata_loaded_oxford_binary.loadOxfordCRC(EXAMPLE_EBSD)
        x_dim = metadata_loaded_oxford_binary.loadedMetadata["xDim"]
        y_dim = metadata_loaded_oxford_binary.loadedMetadata["yDim"]
        assert isinstance(metadata_loaded_oxford_binary.loadedData['bandContrast'], np.ndarray)
        assert metadata_loaded_oxford_binary.loadedData['bandContrast'].shape == (y_dim, x_dim)
        assert isinstance(metadata_loaded_oxford_binary.loadedData['bandContrast'][0, 0], np.uint8)

        assert isinstance(metadata_loaded_oxford_binary.loadedData['phase'], np.ndarray)
        assert metadata_loaded_oxford_binary.loadedData['phase'].shape == (y_dim, x_dim)
        assert isinstance(metadata_loaded_oxford_binary.loadedData['phase'][0, 0], np.uint8)

        assert isinstance(metadata_loaded_oxford_binary.loadedData['eulerAngle'], np.ndarray)
        assert metadata_loaded_oxford_binary.loadedData['eulerAngle'].shape == (3, y_dim, x_dim)
        assert isinstance(metadata_loaded_oxford_binary.loadedData['eulerAngle'][0, 0, 0], np.float64)

    @staticmethod
    def test_load_oxford_crc_bad(metadata_loaded_oxford_binary):
        with pytest.raises(FileNotFoundError):
            metadata_loaded_oxford_binary.loadOxfordCRC("badger")


class TestDICDataLoader:

    @staticmethod
    @pytest.fixture
    def dic_loader():
        return defdap.file_readers.DICDataLoader()

    @staticmethod
    @pytest.fixture
    def dic_metadata_loaded(dic_loader):
        dic_loader.loadDavisMetadata(EXAMPLE_DIC)
        return dic_loader

    @staticmethod
    @pytest.fixture
    def dic_data_loaded(dic_metadata_loaded):
        dic_metadata_loaded.loadDavisData(EXAMPLE_DIC)
        return dic_metadata_loaded

    @staticmethod
    def test_init(dic_loader):
        assert isinstance(dic_loader.loadedMetadata, dict)
        assert isinstance(dic_loader.loadedData, dict)

    @staticmethod
    def test_load_davis_metadata(dic_loader):
        dic_loader.loadDavisMetadata(EXAMPLE_DIC)
        metadata = dic_loader.loadedMetadata
        assert metadata['format'] == "DaVis"
        assert metadata['version'] == "8.4.0"
        assert metadata['binning'] == 12
        assert metadata['xDim'] == 300
        assert metadata['yDim'] == 200

    @staticmethod
    def test_load_davis_metadata_bad_file(dic_loader):
        with pytest.raises(FileNotFoundError):
            dic_loader.loadDavisMetadata("badger")

    @staticmethod
    def test_load_davis_data(dic_metadata_loaded):
        dic_metadata_loaded.loadDavisData(EXAMPLE_DIC)
        data = dic_metadata_loaded.loadedData
        num_elements = dic_metadata_loaded.loadedMetadata["xDim"] * \
                       dic_metadata_loaded.loadedMetadata["yDim"]
        assert data['xc'].shape[0] == num_elements
        assert data['yc'].shape[0] == num_elements
        assert data['xd'].shape[0] == num_elements
        assert data['yd'].shape[0] == num_elements

    @staticmethod
    def test_load_davis_data_bad_file(dic_metadata_loaded):
        with pytest.raises(FileNotFoundError):
            dic_metadata_loaded.loadDavisData("badger")

    @staticmethod
    def test_check_davis_data(dic_data_loaded):
        assert dic_data_loaded.checkData() is None

    @staticmethod
    def test_check__bad_davis_data(dic_data_loaded):
        dic_data_loaded.loadedMetadata["xDim"] = 42
        with pytest.raises(AssertionError):
            dic_data_loaded.checkData()