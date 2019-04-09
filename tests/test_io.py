import pytest
import numpy as np

import defdap.file_readers

EXAMPLE_DIC = "../example_data/Map Data 2-DIC area"
EXAMPLE_TXT = "../example_data/B00005.txt"


class TestEBSDDataLoader:

    @staticmethod
    @pytest.fixture
    def data_loader():
        return defdap.file_readers.EBSDDataLoader()

    @staticmethod
    @pytest.fixture
    def metadata_loaded(data_loader):
        data_loader.loadOxfordCPR(EXAMPLE_DIC)
        return data_loader

    @staticmethod
    def test_init(data_loader):
        assert isinstance(data_loader.loadedMetadata, dict)
        assert isinstance(data_loader.loadedData, dict)

    @staticmethod
    def test_check_metadata_good(data_loader):
        """The check_metadata method should pass silently if phaseNames
        and numPhases match."""
        data_loader.loadedMetadata["phaseNames"] = ["1", "2", "3"]
        data_loader.loadedMetadata["numPhases"] = 3
        assert data_loader.checkMetadata() is None

    @staticmethod
    def test_check_metadata_bad(data_loader):
        """The check_metadata method should fail if phaseNames and
        numPhases do not match."""
        data_loader.loadedMetadata["phaseNames"] = ["1", "2"]
        data_loader.loadedMetadata["numPhases"] = 3
        with pytest.raises(AssertionError):
            data_loader.checkMetadata()

    @staticmethod
    def test_load_oxford_cpr_good_file(data_loader):
        data_loader.loadOxfordCPR(EXAMPLE_DIC)
        metadata = data_loader.loadedMetadata
        assert metadata["xDim"] == 1006
        assert metadata["yDim"] == 996
        # Testing for floating point equality so use approx
        assert metadata["stepSize"] == pytest.approx(0.1)
        assert metadata["numPhases"] == 1
        assert metadata["phaseNames"] == ["Ni-superalloy"]

    @staticmethod
    def test_load_oxford_cpr_bad_file(data_loader):
        with pytest.raises(FileNotFoundError):
            data_loader.loadOxfordCPR("badger")

    @staticmethod
    def test_load_oxford_crc_good_file(metadata_loaded):
        metadata_loaded.loadOxfordCRC(EXAMPLE_DIC)
        x_dim = metadata_loaded.loadedMetadata["xDim"]
        y_dim = metadata_loaded.loadedMetadata["yDim"]
        assert metadata_loaded.loadedData['bandContrast'].shape == (y_dim, x_dim)
        assert isinstance(metadata_loaded.loadedData['bandContrast'][0][0], np.uint8)

        assert metadata_loaded.loadedData['phase'].shape == (y_dim, x_dim)
        assert isinstance(metadata_loaded.loadedData['phase'][0][0], np.int8)

        assert metadata_loaded.loadedData['eulerAngle'].shape == (3, y_dim, x_dim)
        assert isinstance(metadata_loaded.loadedData['eulerAngle'][0], np.ndarray)
        assert isinstance(metadata_loaded.loadedData['eulerAngle'][0][0], np.ndarray)
        assert isinstance(metadata_loaded.loadedData['eulerAngle'][0][0][0], np.float64)

    @staticmethod
    def test_load_oxford_crc_bad(metadata_loaded):
        with pytest.raises(FileNotFoundError):
            metadata_loaded.loadOxfordCRC("badger")


class TestDICDataLoader:

    @staticmethod
    @pytest.fixture
    def dic_loader():
        return defdap.file_readers.DICDataLoader()

    @staticmethod
    @pytest.fixture
    def dic_metadata_loaded(dic_loader):
        dic_loader.loadDavisMetadata(EXAMPLE_TXT)
        return dic_loader

    @staticmethod
    @pytest.fixture
    def dic_data_loaded(dic_metadata_loaded):
        dic_metadata_loaded.loadDavisData(EXAMPLE_TXT)
        return dic_metadata_loaded

    @staticmethod
    def test_init(dic_loader):
        assert isinstance(dic_loader.loadedMetadata, dict)
        assert isinstance(dic_loader.loadedData, dict)

    @staticmethod
    def test_load_davis_metadata(dic_loader):
        dic_loader.loadDavisMetadata(EXAMPLE_TXT)
        metadata = dic_loader.loadedMetadata
        assert metadata['format'] == "DaVis"
        assert metadata['version'] == "8.1.5"
        assert metadata['binning'] == 12
        assert metadata['xDim'] == 586
        assert metadata['yDim'] == 510

    @staticmethod
    def test_load_davis_metadata_bad_file(dic_loader):
        with pytest.raises(FileNotFoundError):
            dic_loader.loadDavisMetadata("badger")

    @staticmethod
    def test_load_davis_data(dic_metadata_loaded):
        dic_metadata_loaded.loadDavisData(EXAMPLE_TXT)
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