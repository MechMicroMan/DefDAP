import pytest
import numpy as np

import defdap.io


class TestEBSDDataLoader:

    @staticmethod
    @pytest.fixture
    def data_loader():
        return defdap.io.EBSDDataLoader()

    @staticmethod
    @pytest.fixture
    def metadata_loaded(data_loader):
        data_loader.loadOxfordCPR("test_data/example")
        return data_loader

    @staticmethod
    def test_init(data_loader):
        assert isinstance(data_loader.loadedMetadata, dict)
        assert isinstance(data_loader.loadedData, dict)

    @staticmethod
    def test_check_metadata_good(data_loader):
        """The check_metadata method should pass silently if phaseNames and numPhases match."""
        data_loader.loadedMetadata["phaseNames"] = ["1", "2", "3"]
        data_loader.loadedMetadata["numPhases"] = 3
        assert data_loader.checkMetadata() is None

    @staticmethod
    def test_check_metadata_bad(data_loader):
        """The check_metadata method should fail if phaseNames and numPhases do not match."""
        data_loader.loadedMetadata["phaseNames"] = ["1", "2"]
        data_loader.loadedMetadata["numPhases"] = 3
        with pytest.raises(AssertionError):
            data_loader.checkMetadata()

    @staticmethod
    def test_load_oxford_cpr_good_file(data_loader):
        data_loader.loadOxfordCPR("test_data/example")
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
    def test_read_crc_good_file(metadata_loaded):
        metadata_loaded.read_crc("../example_data/Map Data 2-DIC area")
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
    def test_read_crc_bad(metadata_loaded):
        with pytest.raises(FileNotFoundError):
            metadata_loaded.read_crc("badger")
