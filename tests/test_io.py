import pytest

import defdap.io


class TestEBSDDataLoader:

    @staticmethod
    @pytest.fixture
    def data_loader():
        return defdap.io.EBSDDataLoader()

    @staticmethod
    @pytest.fixture
    def metadata_loaded(data_loader):
        return data_loader.loadOxfordCPR("test_data/example.cpr")

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
    def test_load_oxford_cpr(data_loader):
        data_loader.loadOxfordCPR("test_data/example")
        metadata = data_loader.loadedMetadata
        assert metadata["xDim"] == 1006
        assert metadata["yDim"] == 996
        # Testing for floating point equality so use approx
        assert metadata["stepSize"] == pytest.approx(0.1)
        assert metadata["numPhases"] == 1
        assert metadata["phaseNames"] == ["Ni-superalloy"]

