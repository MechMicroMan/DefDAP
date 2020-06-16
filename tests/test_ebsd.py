import pytest
from pytest import approx
from unittest.mock import Mock

import numpy as np
import defdap.ebsd as ebsd


DATA_DIR = "data/"
EXPECTED_RESULTS_DIR = DATA_DIR + "expected_output/"
EXAMPLE_EBSD = DATA_DIR + "testDataEBSD"


@pytest.fixture(scope="module")
def good_map():
    ebsd_map = ebsd.Map(EXAMPLE_EBSD, "cubic")

    return ebsd_map


@pytest.fixture(scope="module")
def good_map_with_quats(good_map):
    good_map.buildQuatArray()

    return good_map


@pytest.fixture(scope="module")
def good_quat_array(good_map_with_quats):
    return good_map_with_quats.quatArray


class TestMapFindBoundaries:
    # Depends on Quat.symEqv, self.crystalSym, self.yDim, self.xDim, self.quatArray
    # Affects self.boundaries

    @staticmethod
    @pytest.fixture
    def mock_map(good_quat_array):
        # create stub object
        mock_map = Mock(spec=ebsd.Map)
        mock_map.quatArray = good_quat_array
        mock_map.yDim, mock_map.xDim = good_quat_array.shape
        mock_map.crystalSym = "cubic"

        return mock_map

    @staticmethod
    def test_return_type(mock_map):
        # run test and collect result
        ebsd.Map.findBoundaries(mock_map, boundDef=10)
        result = mock_map.boundaries

        assert type(result) is np.ndarray
        assert result.dtype is np.dtype(np.int64)
        assert result.shape == mock_map.quatArray.shape
        assert result.max() == 0
        assert result.min() == -1

    @staticmethod
    @pytest.mark.parametrize('bound_def', [5, 10])
    def test_calc(mock_map, bound_def):
        # run test and collect result
        ebsd.Map.findBoundaries(mock_map, boundDef=bound_def)
        result = mock_map.boundaries

        # load expected
        expected = -np.loadtxt(
            "{:}boundaries_{:}deg.txt".format(EXPECTED_RESULTS_DIR, bound_def),
            dtype=int
        )

        assert np.allclose(result, expected)




''' Functions left to test
Map:
__init__
plotDefault
loadData
scale
transformData
plotBandContrastMap
plotEulerMap
plotIPFMap
plotPhaseMap
calcKam
plotKamMap
calcNye
plotGNDMap
checkDataLoaded
buildQuatArray
findPhaseBoundaries
plotPhaseBoundaryMap
plotBoundaryMap
findGrains
plotGrainMap
floodFill
calcGrainAvOris
calcGrainMisOri
plotMisOriMap
loadSlipSystems
printSlipSystems
calcAverageGrainSchmidFactors
plotAverageGrainSchmidFactorsMap

Grain:
__init__
addPoint
calcAverageOri
buildMisOriList
plotRefOri
plotOriSpread
plotUnitCell
plotMisOri
calcAverageSchmidFactors
slipTraces
printSlipTraces
calcSlipTraces

'''