import pytest
from pytest import approx
from unittest.mock import Mock

import numpy as np
import defdap.ebsd as ebsd
import defdap.crystal as crystal
from defdap.utils import Datastore


DATA_DIR = "tests/data/"
EXPECTED_RESULTS_DIR = DATA_DIR + "expected_output/"
EXAMPLE_EBSD = DATA_DIR + "testDataEBSD"


@pytest.fixture(scope="module")
def good_map():
    ebsd_map = ebsd.Map(EXAMPLE_EBSD)

    return ebsd_map


@pytest.fixture(scope="module")
def good_map_with_quats(good_map):
    good_map.calc_quat_array()

    return good_map


@pytest.fixture(scope="module")
def good_quat_array(good_map_with_quats):
    return good_map_with_quats.data.orientation


@pytest.fixture(scope="module")
def good_phase_array(good_map_with_quats):
    return good_map_with_quats.data.phase


class TestMapFindBoundaries:
    # Depends on Quat.symEqv, self.crystal_sym, self.yDim, self.xDim,
    # self.quatArray, self.phaseArray
    # Affects self.boundaries

    @staticmethod
    @pytest.fixture
    def mock_map(good_quat_array, good_phase_array):
        # create stub object
        mock_map = Mock(spec=ebsd.Map)
        mock_datastore = Mock(spec=Datastore)
        mock_datastore.orientation = good_quat_array
        mock_datastore.phase = good_phase_array
        mock_map.data = mock_datastore
        mock_map.shape = good_quat_array.shape

        mock_phase = Mock(spec=crystal.Phase)
        mock_phase.crystalStructure = crystal.crystalStructures['cubic']
        mock_map.primaryPhase = mock_phase

        return mock_map

    @staticmethod
    def test_return_type(mock_map):
        # run test and collect result
        ebsd.Map.find_boundaries(mock_map, misori_tol=10)
        result = mock_map.data.grain_boundaries.image

        assert type(result) is np.ndarray
        assert result.dtype is np.dtype(np.int64)
        assert result.shape == mock_map.shape
        assert result.max() == 0
        assert result.min() == -1

    @staticmethod
    @pytest.mark.parametrize('bound_def', [5, 10])
    def test_calc(mock_map, bound_def):
        # run test and collect result
        ebsd.Map.find_boundaries(mock_map, misori_tol=bound_def)
        result = mock_map.data.grain_boundaries.image

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
calc_slip_traces

'''