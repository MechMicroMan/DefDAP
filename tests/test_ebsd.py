import pytest
from pytest import approx
from unittest.mock import Mock

import numpy as np
import defdap.ebsd as ebsd
import defdap.crystal as crystal
from defdap.quat import Quat
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


@pytest.fixture(scope="module")
def good_symmetries():
    over_root2 = np.sqrt(2) / 2
    return [
        Quat(1.0, 0.0, 0.0, 0.0),
        Quat(over_root2, over_root2, 0.0, 0.0),
        Quat(0.0, 1.0, 0.0, 0.0),
        Quat(over_root2, -over_root2, 0.0, 0.0),
        Quat(over_root2, 0.0, over_root2, 0.0),
        Quat(0.0, 0.0, 1.0, 0.0),
        Quat(over_root2, 0.0, -over_root2, 0.0),
        Quat(over_root2, 0.0, 0.0, over_root2),
        Quat(0.0, 0.0, 0.0, 1.0),
        Quat(over_root2, 0.0, 0.0, -over_root2),
        Quat(0.0, over_root2, over_root2, 0.0),
        Quat(0.0, -over_root2, over_root2, 0.0),
        Quat(0.0, over_root2, 0.0, over_root2),
        Quat(0.0, -over_root2, 0.0, over_root2),
        Quat(0.0, 0.0, over_root2, over_root2),
        Quat(0.0, 0.0, -over_root2, over_root2),
        Quat(0.5, 0.5, 0.5, 0.5),
        Quat(0.5, -0.5, -0.5, -0.5),
        Quat(0.5, -0.5, 0.5, 0.5),
        Quat(0.5, 0.5, -0.5, -0.5),
        Quat(0.5, 0.5, -0.5, 0.5),
        Quat(0.5, -0.5, 0.5, -0.5),
        Quat(0.5, 0.5, 0.5, -0.5),
        Quat(0.5, -0.5, -0.5, 0.5)
    ]


class TestMapFindBoundaries:
    # Depends on Quat.sym_eqv, self.crystal_sym, self.y_dim, self.x_dim,
    # self.quatArray, self.phaseArray
    # Affects self.boundaries

    @staticmethod
    @pytest.fixture
    def mock_map(good_quat_array, good_phase_array, good_symmetries):
        # create stub object
        mock_map = Mock(spec=ebsd.Map)
        mock_datastore = Mock(spec=Datastore)
        mock_datastore.orientation = good_quat_array
        mock_datastore.phase = good_phase_array
        mock_map.data = mock_datastore
        mock_map.shape = good_quat_array.shape

        mock_crystal_structure = Mock(spec=crystal.CrystalStructure)
        mock_crystal_structure.symmetries = good_symmetries
        mock_phase = Mock(spec=crystal.Phase)
        mock_phase.crystal_structure = mock_crystal_structure

        # mock_phase = Mock(spec=crystal.Phase)
        # mock_phase.crystal_structure = crystal.crystalStructures['cubic']

        mock_map.primary_phase = mock_phase

        return mock_map

    @staticmethod
    def test_return_type(mock_map):
        # run test and collect result
        result = ebsd.Map.find_boundaries(mock_map, misori_tol=10)

        assert isinstance(result, tuple)
        assert len(result) == 2
        for boundaries in result:
            assert isinstance(boundaries, ebsd.BoundarySet)

    @staticmethod
    @pytest.mark.parametrize('bound_def', [5, 10])
    def test_calc(mock_map, bound_def):
        # run test and collect result
        _, result = ebsd.Map.find_boundaries(
            mock_map, misori_tol=bound_def
        )

        # load expected
        expected = np.loadtxt(
            "{:}boundaries_{:}deg.txt".format(EXPECTED_RESULTS_DIR, bound_def),
            dtype=int
        )

        assert np.allclose(result.image, expected)




''' Functions left to test
Map:
__init__
plot_default
load_data
scale
transformData
plotBandContrastMap
plot_euler_map
plot_ipf_map
plot_phase_map
calcKam
plotKamMap
calcNye
plotGNDMap
checkDataLoaded
buildQuatArray
findPhaseBoundaries
plot_phase_boundary_map
plot_boundary_map
findGrains
plot_grain_map
floodFill
calc_grain_av_oris
calc_grain_mis_ori
plot_mis_ori_map
loadSlipSystems
print_slip_systems
calc_average_grain_schmid_factors
plot_average_grain_schmid_factors_map

Grain:
__init__
add_point
calc_average_ori
build_mis_ori_list
plot_ref_ori
plot_ori_spread
plotUnitCell
plot_mis_ori
calc_average_schmid_factors
slip_traces
print_slip_traces
calc_slip_traces

'''