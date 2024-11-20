import pytest
from unittest.mock import Mock, MagicMock

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
def good_grain_boundaries(good_map):
    expected = np.load(
        f'{EXPECTED_RESULTS_DIR}/ebsd_grain_boundaries_5deg.npz'
    )
    return ebsd.BoundarySet(
        good_map,
        [tuple(row) for row in expected['points_x']],
        [tuple(row) for row in expected['points_y']]
    )


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
        expected = np.load(
            f'{EXPECTED_RESULTS_DIR}/ebsd_grain_boundaries_{bound_def}deg.npz'
        )
        expected_x = set([tuple(row) for row in expected['points_x']])
        expected_y = set([tuple(row) for row in expected['points_y']])

        assert result.points_x == expected_x
        assert result.points_y == expected_y


class TestMapFindGrains:
    # Depends on self.data.grain_boundaries.image_*, self.data.phase,
    # self.flood_fill, self.num_phases, self.phases
    # Affects self.boundaries

    @staticmethod
    @pytest.fixture
    def mock_map(good_grain_boundaries, good_phase_array):
        # create stub object
        mock_map = Mock(spec=ebsd.Map)
        mock_datastore = MagicMock(spec=Datastore)
        mock_datastore.phase = good_phase_array
        mock_datastore.grain_boundaries = good_grain_boundaries
        mock_datastore.generate_id = Mock(return_value=1)
        mock_map.data = mock_datastore
        mock_map.shape = good_phase_array.shape
        mock_map.num_phases = 1
        mock_map.phases = [Mock(crystal.Phase)]

        return mock_map

    @staticmethod
    def test_return_type(mock_map):
        # run test and collect result
        result = ebsd.Map.find_grains(mock_map, min_grain_size=10)

        assert isinstance(result, np.ndarray)
        assert result.shape == mock_map.shape
        assert result.dtype == np.int64

    @staticmethod
    @pytest.mark.parametrize('min_grain_size', [0, 10, 100])
    def test_calc(mock_map, min_grain_size):
        # run test and collect result
        result = ebsd.Map.find_grains(mock_map, min_grain_size=min_grain_size)

        # load expected
        expected = np.load(
            f'{EXPECTED_RESULTS_DIR}/ebsd_grains_5deg_{min_grain_size}.npz'
        )['grains']

        assert np.all(result == expected)

    @staticmethod
    def test_add_derivative(mock_map):
        mock_add_derivative = Mock()
        mock_map.data.add_derivative = mock_add_derivative
        # run test and collect result
        ebsd.Map.find_grains(mock_map, min_grain_size=10)

        mock_add_derivative.assert_called_once()

    @staticmethod
    def test_grain_list_type(mock_map):
        ebsd.Map.find_grains(mock_map, min_grain_size=10)
        result = mock_map._grains

        assert isinstance(result, list)
        for g in result:
            assert isinstance(g, ebsd.Grain)

    @staticmethod
    @pytest.mark.parametrize('min_grain_size, expected_len', [
        (0, 141), (10, 109), (100, 76)
    ])
    def test_grain_list_size(mock_map, min_grain_size, expected_len):
        ebsd.Map.find_grains(mock_map, min_grain_size=min_grain_size)
        result = mock_map._grains

        assert len(result) == expected_len

    @staticmethod
    @pytest.mark.parametrize('min_grain_size', [0, 10, 100])
    def test_grain_points(mock_map, min_grain_size):
        ebsd.Map.find_grains(mock_map, min_grain_size=min_grain_size)
        result = mock_map._grains

        # load expected
        expected_grains = np.load(
            f'{EXPECTED_RESULTS_DIR}/ebsd_grains_5deg_{min_grain_size}.npz'
        )['grains']

        # transform both to set of tuples so order of points is ignored
        for i in range(expected_grains.max()):
            expected_point = set(zip(*np.nonzero(expected_grains == i+1)[::-1]))

            assert set([(*r, ) for r in result[i].data.point]) == expected_point


class TestMapCalcProxigram:
    @staticmethod
    @pytest.fixture
    def mock_map(good_grain_boundaries, good_phase_array):
        # create stub object
        mock_map = Mock(spec=ebsd.Map)
        mock_datastore = MagicMock(spec=Datastore)
        mock_datastore.grain_boundaries = good_grain_boundaries
        mock_map.data = mock_datastore
        mock_map.shape = good_grain_boundaries.ebsd_map.shape

        return mock_map

    @staticmethod
    def test_return_type(mock_map):
        # run test and collect result
        result = ebsd.Map.calc_proxigram(mock_map)

        assert isinstance(result, np.ndarray)
        assert result.shape == mock_map.shape
        assert result.dtype == float

    @staticmethod
    def test_calc(mock_map):
        # run test and collect result
        result = ebsd.Map.calc_proxigram(mock_map)

        # load expected
        expected = np.load(
            f'{EXPECTED_RESULTS_DIR}/ebsd_proxigram.npz'
        )['proxigram']

        np.testing.assert_array_equal(result, expected)


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