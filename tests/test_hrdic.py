import pytest
from unittest.mock import Mock, MagicMock

import numpy as np

import defdap.ebsd as ebsd
import defdap.hrdic as hrdic
from defdap.utils import Datastore


DATA_DIR = "tests/data/"
EXPECTED_RESULTS_DIR = DATA_DIR + "expected_output/"
EXAMPLE_EBSD = DATA_DIR + "testDataEBSD"


@pytest.fixture(scope="module")
def good_grain_boundaries():
    expected = np.load(
        f'{EXPECTED_RESULTS_DIR}/hrdic_grain_boundaries_5deg.npz'
    )
    mock_map = Mock(spec=hrdic.Map)
    mock_map.shape = (200, 300)
    return hrdic.BoundarySet(
        mock_map,
        [tuple(row) for row in expected['points']],
        None
    )


@pytest.fixture(scope="module")
def good_warped_ebsd_grains():
    return np.load(
        f'{EXPECTED_RESULTS_DIR}/ebsd_grains_warped_5deg_0.npz'
    )['grains']


@pytest.fixture(scope="module")
def good_warped_dic_grains():
    return np.load(
        f'{EXPECTED_RESULTS_DIR}/hrdic_grains_warped.npz'
    )['grains']


@pytest.fixture(scope="module")
def good_ebsd_grains():
    return np.load(
            f'{EXPECTED_RESULTS_DIR}/ebsd_grains_5deg_0.npz'
        )['grains']


class TestMapFindGrains:
    # for warp depends on
    # check_ebsd_linked, warp_to_dic_frame, shape, ebsd_map
    # does not depend on min_grain_size

    # Affects self.boundaries

    @staticmethod
    @pytest.fixture
    def mock_map(good_warped_ebsd_grains, good_grain_boundaries,
                 good_warped_dic_grains, good_ebsd_grains):
        # create stub object
        mock_map = Mock(spec=hrdic.Map)
        mock_map.check_ebsd_linked = Mock(return_value=True)
        mock_map.warp_to_dic_frame = Mock(return_value=good_warped_ebsd_grains)
        mock_map.shape = good_warped_ebsd_grains.shape
        mock_map.data = MagicMock(spec=Datastore)
        mock_map.data.generate_id = Mock(return_value=1)
        mock_map.ebsd_map = MagicMock(spec=ebsd.Map)
        mock_map.ebsd_map.__getitem__ = lambda self, k: k
        mock_map.ebsd_map.data = Mock(spec=Datastore)

        mock_map.data.grain_boundaries = good_grain_boundaries

        mock_map.experiment = Mock()
        mock_map.experiment.warp_image = Mock(
            return_value=good_warped_dic_grains
        )
        mock_map.frame = Mock()
        mock_map.ebsd_map.frame = Mock()
        mock_map.ebsd_map.shape = good_warped_dic_grains.shape
        mock_map.ebsd_map.data.grains = good_ebsd_grains

        return mock_map

    @staticmethod
    @pytest.mark.parametrize('algorithm', ['warp', 'floodfill'])
    def test_return_type(mock_map, algorithm):
        # algorithm = 'warp'
        # run test and collect result
        result = hrdic.Map.find_grains(mock_map, algorithm=algorithm)

        assert isinstance(result, np.ndarray)
        assert result.shape == mock_map.shape
        assert result.dtype == np.int64

    @staticmethod
    @pytest.mark.parametrize('algorithm, min_grain_size', [
        ('warp', None),
        ('floodfill', 0),
        ('floodfill', 10),
        ('floodfill', 100),
    ])
    def test_calc_warp(mock_map, algorithm, min_grain_size):
        # algorithm = 'warp'
        # run test and collect result
        result = hrdic.Map.find_grains(
            mock_map, algorithm=algorithm, min_grain_size=min_grain_size
        )

        # load expected
        min_grain_size = '' if min_grain_size is None else f'_{min_grain_size}'
        expected = np.load(
            f'{EXPECTED_RESULTS_DIR}/hrdic_grains_{algorithm}{min_grain_size}.npz'
        )['grains']

        assert np.all(result == expected)

    @staticmethod
    def test_add_derivative(mock_map):
        algorithm = 'warp'
        mock_add_derivative = Mock()
        mock_map.data.add_derivative = mock_add_derivative
        # run test and collect result
        hrdic.Map.find_grains(mock_map, algorithm=algorithm)

        mock_add_derivative.assert_called_once()

    @staticmethod
    @pytest.mark.parametrize('algorithm', ['warp', 'floodfill'])
    def test_grain_list_type(mock_map, algorithm):
        algorithm = 'warp'
        hrdic.Map.find_grains(mock_map, algorithm=algorithm)
        result = mock_map._grains

        assert isinstance(result, list)
        for g in result:
            assert isinstance(g, hrdic.Grain)

    @staticmethod
    @pytest.mark.parametrize('algorithm, expected', [
        ('warp', 111), ('floodfill', 80)
    ])
    def test_grain_list_size(mock_map, algorithm, expected):
        hrdic.Map.find_grains(mock_map, algorithm=algorithm, min_grain_size=10)
        result = mock_map._grains

        assert len(result) == expected

    @staticmethod
    @pytest.mark.parametrize('algorithm, min_grain_size', [
        ('warp', None),
        ('floodfill', 0),
        ('floodfill', 10),
        ('floodfill', 100),
    ])
    def test_grain_points(mock_map, algorithm, min_grain_size):
        hrdic.Map.find_grains(
            mock_map, algorithm=algorithm, min_grain_size=min_grain_size
        )
        result = mock_map._grains

        # load expected
        min_grain_size = '' if min_grain_size is None else f'_{min_grain_size}'
        expected_grains = np.load(
            f'{EXPECTED_RESULTS_DIR}/hrdic_grains_{algorithm}{min_grain_size}.npz'
        )['grains']

        # transform both to set of tuples so order of points is ignored
        for i in range(expected_grains.max()):
            expected_point = set(zip(*np.nonzero(expected_grains == i+1)[::-1]))

            assert set([(*r, ) for r in result[i].data.point]) == expected_point

    @staticmethod
    def test_call_warp_to_dic_frame(mock_map, good_ebsd_grains):
        hrdic.Map.find_grains(mock_map, algorithm='warp')

        mock_map.warp_to_dic_frame.assert_called_once()
        mock_map.warp_to_dic_frame.assert_called_with(
            good_ebsd_grains, order=0, preserve_range=True
        )

    @staticmethod
    def test_call_experiment_warp_image(mock_map, good_ebsd_grains):
        hrdic.Map.find_grains(mock_map, algorithm='floodfill', min_grain_size=10)

        good_grains = np.load(
            f'{EXPECTED_RESULTS_DIR}/hrdic_grains_floodfill_10.npz'
        )['grains']

        mock_map.experiment.warp_image.assert_called_once()
        call_args = mock_map.experiment.warp_image.call_args
        np.testing.assert_array_equal(
            good_grains.astype(float), call_args[0][0]
        )
        assert call_args[0][1] == mock_map.frame
        assert call_args[0][2] == mock_map.ebsd_map.frame
        assert call_args[1]['output_shape'] == mock_map.ebsd_map.shape
        assert call_args[1]['order'] == 0

    @staticmethod
    @pytest.mark.parametrize('algorithm, expected', [
        ('warp', [
            1, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
            23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
            57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 68, 69, 71, 72, 75, 77, 78,
            79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 92, 93, 95, 96, 97,
            98, 99, 100, 101, 102, 103, 105, 106, 107, 108, 109, 111, 112, 113,
            114, 115, 116, 117, 119, 120, 121, 122, 123, 125, 126
        ]),
        ('floodfill', [
            1, 13, 5, 6, 7, 8, 9, 11, 15, 14, 16, 17, 20, 21, 22, 18, 23, 25,
            19, 24, 28, 29, 32, 31, 30, 34, 35, 36, 33, 38, 39, 41, 40, 50, 44,
            39, 52, 47, 51, 48, 37, 57, 58, 65, 61, 62, 64, 72, 77, 79, 81, 80,
            75, 86, 90, 85, 87, 57, 91, 93, 92, 99, 99, 95, 97, 96, 100, 106,
            102, 97, 107, 108, 111, 112, 115, 117, 114, 120, 122, 123
        ])
    ])
    def test_grain_assigned_ebsd_grains(mock_map, algorithm, expected):
        hrdic.Map.find_grains(mock_map, algorithm=algorithm, min_grain_size=10)
        result = [g.ebsd_grain for g in mock_map._grains]

        assert result == expected

# methods to test
# '_grad',
# '_map',
# 'binning',
# 'boundaries',
# 'bseScale',
# 'buildNeighbourNetwork',
# 'calc_grain_average',
# 'calc_proxigram',
# 'checkEbsdLinked',
# 'check_grains_detected',
# 'click_grain_id',
# 'click_grain_neighbours',
# 'clickHomog',
# 'clickSaveHomog',
# 'crop',
# 'cropDists',
# 'crystal_sym',
# 'currGrainId',
# 'display_neighbours',
# 'ebsdGrainIds',
# 'ebsdMap',
# 'ebsdTransform',
# 'ebsdTransformInv',
# 'f11',
# 'f12',
# 'f21',
# 'f22',
# 'findGrains',
# 'floodFill',
# 'fname',
# 'format',
# 'grainList',
# 'grainPlot',
# 'grains',
# 'highlight_alpha',
# 'homogPoints',
# 'linkEbsdMap',
# 'load_data',
# 'locateGrainID',
# 'mapshape',
# 'max_shear',
# 'patScale',
# 'path',
# 'patternImPath',
# 'plot_default',
# 'plot_grain_av_max_shear',
# 'plot_grain_data_ipf',
# 'plot_grain_data_map',
# 'plot_grain_numbers',
# 'plotHomog',
# 'plot_max_shear',
# 'plotPattern',
# 'print_stats_table',
# 'proxigram',
# 'proxigramArr',
# 'retrieve_name',
# 'scale',
# 'setCrop',
# 'setHomogPoint',
# 'setPatternPath',
# 'setScale',
# 'updateHomogPoint',
# 'version',
# 'warpToDicFrame',
# 'x_dim',
# 'x_map',
# 'xc',
# 'xd',
# 'xdim',
# 'y_dim',
# 'y_map',
# 'yc',
# 'yd',
# 'ydim'