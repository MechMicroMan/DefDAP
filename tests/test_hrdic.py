import pytest
from pytest import approx
from unittest.mock import Mock, MagicMock
from functools import partial

import numpy as np
import defdap.ebsd as ebsd
import defdap.hrdic as hrdic
from defdap.utils import Datastore


DATA_DIR = "tests/data/"
EXPECTED_RESULTS_DIR = DATA_DIR + "expected_output/"
EXAMPLE_EBSD = DATA_DIR + "testDataEBSD"


@pytest.fixture(scope="module")
def good_warped_grains():
    return np.load(
        f'{EXPECTED_RESULTS_DIR}/ebsd_grains_warped_5deg_0.npz'
    )['grains']


class TestMapFindGrains:
    # for warp depends on
    # check_ebsd_linked, warp_to_dic_frame, shape, ebsd_map
    # does not depend on min_grain_size

    # Affects self.boundaries

    @staticmethod
    @pytest.fixture
    def mock_map(good_warped_grains):
        # create stub object
        mock_map = Mock(spec=hrdic.Map)
        mock_map.check_ebsd_linked = Mock(return_value=True)
        mock_map.warp_to_dic_frame = Mock(return_value=good_warped_grains)
        mock_map.shape = good_warped_grains.shape
        mock_map.data = MagicMock(spec=Datastore)
        mock_map.data.generate_id = Mock(return_value=1)

        mock_map.ebsd_map = MagicMock(spec=ebsd.Map)
        mock_map.ebsd_map.data = Mock(spec=Datastore)
        mock_map.ebsd_map.data.grains = 'ebsd_grains'

        # mock_map.flood_fill = partial(hrdic.Map.flood_fill, mock_map)

        return mock_map

    @staticmethod
    def test_return_type(mock_map):
        algorithm = 'warp'
        # run test and collect result
        result = hrdic.Map.find_grains(mock_map, algorithm=algorithm)

        assert isinstance(result, np.ndarray)
        assert result.shape == mock_map.shape
        assert result.dtype == np.int64

    @staticmethod
    def test_calc_warp(mock_map):
        algorithm = 'warp'
        # run test and collect result
        result = hrdic.Map.find_grains(mock_map, algorithm=algorithm)

        # load expected
        expected = np.load(
            f'{EXPECTED_RESULTS_DIR}/hrdic_grains_{algorithm}.npz'
        )['grains']

        assert np.alltrue(result == expected)

    @staticmethod
    def test_add_derivative(mock_map):
        algorithm = 'warp'
        mock_add_derivative = Mock()
        mock_map.data.add_derivative = mock_add_derivative
        # run test and collect result
        hrdic.Map.find_grains(mock_map, algorithm=algorithm)

        mock_add_derivative.assert_called_once()

    @staticmethod
    def test_grain_list_type(mock_map):
        algorithm = 'warp'
        hrdic.Map.find_grains(mock_map, algorithm=algorithm)
        result = mock_map._grains

        assert isinstance(result, list)
        for g in result:
            assert isinstance(g, hrdic.Grain)

    @staticmethod
    def test_grain_list_size(mock_map):
        algorithm = 'warp'
        hrdic.Map.find_grains(mock_map, algorithm=algorithm)
        result = mock_map._grains

        assert len(result) == 111

    @staticmethod
    @pytest.mark.parametrize('min_grain_size', [0, 10, 100])
    def test_grain_points(mock_map, min_grain_size):
        algorithm = 'warp'
        hrdic.Map.find_grains(mock_map, algorithm=algorithm)
        result = mock_map._grains

        # load expected
        expected_grains = np.load(
            f'{EXPECTED_RESULTS_DIR}/hrdic_grains_{algorithm}.npz'
        )['grains']

        for i in range(expected_grains.max()):
            expected_point = zip(*np.nonzero(expected_grains == i+1)[::-1])

            assert set(result[i].data.point) == set(expected_point)


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