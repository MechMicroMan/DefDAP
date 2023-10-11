# Copyright 2021 Mechanics of Microstructures Group
#    at The University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import networkx as nx

import defdap
from defdap.quat import Quat
from defdap import plotting
from defdap.plotting import Plot, MapPlot, GrainPlot

from skimage.measure import profile_line

from defdap.utils import report_progress, Datastore
from defdap.experiment import Frame


class Map(object):
    """
    Base class for a map. Contains common functionality for all maps.

    Attributes
    ----------

    _grains : list of defdap.base.Grain
        List of grains.
    sel_grain : defdap.base.grain
        The last selected grain

    """
    def __init__(self, experiment=None, increment=None, frame=None):

        self.data = Datastore(crop_func=self.crop)
        self.frame = frame if frame is not None else Frame()
        if experiment is None:
            self.experiment = defdap.anonymous_experiment
            self.increment = self.experiment.add_increment()
        else:
            self.experiment = experiment
            self.increment = experiment.add_increment() if increment is None else increment

        self.shape = (0, 0)

        self._grains = None

        self.sel_grain = None

        self.proxigramArr = None
        self.neighbour_network = None

        self.grainPlot = None
        self.profilePlot = None

    def __len__(self):
        return len(self.grains)

    # allow array like getting of grains
    def __getitem__(self, key):
        # Check that grains have been detected in the map
        # self.check_grains_detected()

        return self.grains[key]

    @property
    def grains(self):
        # try to access grains image to generate grains if necessary
        self.data.grains
        return self._grains

    @property
    def xDim(self):
        return self.shape[1]

    @property
    def yDim(self):
        return self.shape[0]

    def crop(self, map_data, **kwargs):
        return map_data

    def set_homog_point(self, **kwargs):
        self.frame.set_homog_point(self, **kwargs)

    def check_grains_detected(self, raise_exc=True):
        """Check if grains have been detected.

        Parameters
        ----------
        raise_exc : bool
            If True then an exception is raised if grains have not been
            detected.

        Returns
        -------
        bool:
            True if grains detected, False otherwise.

        Raises
        -------
        Exception
            If grains not detected.

        """

        if (self._grains is None or
                type(self._grains) is not list or
                len(self._grains) < 1):
            if raise_exc:
                raise Exception("No grains detected.")
            else:
                return False
        return True

    def plot_grain_numbers(self, dilate_boundaries=False, ax=None, **kwargs):
        """Plot a map with grains numbered.

        Parameters
        ----------
        dilate_boundaries : bool, optional
            Set to true to dilate boundaries.
        ax : matplotlib.axes.Axes, optional
            axis to plot on, if not provided the current active axis is used.
        kwargs : dict, optional
            Keyword arguments passed to :func:`defdap.plotting.MapPlot.addGrainNumbers`

        Returns
        -------
        defdap.plotting.MapPlot

        """

        plot = plotting.MapPlot(self, ax=ax)
        plot.add_grain_boundaries(colour='black', dilate=dilate_boundaries)
        plot.addGrainNumbers(**kwargs)

        return plot

    def locate_grain(self, click_event=None, display_grain=False, **kwargs):
        """Interactive plot for identifying grains.

        Parameters
        ----------
        click_event : optional
            Click handler to use.
        display_grain : bool, optional
            If true, plot slip traces for grain selected by click.
        kwargs : dict, optional
            Keyword arguments passed to :func:`defdap.base.Map.plotDefault`

        """
        # Check that grains have been detected in the map
        self.check_grains_detected()

        # reset current selected grain and plot euler map with click handler
        plot = self.plotDefault(make_interactive=True, **kwargs)
        if click_event is None:
            # default click handler which highlights grain and prints id
            plot.add_event_handler(
                'button_press_event',
                lambda e, p: self.click_grain_id(e, p, display_grain)
            )
        else:
            # click handler loaded in as parameter. Pass current map
            # object to it.
            plot.add_event_handler('button_press_event', click_event)

        return plot

    def click_grain_id(self, event, plot, displaySelected):
        """Event handler to capture clicking on a map.

        Parameters
        ----------
        event :
            Click event.
        plot : defdap.plotting.MapPlot
            Plot to capture clicks from.
        displaySelected : bool
            If true, plot the selected grain alone in pop-out window.

        """
        # check if click was on the map
        if event.inaxes is not plot.ax:
            return

        # grain id of selected grain
        grain_id = self.data.grains[int(event.ydata), int(event.xdata)] - 1
        if grain_id < 0:
            return
        grain = self[grain_id]
        self.sel_grain = grain
        print("Grain ID: {}".format(grain_id))

        # update the grain highlights layer in the plot
        plot.add_grain_highlights([grain_id], alpha=self.highlightAlpha)

        if displaySelected:
            if self.grainPlot is None or not self.grainPlot.exists:
                self.grainPlot = grain.plotDefault(make_interactive=True)
            else:
                self.grainPlot.clear()
                self.grainPlot.callingGrain = grain
                grain.plotDefault(plot=self.grainPlot)
                self.grainPlot.draw()

    def draw_line_profile(self, **kwargs):
        """Interactive plot for drawing a line profile of data.

        Parameters
        ----------
        kwargs : dict, optional
            Keyword arguments passed to :func:`defdap.base.Map.plotDefault`

        """
        plot = self.plotDefault(make_interactive=True, **kwargs)

        plot.add_event_handler('button_press_event', plot.line_slice)
        plot.add_event_handler('button_release_event', lambda e, p: plot.line_slice(e, p,
                                                                                    action=self.calc_line_profile))

        return plot

    def calc_line_profile(self, plot, start_end, **kwargs):
        """Calculate and plot the line profile.

        Parameters
        ----------
        plot : defdap.plotting.MapPlot
            Plot to calculate the line profile for.
        start_end : array_like
            Selected points (x0, y0, x1, y1).
        kwargs : dict, optional
            Keyword arguments passed to :func:`matplotlib.pyplot.plot`

        """
        x0, y0 = start_end[0:2]
        x1, y1 = start_end[2:4]
        profile_length = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)

        # Extract the values along the line
        zi = profile_line(
            plot.img_layers[0].get_array(),
            (start_end[1], start_end[0]),
            (start_end[3], start_end[2]),
            mode='nearest'
        )
        xi = np.linspace(0, profile_length, len(zi))

        if self.profilePlot is None or not self.profilePlot.exists:
            self.profilePlot = Plot(make_interactive=True)
        else:
            self.profilePlot.clear()

        self.profilePlot.ax.plot(xi, zi, **kwargs)
        self.profilePlot.ax.set_xlabel('Distance (pixels)')
        self.profilePlot.ax.set_ylabel('Intensity')
        self.profilePlot.draw()

    @report_progress("constructing neighbour network")
    def build_neighbour_network(self):
        """Construct a list of neighbours

        """
        ## TODO: fix HRDIC NN
        # create network
        nn = nx.Graph()
        nn.add_nodes_from(self.grains)

        yLocs, xLocs = np.nonzero(self.boundaries)
        totalPoints = len(xLocs)

        for iPoint, (x, y) in enumerate(zip(xLocs, yLocs)):
            # report progress
            yield iPoint / totalPoints

            if (x == 0 or y == 0 or x == self.data.grains.shape[1] - 1 or
                    y == self.data.grains.shape[0] - 1):
                # exclude boundary pixels of map
                continue

            # use 4 nearest neighbour points as potential neighbour grains
            # (this maybe needs changing considering the position of
            # boundary pixels relative to the actual edges)
            # use sets as they do not allow duplicate elements
            # minus 1 on all as the grain image starts labeling at 1
            neighbours = {
                self.data.grains[y + 1, x] - 1,
                self.data.grains[y - 1, x] - 1,
                self.data.grains[y, x + 1] - 1,
                self.data.grains[y, x - 1] - 1
            }
            # neighbours = set(neighbours)
            # remove boundary points (-2) and points in small
            # grains (-3) (Normally -1 and -2)
            neighbours.discard(-2)
            neighbours.discard(-3)

            neighbours = tuple(neighbours)
            nunNeig = len(neighbours)
            if nunNeig <= 1:
                continue
            for i in range(nunNeig):
                for j in range(i + 1, nunNeig):
                    # Add  to network
                    grain = self[neighbours[i]]
                    neiGrain = self[neighbours[j]]
                    try:
                        # look up boundary
                        nn[grain][neiGrain]
                    except KeyError:
                        # neighbour relation doesn't exist so add it
                        nn.add_edge(grain, neiGrain)

        self.neighbourNetwork = nn

    def display_neighbours(self, **kwargs):
        return self.locate_grain(
            click_event=self.click_grain_neighbours, **kwargs
        )

    def click_grain_neighbours(self, event, plot):
        """Event handler to capture clicking and show neighbours of selected grain.

        Parameters
        ----------
        event :
            Click event.
        plot : defdap.plotting.MapPlot
            Plot to monitor.

        """
        # check if click was on the map
        if event.inaxes is not plot.ax:
            return

        # grain id of selected grain
        grain_id = self.data.grains[int(event.ydata), int(event.xdata)] - 1
        if grain_id < 0:
            return
        grain = self[grain_id]
        self.sel_grain = grain

        # find first and second nearest neighbours
        firstNeighbours = list(self.neighbourNetwork.neighbors(grain))
        highlightGrains = [grain] + firstNeighbours

        secondNeighbours = []
        for firstNeighbour in firstNeighbours:
            trialSecondNeighbours = list(
                self.neighbourNetwork.neighbors(firstNeighbour)
            )
            for secondNeighbour in trialSecondNeighbours:
                if (secondNeighbour not in highlightGrains and
                        secondNeighbour not in secondNeighbours):
                    secondNeighbours.append(secondNeighbour)
        highlightGrains.extend(secondNeighbours)

        highlightGrains = [grain.grain_id for grain in highlightGrains]
        highlightColours = ['white']
        highlightColours.extend(['yellow'] * len(firstNeighbours))
        highlightColours.append('green')

        # update the grain highlights layer in the plot
        plot.add_grain_highlights(highlightGrains, grain_colours=highlightColours)

    @property
    def proxigram(self):
        """Proxigram for a map.

        Returns
        -------
        numpy.ndarray
            Distance from a grain boundary at each point in map.

        """
        self.calc_proxigram(forceCalc=False)

        return self.proxigramArr

    @report_progress("calculating proxigram")
    def calc_proxigram(self, numTrials=500, forceCalc=True):
        """Calculate distance from a grain boundary at each point in map.

        Parameters
        ----------
        numTrials : int, optional
            number of trials.
        forceCalc : bool, optional
            Force calculation even is proxigramArr is populated.

        """
        # TODO: fix proxigram
        if self.proxigramArr is not None and not forceCalc:
            return

        proxBoundaries = np.copy(self.boundaries)
        proxShape = proxBoundaries.shape

        # ebsd boundary arrays have extra boundary along right and
        # bottom edge. These need to be removed right edge
        if np.all(proxBoundaries[:, -1] == -1):
            proxBoundaries[:, -1] = proxBoundaries[:, -2]
        # bottom edge
        if np.all(proxBoundaries[-1, :] == -1):
            proxBoundaries[-1, :] = proxBoundaries[-2, :]

        # create list of positions of each boundary point
        indexBoundaries = []
        for index, value in np.ndenumerate(proxBoundaries):
            if value == -1:
                indexBoundaries.append(index)
        # add 0.5 to boundary coordinates as they are placed on the
        # bottom right edge pixels of grains
        indexBoundaries = np.array(indexBoundaries) + 0.5

        # array of x and y coordinate of each pixel in the map
        coords = np.zeros((2, proxShape[0], proxShape[1]), dtype=float)
        coords[0], coords[1] = np.meshgrid(
            range(proxShape[0]), range(proxShape[1]), indexing='ij'
        )

        # array to store trial distance from each boundary point
        trialDistances = np.full((numTrials + 1, proxShape[0], proxShape[1]),
                                 1000, dtype=float)

        # loop over each boundary point (p) and calculate distance from
        # p to all points in the map store minimum once numTrails have
        # been made and start a new batch of trials
        numBoundaryPoints = len(indexBoundaries)
        j = 1
        for i, indexBoundary in enumerate(indexBoundaries):
            trialDistances[j] = np.sqrt((coords[0] - indexBoundary[0])**2
                                        + (coords[1] - indexBoundary[1])**2)

            if j == numTrials:
                # find current minimum distances and store
                trialDistances[0] = trialDistances.min(axis=0)
                j = 0
                # report progress
                yield i / numBoundaryPoints
            j += 1

        # find final minimum distances to a boundary
        self.proxigramArr = trialDistances.min(axis=0)

        trialDistances = None

    def _validate_map(self, map_name):
        """Check the name exists and is a map data.

        Parameters
        ----------
        map_name : str

        """
        if map_name not in self.data:
            raise ValueError(f'`{map_name}` does not exist.')
        if (self.data.get_metadata(map_name, 'type') != 'map' or
                self.data.get_metadata(map_name, 'order') is None):
            raise ValueError(f'`{map_name}` is not a valid map.')

    def _validate_component(self, map_name, comp):
        """

        Parameters
        ----------
        map_name : str
        comp : int or tuple of int or str
            Component of the map data. This is either the
            tensor component (tuple of ints) or the name of a calculation
            to be applied e.g. 'norm', 'all_euler' or 'IPF_x'.

        Returns
        -------
        tuple of int or str

        """
        order = self.data[map_name, 'order']
        if comp is None:
            comp = self.data.get_metadata(map_name, 'default_component')
            if comp is not None:
                print(f'Using default component: `{comp}`')

        if comp is None:
            if order != 0:
                raise ValueError('`comp` must be specified.')
            else:
                return comp

        if isinstance(comp, int):
            comp = (comp,)
        if isinstance(comp, tuple) and len(comp) != order:
            raise ValueError(f'Component length does not match data, expected '
                             f'{self.data[map_name, "order"]} values but got '
                             f'{len(comp)}.')

        return comp

    def _extract_component(self, map_data, comp):
        """Extract a component from the data.

        Parameters
        ----------
        map_data : numpy.ndarray
            Map data to extract from.
        comp : tuple of int or str
            Component of the map data to extract. This is either the
            tensor component (tuple of ints) or the name of a calculation
            to be applied e.g. 'norm', 'all_euler' or 'IPF_x'.

        Returns
        -------
        numpy.ndarray

        """
        if comp is None:
            return map_data
        if isinstance(comp, tuple):
            return map_data[comp]
        if isinstance(comp, str):
            comp = comp.lower()
            if comp == 'norm':
                if len(map_data.shape) == 3:
                    axis = 0
                elif len(map_data.shape) == 4:
                    axis = (0, 1)
                else:
                    raise ValueError('Unsupported data for norm.')

                return np.linalg.norm(map_data, axis=axis)

            if comp == 'all_euler':
                return self.calc_euler_colour(map_data)

            if comp.startswith('ipf'):
                direction = comp.split('_')[1]
                direction = {
                    'x': np.array([1, 0, 0]),
                    'y': np.array([0, 1, 0]),
                    'z': np.array([0, 0, 1]),
                }[direction]
                return self.calc_ipf_colour(map_data, direction)

        raise ValueError(f'Invalid component `{comp}`')

    def plot_map(self, map_name, component=None, **kwargs):
        """Plot a map from the DIC data.

        Parameters
        ----------
        map_name : str
            Map data name to plot i.e. e, max_shear, euler_angle, orientation.
        component : int or tuple of int or str
            Component of the map data to plot. This is either the tensor
            component (int or tuple of ints) or the name of a calculation
            to be applied e.g. 'norm', 'all_euler' or 'IPF_x'.
        kwargs
            All arguments are passed to :func:`defdap.plotting.MapPlot.create`.

        Returns
        -------
        defdap.plotting.MapPlot
            Plot containing map.

        """
        self._validate_map(map_name)
        comp = self._validate_component(map_name, component)

        # Set default plot parameters then update with any input
        plot_params = {}   # should load default plotting params
        plot_params.update(self.data.get_metadata(map_name, 'plot_params', {}))

        # Add extra info to label
        clabel = plot_params.get('clabel')
        if clabel is not None:
            # tensor component
            if isinstance(comp, tuple):
                comp_fmt = ' (' + '{}' * len(comp) + ')'
                clabel += comp_fmt.format(*(i+1 for i in comp))
            elif isinstance(comp, str):
                clabel += f' ({comp.replace("_", " ")})'
            # unit
            unit = self.data.get_metadata(map_name, 'unit')
            if unit is not None and unit != '':
                clabel += f' ({unit})'

            plot_params['clabel'] = clabel

        if self.scale is not None:
            binning = self.data.get_metadata(map_name, 'binning', 1)
            plot_params['scale'] = self.scale / binning

        plot_params.update(kwargs)

        map_data = self._extract_component(self.data[map_name], comp)

        return MapPlot.create(self, map_data, **plot_params)

    def calc_grain_average(self, map_data, grainIds=-1):
        """Calculate grain average of any DIC map data.

        Parameters
        ----------
        map_data : numpy.ndarray
            Array of map data to grain average. This must be cropped!
        grainIds : list, optional
            grainIDs to perform operation on, set to -1 for all grains.

        Returns
        -------
        numpy.ndarray
            Array containing the grain average values.

        """

        # Check that grains have been detected in the map
        self.check_grains_detected()

        if type(grainIds) is int and grainIds == -1:
            grainIds = range(len(self))

        grain_average_data = np.zeros(len(grainIds))

        for i, grainId in enumerate(grainIds):
            grain = self[grainId]
            grainData = grain.grainData(map_data)
            grain_average_data[i] = grainData.mean()

        return grain_average_data

    def grain_data_to_map(self, name):
        map_data = np.zeros(self[0].data[name].shape[:-1] + self.shape)
        for grain in self:
            for i, point in enumerate(grain.data.point):
                map_data[..., point[1], point[0]] = grain.data[name][..., i]

        return map_data

    def grain_data_to_map_data(self, grain_data, grainIds=-1, bg=0):
        """Create a map array with each grain filled with the given
        values.

        Parameters
        ----------
        grain_data : list or numpy.ndarray
            Grain values. This can be a single value per grain or RGB
            values.
        grainIds : list of int or int, optional
            IDs of grains to plot for. Use -1 for all grains in the map.
        bg : int or real, optional
            Value to fill the background with.

        Returns
        -------
        grainMap: numpy.ndarray
            Array filled with grain data values

        """
        # Check that grains have been detected in the map
        self.check_grains_detected()

        if type(grainIds) is int:
            if grainIds == -1:
                grainIds = range(len(self))
            else:
                grainIds = [grainIds]

        grain_data = np.array(grain_data)
        if grain_data.shape[0] != len(grainIds):
            raise ValueError("The length of supplied grain data does not"
                             "match the number of grains.")
        if len(grain_data.shape) == 1:
            mapShape = [self.yDim, self.xDim]
        elif len(grain_data.shape) == 2 and grain_data.shape[1] == 3:
            mapShape = [self.yDim, self.xDim, 3]
        else:
            raise ValueError("The grain data supplied must be either a"
                             "single value or RGB values per grain.")

        grainMap = np.full(mapShape, bg, dtype=grain_data.dtype)
        for grainId, grainValue in zip(grainIds, grain_data):
            for point in self[grainId].data.point:
                grainMap[point[1], point[0]] = grainValue

        return grainMap

    def plot_grain_data_map(
        self, map_data=None, grain_data=None, grainIds=-1, bg=0, **kwargs
    ):
        """Plot a grain map with grains coloured by given data. The data
        can be provided as a list of values per grain or as a map which
        a grain average will be applied.

        Parameters
        ----------
        map_data : numpy.ndarray, optional
            Array of map data. This must be cropped! Either mapData or 
            grainData must be supplied.
        grain_data : list or np.array, optional
            Grain values. This an be a single value per grain or RGB
            values. You must supply either mapData or grainData.
        grainIds: list of int or int, optional
            IDs of grains to plot for. Use -1 for all grains in the map.
        bg: int or real, optional
            Value to fill the background with.
        kwargs : dict, optional
            Keyword arguments passed to :func:`defdap.plotting.MapPlot.create`

        Returns
        -------
        plot: defdap.plotting.MapPlot
            Plot object created

        """
        # Set default plot parameters then update with any input
        plotParams = {}
        plotParams.update(kwargs)

        if grain_data is None:
            if map_data is None:
                raise ValueError("Either 'mapData' or 'grainData' must "
                                 "be supplied.")
            else:
                grain_data = self.calc_grain_average(map_data, grainIds=grainIds)

        grainMap = self.grain_data_to_map_data(grain_data, grainIds=grainIds,
                                               bg=bg)

        plot = MapPlot.create(self, grainMap, **plotParams)

        return plot

    def plotGrainDataIPF(
        self, direction, mapData=None, grainData=None, grainIds=-1,
        **kwargs
    ):
        """
        Plot IPF of grain reference (average) orientations with
        points coloured by grain average values from map data.

        Parameters
        ----------
        direction : numpy.ndarray
            Vector of reference direction for the IPF.
        mapData : numpy.ndarray
            Array of map data. This must be cropped! Either mapData or
            grainData must be supplied.
        grainData : list or np.array, optional
            Grain values. This an be a single value per grain or RGB
            values. You must supply either mapData or grainData.
        grainIds: list of int or int, optional
            IDs of grains to plot for. Use -1 for all grains in the map.
        kwargs : dict, optional
            Keyword arguments passed to :func:`defdap.quat.Quat.plotIPF`

        """
        # Set default plot parameters then update with any input
        plotParams = {}
        plotParams.update(kwargs)

        if grainData is None:
            if mapData is None:
                raise ValueError("Either 'mapData' or 'grainData' must "
                                 "be supplied.")
            else:
                grainData = self.calc_grain_average(mapData, grainIds=grainIds)

        # Check that grains have been detected in the map
        self.check_grains_detected()

        if type(grainIds) is int and grainIds == -1:
            grainIds = range(len(self))

        if len(grainData) != len(grainIds):
            raise Exception("Must be 1 value for each grain in grainData.")

        grainOri = np.empty(len(grainIds), dtype=Quat)

        for i, grainId in enumerate(grainIds):
            grain = self[grainId]
            grainOri[i] = grain.refOri

        plot = Quat.plotIPF(grainOri, direction, self.crystalSym,
                            c=grainData, **plotParams)

        return plot


class Grain(object):
    """
    Base class for a grain.

    Attributes
    ----------
    grainID : int

    ownerMap : defdap.base.Map

    """
    def __init__(self, grainID, ownerMap, group_id):
        self.data = Datastore(group_id=group_id)
        self.data.add_derivative(
            ownerMap.data, self.grainData,
            in_props={
                'type': 'map'
            },
            out_props={
                'type': 'list'
            }
        )
        self.data.add(
            'point', [],
            unit='', type='list', order=1
        )

        # list of coords stored as tuples (x, y). These are coords in a
        # cropped image if crop exists.
        self.grainID = grainID
        self.ownerMap = ownerMap

    def __len__(self):
        return len(self.data.point)

    def __str__(self):
        return f"Grain(ID={self.grainID})"

    def addPoint(self, point):
        """Append a coordinate and a quat to a grain.

        Parameters
        ----------
        point : tuple
            (x,y) coordinate to append

        """
        self.data.point.append(point)

    @property
    def extreme_coords(self):
        """Coordinates of the bounding box for a grain.

        Returns
        -------
        int, int, int, int
            minimum x, minimum y, maximum x, maximum y.

        """
        points = np.array(self.data.point, dtype=int)

        x0, y0 = points.min(axis=0)
        xmax, ymax = points.max(axis=0)

        return x0, y0, xmax, ymax

    def centreCoords(self, centreType="box", grainCoords=True):
        """
        Calculates the centre of the grain, either as the centre of the
        bounding box or the grains centre of mass.

        Parameters
        ----------
        centreType : str, optional, {'box', 'com'}
            Set how to calculate the centre. Either 'box' for centre of
            bounding box or 'com' for centre of mass. Default is 'box'.
        grainCoords : bool, optional
            If set True the centre is returned in the grain coordinates
            otherwise in the map coordinates. Defaults is grain.

        Returns
        -------
        int, int
            Coordinates of centre of grain.

        """
        x0, y0, xmax, ymax = self.extreme_coords
        if centreType == "box":
            xCentre = round((xmax + x0) / 2)
            yCentre = round((ymax + y0) / 2)
        elif centreType == "com":
            xCentre, yCentre = np.array(self.data.point).mean(axis=0).round()
        else:
            raise ValueError("centreType must be box or com")

        if grainCoords:
            xCentre -= x0
            yCentre -= y0

        return int(xCentre), int(yCentre)

    def grainOutline(self, bg=np.nan, fg=0):
        """Generate an array of the grain outline.

        Parameters
        ----------
        bg : int
            Value for points not within grain.
        fg : int
            Value for points within grain.

        Returns
        -------
        numpy.ndarray
            Bounding box for grain with :obj:`~numpy.nan` outside the grain and given number within.

        """
        x0, y0, xmax, ymax = self.extreme_coords

        # initialise array with nans so area not in grain displays white
        outline = np.full((ymax - y0 + 1, xmax - x0 + 1), bg, dtype=int)

        for coord in self.data.point:
            outline[coord[1] - y0, coord[0] - x0] = fg

        return outline

    def plotOutline(self, ax=None, plot_scale_bar=False, **kwargs):
        """Plot the outline of the grain.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            axis to plot on, if not provided the current active axis is used.
        plot_scale_bar : bool
            plots the scale bar on the grain if true.
        kwargs : dict
            keyword arguments passed to :func:`defdap.plotting.GrainPlot.add_map`

        Returns
        -------
        defdap.plotting.GrainPlot

        """
        plot = plotting.GrainPlot(self, ax=ax)
        plot.addMap(self.grainOutline(), **kwargs)

        if plot_scale_bar:
            plot.add_scale_bar()

        return plot

    def grainData(self, mapData):
        """Extract this grains data from the given map data.

        Parameters
        ----------
        mapData : numpy.ndarray
            Array of map data. This must be cropped!

        Returns
        -------
        numpy.ndarray
            Array containing this grains values from the given map data.

        """
        grainData = np.zeros(len(self), dtype=mapData.dtype)

        for i, coord in enumerate(self.data.point):
            grainData[i] = mapData[coord[1], coord[0]]

        return grainData

    def grainMapData(self, mapData=None, grainData=None, bg=np.nan):
        """Extract a single grain map from the given map data.

        Parameters
        ----------
        mapData : numpy.ndarray
            Array of map data. This must be cropped! Either this or
            'grainData' must be supplied and 'grainData' takes precedence.
        grainData : numpy.ndarray
            Array of data at each point in the grain. Either this or
            'mapData' must be supplied and 'grainData' takes precedence.
        bg : various, optional
            Value to fill the background with. Must be same dtype as
            input array.

        Returns
        -------
        numpy.ndarray
            Grain map extracted from given data.

        """
        if grainData is None:
            if mapData is None:
                raise ValueError("Either 'mapData' or 'grainData' must "
                                 "be supplied.")
            else:
                grainData = self.grainData(mapData)
        x0, y0, xmax, ymax = self.extreme_coords

        grainMapData = np.full((ymax - y0 + 1, xmax - x0 + 1), bg,
                               dtype=type(grainData[0]))

        for coord, data in zip(self.data.point, grainData):
            grainMapData[coord[1] - y0, coord[0] - x0] = data

        return grainMapData

    def grainMapDataCoarse(self, mapData=None, grainData=None,
                           kernelSize=2, bg=np.nan):
        """
        Create a coarsened data map of this grain only from the given map
        data. Data is coarsened using a kernel at each pixel in the
        grain using only data in this grain.

        Parameters
        ----------
        mapData : numpy.ndarray
            Array of map data. This must be cropped! Either this or
            'grainData' must be supplied and 'grainData' takes precedence.
        grainData : numpy.ndarray
            List of data at each point in the grain. Either this or
            'mapData' must be supplied and 'grainData' takes precedence.
        kernelSize : int, optional
            Size of kernel as the number of pixels to dilate by i.e 1
            gives a 3x3 kernel.
        bg : various, optional
            Value to fill the background with. Must be same dtype as
            input array.

        Returns
        -------
        numpy.ndarray
            Map of this grains coarsened data.

        """
        grainMapData = self.grainMapData(mapData=mapData, grainData=grainData)
        grainMapDataCoarse = np.full_like(grainMapData, np.nan)

        for i, j in np.ndindex(grainMapData.shape):
            if np.isnan(grainMapData[i, j]):
                grainMapDataCoarse[i, j] = bg
            else:
                coarseValue = 0

                if i - kernelSize >= 0:
                    yLow = i - kernelSize
                else:
                    yLow = 0
                if i + kernelSize + 1 <= grainMapData.shape[0]:
                    yHigh = i + kernelSize + 1
                else:
                    yHigh = grainMapData.shape[0]
                if j - kernelSize >= 0:
                    xLow = j - kernelSize
                else:
                    xLow = 0
                if j + kernelSize + 1 <= grainMapData.shape[1]:
                    xHigh = j + kernelSize + 1
                else:
                    xHigh = grainMapData.shape[1]

                numPoints = 0
                for k in range(yLow, yHigh):
                    for l in range(xLow, xHigh):
                        if not np.isnan(grainMapData[k, l]):
                            coarseValue += grainMapData[k, l]
                            numPoints += 1

                if numPoints > 0:
                    grainMapDataCoarse[i, j] = coarseValue / numPoints
                else:
                    grainMapDataCoarse[i, j] = np.nan

        return grainMapDataCoarse

    def plot_grain_data(self, map_data=None, grain_data=None, **kwargs):
        """
        Plot a map of this grain from the given map data.

        Parameters
        ----------
        map_data : numpy.ndarray
            Array of map data. This must be cropped! Either this or
            'grainData' must be supplied and 'grainData' takes precedence.
        grain_data : numpy.ndarray
            List of data at each point in the grain. Either this or
            'mapData' must be supplied and 'grainData' takes precedence.
        kwargs : dict, optional
            Keyword arguments passed to :func:`defdap.plotting.GrainPlot.create`

        """
        # Set default plot parameters then update with any input
        plotParams = {}
        plotParams.update(kwargs)

        grainMapData = self.grainMapData(mapData=map_data, grainData=grain_data)

        plot = GrainPlot.create(self, grainMapData, **plotParams)

        return plot
