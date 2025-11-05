# Copyright 2025 Mechanics of Microstructures Group
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

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import networkx as nx

import defdap
from defdap.quat import Quat
from defdap import plotting
from defdap.plotting import Plot, MapPlot, GrainPlot

from skimage.measure import profile_line

from defdap.utils import report_progress, Datastore
from defdap.experiment import Frame


class Map(ABC):
    """
    Base class for a map. Contains common functionality for all maps.

    Attributes
    ----------

    _grains : list of defdap.base.Grain
        List of grains.
    sel_grain : defdap.base.grain
        The last selected grain

    """
    def __init__(self, file_name, data_type=None, experiment=None,
                 increment=None, frame=None, map_name=None):
        """

        Parameters
        ----------
        file_name : str
            Path to EBSD file, including name, excluding extension.
        data_type : str, {'OxfordBinary', 'OxfordText'}
            Format of EBSD data file.

        """

        self.data = Datastore(crop_func=self.crop, mask_func=self.mask)
        self.frame = frame if frame is not None else Frame()
        if increment is not None:
            self.increment = increment
            self.experiment = self.increment.experiment
            if experiment is not None:
                assert self.experiment is experiment
        else:
            self.experiment = experiment
            if experiment is None:
                self.experiment = defdap.anonymous_experiment
            self.increment = self.experiment.add_increment()
        map_name = self.MAPNAME if map_name is None else map_name
        self.increment.add_map(map_name, self)

        self.shape = (0, 0)

        self._grains = None

        self.sel_grain = None

        self.proxigram_arr = None
        self.neighbour_network = None

        self.grain_plot = None
        self.profile_plot = None

        self.file_name = Path(file_name)
        self.load_data(self.file_name, data_type=data_type)

        self.data.add_generator(
            'proxigram', self.calc_proxigram, unit='', type='map', order=0,
            cropped=True
        )

    @abstractmethod
    def load_data(self, file_name, data_type=None):
        pass

    def __len__(self):
        return len(self.grains)

    # allow array like getting of grains
    def __getitem__(self, key):
        return self.grains[key]

    @property
    def grains(self):
        # try to access grains image to generate grains if necessary
        self.data.grains
        return self._grains

    @property
    def x_dim(self):
        return self.shape[1]

    @property
    def y_dim(self):
        return self.shape[0]

    def crop(self, map_data, **kwargs):
        return map_data
    
    def mask(self, map_data, **kwargs):
        return map_data

    def set_homog_point(self, **kwargs):
        return self.frame.set_homog_point(self, **kwargs)

    def plot_grain_numbers(self, dilate_boundaries=False, ax=None, **kwargs):
        """Plot a map with grains numbered.

        Parameters
        ----------
        dilate_boundaries : bool, optional
            Set to true to dilate boundaries.
        ax : matplotlib.axes.Axes, optional
            axis to plot on, if not provided the current active axis is used.
        kwargs : dict, optional
            Keyword arguments passed to :func:`defdap.plotting.MapPlot.add_grain_numbers`

        Returns
        -------
        defdap.plotting.MapPlot

        """

        plot = plotting.MapPlot(self, ax=ax)
        plot.add_grain_boundaries(colour='black', dilate=dilate_boundaries)
        plot.add_grain_numbers(**kwargs)

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
            Keyword arguments passed to :func:`defdap.base.Map.plot_default`

        """
        # reset current selected grain and plot euler map with click handler
        plot = self.plot_default(make_interactive=True, **kwargs)
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
        if display_grain:
            self.grain_plot = None

        return plot

    def click_grain_id(self, event, plot, display_grain):
        """Event handler to capture clicking on a map.

        Parameters
        ----------
        event :
            Click event.
        plot : defdap.plotting.MapPlot
            Plot to capture clicks from.
        display_grain : bool
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
        plot.add_grain_highlights([grain_id], alpha=self.highlight_alpha)

        if display_grain:
            if self.grain_plot is None or not self.grain_plot.exists:
                self.grain_plot = grain.plot_default(make_interactive=True)
            else:
                self.grain_plot.clear()
                self.grain_plot.calling_grain = grain
                grain.plot_default(plot=self.grain_plot)
                self.grain_plot.draw()

    def draw_line_profile(self, **kwargs):
        """Interactive plot for drawing a line profile of data.

        Parameters
        ----------
        kwargs : dict, optional
            Keyword arguments passed to :func:`defdap.base.Map.plot_default`

        """
        plot = self.plot_default(make_interactive=True, **kwargs)

        plot.add_event_handler('button_press_event', plot.line_slice)
        plot.add_event_handler(
            'button_release_event',
            lambda e, p: plot.line_slice(e, p, action=self.calc_line_profile)
        )

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

        if self.profile_plot is None or not self.profile_plot.exists:
            self.profile_plot = Plot(make_interactive=True)
        else:
            self.profile_plot.clear()

        self.profile_plot.ax.plot(xi, zi, **kwargs)
        self.profile_plot.ax.set_xlabel('Distance (pixels)')
        self.profile_plot.ax.set_ylabel('Intensity')
        self.profile_plot.draw()

    @report_progress("constructing neighbour network")
    def build_neighbour_network(self):
        """Construct a list of neighbours

        """
        ## TODO: fix HRDIC NN
        # create network
        nn = nx.Graph()
        nn.add_nodes_from(self.grains)

        y_locs, x_locs = np.nonzero(self.boundaries)
        total_points = len(x_locs)

        for i_point, (x, y) in enumerate(zip(x_locs, y_locs)):
            # report progress
            yield i_point / total_points

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
            num_neigh = len(neighbours)
            if num_neigh <= 1:
                continue
            for i in range(num_neigh):
                for j in range(i + 1, num_neigh):
                    # Add  to network
                    grain = self[neighbours[i]]
                    neigh_grain = self[neighbours[j]]
                    try:
                        # look up boundary
                        nn[grain][neigh_grain]
                    except KeyError:
                        # neighbour relation doesn't exist so add it
                        nn.add_edge(grain, neigh_grain)

        self.neighbour_network = nn

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
        first_neighbours = list(self.neighbour_network.neighbors(grain))
        highlight_grains = [grain] + first_neighbours

        second_neighbours = []
        for firstNeighbour in first_neighbours:
            trial_second_neighbours = list(
                self.neighbour_network.neighbors(firstNeighbour)
            )
            for second_neighbour in trial_second_neighbours:
                if (second_neighbour not in highlight_grains and
                        second_neighbour not in second_neighbours):
                    second_neighbours.append(second_neighbour)
        highlight_grains.extend(second_neighbours)

        highlight_grains = [grain.grain_id for grain in highlight_grains]
        highlight_colours = ['white']
        highlight_colours.extend(['yellow'] * len(first_neighbours))
        highlight_colours.append('green')

        # update the grain highlights layer in the plot
        plot.add_grain_highlights(highlight_grains,
                                  grain_colours=highlight_colours)

    @property
    def proxigram(self):
        """Proxigram for a map.

        Returns
        -------
        numpy.ndarray
            Distance from a grain boundary at each point in map.

        """
        self.calc_proxigram(force_calc=False)

        return self.proxigram_arr

    @report_progress("calculating proxigram")
    def calc_proxigram(self, num_trials=500):
        """Calculate distance from a grain boundary at each point in map.

        Parameters
        ----------
        num_trials : int, optional
            number of trials.

        """
        # add 0.5 to boundary coordinates as they are placed on the
        # bottom right edge pixels of grains
        index_boundaries = [t[::-1] for t in self.data.grain_boundaries.points]
        index_boundaries = np.array(index_boundaries) + 0.5

        # array of x and y coordinate of each pixel in the map
        coords = np.zeros((2,) + self.shape, dtype=float)
        coords[0], coords[1] = np.meshgrid(
            range(self.shape[0]), range(self.shape[1]), indexing='ij'
        )

        # array to store trial distance from each boundary point
        trial_distances = np.full((num_trials + 1,) + self.shape,
                                  1000, dtype=float)

        # loop over each boundary point (p) and calculate distance from
        # p to all points in the map store minimum once numTrails have
        # been made and start a new batch of trials
        num_boundary_points = len(index_boundaries)
        j = 1
        for i, index_boundary in enumerate(index_boundaries):
            trial_distances[j] = np.sqrt((coords[0] - index_boundary[0])**2
                                        + (coords[1] - index_boundary[1])**2)

            if j == num_trials:
                # find current minimum distances and store
                trial_distances[0] = trial_distances.min(axis=0)
                j = 0
                # report progress
                yield i / num_boundary_points
            j += 1

        # find final minimum distances to a boundary
        return trial_distances.min(axis=0)

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
        """Plot a map of the data.

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

    def calc_grain_average(self, map_data, grain_ids=-1):
        """Calculate grain average of any DIC map data.

        Parameters
        ----------
        map_data : numpy.ndarray
            Array of map data to grain average. This must be cropped!
        grain_ids : list, optional
            grain_ids to perform operation on, set to -1 for all grains.

        Returns
        -------
        numpy.ndarray
            Array containing the grain average values.

        """
        if type(grain_ids) is int and grain_ids == -1:
            grain_ids = range(len(self))

        grain_average_data = np.zeros(len(grain_ids))

        for i, grainId in enumerate(grain_ids):
            grain = self[grainId]
            grainData = grain.grain_data(map_data)
            grain_average_data[i] = grainData.mean()

        return grain_average_data

    def grain_data_to_map(self, name):
        map_data = np.zeros(self[0].data[name].shape[:-1] + self.shape)
        for grain in self:
            for i, point in enumerate(grain.data.point):
                map_data[..., point[1], point[0]] = grain.data[name][..., i]

        return map_data

    def grain_data_to_map_data(self, grain_data, grain_ids=-1, bg=0):
        """Create a map array with each grain filled with the given
        values.

        Parameters
        ----------
        grain_data : list or numpy.ndarray
            Grain values. This can be a single value per grain or RGB
            values.
        grain_ids : list of int or int, optional
            IDs of grains to plot for. Use -1 for all grains in the map.
        bg : int or real, optional
            Value to fill the background with.

        Returns
        -------
        grain_map: numpy.ndarray
            Array filled with grain data values

        """
        if type(grain_ids) is int:
            if grain_ids == -1:
                grain_ids = range(len(self))
            else:
                grain_ids = [grain_ids]

        grain_data = np.array(grain_data)
        if grain_data.shape[0] != len(grain_ids):
            raise ValueError("The length of supplied grain data does not"
                             "match the number of grains.")
        if len(grain_data.shape) == 1:
            mapShape = [self.y_dim, self.x_dim]
        elif len(grain_data.shape) == 2 and grain_data.shape[1] == 3:
            mapShape = [self.y_dim, self.x_dim, 3]
        else:
            raise ValueError("The grain data supplied must be either a"
                             "single value or RGB values per grain.")

        grain_map = np.full(mapShape, bg, dtype=grain_data.dtype)
        for grainId, grain_value in zip(grain_ids, grain_data):
            for point in self[grainId].data.point:
                grain_map[point[1], point[0]] = grain_value

        return grain_map

    def plot_grain_data_map(
        self, map_data=None, grain_data=None, grain_ids=-1, bg=0, **kwargs
    ):
        """Plot a grain map with grains coloured by given data. The data
        can be provided as a list of values per grain or as a map which
        a grain average will be applied.

        Parameters
        ----------
        map_data : numpy.ndarray, optional
            Array of map data. This must be cropped! Either mapData or 
            grain_data must be supplied.
        grain_data : list or np.array, optional
            Grain values. This an be a single value per grain or RGB
            values. You must supply either mapData or grain_data.
        grain_ids: list of int or int, optional
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
        plot_params = {}
        plot_params.update(kwargs)

        if grain_data is None:
            if map_data is None:
                raise ValueError("Either 'mapData' or 'grain_data' must "
                                 "be supplied.")
            else:
                grain_data = self.calc_grain_average(map_data, grain_ids=grain_ids)

        grain_map = self.grain_data_to_map_data(grain_data, grain_ids=grain_ids,
                                               bg=bg)

        plot = MapPlot.create(self, grain_map, **plot_params)

        return plot

    def plot_grain_data_ipf(
        self, direction, map_data=None, grain_data=None, grain_ids=-1,
        **kwargs
    ):
        """
        Plot IPF of grain reference (average) orientations with
        points coloured by grain average values from map data.

        Parameters
        ----------
        direction : numpy.ndarray
            Vector of reference direction for the IPF.
        map_data : numpy.ndarray
            Array of map data. This must be cropped! Either mapData or
            grain_data must be supplied.
        grain_data : list or np.array, optional
            Grain values. This an be a single value per grain or RGB
            values. You must supply either mapData or grain_data.
        grain_ids: list of int or int, optional
            IDs of grains to plot for. Use -1 for all grains in the map.
        kwargs : dict, optional
            Keyword arguments passed to :func:`defdap.quat.Quat.plot_ipf`

        """
        # Set default plot parameters then update with any input
        plot_params = {}
        plot_params.update(kwargs)

        if grain_data is None:
            if map_data is None:
                raise ValueError("Either 'mapData' or 'grain_data' must "
                                 "be supplied.")
            else:
                grain_data = self.calc_grain_average(map_data, grain_ids=grain_ids)

        if type(grain_ids) is int and grain_ids == -1:
            grain_ids = range(len(self))

        if len(grain_data) != len(grain_ids):
            raise Exception("Must be 1 value for each grain in grain_data.")

        grain_ori = np.empty(len(grain_ids), dtype=Quat)

        for i, grainId in enumerate(grain_ids):
            grain = self[grainId]
            grain_ori[i] = grain.ref_ori

        plot = Quat.plot_ipf(grain_ori, direction, self.crystal_sym,
                             c=grain_data, **plot_params)

        return plot


class Grain(ABC):
    """
    Base class for a grain.

    Attributes
    ----------
    grain_id : int

    owner_map : defdap.base.Map

    """
    def __init__(self, grain_id, owner_map, group_id):
        self.data = Datastore(group_id=group_id)
        self.data.add_derivative(
            owner_map.data, self.grain_data,
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
        self.grain_id = grain_id
        self.owner_map = owner_map

    def __len__(self):
        return len(self.data.point)

    def __str__(self):
        return f"Grain(ID={self.grain_id})"

    @property
    def extreme_coords(self):
        """Coordinates of the bounding box for a grain.

        Returns
        -------
        int, int, int, int
            minimum x, minimum y, maximum x, maximum y.

        """
        return *self.data.point.min(axis=0), *self.data.point.max(axis=0)

    def centre_coords(self, centre_type="box", grain_coords=True):
        """
        Calculates the centre of the grain, either as the centre of the
        bounding box or the grains centre of mass.

        Parameters
        ----------
        centre_type : str, optional, {'box', 'com'}
            Set how to calculate the centre. Either 'box' for centre of
            bounding box or 'com' for centre of mass. Default is 'box'.
        grain_coords : bool, optional
            If set True the centre is returned in the grain coordinates
            otherwise in the map coordinates. Defaults is grain.

        Returns
        -------
        int, int
            Coordinates of centre of grain.

        """
        x0, y0, xmax, ymax = self.extreme_coords
        if centre_type == "box":
            x_centre = round((xmax + x0) / 2)
            y_centre = round((ymax + y0) / 2)
        elif centre_type == "com":
            x_centre, y_centre = self.data.point.mean(axis=0).round()
        else:
            raise ValueError("centreType must be box or com")

        if grain_coords:
            x_centre -= x0
            y_centre -= y0

        return int(x_centre), int(y_centre)

    def grain_outline(self, bg=np.nan, fg=0):
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

    def plot_outline(self, ax=None, plot_scale_bar=False, **kwargs):
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
        plot.addMap(self.grain_outline(), **kwargs)

        if plot_scale_bar:
            plot.add_scale_bar()

        return plot

    def grain_data(self, map_data):
        """Extract this grains data from the given map data.

        Parameters
        ----------
        map_data : numpy.ndarray
            Array of map data. This must be cropped!

        Returns
        -------
        numpy.ndarray
            Array containing this grains values from the given map data.

        """
        return map_data[..., self.data.point[:, 1], self.data.point[:, 0]]

    def grain_map_data(self, map_data=None, grain_data=None, bg=np.nan):
        """Extract a single grain map from the given map data.

        Parameters
        ----------
        map_data : numpy.ndarray
            Array of map data. This must be cropped! Either this or
            'grain_data' must be supplied and 'grain_data' takes precedence.
        grain_data : numpy.ndarray
            Array of data at each point in the grain. Either this or
            'mapData' must be supplied and 'grain_data' takes precedence.
        bg : various, optional
            Value to fill the background with. Must be same dtype as
            input array.

        Returns
        -------
        numpy.ndarray
            Grain map extracted from given data.

        """
        if grain_data is None:
            if map_data is None:
                raise ValueError("Either 'mapData' or 'grain_data' must "
                                 "be supplied.")
            else:
                grain_data = self.grain_data(map_data)
        x0, y0, xmax, ymax = self.extreme_coords

        grain_map_data = np.full((ymax - y0 + 1, xmax - x0 + 1), bg,
                               dtype=type(grain_data[0]))

        for coord, data in zip(self.data.point, grain_data):
            grain_map_data[coord[1] - y0, coord[0] - x0] = data

        return grain_map_data

    def grain_map_data_coarse(self, map_data=None, grain_data=None,
                              kernel_size=2, bg=np.nan):
        """
        Create a coarsened data map of this grain only from the given map
        data. Data is coarsened using a kernel at each pixel in the
        grain using only data in this grain.

        Parameters
        ----------
        map_data : numpy.ndarray
            Array of map data. This must be cropped! Either this or
            'grain_data' must be supplied and 'grain_data' takes precedence.
        grain_data : numpy.ndarray
            List of data at each point in the grain. Either this or
            'mapData' must be supplied and 'grain_data' takes precedence.
        kernel_size : int, optional
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
        grain_map_data = self.grain_map_data(map_data=map_data, grain_data=grain_data)
        grain_map_data_coarse = np.full_like(grain_map_data, np.nan)

        for i, j in np.ndindex(grain_map_data.shape):
            if np.isnan(grain_map_data[i, j]):
                grain_map_data_coarse[i, j] = bg
            else:
                coarse_value = 0

                if i - kernel_size >= 0:
                    yLow = i - kernel_size
                else:
                    yLow = 0
                if i + kernel_size + 1 <= grain_map_data.shape[0]:
                    yHigh = i + kernel_size + 1
                else:
                    yHigh = grain_map_data.shape[0]
                if j - kernel_size >= 0:
                    x_low = j - kernel_size
                else:
                    x_low = 0
                if j + kernel_size + 1 <= grain_map_data.shape[1]:
                    x_high = j + kernel_size + 1
                else:
                    x_high = grain_map_data.shape[1]

                num_points = 0
                for k in range(yLow, yHigh):
                    for l in range(x_low, x_high):
                        if not np.isnan(grain_map_data[k, l]):
                            coarse_value += grain_map_data[k, l]
                            num_points += 1

                if num_points > 0:
                    grain_map_data_coarse[i, j] = coarse_value / num_points
                else:
                    grain_map_data_coarse[i, j] = np.nan

        return grain_map_data_coarse

    def plot_grain_data(self, map_data=None, grain_data=None, **kwargs):
        """
        Plot a map of this grain from the given map data.

        Parameters
        ----------
        map_data : numpy.ndarray
            Array of map data. This must be cropped! Either this or
            'grain_data' must be supplied and 'grain_data' takes precedence.
        grain_data : numpy.ndarray
            List of data at each point in the grain. Either this or
            'mapData' must be supplied and 'grain_data' takes precedence.
        kwargs : dict, optional
            Keyword arguments passed to :func:`defdap.plotting.GrainPlot.create`

        """
        # Set default plot parameters then update with any input
        plot_params = {}
        plot_params.update(kwargs)

        grain_map_data = self.grain_map_data(map_data=map_data, grain_data=grain_data)

        plot = GrainPlot.create(self, grain_map_data, **plot_params)

        return plot
    
    def _validate_list(self, list_name):
        """Check the name exists and is a list data.

        Parameters
        ----------
        list_name : str

        """
        if list_name not in self.data:
            raise ValueError(f'`{list_name}` does not exist.')
        if (self.data.get_metadata(list_name, 'type') != 'list' or
                self.data.get_metadata(list_name, 'order') is None):
            raise ValueError(f'`{list_name}` is not a valid data.')

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
                if len(map_data.shape) == 2:
                    axis = 0
                elif len(map_data.shape) == 3:
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
        """Plot a map of the data.

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
        self._validate_list(map_name)
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

        if self.owner_map.scale is not None:
            binning = self.data.get_metadata(map_name, 'binning', 1)
            plot_params['scale'] = self.owner_map.scale / binning

        plot_params.update(kwargs)

        list_data = self._extract_component(self.data[map_name], comp)

        return self.plot_grain_data(grain_data=list_data, **plot_params)
