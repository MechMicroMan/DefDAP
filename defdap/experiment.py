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

import numpy as np
from skimage import transform as tf
from skimage import morphology as mph
import networkx as nx

point_type = tuple[int | float, int | float]
line_type = tuple[point_type, point_type]


class Experiment(object):
    def __init__(self):
        self.frame_relations = nx.Graph()
        self.increments = []

    def __getitem__(self, key):
        return self.increments[key]

    def add_increment(self, **kwargs):
        inc = Increment(self, **kwargs)
        self.increments.append(inc)
        return inc
    
    def add_frame(self):
        return Frame(self)

    def iter_over_maps(self, map_name):
        for i, inc in enumerate(self.increments):
            map_obj = inc.maps.get(map_name)
            if map_obj is None:
                continue
            yield i, map_obj

class Maps(object):
    def __init__(self, allow_multiple: bool):
        self.allow_multiple = allow_multiple
        self.maps: dict[str, list["Map"]] = {}

    def add_map(self, map_name, map_obj, primary: bool=False):
        if map_name in self.maps:
            if not self.allow_multiple:
                raise ValueError(f"Map with name {map_name} already exists.")
            self.maps[map_name].append(map_obj)
        else:
            self.maps[map_name] = [map_obj]
    
    def get_map(self, map_name):
        pass

    def __getitem__(self, key):
        if key not in self.maps:
            raise KeyError(f"Map with name `{key}` does not exist.")
        maps = self.maps[key]
        if len(maps) == 1:
            return maps[0]
        raise ValueError(f"More than one map exists with name `{key}`")
    
    def get(self, key, value=None):
        try:
            return self[key]
        except KeyError:
            return value
        
    def get_all(self):
        return sum(self.maps.values(), start=[])


class Increment(object):
    # def __init__(self, experiment, **kwargs):
    def __init__(self, experiment, **kwargs):
        self.experiment = experiment
        self.maps = Maps(False)
        # ex: (name, map, frame)
        # default behaviour for no frame, different frame for
        # each EBSD map, initial increment frame for DIC maps

        self.metadata = kwargs

    @property
    def inc_id(self):
        return self.experiment.increments.index(self)

    def __str__(self):
        return f"Increment({self.inc_id})"

    def add_map(self, map_name, map_obj):
        self.maps.add_map(map_name, map_obj)


class Frame(object):
    def __init__(self, experiment):
        self.experiment = experiment
        # self.maps = []
        self.maps = Maps(True)
        self.homog_points = {}

    def add_map(self, map_name, map_obj):
        self.maps.add_map(map_name, map_obj)

    def link_frames(
        self, other, points_names: tuple[str, str], 
        transform_type=None, **kwargs
    ):
        if self.experiment != other.experiment:
            raise ValueError('Frames are in different experiments.')
        
        if transform_type is None:
            transform_type = "affine"
        kwargs.update({'type': transform_type.lower()})

        edge_props = {
            'start': self,
            'transform_props': kwargs,
            'points_names': points_names,
        }
        if self.experiment.frame_relations.has_edge(self, other):
            if edge_props != self.experiment.frame_relations[self][other]:
                print("Overwriting transform")
        self.experiment.frame_relations.add_edge(self, other, **edge_props)

    def get_frame_transform(self, other):
        transform_lookup = {
            'piecewise_affine': tf.PiecewiseAffineTransform,
            'projective': tf.ProjectiveTransform,
            'polynomial': tf.PolynomialTransform,
            'affine': tf.AffineTransform,
        }

        try:
            frame_relation = self.experiment.frame_relations[self][other]
        except KeyError:
            raise ValueError('Frames are not linked.')
        
        transform_props = frame_relation['transform_props']
        transform = transform_lookup[transform_props['type']]()
        points_names = frame_relation['points_names']

        frames = (self, other)
        if frame_relation['start'] is not self:
            points_names = points_names[::-1]
        transform.estimate(
            np.array(frames[0].homog_points[points_names[0]]),
            np.array(frames[1].homog_points[points_names[1]]),
            **{k: v for k, v in transform_props.items() if k != 'type'}
        )
        return transform

        invert = (frame_relation['start'] is not self 
                  and transform_props['type'] != 'polynomial')
        if invert:
            frames = frames[::-1]
            points_names = points_names[::-1]
        transform.estimate(
            np.array(frames[0].homog_points[points_names[0]]),
            np.array(frames[1].homog_points[points_names[1]]),
            **{k: v for k, v in transform_props.items() if k != 'type'}
        )
        if invert:
            transform = transform.inverse
        
        return transform
    
    def get_linked_frames(self):
        try:
            return list(self.experiment.frame_relations[self])
        except KeyError:
            return []

    def get_linked_maps(self, map_type=None):
        if map_type is None:
            map_type = object

        return [m for f in self.get_linked_frames() for m in f.maps.get_all() 
                if isinstance(m, map_type)]

    def warp_image(self, other, map_data, crop=True, **kwargs):
        """Warps a map to the `other` frame.

        Parameters
        ----------
        map_data : numpy.ndarray
            Data to warp.
        crop : bool, optional
            Crop to size of DIC map if true.
        kwargs
            All other arguments passed to :func:`skimage.transform.warp`.

        Returns
        ----------
        numpy.ndarray
            Map (i.e. EBSD map data) warped to the DIC frame.

        """
        transform = other.get_frame_transform(self)

        if not crop and isinstance(transform, tf.AffineTransform):
            # copy transform and change translation to give an extra
            # 5% border to show the entire image after rotation/shearing
            input_shape = np.array(map_data.shape)
            transform = tf.AffineTransform(matrix=np.copy(transform.params))
            transform.params[0:2, 2] = -0.05 * input_shape
            output_shape = input_shape * 1.4 / transform.scale
            kwargs['output_shape'] = output_shape.astype(int)

        return tf.warp(map_data, transform, **kwargs)
    
    def warp_points(
            self, 
            other, 
            points: list[point_type], 
            round=True
        ) -> list[point_type]:
        """Warp a list of points to the `other` reference frame.

        Parameters
        ----------
        points : list of tuples
            Points to warp. Each line is represented as a tuple of start
            and end coordinates (x, y).

        Returns
        -------
        list of tuples
            List of warped lines with same representation as input.

        """
        # Transform
        transform = self.get_frame_transform(other)
        points = transform(np.array(points))
        # Round to nearest
        if round:
            points = np.round(points - 0.5) + 0.5
        points = list(map(tuple, points))
        return points

    def warp_lines(
        self, 
        other, 
        lines: list[line_type], 
        round=True
    ) -> list[line_type]:
        """Warp a list of lines to the `other` reference frame.

        Parameters
        ----------
        lines : list of tuple of tuple
            Lines to warp. Each line is represented as a tuple of start
            and end coordinates (x, y).

        Returns
        -------
        list of tuple of tuple
            List of warped lines with same representation as input.

        """
        # Transform
        transform = self.get_frame_transform(other)
        points = transform(np.array(lines).reshape(-1, 2))
        lines = points.reshape(-1, 2, 2)

        # Remove any lines with an invalid point (-1, -1)
        if isinstance(transform, tf.PiecewiseAffineTransform):
            bad_points = np.nonzero(np.all(points == -1, axis=1))[0]
            good_lines = np.ones(len(points) // 2, dtype=bool)
            good_lines[bad_points // 2] = False
            lines = lines[good_lines]
   
        # Round to nearest
        if round:
            lines = np.round(lines - 0.5) + 0.5
        lines = [(tuple(line[0]), tuple(line[1])) for line in lines]
        return lines

    def warp_points_img(self, other, points_img, **kwargs):
        input_shape = np.array(points_img.shape)
        points_img = self.warp_image(other, points_img, crop=False, **kwargs)

        points_img = mph.skeletonize(points_img > 0.1)
        mph.remove_small_objects(points_img, min_size=10, connectivity=2,
                                 out=points_img)

        # remove 5% border if required
        transform = other.get_frame_transform(self)
        if isinstance(transform, tf.AffineTransform):
            # the crop is defined in EBSD coords so need to transform it
            crop = np.matmul(
                np.linalg.inv(transform.params[0:2, 0:2]),
                transform.params[0:2, 2] + 0.05*input_shape
            )
            crop = crop.round().astype(int)
            points_img = points_img[crop[1]:crop[1] + kwargs['output_shape'][0],
                                    crop[0]:crop[0] + kwargs['output_shape'][1]]

        return zip(*points_img.transpose().nonzero())

    def set_homog_points(self, points_name, points):
        """

        Parameters
        ----------
        points : numpy.ndarray, optional
            Array of (x,y) homologous points to set explicitly.
        """
        self.homog_points[points_name] = points

    def set_homog_point(self, map_obj, points_name=None, map_name=None, **kwargs):
        """
        Interactive tool to set homologous points. Right-click on a point
        then click 'save point' to append to the homologous points list.

        Parameters
        ----------
        map_name : str, optional
            Map data to plot for selecting points.
        points : numpy.ndarray, optional
            Array of (x,y) homologous points to set explicitly.
        kwargs : dict, optional
            Keyword arguments passed to :func:`defdap.base.Map.plotHomog`

        """
        if map_name is None:
            map_name = map_obj.homog_map_name

        if points_name is None:
            points_name = map_obj.name

        binning = map_obj.data.get_metadata(map_name, 'binning', 1)
        plot = map_obj.plot_map(map_name, make_interactive=True, **kwargs)

        # Plot stored homog points if there are any
        homog_points = self.homog_points.get(points_name, [])
        if len(homog_points) > 0:
            homog_points = np.array(homog_points) * binning
            plot.add_points(homog_points[:, 0], homog_points[:, 1], c='y', s=60)
        else:
            # add empty points layer to update later
            plot.add_points([None], [None], c='y', s=60)

        # add empty points layer for current selected point
        plot.add_points([None], [None], c='w', s=60, marker='x')

        plot.add_event_handler('button_press_event', self.homog_click)
        plot.add_event_handler('key_press_event', self.homog_key)
        plot.add_button("Save point",
                        lambda e, p: self.homog_click_save(e, p, points_name, binning),
                        color="0.85", hovercolor="blue")

        return plot

    @staticmethod
    def homog_click(event, plot):
        """Event handler for capturing position when clicking on a map.

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

        # right mouse click or shift + left mouse click
        # shift click doesn't work in osx backend
        if event.button == 3 or (event.button == 1 and event.key == 'shift'):
            plot.add_points([int(event.xdata)], [int(event.ydata)], update_layer=1)

    @staticmethod
    def homog_key(event, plot):
        """Event handler for moving position using keyboard after clicking on
        a map.

        Parameters
        ----------
        event :
            Keypress event.
        plot : defdap.plotting.MapPlot
            Plot to monitor.

        """
        arrow_keys = ['left', 'right', 'up', 'down']
        keys = event.key.split('+')
        key = keys[-1]
        if key not in arrow_keys:
            return

        # get the selected point
        sel_point = plot.img_layers[plot.points_layer_ids[1]].get_offsets()[0]
        if sel_point[0] is None or sel_point[1] is None:
            return

        move = 10 if len(keys) == 2 and keys[0] == 'shift' else 1
        if key == arrow_keys[0]:
            sel_point[0] -= move
        elif key == arrow_keys[1]:
            sel_point[0] += move
        elif key == arrow_keys[2]:
            sel_point[1] -= move
        elif key == arrow_keys[3]:
            sel_point[1] += move

        plot.add_points([sel_point[0]], [sel_point[1]], update_layer=1)

    def homog_click_save(self, event, plot, points_name, binning):
        """Append the selected point on the map to homogPoints.

        Parameters
        ----------
        event :
            Button click event.
        plot : defdap.plotting.MapPlot
            Plot to monitor.
        binning : int, optional
            Binning applied to image, if applicable.

        """
        # get the selected point
        sel_point = plot.img_layers[plot.points_layer_ids[1]].get_offsets()[0]
        if any(np.isnan(sel_point)) or sel_point[0] is None or sel_point[1] is None:
            return

        # remove selected point from plot
        plot.add_points([None], [None], update_layer=1)

        # then scale and add to homog points list
        sel_point = tuple((sel_point / binning).round().astype(int).tolist())
        homog_points = self.homog_points.get(points_name, [])
        homog_points.append(sel_point)
        self.homog_points[points_name] = homog_points

        # update the plotted homog points
        homog_points = np.array(homog_points) * binning
        plot.add_points(homog_points[:, 0], homog_points[:, 1], update_layer=0)

    def update_homog_points(self, points_name, homog_idx, new_point=None, delta=None):
        """
        Update a homog point by either over writing it with a new point or
        incrementing the current values.

        Parameters
        ----------
        homog_idx : int
            ID (place in list) of point to update or -1 for all.
        new_point : tuple, optional
            (x, y) coordinates of new point.
        delta : tuple, optional
            Increments to current point (dx, dy).

        """
        homog_points = self.homog_points.get(points_name, [])

        if type(homog_idx) is not int:
            raise Exception("homog_idx must be an integer.")
        if homog_idx >= len(homog_points):
            raise Exception("homog_idx is out of range.")

        # Update all points
        if homog_idx < 0:
            for i in range(len(homog_points)):
                self.update_homog_points(points_name, i, delta=delta)
            return

        # Update a single point
        # overwrite point
        if new_point is not None:
            if type(new_point) is not tuple and len(new_point) != 2:
                raise Exception("newPoint must be a 2 component tuple")
        # increment current point
        elif delta is not None:
            if type(delta) is not tuple and len(delta) != 2:
                raise Exception("delta must be a 2 component tuple")
            new_point = tuple(x+d for x, d in zip(homog_points[homog_idx], delta))

        homog_points[homog_idx] = new_point
        self.homog_points[points_name] = homog_points
