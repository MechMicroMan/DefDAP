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


class Experiment(object):
    def __init__(self):
        self.frame_relations = {}
        self.increments = []

    def __getitem__(self, key):
        return self.increments[key]

    def add_increment(self, **kwargs):
        inc = Increment(self, **kwargs)
        self.increments.append(inc)
        return inc

    def iter_over_maps(self, map_name):
        for i, inc in enumerate(self.increments):
            map_obj = inc.maps.get(map_name)
            if map_obj is None:
                continue
            yield i, map_obj

    def link_frames(self, frame_1, frame_2, transform_props):
        self.frame_relations[(frame_1, frame_2)] = transform_props

    def get_frame_transform(self, frame_1, frame_2):
        transform_lookup = {
            'piecewise_affine': tf.PiecewiseAffineTransform,
            'projective': tf.ProjectiveTransform,
            'polynomial': tf.PolynomialTransform,
            'affine': tf.AffineTransform,
        }

        forward = (frame_1, frame_2) in self.frame_relations
        reverse = (frame_2, frame_1) in self.frame_relations
        if forward and reverse:
            raise ValueError('Why are frame relations in both senses stored?')
        if not (forward or reverse):
            raise ValueError('Frames are not linked.')

        frames = (frame_1, frame_2) if forward else (frame_2, frame_1)
        transform_props = self.frame_relations[frames]
        calc_inverse = transform_props['type'] == 'polynomial'
        transform = transform_lookup[transform_props['type']]()

        if reverse and calc_inverse:
            frames = frames[::-1]

        transform.estimate(
            np.array(frames[0].homog_points),
            np.array(frames[1].homog_points),
            **{k: v for k, v in transform_props.items() if k != 'type'}
        )

        if reverse and not calc_inverse:
            transform = transform.inverse

        return transform

    def warp_image(self, map_data, frame_1, frame_2, crop=True, **kwargs):
        """Warps a map to the DIC frame.

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
        transform = self.get_frame_transform(frame_2, frame_1)

        if not crop and isinstance(transform, tf.AffineTransform):
            # copy transform and change translation to give an extra
            # 5% border to show the entire image after rotation/shearing
            input_shape = np.array(map_data.shape)
            transform = tf.AffineTransform(matrix=np.copy(transform.params))
            transform.params[0:2, 2] = -0.05 * input_shape
            output_shape = input_shape * 1.4 / transform.scale
            kwargs['output_shape'] = output_shape.astype(int)

        return tf.warp(map_data, transform, **kwargs)

    def warp_lines(self, lines, frame_1, frame_2):
        """Warp a set of lines to the DIC reference frame.

        Parameters
        ----------
        lines : list of tuples
            Lines to warp. Each line is represented as a tuple of start
            and end coordinates (x, y).

        Returns
        -------
        list of tuples
            List of warped lines with same representation as input.

        """
        # Transform
        transform = self.get_frame_transform(frame_1, frame_2)
        lines = transform(np.array(lines).reshape(-1, 2)).reshape(-1, 2, 2)
        # Round to nearest
        lines = np.round(lines - 0.5) + 0.5
        lines = [(tuple(line[0]), tuple(line[1])) for line in lines]
        return lines

    def warp_points(self, points_img, frame_1, frame_2, **kwargs):
        input_shape = np.array(points_img.shape)
        points_img = self.warp_image(points_img, frame_1, frame_2, crop=False,
                                     **kwargs)

        points_img = mph.skeletonize(points_img > 0.1)
        mph.remove_small_objects(points_img, min_size=10, connectivity=2,
                                 out=points_img)

        # remove 5% border if required
        transform = self.get_frame_transform(frame_2, frame_1)
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


class Increment(object):
    # def __init__(self, experiment, **kwargs):
    def __init__(self, experiment, **kwargs):

        self.maps = {}
        # ex: (name, map, frame)
        # default behaviour for no frame, different frame for
        # each EBSD map, initial increment frame for DIC maps

        self.experiment = experiment
        self.metadata = kwargs

    def add_map(self, name, map_obj):
        self.maps[name] = map_obj


class Frame(object):
    def __init__(self):
        # self.maps = []
        self.homog_points = []

    def set_homog_points(self, points):
        """

        Parameters
        ----------
        points : numpy.ndarray, optional
            Array of (x,y) homologous points to set explicitly.
        """
        self.homog_points = points

    def set_homog_point(self, map_obj, map_name=None, **kwargs):
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

        binning = map_obj.data.get_metadata(map_name, 'binning', 1)
        plot = map_obj.plot_map(map_name, make_interactive=True, **kwargs)

        # Plot stored homog points if there are any
        if len(self.homog_points) > 0:
            homog_points = np.array(self.homog_points) * binning
            plot.add_points(homog_points[:, 0], homog_points[:, 1], c='y', s=60)
        else:
            # add empty points layer to update later
            plot.add_points([None], [None], c='y', s=60)

        # add empty points layer for current selected point
        plot.add_points([None], [None], c='w', s=60, marker='x')

        plot.add_event_handler('button_press_event', self.homog_click)
        plot.add_event_handler('key_press_event', self.homog_key)
        plot.add_button("Save point",
                        lambda e, p: self.homog_click_save(e, p, binning),
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

    def homog_click_save(self, event, plot, binning):
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
        self.homog_points.append(sel_point)

        # update the plotted homog points
        homog_points = np.array(self.homog_points) * binning
        plot.add_points(homog_points[:, 0], homog_points[:, 1], update_layer=0)

    def update_homog_points(self, homog_idx, new_point=None, delta=None):
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
        if type(homog_idx) is not int:
            raise Exception("homog_idx must be an integer.")
        if homog_idx >= len(self.homog_points):
            raise Exception("homog_idx is out of range.")

        # Update all points
        if homog_idx < 0:
            for i in range(len(self.homog_points)):
                self.update_homog_points(homog_idx=i, delta=delta)
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
            new_point = list(self.homog_points[homog_idx])
            new_point[0] += delta[0]
            new_point[1] += delta[1]
            new_point = tuple(new_point)

        self.homog_points[homog_idx] = new_point
