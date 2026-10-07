# Copyright 2026 Mechanics of Microstructures Group
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
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import profile_line
import skimage.draw as draw

from defdap import defaults, ebsd
from defdap.quat import Quat
from defdap.boundaries import clip_boundary_lines

point_type = tuple[int | float, int | float]


class Feature(ABC):
    def __init__(self, frame):
        self.frame = frame

    @property
    def experiment(self):
        return self.frame.experiment

    @abstractmethod
    def get_map_data(self, map_obj, map_name, component=None):
        pass

    @abstractmethod
    def plot_map_data(self, map_obj, map_name, component=None):
        pass


class Line(Feature):
    def __init__(self, frame, start: point_type, end: point_type):
        super().__init__(frame)
        self.start = start
        self.end = end
    
    def _get_point(self, frame, point):
        if frame is self.frame:
            return point
        return self.frame.warp_points(frame, [point], round=False)[0]

    def get_start(self, frame):
        return self._get_point(frame, self.start)

    def get_end(self, frame):
        return self._get_point(frame, self.end)

    def get_map_data(
        self, map_obj, map_name, component=None, 
        line_param: Literal["pixel", "fraction"] = "pixel"
    ):
        map_data = map_obj.get_map_data(map_name, component=component)
        start, end = self.get_start(map_obj.frame), self.get_end(map_obj.frame)
        line_vals = profile_line(map_data, start[::-1], end[::-1], mode="nearest")
        if line_param == 'fraction':
            line_param = np.linspace(0, 1, len(line_vals))
        else:
            length = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            line_param = np.linspace(0, length, len(line_vals))
        return line_param, line_vals

    def plot_map_data(self, map_obj, map_name, component=None):
        line_data = self.get_map_data(
            map_obj, map_name, component=component
        )
        fig, ax = plt.subplots(1, 1)
        ax.plot(*line_data)


class Area(Feature):
    def __init__(self, frame):
        super().__init__(frame)
    
    @abstractmethod
    def get_mask(self, frame, shape):
        pass

    def get_data(self, map_obj, map_name):
        map_data = map_obj.data[map_name]
        mask = self.get_mask(map_obj.frame, map_obj.shape)
        return map_data[..., mask]

    def get_map_data(
        self, map_obj, map_name, component=None, 
        return_type: Literal["values", "image"] = "values"
    ):
        map_data = map_obj.get_map_data(map_name, component=component)
        mask = self.get_mask(map_obj.frame, map_obj.shape)
        data_vals = map_data[mask]
        if return_type != "image":
            return data_vals
        y, x = np.where(mask)
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        image = np.full_like(
            map_data, np.nan, 
            shape=(ymax - ymin + 1, xmax - xmin + 1) + map_data.shape[2:]
        )
        image[mask[ymin:ymax + 1, xmin:xmax + 1]] = data_vals
        return image

    def plot_map_data(self, map_obj, map_name, component=None):
        pass

    def get_grain_data(
        self, 
        map_obj, 
        centre_type: Literal["box", "com"] = "box",
        ori_frame: Literal["native", "spatial"] = "native",
        ori_hex_ortho_conv: Literal["hkl", "tsl"] = "hkl",
    ):
        """Create an array containing for each grain within the feature: 
         - Grain ID from `map_obj`
         - Grain centre (x, y)
         - Mean orientation as Bunge Euler angles (ph1, Phi, ph2)
         - Phase ID

        Parameters
        ----------
        map_obj : Map
            Map to extract data from
        centre_type : {"box", "com"}, optional
            How to calculate the grain centre, either "box" for centre of
            bounding box or "com" for centre of mass. Default is "box".
        ori_frame : {"native", "spatial"}, optional
            Reference frame for the orientation, either "spatial" for same 
            frame as image or "native" for 180 degree rotation around z from 
            spatial frame (top-left to bottom-left). Default is "native".
        ori_hex_ortho_conv : {"hkl", "tsl"}, optional
            Convention for hexagonal to orthonormal conversion, either "hkl" 
            for x // [10-10], y // a2 [-12-10] or "tsl" for x // a1 [2-1-10], 
            y // [01-10]. Default is "hkl".

        """
        grain_image = self.get_map_data(
            map_obj, 'grains', return_type='image'
        )
        grain_idxs = np.unique(grain_image)
        grain_idxs = grain_idxs[grain_idxs > 0]
        grains = [map_obj[grain_idx - 1] for grain_idx in grain_idxs]
        ebsd_grains = grains
        if not isinstance(map_obj, ebsd.Map):
            ebsd_grains = [grain.ebsd_grain for grain in grains]
        n_grains = len(grains)

        # Property: grain centre
        if centre_type == "box":
            grain_centres = np.zeros((n_grains, 2))
            for i, grain_idx in enumerate(grain_idxs):
                points = np.asarray(np.nonzero(grain_image == grain_idx))[::-1]
                grain_centres[i] = (points.max(axis=1) + points.min(axis=1)) / 2
        elif centre_type == "com":
            grain_centres = np.array([
                np.asarray(np.nonzero(grain_image == grain_idx)).mean(axis=1)[::-1]
                for grain_idx in grain_idxs
            ])
        else:
            raise ValueError(f"Unknown `centre_type` {centre_type}")

        # Property: grain orientation
        # Transformation from EBSD orientation reference frame to EBSD spatial 
        # reference frame
        if ori_frame == "spatial":
            frame_transform = Quat.from_axis_angle(np.array((1, 0, 0)), np.pi)
        elif ori_frame == "native":
            frame_transform = Quat(1, 0, 0, 0)
        else:
            raise ValueError(f"Unknown `ori_frame` {ori_frame}")
        # Transformation for hex convention from y // a2 of EBSD map to x // a1
        if ori_hex_ortho_conv not in ["hkl", "tsl"]:
            raise ValueError(
                f"Unknown `ori_hex_ortho_conv` {ori_hex_ortho_conv}"
            )
        hex_transform = Quat(1, 0, 0, 0)
        if defaults["crystal_ortho_conv"] != ori_hex_ortho_conv:
            sense = -1 if ori_hex_ortho_conv == "tsl" else 1
            hex_transform = Quat.from_axis_angle(
                np.array([0, 0, 1]), sense * np.pi / 6
            )
        grain_eulers = np.zeros((n_grains, 3))
        for i, ebsd_grain in enumerate(ebsd_grains):
            grain_ori = ebsd_grain.ref_ori * frame_transform
            if ebsd_grain.phase.crystal_structure.name == "hexagonal":
                grain_ori = hex_transform * grain_ori
            grain_eulers[i] = grain_ori.euler_angles()
        

        # Property: grain phase
        grain_phases = np.array([
            ebsd_grain.phase_id for ebsd_grain in ebsd_grains
        ])

        return np.hstack([
            grain_idxs[:, None],
            grain_centres,
            grain_eulers,
            grain_phases[:, None],
        ])


class Polygon(Area):
    def __init__(self, frame, polygon_points: list[point_type]):
        super().__init__(frame)
        self.points = polygon_points

    def get_mask(self, frame, shape):
        points = self.points
        if frame is not self.frame:
            points = self.frame.warp_points(frame, points, round=False)

        mask_points = draw.polygon(*np.array(points).T[::-1], shape=shape)
        mask = np.zeros(shape, dtype=bool)
        mask[mask_points[0], mask_points[1]] = True
        return mask


class Rectangle(Polygon):
    def __init__(self, frame, top_left: point_type, size: point_type):
        self.top_left = top_left
        self.size = size
        super().__init__(frame, self.polygon_points)

    @classmethod
    def from_plot(cls, plot):
        plot_roi = np.array([plot.ax.get_xlim(), plot.ax.get_ylim()[::-1]])
        top_left = tuple(plot_roi[:, 0].round().astype(int).tolist())
        size = plot_roi[:, 1] - plot_roi[:, 0]
        size = tuple(size.round().astype(int).tolist())
        return cls(plot.calling_map.frame, top_left, size)

    @property
    def polygon_points(self):
        return [
            self.top_left,
            (self.top_left[0] + self.size[0] - 1, self.top_left[1]),
            (self.top_left[0] + self.size[0] - 1, self.top_left[1] + self.size[1] - 1),
            (self.top_left[0], self.top_left[1] + self.size[1] - 1),
        ]

    def get_data(self, map_obj, map_name):
        data_vals = super().get_data(map_obj, map_name)
        return data_vals.reshape(data_vals.shape[:-2] + self.size)

    def clip_lines(self, lines, **kwargs):
        roi_lines = clip_boundary_lines(
            lines,
            (
                self.top_left[0] + self.size[0] - 1, 
                self.top_left[1] + self.size[1] - 1
            ),
            min_bounds=self.top_left,
            **kwargs,
        )
        return np.copy(roi_lines) - self.top_left


class Circle(Area):
    def __init__(self, frame, center: point_type, radius: int | float):
        super().__init__(frame)
        self.centre = center
        self.radius = radius
    
    def get_mask(self, frame, shape):
        if frame is self.frame:
            mask_points = draw.disk(self.centre[::-1], self.radius, shape=shape)
            mask = np.zeros(shape, dtype=bool)
            mask[mask_points[0], mask_points[1]] = True
            return mask
    
        mask_points = draw.disk(self.centre[::-1], self.radius)
        shape_other = (mask_points[0].max() + 10, mask_points[1].max() + 10)
        mask = np.zeros(shape_other, dtype=bool)
        mask[mask_points[0], mask_points[1]] = True
        mask = self.frame.warp_image(frame, mask, output_shape=shape)
        return mask
