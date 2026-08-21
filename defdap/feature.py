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

point_type = tuple[int | float, int | float]


class Feature(ABC):
    def __init__(self, frame):
        self.frame = frame

    @property
    def experiment(self):
        return self.frame.experiment

    @abstractmethod
    def extract_map_data(self, map_obj, map_name, component=None):
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
        return self.frame.warp_points(frame, [point])[0]

    def get_start(self, frame):
        return self._get_point(frame, self.start)

    def get_end(self, frame):
        return self._get_point(frame, self.end)

    def extract_map_data(
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
        line_data = self.extract_map_data(
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
    
    def extract_map_data(
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
            map_data, np.nan, shape=(ymax - ymin + 1, xmax - xmin + 1)
        )
        image[mask[ymin:ymax + 1, xmin:xmax + 1]] = data_vals
        return image

    def plot_map_data(self, map_obj, map_name, component=None):
        pass


class Polygon(Area):
    def __init__(self, frame, polygon_points: list[point_type]):
        super().__init__(frame)
        self.points = polygon_points
    
    def get_mask(self, frame, shape):
        points = self.points
        if frame is not self.frame:
            points = self.frame.warp_points(frame, points)

        mask_points = draw.polygon(*np.array(points).T[::-1], shape=shape)
        mask = np.zeros(shape, dtype=bool)
        mask[mask_points[0], mask_points[1]] = True
        return mask


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
