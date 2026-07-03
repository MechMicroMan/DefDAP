from abc import ABC
from copy import copy, deepcopy
from typing import Literal

import numpy as np
import networkx as nx

point_type = tuple[int | float, int | float]
line_type = tuple[point_type, point_type]


class BoundarySegment(object):
    def __init__(self, ebsdMap, grain1, grain2):
        self.ebsdMap = ebsdMap

        self.grain1 = grain1
        self.grain2 = grain2

        # list of boundary points (x, y) for horizontal (X) and
        # vertical (Y) boundaries
        self.boundary_points_x = []
        self.boundary_points_y = []
        # Boolean value for each point above, True if boundary point is
        # in grain1 and False if in grain2
        self.boundary_point_owners_x = []
        self.boundary_point_owners_y = []

    def __eq__(self, right):
        if type(self) is not type(right):
            raise NotImplementedError()

        return ((self.grain1 is right.grain1 and
                self.grain2 is right.grain2) or
                (self.grain1 is right.grain2 and
                 self.grain2 is right.grain1))

    def __len__(self):
        return len(self.boundary_points_x) + len(self.boundary_points_y)

    def add_boundary_point(self, point, kind, owner_grain):
        if kind == 0:
            self.boundary_points_x.append(point)
            self.boundary_point_owners_x.append(owner_grain is self.grain1)
        elif kind == 1:
            self.boundary_points_y.append(point)
            self.boundary_point_owners_y.append(owner_grain is self.grain1)
        else:
            raise ValueError("Boundary point kind is 0 for x and 1 for y")

    def boundary_point_pairs(self, kind):
        """Return pairs of points either side of the boundary. The first
        point is always in grain1
        """
        if kind == 0:
            boundary_points = self.boundary_points_x
            boundary_point_owners = self.boundary_point_owners_x
            delta = (1, 0)
        else:
            boundary_points = self.boundary_points_y
            boundary_point_owners = self.boundary_point_owners_y
            delta = (0, 1)

        boundary_point_pairs = []
        for point, owner in zip(boundary_points, boundary_point_owners):
            other_point = (point[0] + delta[0], point[1] + delta[1])
            if owner:
                boundary_point_pairs.append((point, other_point))
            else:
                boundary_point_pairs.append((other_point, point))

        return boundary_point_pairs

    @property
    def boundary_point_pairs_x(self):
        """Return pairs of points either side of the boundary. The first
        point is always in grain1
        """
        return self.boundary_point_pairs(0)

    @property
    def boundary_point_pairs_y(self):
        """Return pairs of points either side of the boundary. The first
        point is always in grain1
        """
        return self.boundary_point_pairs(1)

    @property
    def boundary_lines(self):
        """Return line points along this boundary segment"""
        _, _, lines = EbsdBoundaries.boundary_points_to_lines(
            boundary_points_x=self.boundary_points_x,
            boundary_points_y=self.boundary_points_y
        )
        return lines

    def misorientation(self):
        mis_ori, min_symm = self.grain1.ref_ori.mis_ori(
            self.grain2.ref_ori, self.ebsdMap.crystal_sym, return_quat=2
        )
        mis_ori = 2 * np.arccos(mis_ori)
        mis_ori_axis = self.grain1.ref_ori.mis_ori_axis(min_symm)
        mis_ori_axis /= np.sqrt(np.dot(mis_ori_axis, mis_ori_axis))

        return mis_ori, mis_ori_axis


class Boundaries(ABC):
    def __init__(self, owner_map):
        self.owner_map = owner_map

    def _image_from_points(self, points):
        image = np.zeros(self.owner_map.shape, dtype=bool)
        image[tuple(zip(*points))[::-1]] = True
        return image

    @property
    def image(self):
        raise NotImplementedError("Image not available for these boundaries.")
    
    @property
    def lines(self):
        raise NotImplementedError("Lines not available for these boundaries.")


class EbsdBoundaries(Boundaries):
    def __init__(self, owner_map, points_x, points_y):
        super().__init__(owner_map)
        self.points_x = set(points_x)
        self.points_y = set(points_y)

    @classmethod
    def from_image(cls, owner_map, image_x, image_y):
        return cls(
            owner_map,
            zip(*image_x.transpose().nonzero()),
            zip(*image_y.transpose().nonzero())
        )

    @classmethod
    def from_boundary_segments(cls, b_segs):
        points_x = []
        points_y = []
        for b_seg in b_segs:
            points_x += b_seg.boundary_points_x
            points_y += b_seg.boundary_points_y

        return cls(b_segs[0].ebsdMap, points_x, points_y)

    @property
    def points(self):
        return self.points_x.union(self.points_y)
    
    @property
    def image(self):
        return self._image_from_points(self.points)

    @property
    def image_x(self):
        return self._image_from_points(self.points_x)

    @property
    def image_y(self):
        return self._image_from_points(self.points_y)

    @property
    def lines(self):
        return self.boundary_points_to_lines(
            boundary_points_x=self.points_x,
            boundary_points_y=self.points_y
        )[2]

    @staticmethod
    def boundary_points_to_lines(*, boundary_points_x=None,
                                 boundary_points_y=None):
        boundary_data = {}
        if boundary_points_x is not None:
            boundary_data['x'] = boundary_points_x
        if boundary_points_y is not None:
            boundary_data['y'] = boundary_points_y
        if not boundary_data:
            raise ValueError("No boundaries provided.")

        deltas = {
            'x': (0.5, -0.5, 0.5, 0.5),
            'y': (-0.5, 0.5, 0.5, 0.5)
        }
        all_lines = []
        for mode, points in boundary_data.items():
            lines = []
            for i, j in points:
                lines.append((
                    (i + deltas[mode][0], j + deltas[mode][1]),
                    (i + deltas[mode][2], j + deltas[mode][3])
                ))
            all_lines.append(lines)

        if len(all_lines) == 2:
            all_lines.append(all_lines[0] + all_lines[1])
            return tuple(all_lines)
        else:
            return all_lines[0]
        

class DerivedBoundaries(object):
    def __init__(self, owner_map, points=None, lines=None, graph=None):
        super().__init__(owner_map)
        self.points = set(points) if points is not None else None
        self.lines = lines
        self.graph = graph

    @classmethod
    def from_warped_boundaries(cls, boundaries, other_map):
        if len(boundaries.points) == 0:
            return cls(other_map, points=[], lines=[])
        assert other_map.ebsd_map == boundaries.owner_map

        points = boundaries.owner_map.frame.warp_points_img(
            other_map.frame, boundaries.image.astype(float),
            output_shape=other_map.shape
        )
        lines = boundaries.owner_map.frame.warp_lines(
             other_map.frame, boundaries.lines
        )
        return cls(other_map, points=points, lines=lines)
    
    @classmethod
    def from_simplified_boundaries(cls, ebsd_map, **kwargs):
        graph, lines = simpify_boundaries(
            ebsd_map.neighbour_network, **kwargs
        )
        return cls(ebsd_map, lines=lines, graph=graph)

    @property
    def image(self):
        if self.points is None:
            raise ValueError("Image not available for these boundaries.")
        return self._image_from_points(self.points)

