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

from abc import ABC
from copy import copy, deepcopy
from typing import Literal

import numpy as np
import networkx as nx
from simplification.cutil import simplify_coords, simplify_coords_vw, simplify_coords_vwp

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
        raise ValueError("Image not available for these boundaries.")
    
    @property
    def lines(self):
        raise ValueError("Lines not available for these boundaries.")


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
    def boundary_points_to_lines(
        *, boundary_points_x=None, boundary_points_y=None
    ):
        if boundary_points_x is None and boundary_points_y is None:
            raise ValueError("No boundaries provided.")

        all_lines = []
        for points, delta in (
            (boundary_points_x, np.array([0.5, -0.5, 0.5, 0.5])), 
            (boundary_points_y, np.array([-0.5, 0.5, 0.5, 0.5])),
        ):
            if points is None:
                continue
            points = np.array(list(points))
            lines = np.concatenate(
                (points + delta[:2], points + delta[2:]), axis=1
            ).reshape(-1, 2, 2)
            all_lines.append(lines)

        if len(all_lines) == 2:
            all_lines.append(np.concatenate(all_lines, axis=0))
            return tuple(all_lines)
        else:
            return all_lines[0]


class DerivedBoundaries(Boundaries):
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
    
    @property
    def lines(self):
        return self._lines
    
    @lines.setter
    def lines(self, value):
        self._lines = value


def order_boundary_lines(boundary_lines : list[line_type]) -> list[list[line_type]]:
    """Sort a list of lines into ordered connected sections"""
    
    def connect_edge(edge: line_type, direction: int):
        if direction > 1 or direction < 0:
            connect_edge(edge, 0)
            connect_edge(edge, 1)
            return

        # find connecting verticies
        new_edge_generator = (
            idx for idx, new_edge in enumerate(boundary_lines) 
            if new_edge[0] == edge[direction] or new_edge[1] == edge[direction]
        )
        # grab the first matching edge
        try:
            idx = next(new_edge_generator)
        except StopIteration:
            # end of line
            return

        new_edge = boundary_lines.pop(idx)
        # switch round edge if in wrong direction
        if (new_edge[0] == edge[direction]) ^ bool(direction):
            new_edge = (new_edge[1], new_edge[0])

        # add to ordered list
        if direction == 1:
            ordered_line.append(new_edge)
        else:
            ordered_line.insert(0, new_edge)

        # continue along the edge
        connect_edge(new_edge, direction)

        for idx in new_edge_generator:
            print("found branching point. this is new")
    
    boundary_lines = copy(boundary_lines)
    ordered_lines = []
    while boundary_lines:
        ordered_line = [boundary_lines.pop(0)]
        connect_edge(ordered_line[0], 2)
        ordered_lines.append(ordered_line)

    return ordered_lines

def simplify_boundary_line(
    ordered_boundary_line : list[line_type], 
    point_type: Literal["centre", "vertex"] = "centre", 
    tol: float = 10., 
    method: Literal["simple", "vw", "vwp"] = "vwp"
) -> list[line_type]:
    if point_type == "vertex":
        boundary_points = [edge[0] for edge in ordered_boundary_line]
        boundary_points.append(ordered_boundary_line[-1][1])
    else:
        boundary_points = [
            ((edge[0][0] + edge[1][0]) / 2, (edge[0][1] + edge[1][1]) / 2) 
            for edge in ordered_boundary_line[1:-1]
        ]
        boundary_points.insert(0, ordered_boundary_line[0][0])
        boundary_points.append(ordered_boundary_line[-1][1])

    simplify = {
        "vw": simplify_coords_vw,
        "vwp": simplify_coords_vwp,
    }.get(method, simplify_coords)
    simple_points = simplify(np.ascontiguousarray(boundary_points), tol)
    return [(tuple(p1.tolist()), tuple(p2.tolist())) 
            for p1, p2 in zip(simple_points[:-1], simple_points[1:])]

def simpify_boundaries(grain_graph, point_type="centre", tol=10., method="vwp"):
    boundary_graph = nx.Graph()
    boundary_lines = []

    for _, _, bseg in grain_graph.edges.data('boundary'):
        for line in order_boundary_lines(bseg.boundary_lines):
            line_simple = simplify_boundary_line(
                line, point_type=point_type, tol=tol, method=method
            )
            for line_seg in line_simple:
                # Some lines end up the same if a grain with < 3 neighbours 
                # collapses to a single line
                try:
                    boundary_graph[line_seg[0]][line_seg[1]]["boundary"]
                    print(f"edge {line_seg[0]}, {line_seg[1]} already exists")
                except KeyError:
                    pass
                boundary_graph.add_edge(*line_seg, boundary=bseg)
                boundary_lines.append(line_seg)

    return boundary_graph, boundary_lines
