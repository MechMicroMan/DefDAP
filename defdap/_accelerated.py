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

from numba import njit
import numpy as np


@njit
def find_first(arr):
    for i in range(len(arr)):
        if arr[i]:
            return i


@njit
def flood_fill(seed, index, points_remaining, grains, boundary_x, boundary_y,
               added_coords):
    """Flood fill algorithm that uses the x and y boundary arrays to
    fill a connected area around the seed point. The points are inserted
    into a grain object and the grain map array is updated.

    Parameters
    ----------
    seed : tuple of 2 int
        Seed point x for flood fill
    index : int
        Value to fill in grain map
    points_remaining : numpy.ndarray
        Boolean map of the points that have not been assigned a grain yet

    Returns
    -------
    grain : defdap.ebsd.Grain
        New grain object with points added
    """
    x, y = seed
    grains[y, x] = index
    points_remaining[y, x] = False
    edge = [seed]
    added_coords[0] = seed
    npoints = 1

    while edge:
        x, y = edge.pop()
        moves = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        # get rid of any that go out of the map area
        if x <= 0:
            moves.pop(1)
        elif x > grains.shape[1] - 2:
            moves.pop(0)
        if y <= 0:
            moves.pop(-1)
        elif y > grains.shape[0] - 2:
            moves.pop(-2)

        for (s, t) in moves:
            if grains[t, s] > 0:
                continue

            add_point = False

            if t == y:
                # moving horizontally
                if s > x:
                    # moving right
                    add_point = not boundary_x[y, x]
                else:
                    # moving left
                    add_point = not boundary_x[t, s]
            else:
                # moving vertically
                if t > y:
                    # moving down
                    add_point = not boundary_y[y, x]
                else:
                    # moving up
                    add_point = not boundary_y[t, s]

            if add_point:
                added_coords[npoints] = s, t
                grains[t, s] = index
                points_remaining[t, s] = False
                npoints += 1
                edge.append((s, t))

    return added_coords[:npoints]


@njit
def flood_fill_dic(seed, index, points_remaining, grains, added_coords):
    """Flood fill algorithm that uses the combined x and y boundary array
    to fill a connected area around the seed point. The points are returned and
    the grain map array is updated.

    Parameters
    ----------
    seed : tuple of 2 int
        Seed point x for flood fill
    index : int
        Value to fill in grain map
    points_remaining : numpy.ndarray
        Boolean map of the points remaining to assign a grain yet
    grains : numpy.ndarray
    added_coords : numpy.ndarray
        Buffer for points in the grain

    Returns
    -------
    numpy.ndarray
        Flooded points (n, 2)

    """
    # add first point to the grain
    x, y = seed
    grains[y, x] = index
    points_remaining[y, x] = False
    edge = [seed]
    added_coords[0] = seed
    npoints = 1

    while edge:
        x, y = edge.pop()

        moves = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        # get rid of any that go out of the map area
        if x <= 0:
            moves.pop(1)
        elif x >= grains.shape[1] - 1:
            moves.pop(0)
        if y <= 0:
            moves.pop(-1)
        elif y >= grains.shape[0] - 1:
            moves.pop(-2)

        for (s, t) in moves:
            add_point = False

            if grains[t, s] == 0:
                add_point = True
                edge.append((s, t))

            elif grains[t, s] == -1 and (s > x or t > y):
                add_point = True

            if add_point:
                added_coords[npoints] = (s, t)
                grains[t, s] = index
                points_remaining[t, s] = False
                npoints += 1

    return added_coords[:npoints]
