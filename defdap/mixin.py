# Copyright 2024 Mechanics of Microstructures Group
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

class Map():  

    def set_crop(self, *, left=None, right=None, top=None, bottom=None,
                 update_homog_points=False):
        """Set a crop for the map.

        Parameters
        ----------
        left : int
            Distance to crop from left in pixels (formally `xMin`)
        right : int
            Distance to crop from right in pixels (formally `xMax`)
        top : int
            Distance to crop from top in pixels (formally `yMin`)
        bottom : int
            Distance to crop from bottom in pixels (formally `yMax`)
        update_homog_points : bool, optional
            If true, change homologous points to reflect crop.

        """
        # changes in homog points
        dx = 0
        dy = 0

        # update crop distances
        if left is not None:
            left = int(left)
            dx = self.crop_dists[0, 0] - left
            self.crop_dists[0, 0] = left
        if right is not None:
            self.crop_dists[0, 1] = int(right)
        if top is not None:
            top = int(top)
            dy = self.crop_dists[1, 0] - top
            self.crop_dists[1, 0] = top
        if bottom is not None:
            self.crop_dists[1, 1] = int(bottom)

        # update homogo points if required
        if update_homog_points and (dx != 0 or dy != 0):
            self.frame.update_homog_points(homog_idx=-1, delta=(dx, dy))

        # set new cropped dimensions
        x_dim = self.xdim - self.crop_dists[0, 0] - self.crop_dists[0, 1]
        y_dim = self.ydim - self.crop_dists[1, 0] - self.crop_dists[1, 1]
        self.shape = (y_dim, x_dim)

    def crop(self, map_data, binning=None):
        """ Crop given data using crop parameters stored in map
        i.e. cropped_data = Map.crop(Map.data_to_crop).

        Parameters
        ----------
        map_data : numpy.ndarray
            Bap data to crop.
        binning : int
            True if mapData is binned i.e. binned BSE pattern.
        """
        binning = 1 if binning is None else binning

        min_y = int(self.crop_dists[1, 0] * binning)
        max_y = int((self.ydim - self.crop_dists[1, 1]) * binning)

        min_x = int(self.crop_dists[0, 0] * binning)
        max_x = int((self.xdim - self.crop_dists[0, 1]) * binning)

        return map_data[..., min_y:max_y, min_x:max_x]
    
    def link_ebsd_map(self, ebsd_map, transform_type="affine", **kwargs):
        """Calculates the transformation required to align EBSD map to this map.

        Parameters
        ----------
        ebsd_map : defdap.ebsd.Map
            EBSD map object to link.
        transform_type : str, optional
            affine, piecewiseAffine or polynomial.
        kwargs
            All arguments are passed to `estimate` method of the transform.

        """
        self.ebsd_map = ebsd_map
        kwargs.update({'type': transform_type.lower()})
        self.experiment.link_frames(self.frame, ebsd_map.frame, kwargs)
        self.data.add_derivative(
            self.ebsd_map.data,
            lambda boundaries: BoundarySet.from_ebsd_boundaries(
                self, boundaries
            ),
            in_props={
                'type': 'boundaries'
            }
        )

    def check_ebsd_linked(self):
        """Check if an EBSD map has been linked.

        Returns
        ----------
        bool
            Returns True if EBSD map linked.

        Raises
        ----------
        Exception
            If EBSD map not linked.

        """
        if self.ebsd_map is None:
            raise Exception("No EBSD map linked.")
        return True

    def warp_to_dic_frame(self, map_data, **kwargs):
        """Warps a map to the this frame.

        Parameters
        ----------
        map_data : numpy.ndarray
            Data to warp.
        kwargs
            All other arguments passed to :func:`defdap.experiment.Experiment.warp_map`.

        Returns
        ----------
        numpy.ndarray
            Map (i.e. EBSD map data) warped to the this frame.

        """
        # Check a EBSD map is linked
        self.check_ebsd_linked()
        return self.experiment.warp_image(
            map_data, self.ebsd_map.frame, self.frame, output_shape=self.shape,
            **kwargs
        )
    

class Grain():
        
    @property
    def ref_ori(self):
        """Returns average grain orientation.

        Returns
        -------
        defdap.quat.Quat

        """
        return self.ebsd_grain.ref_ori

    @property
    def slip_traces(self):
        """Returns list of slip trace angles based on EBSD grain orientation.

        Returns
        -------
        list

        """
        return self.ebsd_grain.slip_traces

    def calc_slip_traces(self, slip_systems=None):
        """Calculates list of slip trace angles based on EBSD grain orientation.

        Parameters
        -------
        slip_systems : defdap.crystal.SlipSystem, optional

        """
        self.ebsd_grain.calc_slip_traces(slip_systems=slip_systems)

class BoundarySet(object):
    def __init__(self, dic_map, points, lines):
        self.dic_map = dic_map
        self.points = set(points)
        self.lines = lines

    @classmethod
    def from_ebsd_boundaries(cls, dic_map, ebsd_boundaries):
        if len(ebsd_boundaries.points) == 0:
            return cls(dic_map, [], [])

        points = dic_map.experiment.warp_points(
            ebsd_boundaries.image.astype(float),
            dic_map.ebsd_map.frame, dic_map.frame,
            output_shape=dic_map.shape
        )
        lines = dic_map.experiment.warp_lines(
            ebsd_boundaries.lines, dic_map.ebsd_map.frame, dic_map.frame
        )
        return cls(dic_map, points, lines)

    def _image(self, points):
        image = np.zeros(self.dic_map.shape, dtype=bool)
        image[tuple(zip(*points))[::-1]] = True
        return image

    @property
    def image(self):
        return self._image(self.points)