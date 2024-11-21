# Copyright 2023 Mechanics of Microstructures Group
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

from skimage import measure

from defdap.file_readers import load_image
from defdap import base
from defdap.utils import Datastore, report_progress
from defdap import defaults
from defdap.plotting import MapPlot
from defdap.inspector import GrainInspector

class Map(base.Map):
    '''
    This class is for import and analysiing optical image data
    such as polarised light images and darkfield images to link
    to EBSD data for slip trace analysis. (No RDR)
    
    Attributes
    ----------------------------
    xdim : int
        Size of map along x (from header).
    ydim : int
        Size of map along y (from header).
    shape : tuple
        Size of map (after cropping, like *Dim).
    corrVal : numpy.ndarray
        Correlation value.
    ebsd_map : defdap.ebsd.Map
        EBSD map linked to DIC map.
    highlight_alpha : float
        Alpha (transparency) of grain highlight.
    path : str
        File path.
    fname : str
        File name.
    crop_dists : numpy.ndarray
        Crop distances (default all zeros).

    data : defdap.utils.Datastore
        Must contain after loading data (maps):
            coordinate : numpy.ndarray
                X and Y coordinates
            pixel_value: fill in -------------------
        Derived data:
            Grain list data to map data from all grains
    '''

    MAPNAME = 'optical'
     
    def __init__(self, *args, **kwargs):
        self.xdim = None        # size of map along x (from header)
        self.ydim = None        # size of map along y (from header)

        # Call base class constructor
        super(Map, self).__init__(*args, **kwargs)
        
        self.ebsd_map = None                 # EBSD map linked to DIC map
        self.highlight_alpha = 0.6
        self.crop_dists = np.array(((0, 0), (0, 0)), dtype=int)
        
        self.plot_default = lambda *args, **kwargs: self.plot_map(
            map_name='image', plot_gbs=True, *args, **kwargs)
        
        self.homog_map_name = 'image'

        self.data.add_generator(
            'grains', self.find_grains, unit='', type='map', order=0,
            cropped=True
        )

    @report_progress("loading optical data")
    def load_data(self, file_name, data_type=None):
        """Load optical data from file.

        Parameters
        ----------
        file_name : pathlib.Path
            Name of file including extension.
        data_type : str,  not sure is relavent?

        """
        metadata_dict, loaded_data = load_image(file_name)
        self.data.update(loaded_data)
        
        self.shape = metadata_dict['shape']
        # *dim are full size of data. shape (old *Dim) are size after cropping
        ## TODO needs updating for all maps. cropped shape is stored as a 
        # tuple, why are this seperate values
        self.xdim = self.shape[1]
        self.ydim = self.shape[0]
        
        # write final status
        yield f"(dimensions: {self.xdim} x {self.ydim} pixels)"
               
    # def load_metadata_from_excel(self, file_path):
    #     """Load metadata from an Excel file and convert it into a list of dictionaries."""
    #     # Read the Excel file into a DataFrame
    #     df = pd.read_excel(file_path)

    #     # Convert each row in the DataFrame to a dictionary and store in a list
    #     self.metadata = df.to_dict(orient='records')

               
    def set_scale(self, scale):
        """Sets the scale of the map.

        Parameters
        ----------
        scale : float
            Length of pixel in original BSE image in micrometres.

        """
        self.optical_scale = scale

    @property
    def scale(self):
        """Returns the number of micrometers per pixel in the DIC map.

        """
        if self.optical_scale is None:
            # raise ValueError("Map scale not set. Set with setScale()")
            return None

        return self.optical_scale 
        
    def set_crop(self, *, left=None, right=None, top=None, bottom=None,
                 update_homog_points=False):
        """Set a crop for the DIC map.

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
        i.e. cropped_data = DicMap.crop(DicMap.data_to_crop).

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
        
    def plot_optical_map(self, **kwargs):
        """
        Plot a map with points coloured in IPF colouring,
        with respect to a given sample direction.

        Parameters
        ----------
        kwargs
            Other arguments passed to :func:`defdap.plotting.MapPlot.create`.

        Returns
        -------
        defdap.plotting.MapPlot
        """
        # Ensure map_data is available in the optical object
        if not hasattr(self, 'map_data'):
            raise ValueError("The optical object must have 'map_data' attribute set.")

        # Set default plot parameters then update with any input
        plot_params = {
            'calling_map': self,  # Assuming `self` is the calling map (instance of base.Map)
            'map_data': self.optical  # Access the map_data from the optical object
        }
        
        plot_params.update(kwargs)

        return MapPlot.create(**plot_params)

    def link_ebsd_map(self, ebsd_map, transform_type="affine", **kwargs):
        """Calculates the transformation required to align EBSD dataset to DIC.

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
        """Warps a map to the DIC frame.

        Parameters
        ----------
        map_data : numpy.ndarray
            Data to warp.
        kwargs
            All other arguments passed to :func:`defdap.experiment.Experiment.warp_map`.

        Returns
        ----------
        numpy.ndarray
            Map (i.e. EBSD map data) warped to the DIC frame.

        """
        # Check a EBSD map is linked
        self.check_ebsd_linked()
        return self.experiment.warp_image(
            map_data, self.ebsd_map.frame, self.frame, output_shape=self.shape,
            **kwargs
        )

    @report_progress("finding grains")
    def find_grains(self, algorithm=None, min_grain_size=10):
        """Finds grains in the DIC map.

        Parameters
        ----------
        algorithm : str {'warp', 'floodfill'}
            Use floodfill or warp algorithm.
        min_grain_size : int
            Minimum grain area in pixels for floodfill algorithm.
        """
        # Check a EBSD map is linked
        self.check_ebsd_linked()

        if algorithm is None:
            algorithm = defaults['hrdic_grain_finding_method']
        algorithm = algorithm.lower()

        grain_list = []
        group_id = Datastore.generate_id()

        if algorithm == 'warp':
            # Warp EBSD grain map to DIC frame
            grains = self.warp_to_dic_frame(
                self.ebsd_map.data.grains, order=0, preserve_range=True
            )

            # Find all unique values (these are the EBSD grain IDs in the DIC area, sorted)
            ebsd_grain_ids = np.unique(grains)
            neg_vals = ebsd_grain_ids[ebsd_grain_ids <= 0]
            ebsd_grain_ids = ebsd_grain_ids[ebsd_grain_ids > 0]

            # Map the EBSD IDs to the DIC IDs (keep the same mapping for values <= 0)
            old = np.concatenate((neg_vals, ebsd_grain_ids))
            new = np.concatenate((neg_vals, np.arange(1, len(ebsd_grain_ids) + 1)))
            index = np.digitize(grains.ravel(), old, right=True)
            grains = new[index].reshape(self.shape)
            grainprops = measure.regionprops(grains)
            props_dict = {prop.label: prop for prop in grainprops}

            for dic_grain_id, ebsd_grain_id in enumerate(ebsd_grain_ids):
                yield dic_grain_id / len(ebsd_grain_ids)

                # Make grain object
                grain = Grain(dic_grain_id, self, group_id)

                # Find (x,y) coordinates and corresponding max shears of grain
                coords = props_dict[dic_grain_id + 1].coords  # (y, x)
                grain.data.point = np.flip(coords, axis=1)  # (x, y)

                # Assign EBSD grain ID to DIC grain and increment grain list
                grain.ebsd_grain = self.ebsd_map[ebsd_grain_id - 1]
                grain.ebsd_map = self.ebsd_map
                grain_list.append(grain)

        elif algorithm == 'floodfill':
            raise NotImplementedError()

        else:
            raise ValueError(f"Unknown grain finding algorithm '{algorithm}'.")

        ## TODO: this will get duplicated if find grains called again
        self.data.add_derivative(
            grain_list[0].data, self.grain_data_to_map, pass_ref=True,
            in_props={
                'type': 'list'
            },
            out_props={
                'type': 'map'
            }
        )

        self._grains = grain_list
        return grains

    def grain_inspector(self):
        """Run the grain inspector interactive tool.

        """
        GrainInspector(selected_map=self)


class Grain(base.Grain):
    """
    Class to encapsulate DIC grain data and useful analysis and plotting
    methods.

    Attributes
    ----------
    dicMap : defdap.optical.Map
        DIC map this grain is a member of
    ownerMap : defdap.hrdic.Map
        DIC map this grain is a member of
    maxShearList : list
        List of maximum shear values for grain.
    ebsd_grain : defdap.ebsd.Grain
        EBSD grain ID that this DIC grain corresponds to.
    ebsd_map : defdap.ebsd.Map
        EBSD map that this DIC grain belongs to.
    points_list : numpy.ndarray
        Start and end points for lines drawn using defdap.inspector.GrainInspector.
    groups_list :
        Groups, angles and slip systems detected for
        lines drawn using defdap.inspector.GrainInspector.

    data : defdap.utils.Datastore
        Must contain after creating:
            point : list of tuples
                (x, y) in cropped map
        Generated data:

        Derived data:
            Map data to list data from the map the grain is part of

    """
    def __init__(self, grain_id, optical_map, group_id):
        # Call base class constructor
        super(Grain, self).__init__(grain_id, optical_map, group_id)

        self.optical_map = self.owner_map     # Optical map this grain is a member of
        self.ebsd_grain = None
        self.ebsd_map = None

        self.points_list = []            # Lines drawn for STA
        self.groups_list = []            # Unique angles drawn for STA

        # self.plot_default = lambda *args, **kwargs: self.plot_max_shear(
        #     plot_colour_bar=True, plot_scale_bar=True, plot_slip_traces=True,
        #     plot_slip_bands=True, *args, **kwargs
        # )
        
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
