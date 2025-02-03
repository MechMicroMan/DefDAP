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

from skimage import measure

from defdap.file_readers import load_image
from defdap import base, mixin
from defdap.utils import Datastore, report_progress
from defdap import defaults
from defdap.inspector import GrainInspector

class Map(base.Map, mixin.Map):
    '''
    This class is for import and analysing optical image data
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
        EBSD map linked to optical map.
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
            image : numpy.ndarray
                2D image data
        Derived data:
            Grain list data to map data from all grains
    '''

    MAPNAME = 'optical'
     
    def __init__(self, *args, **kwargs):
        self.xdim = None        # size of map along x (from header)
        self.ydim = None        # size of map along y (from header)

        # Call base class constructor
        super(Map, self).__init__(*args, **kwargs)
        
        self.ebsd_map = None                 # EBSD map linked to optical map
        self.highlight_alpha = 0.6
        self.crop_dists = np.array(((0, 0), (0, 0)), dtype=int)
        
        self.plot_default = lambda *args, **kwargs: self.plot_map(
            map_name='image', plot_gbs=True, *args, **kwargs)
        
        self.homog_map_name = 'image'

        self.data.add_generator(
            'grains', self.find_grains, unit='', type='map', order=0,
            cropped=True
        )

        self.plot_default = lambda *args, **kwargs: self.plot_map(map_name='image',
            plot_gbs=True, *args, **kwargs
        )    
        
    @report_progress("loading optical data")
    def load_data(self, file_name, data_type=None):
        """Load optical data from file.

        Parameters
        ----------
        file_name : pathlib.Path
            Name of file including extension.

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
        """Returns the number of micrometers per pixel in the optical map.

        """
        if self.optical_scale is None:
            # raise ValueError("Map scale not set. Set with setScale()")
            return None

        return self.optical_scale 

    @report_progress("finding grains")
    def find_grains(self, algorithm=None, min_grain_size=10):
        """Finds grains in the optical map.

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
            # Warp EBSD grain map to optical frame
            grains = self.warp_to_dic_frame(
                self.ebsd_map.data.grains, order=0, preserve_range=True
            )

            # Find all unique values (these are the EBSD grain IDs in the optical area, sorted)
            ebsd_grain_ids = np.unique(grains)
            neg_vals = ebsd_grain_ids[ebsd_grain_ids <= 0]
            ebsd_grain_ids = ebsd_grain_ids[ebsd_grain_ids > 0]

            # Map the EBSD IDs to the optical IDs (keep the same mapping for values <= 0)
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

                # Assign EBSD grain ID to optical grain and increment grain list
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

    def grain_inspector(self, correction_angle=0):
        """Run the grain inspector interactive tool.
        Parameters
        ----------
        correction_angle: float
            Correction angle in degrees to subtract from measured angles to account
            for small rotation between optical and EBSD frames. Approximately the rotation
            component of affine transform.
        """
        GrainInspector(selected_map=self, correction_angle=correction_angle)


class Grain(base.Grain):
    """
    Class to encapsulate optical grain data and useful analysis and plotting
    methods.

    Attributes
    ----------
    dicMap : defdap.optical.Map
        Optical map this grain is a member of
    ownerMap : defdap.hrdic.Map
        Optical map this grain is a member of
    maxShearList : list
        List of maximum shear values for grain.
    ebsd_grain : defdap.ebsd.Grain
        EBSD grain ID that this optical grain corresponds to.
    ebsd_map : defdap.ebsd.Map
        EBSD map that this optical grain belongs to.
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

        self.plot_default = lambda *args, **kwargs: self.plot_map(
            plot_colour_bar=True, plot_scale_bar=True, *args, **kwargs
        )

    def plot_map(self, **kwargs):
        """Plot the image for an individual grain.

        Parameters
        ----------
        kwargs
            All arguments are passed to :func:`defdap.base.plot_grain_data`.

        Returns
        -------
        defdap.plotting.GrainPlot

        """
        # Set default plot parameters then update with any input
        plot_params = {
        }
        plot_params.update(kwargs)

        plot = self.plot_grain_data(grain_data=self.data.image, **plot_params)

        return plot