from pathlib import Path
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
from defdap.file_readers import OpticalDataLoader, MatplotlibLoader
from defdap import base
from defdap.plotting import MapPlot
from defdap.utils import report_progress
import numpy as np

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
        
        self.corr_val = None     # correlation value ------------not sure if i need this??

        self.ebsd_map = None                 # EBSD map linked to DIC map
        self.highlight_alpha = 0.6
        self.bse_scale = None                # size of pixels in pattern images
        self.bse_scale = None                # size of pixels in pattern images
        self.crop_dists = np.array(((0, 0), (0, 0)), dtype=int)
        self.file_name = None
        self.shape= None
        
    @report_progress("loading Optical data")
    def load_data(self, file_name, data_type=None):
        """Load DIC data from file.

        Parameters
        ----------
        file_name : pathlib.Path
            Name of file including extension.
        data_type : str,  {'Davis', 'OpenPIV'}
            Type of data file.

        """
        loader = MatplotlibLoader(file_name)

        # *dim are full size of data. shape (old *Dim) are size after cropping
        # *dim are full size of data. shape (old *Dim) are size after cropping
        self.optical = loader.load_image()
        self.shape = np.shape(self.optical)
        self.xdim = self.shape[1]     # size of map along x (from header)
        self.ydim = self.shape[0]     # size of map along y (from header)
        
        # write final status
        yield (
               f"(dimensions: {self.xdim} x {self.ydim} pixels) "
               )
               
               
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
        
    def plot_optical_image(self, **kwargs):
        """Uses the Plot class to display the optical image."""
        if self.optical is not None:
            # Pass the optical data into the create method of Plot
            plot_instance = MapPlot.create(
                calling_map=self,  # Pass the current instance of Map
                map_data=self.optical,  # Pass the loaded optical data
                **kwargs  # Additional parameters can be passed here
            )
            return plot_instance   

    '''        
    def plot_optical_image(self,map_data, **kwargs):
        """Uses the Plot class to display the optical image."""
        plot_params = {}
        plot_params.update(kwargs)
        return MapPlot.create(self, **plot_params)
    '''
     
  