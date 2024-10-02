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
        self.binning = 1
        
        self.plot_default = lambda *args, **kwargs: self.plot_map(map_name='optical',
            plot_gbs=True, *args, **kwargs)
        
        self.homog_map_name = 'optical'
        
                
        self.data.add_generator(
            'grains', self.find_grains, unit='', type='map', order=0,
            cropped=True
        )
        
        self.data.add_generator(
            'optical', self.load_data,  # This should point to your load_data method
            unit='', type='map', order=0,
            save=False,
            plot_params={'cmap': 'gray'}
        )

        
    @report_progress("loading Optical data")
    def load_data(self, file_name, data_type=None):
        """Load optical data from file.

        Parameters
        ----------
        file_name : pathlib.Path
            Name of file including extension.
        data_type : str,  not sure is relavent?

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
               
    def load_metadata_from_excel(self, file_path):
        """Load metadata from an Excel file and convert it into a list of dictionaries."""
        # Read the Excel file into a DataFrame
        df = pd.read_excel(file_path)

        # Convert each row in the DataFrame to a dictionary and store in a list
        self.metadata = df.to_dict(orient='records')

               
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
            # Initialise the grain map
            grains = -np.copy(self.data.grain_boundaries.image.astype(int))

            # List of points where no grain has been set yet
            points_left = grains == 0
            coords_buffer = np.zeros((points_left.size, 2), dtype=np.intp)
            total_points = points_left.sum()
            found_point = 0
            next_point = points_left.tobytes().find(b'\x01')

            # Start counter for grains
            grain_index = 1
            # Loop until all points (except boundaries) have been assigned
            # to a grain or ignored
            i = 0
            while found_point >= 0:
                # Flood fill first unknown point and return grain object
                seed = np.unravel_index(next_point, self.shape)

                grain = Grain(grain_index - 1, self, group_id)
                grain.data.point = flood_fill_dic(
                    (seed[1], seed[0]), grain_index, points_left,
                    grains, coords_buffer
                )
                coords_buffer = coords_buffer[len(grain.data.point):]

                if len(grain) < min_grain_size:
                    # if grain size less than minimum, ignore grain and set
                    # values in grain map to -2
                    for point in grain.data.point:
                        grains[point[1], point[0]] = -2
                else:
                    # add grain to list and increment grain index
                    grain_list.append(grain)
                    grain_index += 1

                # find next search point
                points_left_sub = points_left.reshape(-1)[next_point + 1:]
                found_point = points_left_sub.tobytes().find(b'\x01')
                next_point += found_point + 1

                # report progress
                i += 1
                if i == defaults['find_grain_report_freq']:
                    yield 1. - points_left_sub.sum() / total_points
                    i = 0

            # Now link grains to those in ebsd Map
            # Warp DIC grain map to EBSD frame
            warped_dic_grains = self.experiment.warp_image(
                grains.astype(float), self.frame, self.ebsd_map.frame,
                output_shape=self.ebsd_map.shape, order=0
            ).astype(int)
            for i, grain in enumerate(grain_list):
                # Find grain by masking the native ebsd grain image with
                # selected grain from the warped dic grain image. The modal
                # value is the EBSD grain label.
                mode_id, _ = mode(
                    self.ebsd_map.data.grains[warped_dic_grains == i+1],
                    keepdims=False
                )
                grain.ebsd_grain = self.ebsd_map[mode_id - 1]
                grain.ebsd_map = self.ebsd_map

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
