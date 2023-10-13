# Copyright 2021 Mechanics of Microstructures Group
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
from matplotlib.pyplot import imread
import inspect

from skimage import transform as tf

from scipy.stats import mode
from scipy.ndimage import binary_dilation

import peakutils

from defdap.utils import Datastore
from defdap.file_readers import DICDataLoader
from defdap import base

from defdap import defaults
from defdap.plotting import MapPlot, GrainPlot
from defdap.inspector import GrainInspector
from defdap.utils import report_progress


class Map(base.Map):
    """
    Class to encapsulate DIC map data and useful analysis and plotting
    methods.

    Attributes
    ----------
    format : str
        Software name.
    version : str
        Software version.
    binning : int
        Sub-window size in pixels.
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
    bseScale : float
        Size of a pixel in the correlated images.
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
            displacement : numpy.ndarray
                X and Y displacements
        Generated data:
            f : numpy.ndarray
                Components of the deformation gradient (0=x, 1=y).
            e : numpy.ndarray
                Components of the green strain (0=x, 1=y).
            max_shear : numpy.ndarray
                Max shear component np.sqrt(((e11 - e22) / 2.)**2 + e12**2).
        Derived data:
            Grain list data to map data from all grains

    """
    def __init__(self, path, fname, dataType=None, **kwargs):
        """Initialise class and import DIC data from file.

        Parameters
        ----------
        path : str
            Path to file.
        fname : str
            Name of file including extension.
        dataType : str
            Type of data file.

        """
        # Call base class constructor
        super(Map, self).__init__(**kwargs)
        self.increment.add_map('dic', self)

        # Initialise variables
        self.format = None      # Software name
        self.version = None     # Software version
        self.binning = None     # Sub-window size in pixels
        self.xdim = None        # size of map along x (from header)
        self.ydim = None        # size of map along y (from header)

        self.corrVal = None     # correlation value

        self.ebsd_map = None                 # EBSD map linked to DIC map
        self.highlight_alpha = 0.6
        self.bse_scale = None                # size of pixels in pattern images
        self.path = path                    # file path
        self.fname = fname                  # file name
        self.crop_dists = np.array(((0, 0), (0, 0)), dtype=int)

        self.loadData(path, fname, dataType=dataType)

        ## TODO: cropping, have metadata to state if saved data is cropped, if
        ## not cropped then crop on accesss. Maybe mark cropped data as invalid
        ## if crop distances change

        # Deformation gradient
        f = np.gradient(self.data.displacement, self.binning, axis=(1, 2))
        f = np.array(f).transpose((1, 0, 2, 3))[:, ::-1]
        f[0, 0] += 1
        f[1, 1] += 1
        self.data.add(
            'f', f, unit='', type='map', order=2, default_component=(0, 0),
            plot_params={
                'plot_colour_bar': True,
                'clabel': 'Deformation gradient',
            }
        )

        # Green strain
        e = 0.5 * (np.einsum('ki...,kj...->ij...', f, f))
        e[0, 0] -= 0.5
        e[1, 1] -= 0.5
        self.data.add(
            'e', e, unit='', type='map', order=2, default_component=(0, 0),
            plot_params={
                'plot_colour_bar': True,
                'clabel': 'Green strain',
            }
        )

        # max shear component
        max_shear = np.sqrt(((e[0, 0] - e[1, 1]) / 2.) ** 2 + e[0, 1] ** 2)
        self.data.add(
            'max_shear', max_shear, unit='', type='map', order=0,
            plot_params={
                'plot_colour_bar': True,
                'clabel': 'Effective shear strain',
            }
        )

        # pattern image
        self.data.add_generator(
            'pattern', self.load_pattern, unit='', type='map', order=0,
            save=False,
            plot_params={
                'cmap': 'gray'
            }
        )

        self.data.add_generator(
            'grains', self.find_grains, unit='', type='map', order=0,
            cropped=True
        )

        self.plot_default = lambda *args, **kwargs: self.plot_map(map_name='max_shear',
            plotGBs=True, *args, **kwargs
        )
        self.homog_map_name = 'max_shear'

    @property
    def original_shape(self):
        return self.ydim, self.xdim

    @property
    def crystal_sym(self):
        return self.ebsd_map.crystal_sym

    @report_progress("loading HRDIC data")
    def loadData(self, fileDir, fileName, dataType=None):
        """Load DIC data from file.

        Parameters
        ----------
        fileDir : str
            Path to file.
        fileName : str
            Name of file including extension.
        dataType : str,  {'DavisText'}
            Type of data file.

        """
        dataLoader = DICDataLoader.get_loader(dataType)
        dataLoader.load(fileName, fileDir)

        metadataDict = dataLoader.loaded_metadata
        self.format = metadataDict['format']      # Software name
        self.version = metadataDict['version']    # Software version
        self.binning = metadataDict['binning']    # Sub-window width in pixels
        # *dim are full size of data. shape (old *Dim) are size after cropping
        # *dim are full size of data. shape (old *Dim) are size after cropping
        self.shape = metadataDict['shape']
        self.xdim = metadataDict['shape'][1]      # size of map along x (from header)
        self.ydim = metadataDict['shape'][0]      # size of map along y (from header)

        self.data.update(dataLoader.loaded_data)

        # write final status
        yield (f"Loaded {self.format} {self.version} data "
               f"(dimensions: {self.xdim} x {self.xdim} pixels, "
               f"sub-window size: {self.binning} x {self.binning} pixels)")

    def loadCorrValData(self, fileDir, fileName, dataType=None):
        """Load correlation value for DIC data

        Parameters
        ----------
        fileDir : str
            Path to file.
        fileName : str
            Name of file including extension.
        dataType : str,  {'DavisImage'}
            Type of data file.

        """
        dataType = "DavisImage" if dataType is None else dataType

        dataLoader = DICDataLoader()
        if dataType == "DavisImage":
            loaded_data = dataLoader.loadDavisImageData(fileName, fileDir)
        else:
            raise Exception("No loader found for this DIC data.")
            
        self.corrVal = loaded_data
        
        assert self.xdim == self.corrVal.shape[1], \
            "Dimensions of imported data and dic data do not match"
        assert self.ydim == self.corrVal.shape[0], \
            "Dimensions of imported data and dic data do not match"

    def retrieveName(self):
        """Gets the first name assigned to the a map, as a string

        """
        for fi in reversed(inspect.stack()):
            names = [key for key, val in fi.frame.f_locals.items() if val is self]
            if len(names) > 0:
                return names[0]

    def set_scale(self, scale):
        """Sets the scale of the map.

        Parameters
        ----------
        scale : float
            Length of pixel in original BSE image in micrometres.

        """
        self.bse_scale = scale

    @property
    def scale(self):
        """Returns the number of micrometers per pixel in the DIC map.

        """
        if self.bse_scale is None:
            # raise ValueError("Map scale not set. Set with setScale()")
            return None

        return self.bse_scale * self.binning

    def print_stats_table(self, percentiles, components):
        """Print out a statistics table for a DIC map

        Parameters
        ----------
        percentiles : list of float
            list of percentiles to print i.e. 0, 50, 99.
        components : list of str
            list of map components to print i.e. e, f, max_shear.

        """

        # Check that components are valid
        if not set(components).issubset(self.data):
            strFormat = '{}, ' * (len(self.data) - 1) + '{}'
            raise Exception("Components must be: " + strFormat.format(*self.data))

        # Print map info
        print('\033[1m', end=''),  # START BOLD
        print("{0} (dimensions: {1} x {2} pixels, sub-window size: {3} "
              "x {3} pixels, number of points: {4})\n".format(
            self.retrieveName(), self.x_dim, self.y_dim,
            self.binning, self.x_dim * self.y_dim
        ))

        # Print header
        strFormat = '{:10} ' + '{:12}' * len(percentiles)
        print(strFormat.format('Component', *percentiles))
        print('\033[0m', end='')  # END BOLD

        # Print table
        strFormat = '{:10} ' + '{:12.4f}' * len(percentiles)
        for c in components:
            # Iterate over tensor components (i.e. e11, e22, e12)
            for i in np.ndindex(self.data[c].shape[:len(np.shape(self.data[c]))-2]):
                per = [np.nanpercentile(self.data[c][i], p) for p in percentiles]
                print(strFormat.format(c+''.join([str(t+1) for t in i]), *per))

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
        xDim = self.xdim - self.crop_dists[0, 0] - self.crop_dists[0, 1]
        yDim = self.ydim - self.crop_dists[1, 0] - self.crop_dists[1, 1]
        self.shape = (yDim, xDim)

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

        minY = int(self.crop_dists[1, 0] * binning)
        maxY = int((self.ydim - self.crop_dists[1, 1]) * binning)

        minX = int(self.crop_dists[0, 0] * binning)
        maxX = int((self.xdim - self.crop_dists[0, 1]) * binning)

        return map_data[..., minY:maxY, minX:maxX]

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

    # TODO: fix component stuff
    def generateThresholdMask(self, mask, dilation=0, preview=True):
        """
        Generate a dilated mask, based on a boolean array and previews the appication of
        this mask to the max shear map.

        Parameters
        ----------
        mask: numpy.array(bool)
            A boolean array where points to be removed are True
        dilation: int, optional
            Number of pixels to dilate the mask by. Useful to remove anomalous points
            around masked values. No dilation applied if not specified.
        preview: bool
            If true, show the mask and preview the masked effective shear strain map.

        Examples
        ----------
        To remove data points in dicMap where `max_shear` is above 0.8, use:

        >>> mask = dicMap.data.max_shear > 0.8

        To remove data points in dicMap where e11 is above 1 or less than -1, use:

        >>> mask = (dicMap.data.e[0, 0] > 1) | (dicMap.data.e[0, 0] < -1)

        To remove data points in dicMap where corrVal is less than 0.4, use:

        >>> mask = dicMap.corrVal < 0.4

        Note: correlation value data needs to be loaded seperately from the DIC map,
        see :func:`defdap.hrdic.loadCorrValData`

        """
        self.mask = mask

        if dilation != 0:
            self.mask = binary_dilation(self.mask, iterations=dilation)

        numRemoved = np.sum(self.mask)
        numTotal = self.xdim * self.ydim
        numRemovedCrop = np.sum(self.crop(self.mask))
        numTotalCrop = self.x_dim * self.y_dim

        print('Filtering will remove {0} \ {1} ({2:.3f} %) datapoints in map'
              .format(numRemoved, numTotal, (numRemoved / numTotal) * 100))
        print(
            'Filtering will remove {0} \ {1} ({2:.3f} %) datapoints in cropped map'
            .format(numRemovedCrop, numTotalCrop,
                    (numRemovedCrop / numTotalCrop * 100)))

        if preview == True:
            plot1 = MapPlot.create(self, self.crop(self.mask), cmap='binary')
            plot1.set_title('Removed datapoints in black')
            plot2 = MapPlot.create(self,
                                   self.crop(
                                       np.where(self.mask == True, np.nan,
                                                self.data.max_shear)),
                                   plot_colour_bar='True',
                                   clabel="Effective shear strain")
            plot2.set_title('Effective shear strain preview')
        print(
            'Use applyThresholdMask function to apply this filtering to data')

    def applyThresholdMask(self):
        """ Apply mask to all DIC map data by setting masked values to nan.

        """
        for comp in ('max_shear',
                     'e11', 'e12', 'e22',
                     'f11', 'f12', 'f21', 'e22',
                     'x_map', 'y_map'):
            # self.data[comp] = np.where(self.mask == True, np.nan, self.data[comp])
            self.data[comp][self.mask] = np.nan

    def set_pattern(self, img_path, window_size):
        """Set the path to the image of the pattern.

        Parameters
        ----------
        path : str
            Path to image.
        window_size : int
            Size of pixel in pattern image relative to pixel size of DIC data
            i.e 1 means they  are the same size and 2 means the pixels in
            the pattern are half the size of the dic data.

        """
        path = self.path + img_path
        self.data['pattern', 'path'] = path
        self.data['pattern', 'binning'] = window_size

    def load_pattern(self):
        print('Loading img')
        path = self.data.get_metadata('pattern', 'path')
        binning = self.data.get_metadata('pattern', 'binning', 1)
        if path is None:
            raise FileNotFoundError("First set path to pattern image.")

        img = imread(path)
        exp_shape = tuple(v * binning for v in self.original_shape)
        if img.shape != exp_shape:
            raise ValueError(
                f'Incorrect size of pattern image. For binning of {binning} '
                f'expected size {exp_shape[::-1]} but got {img.shape[::-1]}'
            )
        return img

    def plotGrainAvMaxShear(self, **kwargs):
        """Plot grain map with grains filled with average value of max shear.
        This uses the max shear values stored in grain objects, to plot other data
        use :func:`~defdap.hrdic.Map.plotGrainAv`.

        Parameters
        ----------
        kwargs
            All arguments are passed to :func:`defdap.base.Map.plot_grain_data_map`.

        """
        # Set default plot parameters then update with any input
        plot_params = {
            'clabel': "Effective shear strain"
        }
        plot_params.update(kwargs)

        plot = self.plot_grain_data_map(
            map_data=self.data.max_shear, **plot_params
        )

        return plot

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

            for dic_grain_id, ebsd_grain_id in enumerate(ebsd_grain_ids):
                yield dic_grain_id / len(ebsd_grain_ids)

                # Make grain object
                grain = Grain(dic_grain_id, self, group_id)

                # Find (x,y) coordinates and corresponding max shears of grain
                coords = np.argwhere(grains == dic_grain_id + 1)  # (y,x)
                grain.data.point = [(x, y) for y, x in coords]

                # Assign EBSD grain ID to DIC grain and increment grain list
                grain.ebsd_grain = self.ebsd_map[ebsd_grain_id - 1]
                grain.ebsd_map = self.ebsd_map
                grain_list.append(grain)

        elif algorithm == 'floodfill':
            # Initialise the grain map
            grains = -np.copy(self.data.grain_boundaries.image.astype(int))

            # List of points where no grain has been set yet
            points_left = grains == 0
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
                grain = self.flood_fill(
                    (seed[1], seed[0]), grain_index, points_left, grains, group_id
                )

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

    def flood_fill(self, seed, index, points_left, grains, group_id):
        """Flood fill algorithm that uses the combined x and y boundary array
        to fill a connected area around the seed point. The points are inserted
        into a grain object and the grain map array is updated.

        Parameters
        ----------
        seed : tuple of 2 int
            Seed point x for flood fill
        index : int
            Value to fill in grain map
        points_left : numpy.ndarray
            Boolean map of the points that have not been assigned a grain yet

        Returns
        -------
        grain : defdap.hrdic.Grain
            New grain object with points added

        """
        # create new grain
        grain = Grain(index - 1, self, group_id)

        # add first point to the grain
        x, y = seed
        grain.addPoint(seed)
        grains[y, x] = index
        points_left[y, x] = False
        edge = [seed]

        while edge:
            x, y = edge.pop(0)

            moves = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            # get rid of any that go out of the map area
            if x <= 0:
                moves.pop(1)
            elif x >= self.shape[1] - 1:
                moves.pop(0)
            if y <= 0:
                moves.pop(-1)
            elif y >= self.shape[0] - 1:
                moves.pop(-2)

            for (s, t) in moves:
                add_point = False

                if grains[t, s] == 0:
                    add_point = True
                    edge.append((s, t))

                elif grains[t, s] == -1 and (s > x or t > y):
                    add_point = True

                if add_point:
                    grain.addPoint((s, t))
                    grains[t, s] = index
                    points_left[t, s] = False

        return grain

    def grain_inspector(self, vmax=0.1, corrAngle=0, RDRlength=3):
        """Run the grain inspector interactive tool.

        Parameters
        ----------
        vmax : float
            Maximum value of the colour map.
        corrAngle: float
            Correction angle in degrees to subtract from measured angles to account
            for small rotation between DIC and EBSD frames. Approximately the rotation
            component of affine transform.
        RDRlength: int
            Length of lines perpendicular to slip trace used to calculate RDR.

        """
        GrainInspector(selected_dic_map=self, vmax=vmax, correction_angle=corrAngle,
                       rdr_line_length=RDRlength)


class Grain(base.Grain):
    """
    Class to encapsulate DIC grain data and useful analysis and plotting
    methods.

    Attributes
    ----------
    dicMap : defdap.hrdic.Map
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
    def __init__(self, grainID, dicMap, group_id):
        # Call base class constructor
        super(Grain, self).__init__(grainID, dicMap, group_id)

        self.dicMap = self.owner_map     # DIC map this grain is a member of
        self.ebsd_grain = None
        self.ebsd_map = None

        self.points_list = []            # Lines drawn for STA
        self.groups_list = []            # Unique angles drawn for STA

        self.plot_default = lambda *args, **kwargs: self.plot_max_shear(
            plot_colour_bar=True, plot_scale_bar=True, plot_slip_traces=True,
            plot_slip_bands=True, *args, **kwargs
        )

    def plot_max_shear(self, **kwargs):
        """Plot a maximum shear map for a grain.

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
            'plot_colour_bar': True,
            'clabel': "Effective shear strain"
        }
        plot_params.update(kwargs)

        plot = self.plot_grain_data(grain_data=self.data.max_shear, **plot_params)

        return plot

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

    def calc_slip_traces(self, slipSystems=None):
        """Calculates list of slip trace angles based on EBSD grain orientation.

        Parameters
        -------
        slipSystems : defdap.crystal.SlipSystem, optional

        """
        self.ebsd_grain.calc_slip_traces(slip_systems=slipSystems)

    def calcSlipBands(self, grainMapData, thres=None, min_dist=None):
        """Use Radon transform to detect slip band angles.

        Parameters
        ----------
        grainMapData : numpy.ndarray
            Data to find bands in.
        thres : float, optional
            Normalised threshold for peaks.
        min_dist : int, optional
            Minimum angle between bands.

        Returns
        ----------
        list(float)
            Detected slip band angles

        """
        if thres is None:
            thres = 0.3
        if min_dist is None:
            min_dist = 30
        grainMapData = np.nan_to_num(grainMapData)

        if grainMapData.min() < 0:
            print("Negative values in data, taking absolute value.")
            # grainMapData = grainMapData**2
            grainMapData = np.abs(grainMapData)
        # array to hold shape / support of grain
        suppGMD = np.zeros(grainMapData.shape)
        suppGMD[grainMapData!=0]=1
        sin_map = tf.radon(grainMapData, circle=False)
        #profile = np.max(sin_map, axis=0) # old method
        supp_map = tf.radon(suppGMD, circle=False)
        supp_1 = np.zeros(supp_map.shape)
        supp_1[supp_map>0]=1
        # minimum diameter of grain
        mindiam = np.min(np.sum(supp_1, axis=0), axis=0)
        crop_map = np.zeros(sin_map.shape)
        # only consider radon rays that cut grain with mindiam*2/3 or more, 
        # and scale by length of the cut
        selection = supp_map > mindiam * 2 / 3
        crop_map[selection] = sin_map[selection] / supp_map[selection] 
        supp_crop = np.zeros(crop_map.shape)
        supp_crop[crop_map>0] = 1
        # raise to power to accentuate local peaks
        profile = np.sum(crop_map**4, axis=0) / np.sum(supp_crop, axis=0)

        x = np.arange(180)

        indexes = peakutils.indexes(profile, thres=thres, min_dist=min_dist)
        peaks = x[indexes]
        # peaks = peakutils.interpolate(x, profile, ind=indexes)
        print("Number of bands detected: {:}".format(len(peaks)))

        slipBandAngles = peaks
        slipBandAngles = slipBandAngles * np.pi / 180
        return slipBandAngles


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
