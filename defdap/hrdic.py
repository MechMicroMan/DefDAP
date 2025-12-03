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

from pathlib import Path

import numpy as np
from matplotlib.pyplot import imread
import inspect

from skimage import transform as tf
from skimage import measure

from scipy.stats import mode
from scipy.ndimage import binary_dilation

import peakutils

from defdap._accelerated import flood_fill_dic
from defdap.utils import Datastore
from defdap.file_readers import DICDataLoader, DavisLoader
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
    MAPNAME = 'hrdic'

    def __init__(self, *args, **kwargs):
        """Initialise class and import DIC data from file.

        Parameters
        ----------
        *args, **kwarg
            Passed to base constructor

        """
        # Initialise variables
        self.format = None      # Software name
        self.version = None     # Software version
        self.binning = None     # Sub-window size in pixels
        self.xdim = None        # size of map along x (from header)
        self.ydim = None        # size of map along y (from header)

        # Call base class constructor
        super(Map, self).__init__(*args, **kwargs)

        self.corr_val = None     # correlation value

        self.ebsd_map = None                 # EBSD map linked to DIC map
        self.highlight_alpha = 0.6
        self.bse_scale = None                # size of pixels in pattern images
        self.bse_scale = None                # size of pixels in pattern images
        self.crop_dists = np.array(((0, 0), (0, 0)), dtype=int)

        self.data.add_generator(
            'mask', self.calc_mask, unit='', type='map', order=0,
            cropped=True, apply_mask=False
        )

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
            },
            apply_mask=True
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
            },
            apply_mask=True
        )
        # max shear component
        max_shear = np.sqrt(((e[0, 0] - e[1, 1]) / 2.) ** 2 + e[0, 1] ** 2)
        self.data.add(
            'max_shear', max_shear, unit='', type='map', order=0,
            plot_params={
                'plot_colour_bar': True,
                'clabel': 'Effective shear strain',
            },
            apply_mask=True
        )
        # pattern image
        self.data.add_generator(
            'pattern', self.load_pattern, unit='', type='map', order=0,
            save=False, apply_mask=False,
            plot_params={
                'cmap': 'gray'
            }
        )
        self.data.add_generator(
            'grains', self.find_grains, unit='', type='map', order=0,
            cropped=True, apply_mask=False
        )

        self.plot_default = lambda *args, **kwargs: self.plot_map(
            map_name='max_shear', plot_gbs=True, *args, **kwargs
        )
        self.homog_map_name = 'max_shear'

    @property
    def original_shape(self):
        return self.ydim, self.xdim

    @property
    def crystal_sym(self):
        return self.ebsd_map.crystal_sym

    @report_progress("loading HRDIC data")
    def load_data(self, file_name, data_type=None):
        """Load DIC data from file.

        Parameters
        ----------
        file_name : pathlib.Path
            Name of file including extension.
        data_type : str,  {'Davis', 'OpenPIV'}
            Type of data file.

        """
        data_loader = DICDataLoader.get_loader(data_type)
        data_loader.load(file_name)

        metadata_dict = data_loader.loaded_metadata
        self.format = metadata_dict['format']      # Software name
        self.version = metadata_dict['version']    # Software version
        self.binning = metadata_dict['binning']    # Sub-window width in pixels
        # *dim are full size of data. shape (old *Dim) are size after cropping
        # *dim are full size of data. shape (old *Dim) are size after cropping
        self.shape = metadata_dict['shape']
        self.xdim = metadata_dict['shape'][1]      # size of map along x (from header)
        self.ydim = metadata_dict['shape'][0]      # size of map along y (from header)

        self.data.update(data_loader.loaded_data)

        # write final status
        yield (f"Loaded {self.format} {self.version} data "
               f"(dimensions: {self.xdim} x {self.ydim} pixels, "
               f"sub-window size: {self.binning} x {self.binning} pixels)")

    def load_corr_val_data(self, file_name, data_type=None):
        """Load correlation value for DIC data

        Parameters
        ----------
        file_name : pathlib.Path or str
            Path to file.
        data_type : str,  {'DavisImage'}
            Type of data file.

        """
        data_type = "DavisImage" if data_type is None else data_type

        data_loader = DavisLoader()
        if data_type == "DavisImage":
            loaded_data = data_loader.load_davis_image_data(Path(file_name))
        else:
            raise Exception("No loader found for this DIC data.")
            
        self.corr_val = loaded_data
        
        assert self.xdim == self.corr_val.shape[1], \
            "Dimensions of imported data and dic data do not match"
        assert self.ydim == self.corr_val.shape[0], \
            "Dimensions of imported data and dic data do not match"

    def retrieve_name(self):
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
            str_format = '{}, ' * (len(self.data) - 1) + '{}'
            raise Exception("Components must be: " + str_format.format(*self.data))

        # Print map info
        print('\033[1m', end=''),  # START BOLD
        print("{0} (dimensions: {1} x {2} pixels, sub-window size: {3} "
              "x {3} pixels, number of points: {4})\n".format(
            self.retrieve_name(), self.x_dim, self.y_dim,
            self.binning, self.x_dim * self.y_dim
        ))

        # Print header
        str_format = '{:10} ' + '{:12}' * len(percentiles)
        print(str_format.format('Component', *percentiles))
        print('\033[0m', end='')  # END BOLD

        # Print table
        str_format = '{:10} ' + '{:12.4f}' * len(percentiles)
        for c in components:
            # Iterate over tensor components (i.e. e11, e22, e12)
            for i in np.ndindex(self.data[c].shape[:len(np.shape(self.data[c]))-2]):
                per = [np.nanpercentile(self.data[c][i], p) for p in percentiles]
                print(str_format.format(c+''.join([str(t+1) for t in i]), *per))

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

    def calc_mask(self, mask=None, dilation=0):
        """
        Generate a dilated mask, based on a boolean array.

        Parameters
        ----------
        mask: numpy.array(bool) or None
            A boolean array where points to be removed are True. Set to None to disable masking.
        dilation: int, optional
            Number of pixels to dilate the mask by. Useful to remove anomalous points
            around masked values. No dilation applied if not specified.

        Examples
        ----------
        
        To disable masking:

        >>> mask = None
               
        To remove data points in dic_map where `max_shear` is above 0.8, use:
        
        >>> mask = dic_map.data.max_shear > 0.8

        To remove data points in dic_map where e11 is above 1 or less than -1, use:

        >>> mask = (dic_map.data.e[0, 0] > 1) | (dic_map.data.e[0, 0] < -1)

        To remove data points in dic_map where corrVal is less than 0.4, use:

        >>> mask = dic_map.corr_val < 0.4

        Note: correlation value data needs to be loaded seperately from the DIC map,
        see :func:`defdap.hrdic.load_corr_val_data`

        """
        if mask is None:
            self.data.mask = None
            return mask
        
        if not isinstance(mask, np.ndarray) or mask.shape != self.shape:
            raise ValueError('The mask must be a numpy array the same shape as '
                             'the cropped map.')

        if dilation != 0:
            mask = binary_dilation(mask, iterations=dilation)

        num_removed = np.sum(mask)
        num_total = self.shape[0] * self.shape[1]
        frac_removed = num_removed / num_total * 100
        print(f'Masking will mask {num_removed} out of {num_total} '
              f'({frac_removed:.2f} %) datapoints in cropped map.')
        
        self.data.mask = mask

        return mask

    def mask(self, map_data):
        """ Values set to False in mask will be set to nan in map.
        """
        if self.data.mask is None:
            return map_data
        else:
            return np.ma.array(map_data, 
                            mask=np.broadcast_to(self.data.mask, np.shape(map_data)))

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
        path = self.file_name.parent / img_path
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

    def plot_grain_av_max_shear(self, **kwargs):
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

    def grain_inspector(self, vmax=0.1, correction_angle=0, rdr_line_length=3):
        """Run the grain inspector interactive tool.

        Parameters
        ----------
        vmax : float
            Maximum value of the colour map.
        correction_angle: float
            Correction angle in degrees to subtract from measured angles to account
            for small rotation between DIC and EBSD frames. Approximately the rotation
            component of affine transform.
        rdr_line_length: int
            Length of lines perpendicular to slip trace used to calculate RDR.

        """
        GrainInspector(selected_dic_map=self, vmax=vmax, correction_angle=correction_angle,
                       rdr_line_length=rdr_line_length)


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
    def __init__(self, grain_id, dicMap, group_id):
        # Call base class constructor
        super(Grain, self).__init__(grain_id, dicMap, group_id)

        self.dic_map = self.owner_map     # DIC map this grain is a member of
        self.ebsd_grain = None
        self.ebsd_map = None

        self.points_list = []            # Lines drawn for STA
        self.groups_list = []            # Unique angles drawn for STA

        self.plot_default = lambda *args, **kwargs: self.plot_map(
            'max_shear', plot_colour_bar=True, plot_scale_bar=True, 
            plot_slip_traces=True, plot_slip_bands=True, *args, **kwargs
        )

    @property
    def ref_ori(self):
        """Returns average grain orientation.

        Returns
        -------
        defdap.quat.Quat

        """
        return self.ebsd_grain.ref_ori
    
    @property
    def phase(self):
        """Returns the phase of the linked ebsd grain.

        Returns
        -------
        defdap.crystal.Phase

        """
        return self.ebsd_grain.phase

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

    def calc_slip_bands(self, grain_map_data, thres=None, min_dist=None):
        """Use Radon transform to detect slip band angles.

        Parameters
        ----------
        grain_map_data : numpy.ndarray
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
        grain_map_data = np.nan_to_num(grain_map_data)

        if grain_map_data.min() < 0:
            print("Negative values in data, taking absolute value.")
            # grain_map_data = grain_map_data**2
            grain_map_data = np.abs(grain_map_data)
        # array to hold shape / support of grain
        supp_gmd = np.zeros(grain_map_data.shape)
        supp_gmd[grain_map_data != 0]=1
        sin_map = tf.radon(grain_map_data, circle=False)
        #profile = np.max(sin_map, axis=0) # old method
        supp_map = tf.radon(supp_gmd, circle=False)
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

        slip_band_angles = peaks
        slip_band_angles = slip_band_angles * np.pi / 180
        return slip_band_angles


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
