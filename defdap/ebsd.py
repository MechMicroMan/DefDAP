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

import numpy as np
from skimage import morphology as mph
import networkx as nx

import copy
from warnings import warn

from defdap.utils import Datastore
from defdap.file_readers import EBSDDataLoader
from defdap.file_writers import EBSDDataWriter
from defdap.quat import Quat
from defdap import base
from defdap._accelerated import flood_fill

from defdap import defaults
from defdap.plotting import MapPlot
from defdap.utils import report_progress


class Map(base.Map):
    """
    Class to encapsulate an EBSD map and useful analysis and plotting
    methods.

    Attributes
    ----------
    step_size : float
        Step size in micron.
    phases : list of defdap.crystal.Phase
        List of phases.
    mis_ori : numpy.ndarray
        Map of misorientation.
    mis_ori_axis : list of numpy.ndarray
        Map of misorientation axis components.
    origin : tuple(int)
        Map origin (x, y). Used by linker class where origin is a
        homologue point of the maps.

    data : defdap.utils.Datastore
        Must contain after loading data (maps):
            phase : numpy.ndarray
                1-based, 0 is non-indexed points
            euler_angle : numpy.ndarray
                stored as (3, y_dim, x_dim) in radians
        Generated data:
            orientation : numpy.ndarray of defdap.quat.Quat
                Quaterion for each point of map. Shape (y_dim, x_dim).
            grain_boundaries : BoundarySet
            phase_boundaries : BoundarySet
            grains : numpy.ndarray of int
                Map of grains. Grain numbers start at 1 here but everywhere else
                grainID starts at 0. Regions that are smaller than the minimum
                grain size are given value -2. Remnant boundary points are -1.
            KAM : numpy.ndarray
                Kernal average misorientaion map.
            GND : numpy.ndarray
                GND scalar map.
            Nye_tensor : numpy.ndarray
                3x3 Nye tensor at each point.
        Derived data:
            grain_data_to_map : numpy.ndarray
                Grain list data to map data from all grains

    """
    MAPNAME = 'ebsd'

    def __init__(self, *args, **kwargs):
        """
        Initialise class and load EBSD data.

        Parameters
        ----------
        *args, **kwarg
            Passed to base constructor

        """
        # Initialise variables
        self.step_size = None
        self.phases = []

        # Call base class constructor
        super(Map, self).__init__(*args, **kwargs)

        self.mis_ori = None
        self.mis_ori_axis = None
        self.origin = (0, 0)

        # Phase used for the maps crystal structure and c_over_a. So old
        # functions still work for the 'main' phase in the map. 0-based
        self.primary_phase_id = 0

        # Use euler map for defining homologous points
        self.plot_default = self.plot_euler_map
        self.homog_map_name = 'band_contrast'
        self.highlight_alpha = 1

        self.data.add_generator(
            'orientation', self.calc_quat_array, unit='', type='map',
            order=0, default_component='IPF_x',
        )
        self.data.add_generator(
            ('phase_boundaries', 'grain_boundaries'), self.find_boundaries,
            type='boundaries',
        )
        self.data.add_generator(
            'grains', self.find_grains, unit='', type='map', order=0
        )
        self.data.add_generator(
            'KAM', self.calc_kam, unit='rad', type='map', order=0,
            plot_params={
                'plot_colour_bar': True,
                'clabel': 'KAM',
            }
        )
        self.data.add_generator(
            ('GND', 'Nye_tensor'), self.calc_nye,
            unit='', type='map',
            metadatas=({
                'order': 0,
                'plot_params': {
                    'plot_colour_bar': True,
                    'clabel': 'GND content',
                }
            }, {
                'order': 2,
                'save': False,
                'default_component': (0, 0),
                'plot_params': {
                    'plot_colour_bar': True,
                    'clabel': 'Nye tensor',
                }
            })
        )

    @report_progress("loading EBSD data")
    def load_data(self, file_name, data_type=None):
        """Load in EBSD data from file.

        Parameters
        ----------
        file_name : pathlib.Path
            Path to EBSD file
        data_type : str, {'OxfordBinary', 'OxfordText', 'EdaxAng', 'PythonDict'}
            Format of EBSD data file.

        """
        data_loader = EBSDDataLoader.get_loader(data_type, file_name)
        data_loader.load(file_name)

        metadata_dict = data_loader.loaded_metadata
        self.shape = metadata_dict['shape']
        self.step_size = metadata_dict['step_size']
        self.phases = metadata_dict['phases']

        self.data.update(data_loader.loaded_data)

        # write final status
        yield (f"Loaded EBSD data (dimensions: {self.x_dim} x {self.y_dim} "
               f"pixels, step size: {self.step_size} um)")

    def save(self, file_name, data_type=None, file_dir=""):
        """Save EBSD map to file.

        Parameters
        ----------
        file_name : str
            Name of file to save to, it must not already exist.
        data_type : str, {'OxfordText'}
            Format of EBSD data file to save.
        file_dir : str
            Directory to save the file to.

        """
        data_writer = EBSDDataWriter.get_writer(data_type)

        data_writer.metadata['shape'] = self.shape
        data_writer.metadata['step_size'] = self.step_size
        data_writer.metadata['phases'] = self.phases

        data_writer.data['phase'] = self.data.phase
        data_writer.data['quat'] = self.data.orientation
        data_writer.data['band_contrast'] = self.data.band_contrast

        data_writer.write(file_name, file_dir=file_dir)

    @property
    def crystal_sym(self):
        """Crystal symmetry of the primary phase.

        Returns
        -------
        str
            Crystal symmetry

        """
        return self.primary_phase.crystal_structure.name

    @property
    def c_over_a(self):
        """C over A ratio of the primary phase

        Returns
        -------
        float or None
            C over A ratio if hexagonal crystal structure otherwise None

        """
        return self.primary_phase.c_over_a

    @property
    def num_phases(self):
        return len(self.phases) or None

    @property
    def primary_phase(self):
        """Primary phase of the EBSD map.

        Returns
        -------
        defdap.crystal.Phase
            Primary phase

        """
        return self.phases[self.primary_phase_id]

    @property
    def scale(self):
        return self.step_size

    @report_progress("rotating EBSD data")
    def rotate_data(self):
        """Rotate map by 180 degrees and transform quats accordingly.

        """

        self.data.euler_angle = self.data.euler_angle[:, ::-1, ::-1]
        self.data.band_contrast = self.data.band_contrast[::-1, ::-1]
        self.data.band_slope = self.data.band_slope[::-1, ::-1]
        self.data.phase = self.data.phase[::-1, ::-1]
        self.calc_quat_array()

        # Rotation from old coord system to new
        transform_quat = Quat.from_axis_angle(np.array([0, 0, 1]), np.pi).conjugate

        # Perform vectorised multiplication
        quats = Quat.multiply_many_quats(self.data.orientation.flatten(), transform_quat)
        self.data.orientation = np.array(quats).reshape(self.shape)

        yield 1.

    def calc_euler_colour(self, map_data, phases=None, bg_colour=None):
        if phases is None:
            phases = self.phases
            phase_ids = range(len(phases))
        else:
            phase_ids = phases
            phases = [self.phases[i] for i in phase_ids]

        if bg_colour is None:
            bg_colour = np.array([0., 0., 0.])

        map_colours = np.tile(bg_colour, self.shape + (1,))

        for phase, phase_id in zip(phases, phase_ids):
            if phase.crystal_structure.name == 'cubic':
                norm = np.array([2 * np.pi, np.pi / 2, np.pi / 2])
            elif phase.crystal_structure.name == 'hexagonal':
                norm = np.array([np.pi, np.pi, np.pi / 3])
            else:
                ValueError("Only hexagonal and cubic symGroup supported")

            # Apply normalisation for each phase
            phase_mask = self.data.phase == phase_id + 1
            map_colours[phase_mask] = map_data[:, phase_mask].T / norm

        return map_colours

    def calc_ipf_colour(self, map_data, direction, phases=None,
                        bg_colour=None):
        if phases is None:
            phases = self.phases
            phase_ids = range(len(phases))
        else:
            phase_ids = phases
            phases = [self.phases[i] for i in phase_ids]

        if bg_colour is None:
            bg_colour = np.array([0., 0., 0.])

        map_colours = np.tile(bg_colour, self.shape + (1,))

        for phase, phase_id in zip(phases, phase_ids):
            # calculate IPF colours for phase
            phase_mask = self.data.phase == phase_id + 1
            map_colours[phase_mask] = Quat.calc_ipf_colours(
                map_data[phase_mask], direction, phase.crystal_structure.name
            ).T

        return map_colours

    def plot_euler_map(self, phases=None, bg_colour=None, **kwargs):
        """Plot an orientation map in Euler colouring

        Parameters
        ----------
        phases : list of int
            Which phases to plot for
        kwargs
            All arguments are passed to :func:`defdap.plotting.MapPlot.create`.

        Returns
        -------
        defdap.plotting.MapPlot

        """
        # Set default plot parameters then update with any input
        plot_params = {}
        plot_params.update(kwargs)

        map_colours = self.calc_euler_colour(
            self.data.euler_angle, phases=phases, bg_colour=bg_colour
        )

        return MapPlot.create(self, map_colours, **plot_params)

    def plot_ipf_map(self, direction, phases=None, bg_colour=None, **kwargs):
        """
        Plot a map with points coloured in IPF colouring,
        with respect to a given sample direction.

        Parameters
        ----------
        direction : np.array len 3
            Sample direction.
        phases : list of int
            Which phases to plot IPF data for.
        bg_colour : np.array len 3
            Colour of background (i.e. for phases not plotted).
        kwargs
            Other arguments passed to :func:`defdap.plotting.MapPlot.create`.

        Returns
        -------
        defdap.plotting.MapPlot

        """
        # Set default plot parameters then update with any input
        plot_params = {}
        plot_params.update(kwargs)

        map_colours = self.calc_ipf_colour(
            self.data.orientation, direction, phases=phases,
            bg_colour=bg_colour
        )

        return MapPlot.create(self, map_colours, **plot_params)

    def plot_phase_map(self, **kwargs):
        """Plot a phase map.

        Parameters
        ----------
        kwargs
            All arguments passed to :func:`defdap.plotting.MapPlot.create`.

        Returns
        -------
        defdap.plotting.MapPlot

        """
        # Set default plot parameters then update with any input
        plot_params = {
            'vmin': 0,
            'vmax': self.num_phases
        }
        plot_params.update(kwargs)

        plot = MapPlot.create(self, self.data.phase, **plot_params)

        # add a legend to the plot
        phase_ids = list(range(0, self.num_phases + 1))
        phase_names = ["Non-indexed"] + [phase.name for phase in self.phases]
        plot.add_legend(phase_ids, phase_names, loc=2, borderaxespad=0.)

        return plot

    @report_progress("calculating KAM")
    def calc_kam(self):
        """
        Calculates Kernel Average Misorientaion (KAM) for the EBSD map,
        based on a 3x3 kernel. Crystal symmetric equivalences are not
        considered. Stores result as `KAM`.

        """
        quat_comps = np.empty((4, ) + self.shape)

        for i, row in enumerate(self.data.orientation):
            for j, quat in enumerate(row):
                quat_comps[:, i, j] = quat.quat_coef

        kam = np.empty(self.shape)

        # Start with rows. Calculate misorientation with neighbouring rows.
        # First and last row only in one direction
        kam[0] = abs(np.einsum("ij,ij->j",
                               quat_comps[:, 0], quat_comps[:, 1]))
        kam[-1] = abs(np.einsum("ij,ij->j",
                                quat_comps[:, -1], quat_comps[:, -2]))
        for i in range(1, self.y_dim - 1):
            kam[i] = (abs(np.einsum("ij,ij->j",
                                    quat_comps[:, i], quat_comps[:, i + 1])) +
                      abs(np.einsum("ij,ij->j",
                                    quat_comps[:, i], quat_comps[:, i - 1]))
                      ) / 2
        kam[kam > 1] = 1

        # Do the same for columns
        kam[:, 0] += abs(np.einsum("ij,ij->j",
                                   quat_comps[:, :, 0], quat_comps[:, :, 1]))
        kam[:, -1] += abs(np.einsum("ij,ij->j",
                                    quat_comps[:, :, -1], quat_comps[:, :, -2]))
        for i in range(1, self.x_dim - 1):
            kam[:, i] += (abs(np.einsum("ij,ij->j",
                                        quat_comps[:, :, i],
                                        quat_comps[:, :, i + 1])) +
                          abs(np.einsum("ij,ij->j",
                                        quat_comps[:, :, i],
                                        quat_comps[:, :, i - 1]))
                          ) / 2
        kam /= 2
        kam[kam > 1] = 1

        yield 1.
        return 2 * np.arccos(kam)

    @report_progress("calculating Nye tensor")
    def calc_nye(self):
        """
        Calculates Nye tensor and related GND density for the EBSD map.
        Stores result as `Nye_tensor` and `GND`. Uses the crystal
        symmetry of the primary phase.

        """
        syms = self.primary_phase.crystal_structure.symmetries
        num_syms = len(syms)

        # array to store quat components of initial and symmetric equivalents
        quat_comps = np.empty((num_syms, 4) + self.shape)

        # populate with initial quat components
        for i, row in enumerate(self.data.orientation):
            for j, quat in enumerate(row):
                quat_comps[0, :, i, j] = quat.quat_coef

        # loop of over symmetries and apply to initial quat components
        # (excluding first symmetry as this is the identity transformation)
        for i, sym in enumerate(syms[1:], start=1):
            # sym[i] * quat for all points (* is quaternion product)
            quat_comps[i, 0] = (quat_comps[0, 0] * sym[0] - quat_comps[0, 1] * sym[1] -
                               quat_comps[0, 2] * sym[2] - quat_comps[0, 3] * sym[3])
            quat_comps[i, 1] = (quat_comps[0, 0] * sym[1] + quat_comps[0, 1] * sym[0] -
                               quat_comps[0, 2] * sym[3] + quat_comps[0, 3] * sym[2])
            quat_comps[i, 2] = (quat_comps[0, 0] * sym[2] + quat_comps[0, 2] * sym[0] -
                               quat_comps[0, 3] * sym[1] + quat_comps[0, 1] * sym[3])
            quat_comps[i, 3] = (quat_comps[0, 0] * sym[3] + quat_comps[0, 3] * sym[0] -
                               quat_comps[0, 1] * sym[2] + quat_comps[0, 2] * sym[1])

            # swap into positive hemisphere if required
            quat_comps[i, :, quat_comps[i, 0] < 0] *= -1

        # Arrays to store neighbour misorientation in positive x and y direction
        mis_ori_x = np.zeros((num_syms,) + self.shape)
        mis_ori_y = np.zeros((num_syms, ) + self.shape)

        # loop over symmetries calculating misorientation to initial
        for i in range(num_syms):
            for j in range(self.x_dim - 1):
                mis_ori_x[i, :, j] = abs(np.einsum("ij,ij->j", quat_comps[0, :, :, j], quat_comps[i, :, :, j + 1]))

            for j in range(self.y_dim - 1):
                mis_ori_y[i, j, :] = abs(np.einsum("ij,ij->j", quat_comps[0, :, j, :], quat_comps[i, :, j + 1, :]))

        mis_ori_x[mis_ori_x > 1] = 1
        mis_ori_y[mis_ori_y > 1] = 1

        # find min misorientation (max here as misorientaion is cos of this)
        arg_mis_ori_x = np.argmax(mis_ori_x, axis=0)
        arg_mis_ori_y = np.argmax(mis_ori_y, axis=0)
        mis_ori_x = np.max(mis_ori_x, axis=0)
        mis_ori_y = np.max(mis_ori_y, axis=0)

        # convert to misorientation in degrees
        mis_ori_x = 360 * np.arccos(mis_ori_x) / np.pi
        mis_ori_y = 360 * np.arccos(mis_ori_y) / np.pi

        # calculate relative elastic distortion tensors at each point in the two directions
        betaderx = np.zeros((3, 3) + self.shape)
        betadery = betaderx
        for i in range(self.x_dim - 1):
            for j in range(self.y_dim - 1):
                q0x = Quat(quat_comps[0, 0, j, i], quat_comps[0, 1, j, i],
                           quat_comps[0, 2, j, i], quat_comps[0, 3, j, i])
                qix = Quat(quat_comps[arg_mis_ori_x[j, i], 0, j, i + 1],
                           quat_comps[arg_mis_ori_x[j, i], 1, j, i + 1],
                           quat_comps[arg_mis_ori_x[j, i], 2, j, i + 1],
                           quat_comps[arg_mis_ori_x[j, i], 3, j, i + 1])
                misoquatx = qix.conjugate * q0x
                # change stepsize to meters
                betaderx[:, :, j, i] = (Quat.rot_matrix(misoquatx) - np.eye(3)) / self.step_size / 1e-6
                q0y = Quat(quat_comps[0, 0, j, i], quat_comps[0, 1, j, i],
                           quat_comps[0, 2, j, i], quat_comps[0, 3, j, i])
                qiy = Quat(quat_comps[arg_mis_ori_y[j, i], 0, j + 1, i],
                           quat_comps[arg_mis_ori_y[j, i], 1, j + 1, i],
                           quat_comps[arg_mis_ori_y[j, i], 2, j + 1, i],
                           quat_comps[arg_mis_ori_y[j, i], 3, j + 1, i])
                misoquaty = qiy.conjugate * q0y
                # change stepsize to meters
                betadery[:, :, j, i] = (Quat.rot_matrix(misoquaty) - np.eye(3)) / self.step_size / 1e-6

        # Calculate the Nye Tensor
        alpha = np.empty((3, 3) + self.shape)
        bavg = 1.4e-10  # Burgers vector
        alpha[0, 2] = (betadery[0, 0] - betaderx[0, 1]) / bavg
        alpha[1, 2] = (betadery[1, 0] - betaderx[1, 1]) / bavg
        alpha[2, 2] = (betadery[2, 0] - betaderx[2, 1]) / bavg
        alpha[:, 1] = betaderx[:, 2] / bavg
        alpha[:, 0] = -1 * betadery[:, 2] / bavg

        # Calculate 3 possible L1 norms of Nye tensor for total
        # disloction density
        alpha_total3 = np.empty(self.shape)
        alpha_total5 = np.empty(self.shape)
        alpha_total9 = np.empty(self.shape)
        alpha_total3 = 30 / 10. * (
                abs(alpha[0, 2]) + abs(alpha[1, 2]) + abs(alpha[2, 2])
        )
        alpha_total5 = 30 / 14. * (
                abs(alpha[0, 2]) + abs(alpha[1, 2]) + abs(alpha[2, 2]) +
                abs(alpha[1, 0]) + abs(alpha[0, 1])
        )
        alpha_total9 = 30 / 20. * (
                abs(alpha[0, 2]) + abs(alpha[1, 2]) + abs(alpha[2, 2]) +
                abs(alpha[0, 0]) + abs(alpha[1, 0]) + abs(alpha[2, 0]) +
                abs(alpha[0, 1]) + abs(alpha[1, 1]) + abs(alpha[2, 1])
        )
        alpha_total3[abs(alpha_total3) < 1] = 1e12
        alpha_total5[abs(alpha_total3) < 1] = 1e12
        alpha_total9[abs(alpha_total3) < 1] = 1e12

        # choose from the different alpha_totals according to preference;
        # see Ruggles GND density paper

        yield 1.
        return alpha_total9, alpha

    @report_progress("building quaternion array")
    def calc_quat_array(self):
        """Build quaternion array

        """
        # create the array of quat objects
        quats = Quat.create_many_quats(self.data.euler_angle)

        yield 1.
        return quats

    def filter_data(self, misori_tol=5):
        # Kuwahara filter
        print("8 quadrants")
        misori_tol *= np.pi / 180
        misori_tol = np.cos(misori_tol / 2)

        # store quat components in array
        quat_comps = np.empty((4,) + self.shape)
        for idx in np.ndindex(self.shape):
            quat_comps[(slice(None),) + idx] = self.data.orientation[idx].quat_coef

        # misorientation in each quadrant surrounding a point
        mis_oris = np.zeros((8,) + self.shape)

        for i in range(2, self.shape[0] - 2):
            for j in range(2, self.shape[1] - 2):

                ref_quat = quat_comps[:, i, j]
                quadrants = [
                    quat_comps[:, i - 2:i + 1, j - 2:j + 1],   # UL
                    quat_comps[:, i - 2:i + 1, j - 1:j + 2],   # UC
                    quat_comps[:, i - 2:i + 1, j:j + 3],       # UR
                    quat_comps[:, i - 1:i + 2, j:j + 3],       # MR
                    quat_comps[:, i:i + 3, j:j + 3],           # LR
                    quat_comps[:, i:i + 3, j - 1:j + 2],       # LC
                    quat_comps[:, i:i + 3, j - 2:j + 1],       # LL
                    quat_comps[:, i - 1:i + 2, j - 2:j + 1]    # ML
                ]

                for k, quats in enumerate(quadrants):
                    mis_oris_quad = np.abs(
                        np.einsum("ijk,i->jk", quats, ref_quat)
                    )
                    mis_oris_quad = mis_oris_quad[mis_oris_quad > misori_tol]
                    mis_oris[k, i, j] = mis_oris_quad.mean()

        min_mis_ori_quadrant = np.argmax(mis_oris, axis=0)
        # minMisOris = np.max(mis_oris, axis=0)
        # minMisOris[minMisOris > 1.] = 1.
        # minMisOris = 2 * np.arccos(minMisOris)

        quat_comps_new = np.copy(quat_comps)

        for i in range(2, self.shape[0] - 2):
            for j in range(2, self.shape[1] - 2):
                # if minMisOris[i, j] < misOriTol:
                #     continue

                ref_quat = quat_comps[:, i, j]
                quadrants = [
                    quat_comps[:, i - 2:i + 1, j - 2:j + 1],   # UL
                    quat_comps[:, i - 2:i + 1, j - 1:j + 2],   # UC
                    quat_comps[:, i - 2:i + 1, j:j + 3],       # UR
                    quat_comps[:, i - 1:i + 2, j:j + 3],       # MR
                    quat_comps[:, i:i + 3, j:j + 3],           # LR
                    quat_comps[:, i:i + 3, j - 1:j + 2],       # LC
                    quat_comps[:, i:i + 3, j - 2:j + 1],       # LL
                    quat_comps[:, i - 1:i + 2, j - 2:j + 1]    # ML
                ]
                quats = quadrants[min_mis_ori_quadrant[i, j]]

                mis_oris_quad = np.abs(
                    np.einsum("ijk,i->jk", quats, ref_quat)
                )
                quats = quats[:, mis_oris_quad > misori_tol]

                avOri = np.einsum("ij->i", quats)
                # avOri /= np.sqrt(np.dot(avOri, avOri))

                quat_comps_new[:, i, j] = avOri

        quat_comps_new /= np.sqrt(np.einsum("ijk,ijk->jk", quat_comps_new, quat_comps_new))

        quat_array_new = np.empty(self.shape, dtype=Quat)

        for idx in np.ndindex(self.shape):
            quat_array_new[idx] = Quat(quat_comps_new[(slice(None),) + idx])

        self.data.orientation = quat_array_new

        return quats

    @report_progress("finding grain boundaries")
    def find_boundaries(self, misori_tol=10):
        """Find grain and phase boundaries

        Parameters
        ----------
        misori_tol : float
            Critical misorientation in degrees.

        """
        # TODO: what happens with non-indexed points
        # TODO: grain boundaries should be calculated per crystal structure
        misori_tol *= np.pi / 180
        syms = self.primary_phase.crystal_structure.symmetries
        num_syms = len(syms)

        # array to store quat components of initial and symmetric equivalents
        quat_comps = np.empty((num_syms, 4) + self.shape)

        # populate with initial quat components
        for i, row in enumerate(self.data.orientation):
            for j, quat in enumerate(row):
                quat_comps[0, :, i, j] = quat.quat_coef

        # loop of over symmetries and apply to initial quat components
        # (excluding first symmetry as this is the identity transformation)
        for i, sym in enumerate(syms[1:], start=1):
            # sym[i] * quat for all points (* is quaternion product)
            quat_comps[i, 0] = (
                quat_comps[0, 0]*sym[0] - quat_comps[0, 1]*sym[1] -
                quat_comps[0, 2]*sym[2] - quat_comps[0, 3]*sym[3]
            )
            quat_comps[i, 1] = (
                quat_comps[0, 0]*sym[1] + quat_comps[0, 1]*sym[0] -
                quat_comps[0, 2]*sym[3] + quat_comps[0, 3]*sym[2]
            )
            quat_comps[i, 2] = (
                quat_comps[0, 0]*sym[2] + quat_comps[0, 2]*sym[0] -
                quat_comps[0, 3]*sym[1] + quat_comps[0, 1]*sym[3]
            )
            quat_comps[i, 3] = (
                quat_comps[0, 0]*sym[3] + quat_comps[0, 3]*sym[0] -
                quat_comps[0, 1]*sym[2] + quat_comps[0, 2]*sym[1]
            )
            # swap into positive hemisphere if required
            quat_comps[i, :, quat_comps[i, 0] < 0] *= -1

        # Arrays to store neighbour misorientation in positive x and y
        # directions
        misori_x = np.ones((num_syms, ) + self.shape)
        misori_y = np.ones((num_syms, ) + self.shape)

        # loop over symmetries calculating misorientation to initial
        for i in range(num_syms):
            for j in range(self.shape[1] - 1):
                misori_x[i, :, j] = abs(np.einsum(
                    "ij,ij->j", quat_comps[0, :, :, j], quat_comps[i, :, :, j+1]
                ))

            for j in range(self.shape[0] - 1):
                misori_y[i, j] = abs(np.einsum(
                    "ij,ij->j", quat_comps[0, :, j], quat_comps[i, :, j+1]
                ))

        misori_x[misori_x > 1] = 1
        misori_y[misori_y > 1] = 1

        # find max dot product and then convert to misorientation angle
        misori_x = 2 * np.arccos(np.max(misori_x, axis=0))
        misori_y = 2 * np.arccos(np.max(misori_y, axis=0))

        # PHASE boundary POINTS
        phase_im = self.data.phase
        pb_im_x = np.not_equal(phase_im, np.roll(phase_im, -1, axis=1))
        pb_im_x[:, -1] = False
        pb_im_y = np.not_equal(phase_im, np.roll(phase_im, -1, axis=0))
        pb_im_y[-1] = False

        phase_boundaries = BoundarySet.from_image(self, pb_im_x, pb_im_y)
        grain_boundaries = BoundarySet.from_image(
            self,
            (misori_x > misori_tol) | pb_im_x,
            (misori_y > misori_tol) | pb_im_y
        )

        yield 1.
        return phase_boundaries, grain_boundaries

    @report_progress("constructing neighbour network")
    def build_neighbour_network(self):
        # create network
        nn = nx.Graph()
        nn.add_nodes_from(self.grains)

        points_x = self.data.grain_boundaries.points_x
        points_y = self.data.grain_boundaries.points_y
        total_points_x = len(points_x)
        total_points = total_points_x + len(points_y)

        for i, points in enumerate((points_x, points_y)):
            for i_point, (x, y) in enumerate(points):
                # report progress
                yield (i_point + i * total_points_x) / total_points

                if (x == 0 or y == 0 or x == self.shape[1] - 1 or
                        y == self.shape[0] - 1):
                    # exclude boundary pixels of map
                    continue

                grain_id = self.data.grains[y, x] - 1
                nei_grain_id = self.data.grains[y + i, x - i + 1] - 1
                if nei_grain_id == grain_id:
                    # ignore if neighbour is same as grain
                    continue
                if nei_grain_id < 0 or grain_id < 0:
                    # ignore if not a grain (boundary points -1 and
                    # points in small grains -2)
                    continue

                grain = self[grain_id]
                nei_grain = self[nei_grain_id]
                try:
                    # look up boundary segment if it exists
                    b_seg = nn[grain][nei_grain]['boundary']
                except KeyError:
                    # neighbour relation doesn't exist so add it
                    b_seg = BoundarySegment(self, grain, nei_grain)
                    nn.add_edge(grain, nei_grain, boundary=b_seg)

                # add the boundary point
                b_seg.addBoundaryPoint((x, y), i, grain)

        self.neighbour_network = nn

    def plot_phase_boundary_map(self, dilate=False, **kwargs):
        """Plot phase boundary map.

        Parameters
        ----------
        dilate : bool
            If true, dilate boundary.
        kwargs
            All other arguments are passed to :func:`defdap.plotting.MapPlot.create`.

        Returns
        -------
        defdap.plotting.MapPlot

        """
        # Set default plot parameters then update with any input
        plot_params = {
            'vmax': 1,
            'plot_colour_bar': True,
            'cmap': 'gray'
        }
        plot_params.update(kwargs)

        boundaries_image = self.data.phase_boundaries.image.astype(int)
        if dilate:
            boundaries_image = mph.binary_dilation(boundaries_image)

        plot = MapPlot.create(self, boundaries_image, **plot_params)

        return plot

    def plot_boundary_map(self, **kwargs):
        """Plot grain boundary map.

        Parameters
        ----------
        kwargs
            All arguments are passed to :func:`defdap.plotting.MapPlot.create`.

        Returns
        -------
        defdap.plotting.MapPlot

        """
        # Set default plot parameters then update with any input
        plot_params = {
            'plot_gbs': True,
            'boundaryColour': 'black'
        }
        plot_params.update(kwargs)

        plot = MapPlot.create(self, None, **plot_params)

        return plot

    @report_progress("finding grains")
    def find_grains(self, min_grain_size=10):
        """Find grains and assign IDs.

        Parameters
        ----------
        min_grain_size : int
            Minimum grain area in pixels.

        """
        # Initialise the grain map
        # TODO: Look at grain map compared to boundary map
        grains = np.zeros(self.shape, dtype=int)
        grain_list = []

        boundary_im_x = self.data.grain_boundaries.image_x
        boundary_im_y = self.data.grain_boundaries.image_y

        # List of points where no grain has be set yet
        points_left = self.data.phase != 0
        total_points = points_left.sum()
        found_point = 0
        next_point = points_left.tobytes().find(b'\x01')

        # Start counter for grains
        grain_index = 1
        group_id = Datastore.generate_id()

        # Loop until all points (except boundaries) have been assigned
        # to a grain or ignored
        i = 0
        coords_buffer = np.zeros((boundary_im_y.size, 2), dtype=np.intp)
        while found_point >= 0:
            # Flood fill first unknown point and return grain object
            seed = np.unravel_index(next_point, self.shape)

            grain = Grain(grain_index - 1, self, group_id)
            grain.data.point = flood_fill(
                (seed[1], seed[0]), grain_index, points_left, grains,
                boundary_im_x, boundary_im_y, coords_buffer
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

        # Assign phase to each grain
        for grain in grain_list:
            phase_vals = grain.grain_data(self.data.phase)
            if np.max(phase_vals) != np.min(phase_vals):
                warn(f"Grain {grain.grain_id} could not be assigned a "
                     f"phase, phase vals not constant.")
                continue
            phase_id = phase_vals[0] - 1
            if not (0 <= phase_id < self.num_phases):
                warn(f"Grain {grain.grain_id} could not be assigned a "
                     f"phase, invalid phase {phase_id}.")
                continue
            grain.phase_id = phase_id
            grain.phase = self.phases[phase_id]

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

    def plot_grain_map(self, **kwargs):
        """Plot a map with grains coloured.

        Parameters
        ----------
        kwargs
            All arguments are passed to :func:`defdap.plotting.MapPlot.create`.

        Returns
        -------
        defdap.plotting.MapPlot

        """
        # Set default plot parameters then update with any input
        plot_params = {
            'clabel': "Grain number"
        }
        plot_params.update(kwargs)

        plot = MapPlot.create(self, self.data.grains, **plot_params)

        return plot

    @report_progress("calculating grain mean orientations")
    def calc_grain_av_oris(self):
        """Calculate the average orientation of grains.

        """
        numGrains = len(self)
        for iGrain, grain in enumerate(self):
            grain.calc_average_ori()

            # report progress
            yield (iGrain + 1) / numGrains

    @report_progress("calculating grain misorientations")
    def calc_grain_mis_ori(self, calc_axis=False):
        """Calculate the misorientation within grains.

        Parameters
        ----------
        calc_axis : bool
            Calculate the misorientation axis if True.

        """
        num_grains = len(self)
        for i_grain, grain in enumerate(self):
            grain.build_mis_ori_list(calc_axis=calc_axis)

            # report progress
            yield (i_grain + 1) / num_grains

    def plot_mis_ori_map(self, component=0, **kwargs):
        """Plot misorientation map.

        Parameters
        ----------
        component : int, {0, 1, 2, 3}
            0 gives misorientation, 1, 2, 3 gives rotation about x, y, z
        kwargs
            All other arguments are passed to :func:`defdap.plotting.MapPlot.create`.

        Returns
        -------
        defdap.plotting.MapPlot

        """
        if component in [1, 2, 3]:
            self.mis_ori = np.zeros(self.shape)
            # Calculate misorientation axis if not calculated
            if np.any([grain.mis_ori_axis_list is None for grain in self]):
                self.calc_grain_mis_ori(calc_axis=True)
            for grain in self:
                for point, mis_ori_axis in zip(grain.data.point, np.array(grain.mis_ori_axis_list)):
                    self.mis_ori[point[1], point[0]] = mis_ori_axis[component - 1]

            mis_ori = self.mis_ori * 180 / np.pi
            clabel = r"Rotation around {:} axis ($^\circ$)".format(
                ['X', 'Y', 'Z'][component-1]
            )
        else:
            self.mis_ori = np.ones(self.shape)
            # Calculate misorientation if not calculated
            if np.any([grain.mis_ori_list is None for grain in self]):
                self.calc_grain_mis_ori(calc_axis=False)
            for grain in self:
                for point, mis_ori in zip(grain.data.point, grain.mis_ori_list):
                    self.mis_ori[point[1], point[0]] = mis_ori

            mis_ori = np.arccos(self.mis_ori) * 360 / np.pi
            clabel = r"Grain reference orienation deviation (GROD) ($^\circ$)"

        # Set default plot parameters then update with any input
        plot_params = {
            'plot_colour_bar': True,
            'clabel': clabel
        }
        plot_params.update(kwargs)

        plot = MapPlot.create(self, mis_ori, **plot_params)

        return plot

    @report_progress("calculating grain average Schmid factors")
    def calc_average_grain_schmid_factors(self, load_vector, slip_systems=None):
        """
        Calculates Schmid factors for all slip systems, for all grains,
        based on average grain orientation.

        Parameters
        ----------
        load_vector :
            Loading vector, e.g. [1, 0, 0].
        slip_systems : list, optional
            Slip planes to calculate Schmid factor for, maximum of all
            planes calculated if not given.

        """
        num_grains = len(self)
        for iGrain, grain in enumerate(self):
            grain.calc_average_schmid_factors(load_vector, slip_systems=slip_systems)

            # report progress
            yield (iGrain + 1) / num_grains

    @report_progress("calculating RDR values")
    def calc_rdr(self):
        """Calculates Relative Displacent Ratio values for all grains"""
        num_grains = len(self)
        for iGrain, grain in enumerate(self):
            grain.calc_rdr()

            # report progress
            yield (iGrain + 1) / num_grains

    def plot_average_grain_schmid_factors_map(self, planes=None, directions=None,
                                              **kwargs):
        """
        Plot maximum Schmid factor map, based on average grain
        orientation (for all slip systems unless specified).

        Parameters
        ----------
        planes : list, optional
            Plane ID(s) to consider. All planes considered if not given.
        directions : list, optional
            Direction ID(s) to consider. All directions considered if not given.
        kwargs
            All other arguments are passed to :func:`defdap.plotting.MapPlot.create`.

        Returns
        -------
        defdap.plotting.MapPlot

        """
        # Set default plot parameters then update with any input
        plot_params = {
            'vmin': 0,
            'vmax': 0.5,
            'cmap': 'gray',
            'plot_colour_bar': True,
            'clabel': "Schmid factor"
        }
        plot_params.update(kwargs)

        if self[0].average_schmid_factors is None:
            raise Exception("Run 'calc_average_grain_schmid_factors' first")

        grains_sf_max = []
        for grain in self:
            current_sf = []

            if planes is not None:
                for plane in planes:
                    if directions is not None:
                        for direction in directions:
                            current_sf.append(
                                grain.average_schmid_factors[plane][direction]
                            )
                    else:
                        current_sf += grain.average_schmid_factors[plane]
            else:
                for sf_group in grain.average_schmid_factors:
                    current_sf += sf_group

            grains_sf_max.append(max(current_sf))

        plot = self.plot_grain_data_map(grain_data=grains_sf_max, bg=0.5,
                                        **plot_params)

        return plot


class Grain(base.Grain):
    """
    Class to encapsulate a grain in an EBSD map and useful analysis and
    plotting methods.

    Attributes
    ----------
    ebsd_map : defdap.ebsd.Map
        EBSD map this grain is a member of.
    owner_map : defdap.ebsd.Map
        EBSD map this grain is a member of.
    phase_id : int

    phase : defdap.crystal.Phase

    data : defdap.utils.Datastore
        Must contain after creating:
            point : list of tuples
                (x, y)
        Generated data:
            GROD : numpy.ndarray
                Grain reference orientation distribution magnitude
            GROD_axis : numpy.ndarray
                Grain reference orientation distribution direction
        Derived data:
            Map data to list data from the map the grain is part of


    mis_ori_list : list
        MisOri at each point in grain.
    mis_ori_axis_list : list
        MisOri axes at each point in grain.
    ref_ori : defdap.quat.Quat
        Average ori of grain
    average_mis_ori
        Average mis_ori of grain.
    average_schmid_factors : list
        List of list Schmid factors (grouped by slip plane).
    slip_trace_angles : list
        Slip trace angles in screen plane.
    slip_trace_inclinations : list
         Angle between slip plane and screen plane.

    """
    def __init__(self, grain_id, ebsdMap, group_id):
        # Call base class constructor
        super(Grain, self).__init__(grain_id, ebsdMap, group_id)

        self.ebsd_map = self.owner_map            # ebsd map this grain is a member of
        self.mis_ori_list = None                  # list of mis_ori at each point in grain
        self.mis_ori_axis_list = None              # list of mis_ori axes at each point in grain
        self.ref_ori = None                      # (quat) average ori of grain
        self.average_mis_ori = None               # average mis_ori of grain

        self.average_schmid_factors = None        # list of list Schmid factors (grouped by slip plane)
        self.slip_trace_angles = None             # list of slip trace angles
        self.slip_trace_inclinations = None

        self.plot_default = self.plot_unit_cell

        self.data.add_generator(
            ('GROD', 'GROD_axis'), self.calc_grod,
            type='list',
            metadatas=({
                'unit': 'rad',
                'order': 0,
                'plot_params': {
                    'plot_colour_bar': True,
                    'clabel': 'GROD',
                }
            }, {
                'unit': '',
                'order': 1,
                'default_component': 0,
                'plot_params': {
                    'plot_colour_bar': True,
                    'clabel': 'GROD axis',
                }
            })
        )

    @property
    def crystal_sym(self):
        """Temporary"""
        return self.phase.crystal_structure.name

    def calc_average_ori(self):
        """Calculate the average orientation of a grain.

        """
        quat_comps_sym = Quat.calc_sym_eqvs(self.data.orientation, self.crystal_sym)

        self.ref_ori = Quat.calc_average_ori(quat_comps_sym)

    def build_mis_ori_list(self, calc_axis=False):
        """Calculate the misorientation within given grain.

        Parameters
        ----------
        calc_axis : bool
            Calculate the misorientation axis if True.

        """
        quat_comps_sym = Quat.calc_sym_eqvs(self.data.orientation, self.crystal_sym)

        if self.ref_ori is None:
            self.ref_ori = Quat.calc_average_ori(quat_comps_sym)

        mis_ori_array, min_quat_comps = Quat.calcMisOri(quat_comps_sym, self.ref_ori)

        self.average_mis_ori = mis_ori_array.mean()
        self.mis_ori_list = list(mis_ori_array)

        if calc_axis:
            # Now for axis calculation
            ref_ori_inv = self.ref_ori.conjugate

            mis_ori_axis = np.empty((3, min_quat_comps.shape[1]))
            dq = np.empty((4, min_quat_comps.shape[1]))

            # ref_ori_inv * minQuat for all points (* is quaternion product)
            # change to minQuat * ref_ori_inv
            dq[0, :] = (ref_ori_inv[0] * min_quat_comps[0, :] - ref_ori_inv[1] * min_quat_comps[1, :] -
                        ref_ori_inv[2] * min_quat_comps[2, :] - ref_ori_inv[3] * min_quat_comps[3, :])

            dq[1, :] = (ref_ori_inv[1] * min_quat_comps[0, :] + ref_ori_inv[0] * min_quat_comps[1, :] +
                        ref_ori_inv[3] * min_quat_comps[2, :] - ref_ori_inv[2] * min_quat_comps[3, :])

            dq[2, :] = (ref_ori_inv[2] * min_quat_comps[0, :] + ref_ori_inv[0] * min_quat_comps[2, :] +
                        ref_ori_inv[1] * min_quat_comps[3, :] - ref_ori_inv[3] * min_quat_comps[1, :])

            dq[3, :] = (ref_ori_inv[3] * min_quat_comps[0, :] + ref_ori_inv[0] * min_quat_comps[3, :] +
                        ref_ori_inv[2] * min_quat_comps[1, :] - ref_ori_inv[1] * min_quat_comps[2, :])

            dq[:, dq[0] < 0] = -dq[:, dq[0] < 0]

            # numpy broadcasting taking care of different array sizes
            mis_ori_axis[:, :] = (2 * dq[1:4, :] * np.arccos(dq[0, :])) / np.sqrt(1 - np.power(dq[0, :], 2))

            # hack it back into a list. Need to change self.*List to be arrays, it was a bad decision to
            # make them lists in the beginning
            self.mis_ori_axis_list = []
            for row in mis_ori_axis.transpose():
                self.mis_ori_axis_list.append(row)

    def calc_grod(self):
        quat_comps = Quat.calc_sym_eqvs(self.data.orientation, self.crystal_sym)

        if self.ref_ori is None:
            self.ref_ori = Quat.calc_average_ori(quat_comps)

        misori, quat_comps = Quat.calcMisOri(quat_comps, self.ref_ori)
        misori = 2 * np.arccos(misori)

        ref_ori_inv = self.ref_ori.conjugate
        dq = np.empty((4, len(self)))
        # ref_ori_inv * quat_comps for all points
        # change to quat_comps * ref_ori_inv
        dq[0] = (ref_ori_inv[0]*quat_comps[0] - ref_ori_inv[1]*quat_comps[1] -
                 ref_ori_inv[2]*quat_comps[2] - ref_ori_inv[3]*quat_comps[3])
        dq[1] = (ref_ori_inv[1]*quat_comps[0] + ref_ori_inv[0]*quat_comps[1] +
                 ref_ori_inv[3]*quat_comps[2] - ref_ori_inv[2]*quat_comps[3])
        dq[2] = (ref_ori_inv[2]*quat_comps[0] + ref_ori_inv[0]*quat_comps[2] +
                 ref_ori_inv[1]*quat_comps[3] - ref_ori_inv[3]*quat_comps[1])
        dq[3] = (ref_ori_inv[3]*quat_comps[0] + ref_ori_inv[0]*quat_comps[3] +
                 ref_ori_inv[2]*quat_comps[1] - ref_ori_inv[1]*quat_comps[2])
        dq[:, dq[0] < 0] *= -1
        misori_axis = (2 * dq[1:4] * np.arccos(dq[0])) / np.sqrt(1 - dq[0]**2)

        return misori, misori_axis

    def plot_ref_ori(self, direction=np.array([0, 0, 1]), **kwargs):
        """Plot the average grain orientation on an IPF.

        Parameters
        ----------
        direction : numpy.ndarray
            Sample direction for IPF.
        kwargs
            All other arguments are passed to :func:`defdap.quat.Quat.plot_ipf`.

        Returns
        -------
        defdap.plotting.PolePlot

        """
        plot_params = {'marker': '+'}
        plot_params.update(kwargs)
        return Quat.plot_ipf([self.ref_ori], direction, self.crystal_sym,
                             **plot_params)

    def plot_ori_spread(self, direction=np.array([0, 0, 1]), **kwargs):
        """Plot all orientations within a given grain, on an IPF.

        Parameters
        ----------
        direction : numpy.ndarray
            Sample direction for IPF.
        kwargs
            All other arguments are passed to :func:`defdap.quat.Quat.plot_ipf`.

        Returns
        -------
        defdap.plotting.PolePlot

        """
        plot_params = {'marker': '.'}
        plot_params.update(kwargs)
        return Quat.plot_ipf(self.data.orientation, direction, self.crystal_sym,
                             **plot_params)

    def plot_unit_cell(self, fig=None, ax=None, plot=None, **kwargs):
        """Plot an unit cell of the average grain orientation.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Matplotlib figure to plot on
        ax : matplotlib.figure.Figure
            Matplotlib figure to plot on
        plot : defdap.plotting.PolePlot
            defdap plot to plot the figure to.
        kwargs
            All other arguments are passed to :func:`defdap.quat.Quat.plot_unit_cell`.

        """
        crystal_structure = self.ebsd_map.phases[self.phase_id].crystal_structure
        plot = Quat.plot_unit_cell(self.ref_ori, fig=fig, ax=ax, plot=plot,
                                   crystal_structure=crystal_structure, **kwargs)

        return plot

    def plot_mis_ori(self, component=0, **kwargs):
        """Plot misorientation map for a given grain.

        Parameters
        ----------
        component : int, {0, 1, 2, 3}
            0 gives misorientation, 1, 2, 3 gives rotation about x, y, z.
        kwargs
            All other arguments are passed to :func:`defdap.ebsd.plot_grain_data`.

        Returns
        -------
        defdap.plotting.GrainPlot

        """
        component = int(component)

        # Set default plot parameters then update with any input
        plot_params = {
            'plot_colour_bar': True
        }
        if component == 0:
            if self.mis_ori_list is None: self.build_mis_ori_list()
            plot_params['clabel'] = r"Grain reference orientation " \
                                   r"deviation (GROD) ($^\circ$)"
            plot_data = np.rad2deg(2 * np.arccos(self.mis_ori_list))

        elif 0 < component < 4:
            if self.mis_ori_axis_list is None: self.build_mis_ori_list(calc_axis=True)
            plot_params['clabel'] = r"Rotation around {:} ($^\circ$)".format(
                ['X', 'Y', 'Z'][component-1]
            )
            plot_data = np.rad2deg(np.array(self.mis_ori_axis_list)[:, component - 1])

        else:
            raise ValueError("Component must between 0 and 3")
        plot_params.update(kwargs)

        plot = self.plot_grain_data(grain_data=plot_data, **plot_params)

        return plot

    # define load axis as unit vector
    def calc_average_schmid_factors(self, load_vector, slip_systems=None):
        """Calculate Schmid factors for grain, using average orientation.

        Parameters
        ----------
        load_vector : numpy.ndarray
            Loading vector, i.e. [1, 0, 0]
        slip_systems : list, optional
            Slip planes to calculate Schmid factor for. Maximum for all planes
            used if not set.

        """
        if slip_systems is None:
            slip_systems = self.phase.slip_systems
        if self.ref_ori is None:
            self.calc_average_ori()

        # orientation of grain
        grain_av_ori = self.ref_ori

        # Transform the load vector into crystal coordinates
        load_vector_crystal = grain_av_ori.transform_vector(load_vector)

        self.average_schmid_factors = []
        # flatten list of lists
        # slip_systems = chain.from_iterable(slip_systems)

        # Loop over groups of slip systems with same slip plane
        for i, slip_system_group in enumerate(slip_systems):
            self.average_schmid_factors.append([])
            # Then loop over individual slip systems
            for slip_system in slip_system_group:
                schmidFactor = abs(np.dot(load_vector_crystal, slip_system.slip_plane) *
                                   np.dot(load_vector_crystal, slip_system.slip_dir))
                self.average_schmid_factors[i].append(schmidFactor)

        return

    def calc_rdr(self):
        """Calculate Relative Displacement Ratio values."""
        self.rdr = []

        # Loop over groups of slip systems with same slip plane
        for i, slip_system_group in enumerate(self.phase.slip_systems):
            self.rdr.append([])
            # Then loop over individual slip systems
            for slip_system in slip_system_group:
                slip_dir_sample = self.ref_ori.conjugate.transform_vector(slip_system.slip_dir)
                self.rdr[i].append(-slip_dir_sample[0] / slip_dir_sample[1])

    @property
    def slip_traces(self):
        """Returns list of slip trace angles.

        Returns
        -------
        list
            Slip trace angles based on grain orientation in calc_slip_traces.

        """
        if self.slip_trace_angles is None:
            self.calc_slip_traces()

        return self.slip_trace_angles

    def print_slip_traces(self):
        """Print a list of slip planes (with colours) and slip directions

        """
        self.calc_slip_traces()

        if self.average_schmid_factors is None:
            raise Exception("Run 'calc_average_grain_schmid_factors' on the EBSD map first")

        for ss_group, colour, sf_group, slip_trace in zip(
            self.phase.slip_systems,
            self.phase.slip_trace_colours,
            self.average_schmid_factors,
            self.slip_traces
        ):
            print('{0}\tColour: {1}\tAngle: {2:.2f}'.format(ss_group[0].slip_plane_label, colour, slip_trace * 180 / np.pi))
            for ss, sf in zip(ss_group, sf_group):
                print('  {0}   SF: {1:.3f}'.format(ss.slip_dir_label, sf))

    def calc_slip_traces(self, slip_systems=None):
        """Calculates list of slip trace angles based on grain orientation.

        Parameters
        -------
        slip_systems : defdap.crystal.SlipSystem, optional

        """
        if slip_systems is None:
            slip_systems = self.phase.slip_systems
        if self.ref_ori is None:
            self.calc_average_ori()

        screen_plane_norm = np.array((0, 0, 1))   # in sample orientation frame

        grain_av_ori = self.ref_ori   # orientation of grain

        screen_plane_norm_crystal = grain_av_ori.transform_vector(screen_plane_norm)

        self.slip_trace_angles = []
        self.slip_trace_inclinations = []
        # Loop over each group of slip systems
        for slip_system_group in slip_systems:
            # Take slip plane from first in group
            slip_plane_norm = slip_system_group[0].slip_plane
            # planeLabel = slip_system_group[0].slip_plane_label

            # Calculate intersection of slip plane with plane of screen
            intersection_crystal = np.cross(screen_plane_norm_crystal, slip_plane_norm)

            # Calculate angle between slip plane and screen plane
            inclination = np.arccos(np.dot(screen_plane_norm_crystal, slip_plane_norm))
            if inclination > np.pi / 2:
                inclination = np.pi - inclination
            # print("{} inclination: {:.1f}".format(planeLabel, inclination * 180 / np.pi))

            # Transform intersection back into sample coordinates and normalise
            intersection = grain_av_ori.conjugate.transform_vector(intersection_crystal)
            intersection = intersection / np.sqrt(np.dot(intersection, intersection))

            # Calculate trace angle. Starting vertical and proceeding
            # counter clockwise
            if intersection[0] > 0:
                intersection *= -1
            trace_angle = np.arccos(np.dot(intersection, np.array([0, 1.0, 0])))

            # Append to list
            self.slip_trace_angles.append(trace_angle)
            self.slip_trace_inclinations.append(inclination)


class BoundarySet(object):
    # boundaries : numpy.ndarray
    #     Map of boundaries. -1 for a boundary, 0 otherwise.
    # phaseBoundaries : numpy.ndarray
    #     Map of phase boundaries. -1 for boundary, 0 otherwise.
    def __init__(self, ebsd_map, points_x, points_y):
        self.ebsd_map = ebsd_map
        self.points_x = set(points_x)
        self.points_y = set(points_y)

    @classmethod
    def from_image(cls, ebsd_map, image_x, image_y):
        return cls(
            ebsd_map,
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

    def _image(self, points):
        image = np.zeros(self.ebsd_map.shape, dtype=bool)
        image[tuple(zip(*points))[::-1]] = True
        return image

    @property
    def image_x(self):
        return self._image(self.points_x)

    @property
    def image_y(self):
        return self._image(self.points_y)

    @property
    def image(self):
        return self._image(self.points)

    @property
    def lines(self):
        _, _, lines = self.boundary_points_to_lines(
            boundary_points_x=self.points_x,
            boundary_points_y=self.points_y
        )
        return lines

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

    def addBoundaryPoint(self, point, kind, owner_grain):
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
        _, _, lines = BoundarySet.boundary_points_to_lines(
            boundary_points_x=self.boundary_points_x,
            boundary_points_y=self.boundary_points_y
        )
        return lines

    def misorientation(self):
        mis_ori, minSymm = self.grain1.ref_ori.mis_ori(
            self.grain2.ref_ori, self.ebsdMap.crystal_sym, return_quat=2
        )
        mis_ori = 2 * np.arccos(mis_ori)
        mis_ori_axis = self.grain1.ref_ori.mis_ori_axis(minSymm)

        # should this be a unit vector already?
        mis_ori_axis /= np.sqrt(np.dot(mis_ori_axis, mis_ori_axis))

        return mis_ori, mis_ori_axis

        # compVector = np.array([1., 1., 1.])
        # deviation = np.arccos(
        #     np.dot(mis_ori_axis, np.array([1., 1., 1.])) /
        #     (np.sqrt(np.dot(mis_ori_axis, mis_ori_axis) * np.dot(compVector,
        #                                                      compVector))))
        # print(deviation * 180 / np.pi)


class Linker(object):
    """Class for linking multiple EBSD maps of the same region for analysis of deformation.

    Attributes
    ----------
    ebsd_maps : list(ebsd.Map)
        List of `ebsd.Map` objects that are linked.
    links : list(tuple(int))
        List of grain link. Each link is stored as a tuple of
        grain IDs (one from each map stored in same order of maps).
    plots : list(plotting.MapPlot)
        List of last opened plot of each map.

    """
    def __init__(self, ebsd_maps):
        """Initialise linker and set ebsd maps

        Parameters
        ----------
        ebsd_maps : list(ebsd.Map)
            List of `ebsd.Map` objects that are linked.

        """
        self.ebsd_maps = ebsd_maps
        self.links = []
        self.plots = None

    def set_origin(self, **kwargs):
        """Interactive tool to set origin of each EBSD map.

        Parameters
        ----------
        kwargs
            Keyword arguments passed to :func:`defdap.ebsd.Map.plot_default`

        """
        self.plots = []
        for ebsd_map in self.ebsd_maps:
            plot = ebsd_map.plot_default(make_interactive=True, **kwargs)
            plot.add_event_handler('button_press_event', self.click_set_origin)
            plot.add_points([ebsd_map.origin[0]], [ebsd_map.origin[1]],
                            c='w', s=60, marker='x')
            self.plots.append(plot)

    def click_set_origin(self, event, plot):
        """Event handler for clicking to set origin of map.

        Parameters
        ----------
        event
            Click event.
        plot : defdap.plotting.MapPlot
            Plot to capture clicks from.

        """
        # check if click was on the map
        if event.inaxes is not plot.ax:
            return

        origin = (int(event.xdata), int(event.ydata))
        plot.calling_map.origin = origin
        plot.add_points([origin[0]], [origin[1]], update_layer=0)
        print(f"Origin set to ({origin[0]}, {origin[1]})")

    def start_linking(self):
        """Start interactive grain linking process of each EBSD map.

        """
        self.plots = []
        for ebsd_map in self.ebsd_maps:
            plot = ebsd_map.locate_grain(click_event=self.click_grain_guess)

            # Add make link button to axes
            plot.add_button('Make link', self.make_link,
                            color='0.85', hovercolor='0.95')

            self.plots.append(plot)

    def click_grain_guess(self, event, plot):
        """Guesses grain position in other maps, given click on one.

        Parameters
        ----------
        event
            Click handler.
        plot : defdap.plotting.Plot
            Plot to capture clicks from.

        """
        # check if click was on the map
        if event.inaxes is not plot.ax:
            return

        curr_ebsd_map = plot.callingMap

        if curr_ebsd_map is self.ebsd_maps[0]:
            # clicked on 'master' map so highlight and guess grain on others

            # set current grain in 'master' ebsd map
            self.ebsd_maps[0].click_grain_id(event, plot, False)

            # guess at grain in other maps
            for ebsd_map, plot in zip(self.ebsd_maps[1:], self.plots[1:]):
                # calculated position relative to set origin of the
                # map, scaled from step size of maps
                x0m = curr_ebsd_map.origin[0]
                y0m = curr_ebsd_map.origin[1]
                x0 = ebsd_map.origin[0]
                y0 = ebsd_map.origin[1]
                scaling = curr_ebsd_map.step_size / ebsd_map.step_size

                x = int((event.xdata - x0m) * scaling + x0)
                y = int((event.ydata - y0m) * scaling + y0)

                grain_id = int(ebsd_map.data.grains[y, x]) - 1
                grain = self[grain_id]
                ebsd_map.sel_grain = grain
                print(grain_id)

                # update the grain highlights layer in the plot
                plot.add_grain_highlights([grain_id],
                                          alpha=ebsd_map.highlight_alpha)

        else:
            # clicked on other map so correct guessed selected grain
            curr_ebsd_map.click_grain_id(event, plot, False)

    def make_link(self, event, plot):
        """Make a link between the EBSD maps after clicking.

        """
        # create empty list for link
        curr_link = []

        for i, ebsd_map in enumerate(self.ebsd_maps):
            if ebsd_map.sel_grain is not None:
                curr_link.append(ebsd_map.sel_grain.grain_id)
            else:
                raise Exception(f"No grain selected in map {i + 1}.")

        curr_link = tuple(curr_link)
        if curr_link not in self.links:
            self.links.append(curr_link)
            print("Link added " + str(curr_link))

    def reset_links(self):
        """Reset links.

        """
        self.links = []

#   Analysis routines

    def set_ref_ori_from_master(self):
        """Loop over each map (not first/reference) and each link.
        Sets refOri of linked grains to refOri of grain in first map.

        """
        for i, ebsd_map in enumerate(self.ebsd_maps[1:], start=1):
            for link in self.links:
                ebsd_map[link[i]].ref_ori = copy.deepcopy(
                    self.ebsd_maps[0][link[0]].ref_ori
                )

    def update_misori(self, calc_axis=False):
        """Recalculate misorientation for linked grain (not for first map)

        Parameters
        ----------
        calc_axis : bool
            Calculate the misorientation axis if True.

        """
        for i, ebsd_map in enumerate(self.ebsd_maps[1:], start=1):
            for link in self.links:
                ebsd_map[link[i]].build_mis_ori_list(calc_axis=calc_axis)
