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

import os
import numpy as np
from numpy.linalg import norm

from defdap import defaults
from defdap.quat import Quat
from defdap.crystal_utils import *


class Phase(object):
    """
    Represents a crystallographic phase.

    Stores phase information including crystal structure, lattice parameters,
    and associated slip systems.

    Attributes
    ----------
    name : str
        Name of the phase.
    laue_group : int
        Laue group number.
    spaceGroup : int
        Space group number.
    lattice_params : tuple of float
        Lattice parameters (a, b, c, alpha, beta, gamma) 
        where lengths are in angstrom and angles are in radians.
    crystal_structure : defdap.crystal.CrystalStructure
        Crystal structure object for the phase.
    slip_systems : list or list of list of SlipSystem, optional
        Slip systems for the phase.
    slip_trace_colours : list of str, optional
        Colors for slip plane traces.

    """
    def __init__(self, name, laue_group, space_group, lattice_params):
        """
        Initialize a Phase object.

        Parameters
        ----------
        name : str
            Name of the phase.
        laue_group : int
            Laue group number (e.g., 9 for hexagonal, 11 for cubic).
        space_group : int
            Space group number.
        lattice_params : tuple of float
            Lattice parameters in order (a, b, c, alpha, beta, gamma)
            where lengths are in angstrom and angles are in radians.

        """
        self.name = name
        self.laue_group = laue_group
        self.spaceGroup = space_group
        self.lattice_params = lattice_params

        try:
            self.crystal_structure = {
                9: crystalStructures['hexagonal'],
                11: crystalStructures['cubic'],
            }[laue_group]
        except KeyError:
            raise ValueError(f"Unknown Laue group key: {laue_group}")

        if self.crystal_structure is crystalStructures['hexagonal']:
            self.ss_file = defaults['slip_system_file']['HCP']
        else:
            try:
                self.ss_file = defaults['slip_system_file'][
                    {225: 'FCC', 229: 'BCC'}[space_group]
                    # See http://pd.chem.ucl.ac.uk/pdnn/symm3/allsgp.htm
                ]
            except KeyError:
                self.ss_file = None

        if self.ss_file is None:
            self.slip_systems = None
            self.slip_trace_colours = None
        else:
            self.slip_systems, self.slip_trace_colours = SlipSystem.load(
                self.ss_file, self.crystal_structure, c_over_a=self.c_over_a
            )

    def __str__(self):
        text = ("Phase: {:}\n  Crystal structure: {:}\n  Lattice params: "
                "({:.3f} Å, {:.3f} Å, {:.3f} Å, {:.0f} °, {:.0f} °, {:.0f} °)\n"
                "  Slip systems: {:}")
        return text.format(self.name, self.crystal_structure.name,
                           *self.lattice_params[:3],
                           *np.array(self.lattice_params[3:]) * 180 / np.pi,
                           self.ss_file)

    @property
    def c_over_a(self):
        """Crystal c over a ratio for hexagonal crystals."""
        if self.crystal_structure is crystalStructures['hexagonal']:
            return self.lattice_params[2] / self.lattice_params[0]
        return None

    def print_slip_systems(self):
        """Print slip plane family and their associated colors. 
        For each slip plane, print corresponding slip directions.

        """
        # TODO: this should be moved to static method of the SlipSystem class
        for i, (ss_group, colour) in enumerate(zip(self.slip_systems,
                                                  self.slip_trace_colours)):
            print('Plane {0}: {1}\tColour: {2}'.format(
                i, ss_group[0].slip_plane_label, colour
            ))
            for j, ss in enumerate(ss_group):
                print('  Direction {0}: {1}'.format(j, ss.slip_dir_label))


class CrystalStructure(object):
    def __init__(self, name, symmetries, vertices, faces):
        self.name = name
        self.symmetries = symmetries
        self.vertices = vertices
        self.faces = faces


over_root2 = np.sqrt(2) / 2
sqrt3over2 = np.sqrt(3) / 2
# Use ideal ratio as only used for plotting unit cell
c_over_a = 1.633 / 2

crystalStructures = {
    "cubic": CrystalStructure(
        "cubic",
        [
            # identity
            Quat(1.0, 0.0, 0.0, 0.0),

            # cubic tetrads(100)
            Quat(over_root2, over_root2, 0.0, 0.0),
            Quat(0.0, 1.0, 0.0, 0.0),
            Quat(over_root2, -over_root2, 0.0, 0.0),
            Quat(over_root2, 0.0, over_root2, 0.0),
            Quat(0.0, 0.0, 1.0, 0.0),
            Quat(over_root2, 0.0, -over_root2, 0.0),
            Quat(over_root2, 0.0, 0.0, over_root2),
            Quat(0.0, 0.0, 0.0, 1.0),
            Quat(over_root2, 0.0, 0.0, -over_root2),

            # cubic dyads (110)
            Quat(0.0, over_root2, over_root2, 0.0),
            Quat(0.0, -over_root2, over_root2, 0.0),
            Quat(0.0, over_root2, 0.0, over_root2),
            Quat(0.0, -over_root2, 0.0, over_root2),
            Quat(0.0, 0.0, over_root2, over_root2),
            Quat(0.0, 0.0, -over_root2, over_root2),

            # cubic triads (111)
            Quat(0.5, 0.5, 0.5, 0.5),
            Quat(0.5, -0.5, -0.5, -0.5),
            Quat(0.5, -0.5, 0.5, 0.5),
            Quat(0.5, 0.5, -0.5, -0.5),
            Quat(0.5, 0.5, -0.5, 0.5),
            Quat(0.5, -0.5, 0.5, -0.5),
            Quat(0.5, 0.5, 0.5, -0.5),
            Quat(0.5, -0.5, -0.5, 0.5)
        ],
        np.array([
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5]
        ]),
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7]
        ]
    ),
    "hexagonal": CrystalStructure(
        "hexagonal",
        [
            # identity
            Quat(1.0, 0.0, 0.0, 0.0),

            Quat(0.0, 1.0, 0.0, 0.0),
            Quat(0.0, 0.0, 1.0, 0.0),
            Quat(0.0, 0.0, 0.0, 1.0),

            # hexagonal hexads
            Quat(sqrt3over2, 0.0, 0.0, 0.5),
            Quat(0.5, 0.0, 0.0, sqrt3over2),
            Quat(0.5, 0.0, 0.0, -sqrt3over2),
            Quat(sqrt3over2, 0.0, 0.0, -0.5),

            # hexagonal diads
            Quat(0.0, -0.5, -sqrt3over2, 0.0),
            Quat(0.0, 0.5, -sqrt3over2, 0.0),
            Quat(0.0, sqrt3over2, -0.5, 0.0),
            Quat(0.0, -sqrt3over2, -0.5, 0.0)
        ],
        np.array([
            [1, 0, -c_over_a],
            [0.5, sqrt3over2, -c_over_a],
            [-0.5, sqrt3over2, -c_over_a],
            [-1, 0, -c_over_a],
            [-0.5, -sqrt3over2, -c_over_a],
            [0.5, -sqrt3over2, -c_over_a],
            [1, 0, c_over_a],
            [0.5, sqrt3over2, c_over_a],
            [-0.5, sqrt3over2, c_over_a],
            [-1, 0, c_over_a],
            [-0.5, -sqrt3over2, c_over_a],
            [0.5, -sqrt3over2, c_over_a]
        ]),
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
            [0, 6, 7, 1],
            [1, 7, 8, 2],
            [2, 8, 9, 3],
            [3, 9, 10, 4],
            [4, 10, 11, 5],
            [5, 11, 6, 0]
        ]
    )
}


class SlipSystem(object):
    """
    Class for defining and performing operations on a slip system.

    Handles slip system operations including plane and direction transformations
    for both cubic and hexagonal crystal structures.
    """
    def __init__(self, slip_plane, slip_dir, crystal_structure, c_over_a=None):
        """Initialise a slip system object.

        Parameters
        ----------
        slip_plane: numpy.ndarray
            Slip plane.
        slip_dir: numpy.ndarray
            Slip direction.
        crystal_structure : defdap.crystal.CrystalStructure
            Crystal structure of the slip system.
        c_over_a : float, optional
            C over a ratio for hexagonal crystals.

        """
        self.crystal_structure = crystal_structure

        # Stored as Miller indices (Miller-Bravais for hexagonal)
        self.plane_idc = tuple(slip_plane)
        self.dir_idc = tuple(slip_dir)

        # Stored as vectors in a cartesian basis
        if self.crystal_structure.name == "cubic":
            self.slip_plane = slip_plane / norm(slip_plane)
            self.slip_dir = slip_dir / norm(slip_dir)
            self.c_over_a = None
        elif self.crystal_structure.name == "hexagonal":
            if c_over_a is None:
                raise Exception("No c over a ratio given")
            self.c_over_a = c_over_a

            # Convert plane and dir from Miller-Bravais to Miller
            slip_plane_m = convert_idc('mb', plane=slip_plane)
            slip_dir_m = convert_idc('mb', dir=slip_dir)

            # Transformation from crystal to orthonormal coords
            l_matrix = create_l_matrix(
                1, 1, c_over_a, np.pi / 2, np.pi / 2, np.pi * 2 / 3
            )
            # Q matrix for transforming planes
            q_matrix = create_q_matrix(l_matrix)

            # Transform into orthonormal basis and then normalise
            self.slip_plane = np.matmul(q_matrix, slip_plane_m)
            self.slip_plane /= norm(self.slip_plane)
            self.slip_dir = np.matmul(l_matrix, slip_dir_m)
            self.slip_dir /= norm(self.slip_dir)
        else:
            raise Exception("Only cubic and hexagonal currently supported.")

    def __eq__(self, right):
        # or one divide the other should be a constant for each place.
        return (pos_idc(self.plane_idc) == pos_idc(right.plane_idc) and
                pos_idc(self.dir_idc) == pos_idc(right.dir_idc))

    def __hash__(self):
        return hash(pos_idc(self.plane_idc) + pos_idc(self.dir_idc))

    def __str__(self):
        return self.slip_plane_label + self.slip_dir_label

    def __repr__(self):
        return (f"SlipSystem(slip_plane={self.slip_plane_label}, "
                f"slip_dir={self.slip_dir_label}, "
                f"symmetry={self.crystal_structure.name})")

    @property
    def slip_plane_label(self):
        """Slip plane label.

        Returns
        -------
        str
            Slip plane label in the format '(hkl)'. For example, '(111)'.

        """
        return idc_to_string(self.plane_idc, '()')

    @property
    def slip_dir_label(self):
        """Slip direction label.

        Returns
        -------
        str
            Slip direction label in the format '[uvw]'. For example, '[110]'.

        """
        return idc_to_string(self.dir_idc, '[]')

    def generate_family(self):
        """
        Generate the family of slip systems to which this system belongs by
        applying symmetry operations to generate equivalent slip systems.

        Returns
        -------
        set of SlipSystem
            Symmetrically equivalent slip systems.
            
        """
        #
        symms = self.crystal_structure.symmetries

        ss_family = set()  # will not preserve order

        plane = self.plane_idc
        dir = self.dir_idc

        if self.crystal_structure.name == 'hexagonal':
            # Transformation from crystal to orthonormal coords
            l_matrix = create_l_matrix(
                1, 1, self.c_over_a, np.pi / 2, np.pi / 2, np.pi * 2 / 3
            )
            # Q matrix for transforming planes
            q_matrix = create_q_matrix(l_matrix)

            # Transform into orthonormal basis
            plane = np.matmul(q_matrix, convert_idc('mb', plane=plane))
            dir = np.matmul(l_matrix, convert_idc('mb', dir=dir))

        for i, symm in enumerate(symms):
            symm = symm.conjugate

            plane_symm = symm.transform_vector(plane)
            dir_symm = symm.transform_vector(dir)

            if self.crystal_structure.name == 'hexagonal':
                # q_matrix inverse is equal to l_matrix transposed and vice-versa
                plane_symm = reduce_idc(convert_idc(
                    'm', plane=safe_int_cast(np.matmul(l_matrix.T, plane_symm))
                ))
                dir_symm = reduce_idc(convert_idc(
                    'm', dir=safe_int_cast(np.matmul(q_matrix.T, dir_symm))
                ))

            ss_family.add(SlipSystem(
                pos_idc(safe_int_cast(plane_symm)),
                pos_idc(safe_int_cast(dir_symm)),
                self.crystal_structure, c_over_a=self.c_over_a
            ))

        return ss_family

    @staticmethod
    def load(name, crystal_structure, c_over_a=None, group_by='plane'):
        """
        Reads slip system definitions from a text file. File should contain
        Miller (or Miller-Bravais for hexagonal) indices: 3 (or 4) integers
        for slip plane normal and 3 (or 4) for slip direction.

        Parameters
        ----------
        name : str
            Slip system file name (without extension) in defdap's 
            slip_systems directory, or full file path.
        crystal_structure : defdap.crystal.CrystalStructure
            Crystal structure of the slip systems.
        c_over_a : float, optional
            C over a ratio (required for hexagonal crystals).
        group_by : str, optional
            Grouping method: 'plane' (default) to group by slip plane,
            'family' to group by slip system family, or None for no grouping.

        Returns
        -------
        tuple
            Slip systems and trace colors.

            - slip_systems : list of list of SlipSystem objects, 
              optionally grouped based on `group_by`.
            - slip_trace_colours : list of str
              RGB color codes for each slip plane group.

        Raises
        ------
        FileNotFoundError
            If slip system file cannot be found.
        IOError
            If file format is invalid (not 6/8 integers per line).

        """
        # try and load from package dir first
        try:
            file_ext = ".txt"
            package_dir, _ = os.path.split(__file__)
            filepath = f"{package_dir}/slip_systems/{name}{file_ext}"

            slip_system_file = open(filepath)

        except FileNotFoundError:
            # if it doesn't exist in the package dir, try and load the path
            try:
                filepath = name

                slip_system_file = open(filepath)

            except FileNotFoundError:
                raise(FileNotFoundError("Couldn't find the slip systems file"))

        slip_system_file.readline()
        slip_trace_colours = slip_system_file.readline().strip().split(',')
        slip_system_file.close()

        if crystal_structure.name == "hexagonal":
            vect_size = 4
        else:
            vect_size = 3

        ss_data = np.loadtxt(filepath, delimiter='\t', skiprows=2,
                            dtype=np.int8)
        if ss_data.shape[1] != 2 * vect_size:
            raise IOError("Slip system file not valid")

        # Create list of slip system objects
        slip_systems = []
        for row in ss_data:
            slip_systems.append(SlipSystem(
                row[0:vect_size], row[vect_size:2 * vect_size],
                crystal_structure, c_over_a=c_over_a
            ))

        # Group slip systems is required
        if group_by is not None:
            slip_systems = SlipSystem.group(slip_systems, group_by)

        return slip_systems, slip_trace_colours

    @staticmethod
    def group(slip_systems, group_by):
        """
        Groups slip systems by their slip plane or family.

        Parameters
        ----------
        slip_systems : list of SlipSystem
            List of slip systems to group.
        group_by : str
            Grouping method - either 'plane' to group by slip plane
            or 'family' to group by slip system equivalence family.

        Returns
        -------
        list of list of SlipSystem
            Grouped slip systems.

        Raises
        ------
        ValueError
            If `group_by` is not 'plane' or 'family'.

        """
        if group_by.lower() == 'plane':
            # Group by slip plane and keep slip plane order from file
            grouped_slip_systems = [[slip_systems[0]]]
            for ss in slip_systems[1:]:
                for i, ssGroup in enumerate(grouped_slip_systems):
                    if pos_idc(ss.plane_idc) == pos_idc(ssGroup[0].plane_idc):
                        grouped_slip_systems[i].append(ss)
                        break
                else:
                    grouped_slip_systems.append([ss])

        elif group_by.lower() == 'family':
            grouped_slip_systems = []
            ssFamilies = []
            for ss in slip_systems:
                for i, ssFamily in enumerate(ssFamilies):
                    if ss in ssFamily:
                        grouped_slip_systems[i].append(ss)
                        break
                else:
                    grouped_slip_systems.append([ss])
                    ssFamilies.append(ss.generate_family())

        else:
            raise ValueError("Slip systems can be grouped by plane or family")

        return grouped_slip_systems

    @staticmethod
    def print_slip_system_directory():
        """
        Prints the location where slip system definition files are stored.

        """
        package_dir, _ = os.path.split(__file__)
        print("Slip system definition files are stored in directory:")
        print(f"{package_dir}/slip_systems/")
