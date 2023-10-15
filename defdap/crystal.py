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

import os
import numpy as np
from numpy.linalg import norm

from defdap import defaults
from defdap.quat import Quat


class Phase(object):
    def __init__(self, name, laue_group, space_group, lattice_params):
        """
        Parameters
        ----------
        name : str
            Name of the phase
        laue_group : int
            Laue group
        space_group : int
            Space group
        lattice_params : tuple
            Lattice parameters in order (a,b,c,alpha,beta,gamma)

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
                "({:.2f}, {:.2f}, {:.2f}, {:.0f}, {:.0f}, {:.0f})\n"
                "  Slip systems: {:}")
        return text.format(self.name, self.crystal_structure.name,
                           *self.lattice_params[:3],
                           *np.array(self.lattice_params[3:]) * 180 / np.pi,
                           self.ss_file)

    @property
    def c_over_a(self):
        if self.crystal_structure is crystalStructures['hexagonal']:
            return self.lattice_params[2] / self.lattice_params[0]
        return None

    def print_slip_systems(self):
        """Print a list of slip planes (with colours) and slip directions.

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

    # TODO: Move these to the phase class where the lattice parameters
    #  can be accessed
    @staticmethod
    def l_matrix(a, b, c, alpha, beta, gamma, convention=None):
        """ Construct L matrix based on Page 22 of
        Randle and Engle - Introduction to texture analysis"""
        l_matrix = np.zeros((3, 3))

        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)
        cos_gamma = np.cos(gamma)

        sin_gamma = np.sin(gamma)

        l_matrix[0, 0] = a
        l_matrix[0, 1] = b * cos_gamma
        l_matrix[0, 2] = c * cos_beta

        l_matrix[1, 1] = b * sin_gamma
        l_matrix[1, 2] = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma

        l_matrix[2, 2] = c * np.sqrt(
            1 + 2 * cos_alpha * cos_beta * cos_gamma -
            cos_alpha**2 - cos_beta**2 - cos_gamma**2
        ) / sin_gamma

        # OI/HKL convention - x // [10-10],     y // a2 [-12-10]
        # TSL    convention - x // a1 [2-1-10], y // [01-10]
        if convention is None:
            convention = defaults['crystal_ortho_conv']

        if convention.lower() in ['hkl', 'oi']:
            # Swap 00 with 11 and 01 with 10 due to how OI orthonormalises
            # From Brad Wynne
            t1 = l_matrix[0, 0]
            t2 = l_matrix[1, 0]

            l_matrix[0, 0] = l_matrix[1, 1]
            l_matrix[1, 0] = l_matrix[0, 1]

            l_matrix[1, 1] = t1
            l_matrix[0, 1] = t2

        elif convention.lower() != 'tsl':
            raise ValueError(
                f"Unknown convention '{convention}' for orthonormalisation of "
                f"crystal structure, can be 'hkl' or 'tsl'"
            )

        # Set small components to 0
        l_matrix[np.abs(l_matrix) < 1e-10] = 0

        return l_matrix

    @staticmethod
    def q_matrix(l_matrix):
        """ Construct matrix of reciprocal lattice vectors to transform
        plane normals See C. T. Young and J. L. Lytton, J. Appl. Phys.,
        vol. 43, no. 4, pp. 1408–1417, 1972."""
        a = l_matrix[:, 0]
        b = l_matrix[:, 1]
        c = l_matrix[:, 2]

        volume = abs(np.dot(a, np.cross(b, c)))
        a_star = np.cross(b, c) / volume
        b_star = np.cross(c, a) / volume
        c_star = np.cross(a, b) / volume

        q_matrix = np.stack((a_star, b_star, c_star), axis=1)

        return q_matrix


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
    """Class used for defining and performing operations on a slip system.

    """
    def __init__(self, slip_plane, slip_dir, crystal_structure, c_over_a=None):
        """Initialise a slip system object.

        Parameters
        ----------
        slip_plane: nunpy.ndarray
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
            l_matrix = CrystalStructure.l_matrix(
                1, 1, c_over_a, np.pi / 2, np.pi / 2, np.pi * 2 / 3
            )
            # Q matrix for transforming planes
            qMatrix = CrystalStructure.q_matrix(l_matrix)

            # Transform into orthonormal basis and then normalise
            self.slip_plane = np.matmul(qMatrix, slip_plane_m)
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
        return (f"SlipSystem(slipPlane={self.slip_plane_label}, "
                f"slipDir={self.slip_dir_label}, "
                f"symmetry={self.crystal_structure.name})")

    @property
    def slip_plane_label(self):
        """Return the slip plane label. For example '(111)'.

        Returns
        -------
        str
            Slip plane label.

        """
        return '(' + ''.join(map(str_idx, self.plane_idc)) + ')'

    @property
    def slip_dir_label(self):
        """Returns the slip direction label. For example '[110]'.

        Returns
        -------
        str
            Slip direction label.

        """
        return '[' + ''.join(map(str_idx, self.dir_idc)) + ']'

    def generate_family(self):
        """Generate the family of slip systems which this system belongs to.

        Returns
        -------
        list of SlipSystem
            The family of slip systems.

        """
        #
        symms = self.crystal_structure.symmetries

        ss_family = set()  # will not preserve order

        plane = self.plane_idc
        dir = self.dir_idc

        if self.crystal_structure.name == 'hexagonal':
            # Transformation from crystal to orthonormal coords
            l_matrix = CrystalStructure.l_matrix(
                1, 1, self.c_over_a, np.pi / 2, np.pi / 2, np.pi * 2 / 3
            )
            # Q matrix for transforming planes
            q_matrix = CrystalStructure.q_matrix(l_matrix)

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
        Load in slip systems from file. 3 integers for slip plane
        normal and 3 for slip direction. Returns a list of list of slip
        systems grouped by slip plane.

        Parameters
        ----------
        name : str
            Name of the slip system file (without file extension)
            stored in the defdap install dir or path to a file.
        crystal_structure : defdap.crystal.CrystalStructure
            Crystal structure of the slip systems.
        c_over_a : float, optional
            C over a ratio for hexagonal crystals.
        group_by : str, optional
            How to group the slip systems, either by slip plane ('plane')
            or slip system family ('family') or don't group (None).

        Returns
        -------
        list of list of SlipSystem
            A list of list of slip systems grouped slip plane.

        Raises
        ------
        IOError
            Raised if not 6/8 integers per line.

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
        Groups slip systems by their slip plane.

        Parameters
        ----------
        slip_systems : list of SlipSystem
            A list of slip systems.
        group_by : str
            How to group the slip systems, either by slip plane ('plane')
            or slip system family ('family').

        Returns
        -------
        list of list of SlipSystem
            A list of list of grouped slip systems.

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


def convert_idc(in_type, *, dir=None, plane=None):
    """
    Convert between Miller and Miller-Bravais indices.

    Parameters
    ----------
    in_type : str {'m', 'mb'}
        Type of indices provided. If 'm' converts from Miller to
        Miller-Bravais, opposite for 'mb'.
    dir : tuple of int or equivalent, optional
        Direction to convert. This OR `plane` must me provided.
    plane : tuple of int or equivalent, optional
        Plane to convert. This OR `direction` must me provided.

    Returns
    -------
    tuple of int
        The converted plane or direction.

    """
    if dir is None and plane is None:
        raise ValueError("One of either `direction` or `plane` must be "
                         "provided.")
    if dir is not None and plane is not None:
        raise ValueError("One of either `direction` or `plane` must be "
                         "provided, not both.")

    def check_len(val, length):
        if len(val) != length:
            raise ValueError(f"Vector must have {length} values.")

    if in_type.lower() == 'm':
        if dir is None:
            # plane M->MB
            check_len(plane, 3)
            out = np.array(plane)[[0, 1, 0, 2]]
            out[2] += plane[1]
            out[2] *= -1

        else:
            # direction M->MB
            check_len(dir, 3)
            u, v, w = dir
            out = np.array([2*u-v, 2*v-u, -u-v, 3*w]) / 3
            try:
                # Attempt to cast to integers
                out = safe_int_cast(out)
            except ValueError:
                pass

    elif in_type.lower() == 'mb':
        if dir is None:
            # plane MB->M
            check_len(plane, 4)
            out = np.array(plane)[[0, 1, 3]]

        else:
            # direction MB->M
            check_len(dir, 4)
            out = np.array(dir)[[0, 1, 3]]
            out[[0, 1]] -= dir[2]

    else:
        raise ValueError("`inType` must be either 'm' or 'mb'.")

    return tuple(out)


def pos_idc(vec):
    """
    Return a consistent positive version of a set of indices.

    Parameters
    ----------
    vec : tuple of int or equivalent
        Indices to convert.

    Returns
    -------
    tuple of int
        Positive version of indices.

    """
    for idx in vec:
        if idx == 0:
            continue
        if idx > 0:
            return tuple(vec)
        else:
            return tuple(-np.array(vec))


def reduce_idc(vec):
    """
    Reduce indices to lowest integers

    Parameters
    ----------
    vec : tuple of int or equivalent
        Indices to reduce.

    Returns
    -------
    tuple of int
        The reduced indices.

    """
    return tuple((np.array(vec) / np.gcd.reduce(vec)).astype(np.int8))


def safe_int_cast(vec, tol=1e-3):
    """
    Cast a tuple of floats to integers, raising an error if rounding is
    over a tolerance.

    Parameters
    ----------
    vec : tuple of float or equivalent
        Vector to cast.
    tol : float
        Tolerance above which an error is raised.

    Returns
    -------
    tuple of int

    Raises
    ------
    ValueError
        If the rounding is over the tolerance for any value.

    """
    vec = np.array(vec)
    vec_rounded = vec.round()

    if np.any(np.abs(vec - vec_rounded) > tol):
        raise ValueError('Rounding too large', np.abs(vec - vec_rounded))

    return tuple(vec_rounded.astype(np.int8))


def str_idx(idx):
    """
    String representation of an index with overbars.

    Parameters
    ----------
    idx : int

    Returns
    -------
    str

    """
    if not isinstance(idx, (int, np.integer)):
        raise ValueError("Index must be an integer.")

    return str(idx) if idx >= 0 else str(-idx) + u'\u0305'
