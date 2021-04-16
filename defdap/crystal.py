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

import os
import numpy as np
from numpy.linalg import norm

from defdap import defaults
from defdap.quat import Quat


class Phase(object):
    def __init__(self, name, laueGroup, latticeParams,
                 slipSystems=None):
        """
        Parameters
        ----------
        name : str
            Name of the phase
        latticeParams : tuple
            Lattice parameters in order (a,b,c,alpha,beta,gamma)
        crystalStructure : defdap.crystal.CrystalStructure
            Crystal structure of this phase
        slipSystems : collection of defdap.crystal.SlipSystem
            Slip systems available in the phase
        """
        self.name = name
        self.laueGroup = laueGroup
        self.latticeParams = latticeParams
        self.slipSystems = slipSystems
        try:
            self.crystalStructure = {
                9: crystalStructures['hexagonal'],
                11: crystalStructures['cubic'],
            }[laueGroup]
        except KeyError:
            raise ValueError(f"Unknown Laue group key: {laueGroup}")

    def __str__(self):
        text = "Phase: {:}\n  Crystal structure: {:}\n  Lattice params: " \
               "({:.2f}, {:.2f}, {:.2f}, {:.0f}, {:.0f}, {:.0f})"
        return text.format(self.name, self.crystalStructure.name,
                           *self.latticeParams[:3],
                           *np.array(self.latticeParams[3:])*180/np.pi)

    @property
    def cOverA(self):
        if self.crystalStructure is crystalStructures['hexagonal']:
            return self.latticeParams[2] / self.latticeParams[0]
        return None


class CrystalStructure(object):
    def __init__(self, name, symmetries, vertices, faces):
        self.name = name
        self.symmetries = symmetries
        self.vertices = vertices
        self.faces = faces

    # TODO: Move these to the phase class where the lattice parameters
    #  can be accessed
    @staticmethod
    def lMatrix(a, b, c, alpha, beta, gamma, convention=None):
        """ Construct L matrix based on Page 22 of
        Randle and Engle - Introduction to texture analysis"""
        lMatrix = np.zeros((3, 3))

        cosAlpha = np.cos(alpha)
        cosBeta = np.cos(beta)
        cosGamma = np.cos(gamma)

        sinGamma = np.sin(gamma)

        lMatrix[0, 0] = a
        lMatrix[0, 1] = b * cosGamma
        lMatrix[0, 2] = c * cosBeta

        lMatrix[1, 1] = b * sinGamma
        lMatrix[1, 2] = c * (cosAlpha - cosBeta * cosGamma) / sinGamma

        lMatrix[2, 2] = c * np.sqrt(
            1 + 2 * cosAlpha * cosBeta * cosGamma -
            cosAlpha**2 - cosBeta**2 - cosGamma**2
        ) / sinGamma

        # OI/HKL convention - x // [10-10],     y // a2 [-12-10]
        # TSL    convention - x // a1 [2-1-10], y // [01-10]
        if convention is None:
            convention = defaults['crystal_ortho_conv']

        if convention.lower() in ['hkl', 'oi']:
            # Swap 00 with 11 and 01 with 10 due to how OI orthonormalises
            # From Brad Wynne
            t1 = lMatrix[0, 0]
            t2 = lMatrix[1, 0]

            lMatrix[0, 0] = lMatrix[1, 1]
            lMatrix[1, 0] = lMatrix[0, 1]

            lMatrix[1, 1] = t1
            lMatrix[0, 1] = t2

        elif convention.lower() != 'tsl':
            raise ValueError(
                f"Unknown convention '{convention}' for orthonormalisation of "
                f"crystal structure, can be 'hkl' or 'tsl'"
            )

        # Set small components to 0
        lMatrix[np.abs(lMatrix) < 1e-10] = 0

        return lMatrix

    @staticmethod
    def qMatrix(lMatrix):
        """ Construct matrix of reciprocal lattice vectors to transform
        plane normals See C. T. Young and J. L. Lytton, J. Appl. Phys.,
        vol. 43, no. 4, pp. 1408–1417, 1972."""
        a = lMatrix[:, 0]
        b = lMatrix[:, 1]
        c = lMatrix[:, 2]

        volume = abs(np.dot(a, np.cross(b, c)))
        aStar = np.cross(b, c) / volume
        bStar = np.cross(c, a) / volume
        cStar = np.cross(a, b) / volume

        qMatrix = np.stack((aStar, bStar, cStar), axis=1)

        return qMatrix


overRoot2 = np.sqrt(2) / 2
sqrt3over2 = np.sqrt(3) / 2
# Use ideal ratio as only used for plotting unit cell
cOverA = 1.633 / 2

crystalStructures = {
    "cubic": CrystalStructure(
        "cubic",
        [
            # identity
            Quat(1.0, 0.0, 0.0, 0.0),

            # cubic tetrads(100)
            Quat(overRoot2, overRoot2, 0.0, 0.0),
            Quat(0.0, 1.0, 0.0, 0.0),
            Quat(overRoot2, -overRoot2, 0.0, 0.0),
            Quat(overRoot2, 0.0, overRoot2, 0.0),
            Quat(0.0, 0.0, 1.0, 0.0),
            Quat(overRoot2, 0.0, -overRoot2, 0.0),
            Quat(overRoot2, 0.0, 0.0, overRoot2),
            Quat(0.0, 0.0, 0.0, 1.0),
            Quat(overRoot2, 0.0, 0.0, -overRoot2),

            # cubic dyads (110)
            Quat(0.0, overRoot2, overRoot2, 0.0),
            Quat(0.0, -overRoot2, overRoot2, 0.0),
            Quat(0.0, overRoot2, 0.0, overRoot2),
            Quat(0.0, -overRoot2, 0.0, overRoot2),
            Quat(0.0, 0.0, overRoot2, overRoot2),
            Quat(0.0, 0.0, -overRoot2, overRoot2),

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
            [1, 0, -cOverA],
            [0.5, sqrt3over2, -cOverA],
            [-0.5, sqrt3over2, -cOverA],
            [-1, 0, -cOverA],
            [-0.5, -sqrt3over2, -cOverA],
            [0.5, -sqrt3over2, -cOverA],
            [1, 0, cOverA],
            [0.5, sqrt3over2, cOverA],
            [-0.5, sqrt3over2, cOverA],
            [-1, 0, cOverA],
            [-0.5, -sqrt3over2, cOverA],
            [0.5, -sqrt3over2, cOverA]
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
    def __init__(self, slipPlane, slipDir, crystalSym, cOverA=None):
        """Initialise a slip system object.

        Parameters
        ----------
        slipPlane: nunpy.ndarray
            Slip plane.
        slipDir: numpy.ndarray
            Slip direction.
        crystalSym : str
            The crystal symmetry ("cubic" or "hexagonal").
        cOverA : float, optional
            C over a ratio for hexagonal crystals.

        """
        self.crystalSym = crystalSym    # symmetry of material

        # Stored as Miller indices (Miller-Bravais for hexagonal)
        self.planeIdc = tuple(slipPlane)
        self.dirIdc = tuple(slipDir)

        # Stored as vectors in a cartesian basis
        if crystalSym == "cubic":
            self.slipPlane = slipPlane / norm(slipPlane)
            self.slipDir = slipDir / norm(slipDir)
            self.cOverA = None
        elif crystalSym == "hexagonal":
            if cOverA is None:
                raise Exception("No c over a ratio given")
            self.cOverA = cOverA

            # Convert plane and dir from Miller-Bravais to Miller
            slipPlaneM = convertIdc('mb', plane=slipPlane)
            slipDirM = convertIdc('mb', dir=slipDir)

            # Transformation from crystal to orthonormal coords
            lMatrix = CrystalStructure.lMatrix(
                1, 1, cOverA, np.pi / 2, np.pi / 2, np.pi * 2 / 3
            )
            # Q matrix for transforming planes
            qMatrix = CrystalStructure.qMatrix(lMatrix)

            # Transform into orthonormal basis and then normalise
            self.slipPlane = np.matmul(qMatrix, slipPlaneM)
            self.slipPlane /= norm(self.slipPlane)
            self.slipDir = np.matmul(lMatrix, slipDirM)
            self.slipDir /= norm(self.slipDir)
        else:
            raise Exception("Only cubic and hexagonal currently supported.")

    def __eq__(self, right):
        # or one divide the other should be a constant for each place.
        return (posIdc(self.planeIdc) == posIdc(right.planeIdc) and
                posIdc(self.dirIdc) == posIdc(right.dirIdc))

    def __hash__(self):
        return hash(posIdc(self.planeIdc) + posIdc(self.dirIdc))

    def __str__(self):
        return self.slipPlaneLabel + self.slipDirLabel

    def __repr__(self):
        return f"SlipSystem(slipPlane={self.slipPlaneLabel}, " \
               f"slipDir={self.slipDirLabel}, crystalSym={self.crystalSym})"

    @property
    def slipPlaneLabel(self):
        """Return the slip plane label. For example '(111)'.

        Returns
        -------
        str
            Slip plane label.

        """
        return '(' + ''.join(map(strIdx, self.planeIdc)) + ')'

    @property
    def slipDirLabel(self):
        """Returns the slip direction label. For example '[110]'.

        Returns
        -------
        str
            Slip direction label.

        """
        return '[' + ''.join(map(strIdx, self.dirIdc)) + ']'

    def generateFamily(self):
        """Generate the family of slip systems which this system belongs to.

        Returns
        -------
        list of SlipSystem
            The family of slip systems.

        """
        #
        symms = Quat.symEqv(self.crystalSym)

        ss_family = set()  # will not preserve order

        plane = self.planeIdc
        dir = self.dirIdc

        if self.crystalSym == 'hexagonal':
            # Transformation from crystal to orthonormal coords
            lMatrix = CrystalStructure.lMatrix(
                1, 1, self.cOverA, np.pi / 2, np.pi / 2, np.pi * 2 / 3
            )
            # Q matrix for transforming planes
            qMatrix = CrystalStructure.qMatrix(lMatrix)

            # Transform into orthonormal basis
            plane = np.matmul(qMatrix, convertIdc('mb', plane=plane))
            dir = np.matmul(lMatrix, convertIdc('mb', dir=dir))

        for i, symm in enumerate(symms):
            symm = symm.conjugate

            plane_symm = symm.transformVector(plane)
            dir_symm = symm.transformVector(dir)

            if self.crystalSym == 'hexagonal':
                # qMatrix inverse is equal to lMatrix transposed and vice-versa
                plane_symm = reduceIdc(convertIdc(
                    'm', plane=safeIntCast(np.matmul(lMatrix.T, plane_symm))
                ))
                dir_symm = reduceIdc(convertIdc(
                    'm', dir=safeIntCast(np.matmul(qMatrix.T, dir_symm))
                ))

            ss_family.add(SlipSystem(
                posIdc(safeIntCast(plane_symm)),
                posIdc(safeIntCast(dir_symm)),
                self.crystalSym, cOverA=self.cOverA
            ))

        return ss_family

    @staticmethod
    def loadSlipSystems(name, crystalSym, cOverA=None, groupBy='plane'):
        """
        Load in slip systems from file. 3 integers for slip plane
        normal and 3 for slip direction. Returns a list of list of slip
        systems grouped by slip plane.

        Parameters
        ----------
        name : str
            Name of the slip system file (without file extension)
            stored in the defdap install dir or path to a file.
        crystalSym : str
            The crystal symmetry ("cubic" or "hexagonal").
        cOverA : float, optional
            C over a ratio for hexagonal crystals.
        groupBy : str, optional
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
            fileExt = ".txt"
            packageDir, _ = os.path.split(__file__)
            filepath = f"{packageDir}/slip_systems/{name}{fileExt}"

            slipSystemFile = open(filepath)

        except FileNotFoundError:
            # if it doesn't exist in the package dir, try and load the path
            try:
                filepath = name

                slipSystemFile = open(filepath)

            except FileNotFoundError:
                raise(FileNotFoundError("Couldn't find the slip systems file"))

        slipSystemFile.readline()
        slipTraceColours = slipSystemFile.readline().strip().split(',')
        slipSystemFile.close()

        if crystalSym == "hexagonal":
            vectSize = 4
        else:
            vectSize = 3

        ssData = np.loadtxt(filepath, delimiter='\t', skiprows=2,
                            dtype=np.int8)
        if ssData.shape[1] != 2 * vectSize:
            raise IOError("Slip system file not valid")

        # Create list of slip system objects
        slipSystems = []
        for row in ssData:
            slipSystems.append(SlipSystem(
                row[0:vectSize], row[vectSize:2 * vectSize],
                crystalSym, cOverA=cOverA
            ))

        # Group slip systems is required
        if groupBy is not None:
            slipSystems = SlipSystem.groupSlipSystems(slipSystems, groupBy)

        return slipSystems, slipTraceColours

    @staticmethod
    def groupSlipSystems(slipSystems, groupBy):
        """
        Groups slip systems by their slip plane.

        Parameters
        ----------
        slipSystems : list of SlipSystem
            A list of slip systems.
        groupBy : str
            How to group the slip systems, either by slip plane ('plane')
            or slip system family ('family').

        Returns
        -------
        list of list of SlipSystem
            A list of list of grouped slip systems.

        """
        if groupBy.lower() == 'plane':
            # Group by slip plane and keep slip plane order from file
            groupedSlipSystems = [[slipSystems[0]]]
            for ss in slipSystems[1:]:
                for i, ssGroup in enumerate(groupedSlipSystems):
                    if posIdc(ss.planeIdc) == posIdc(ssGroup[0].planeIdc):
                        groupedSlipSystems[i].append(ss)
                        break
                else:
                    groupedSlipSystems.append([ss])

        elif groupBy.lower() == 'family':
            groupedSlipSystems = []
            ssFamilies = []
            for ss in slipSystems:
                for i, ssFamily in enumerate(ssFamilies):
                    if ss in ssFamily:
                        groupedSlipSystems[i].append(ss)
                        break
                else:
                    groupedSlipSystems.append([ss])
                    ssFamilies.append(ss.generateFamily())

        else:
            raise ValueError("Slip systems can be grouped by plane or family")

        return groupedSlipSystems

    @staticmethod
    def printSlipSystemDirectory():
        """
        Prints the location where slip system definition files are stored.

        """
        packageDir, _ = os.path.split(__file__)
        print("Slip system definition files are stored in directory:")
        print(f"{packageDir}/slip_systems/")


def convertIdc(inType, *, dir=None, plane=None):
    """
    Convert between Miller and Miller-Bravais indices.

    Parameters
    ----------
    inType : str {'m', 'mb'}
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

    def checkLen(val, length):
        if len(val) != length:
            raise ValueError(f"Vector must have {length} values.")

    if inType.lower() == 'm':
        if dir is None:
            # plane M->MB
            checkLen(plane, 3)
            out = np.array(plane)[[0, 1, 0, 2]]
            out[2] += plane[1]
            out[2] *= -1

        else:
            # direction M->MB
            checkLen(dir, 3)
            u, v, w = dir
            out = np.array([2*u-v, 2*v-u, -u-v, 3*w]) / 3
            try:
                # Attempt to cast to integers
                out = safeIntCast(out)
            except ValueError:
                pass

    elif inType.lower() == 'mb':
        if dir is None:
            # plane MB->M
            checkLen(plane, 4)
            out = np.array(plane)[[0, 1, 3]]

        else:
            # direction MB->M
            checkLen(dir, 4)
            out = np.array(dir)[[0, 1, 3]]
            out[[0, 1]] -= dir[2]

    else:
        raise ValueError("`inType` must be either 'm' or 'mb'.")

    return tuple(out)


def posIdc(vec):
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


def reduceIdc(vec):
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


def safeIntCast(vec, tol=1e-3):
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


def strIdx(idx):
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
