# Copyright 2019 Mechanics of Microstructures Group
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

from defdap.quat import Quat


class Phase(object):
    def __init__(self, name, crystalStructure, latticeParams,
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
        self.crystalStructure = crystalStructure
        self.latticeParams = latticeParams
        self.slipSystems = slipSystems


class CrystalStructure(object):
    def __init__(self, name, symmetries, vertices, faces):
        self.name = name
        self._symmetries = symmetries
        self._vertices = vertices
        self._faces = faces

    @property
    def symmetries(self):
        return self.symmetries

    @staticmethod
    def lMatrix(a, b, c, alpha, beta, gamma):
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

        # Swap 00 with 11 and 01 with 10 due to how OI orthonormalises
        # From Brad Wynne
        t1 = lMatrix[0, 0]
        t2 = lMatrix[1, 0]

        lMatrix[0, 0] = lMatrix[1, 1]
        lMatrix[1, 0] = lMatrix[0, 1]

        lMatrix[1, 1] = t1
        lMatrix[0, 1] = t2

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
    def __init__(self, slipPlane, slipDir, crystalSym, cOverA=None):
        # Currently only for cubic
        self.crystalSym = crystalSym    # symmetry of material

        # Stored as Miller indicies (Miller-Bravais for hexagonal)
        self.slipPlaneMiller = slipPlane
        self.slipDirMiller = slipDir

        # Stored as vectors in a cartesian basis
        if crystalSym == "cubic":
            self.slipPlaneOrtho = slipPlane / np.sqrt(
                np.dot(slipPlane, slipPlane)
            )
            self.slipDirOrtho = slipDir / np.sqrt(np.dot(slipDir, slipDir))
        elif crystalSym == "hexagonal":
            if cOverA is None:
                raise Exception("No c over a ratio given")
            self.cOverA = cOverA

            # Convert plane and dir from Miller-Bravais to Miller
            slipPlaneM = slipPlane[[0, 1, 3]]
            slipDirM = slipDir[[0, 1, 3]]
            slipDirM[[0, 1]] -= slipDir[2]

            # Transformation from crystal to orthonormal coords
            lMatrix = CrystalStructure.lMatrix(
                1, 1, cOverA, np.pi / 2, np.pi / 2, np.pi * 2 / 3
            )

            # Q matrix for transforming planes
            qMatrix = CrystalStructure.qMatrix(lMatrix)

            # Transform into orthonormal basis and then normalise
            self.slipPlaneOrtho = np.matmul(qMatrix, slipPlaneM)
            self.slipDirOrtho = np.matmul(lMatrix, slipDirM)
            self.slipPlaneOrtho /= np.sqrt(
                np.dot(self.slipPlaneOrtho, self.slipPlaneOrtho)
            )
            self.slipDirOrtho /= np.sqrt(
                np.dot(self.slipDirOrtho, self.slipDirOrtho)
            )
        else:
            raise Exception("Only cubic and hexagonal currently supported.")

    # overload ==. Two slip systems are equal if they have the same slip
    # plane in miller
    def __eq__(self, right):
        return np.all(self.slipPlaneMiller == right.slipPlaneMiller)

    @property
    def slipPlane(self):
        return self.slipPlaneOrtho

    @property
    def slipDir(self):
        return self.slipDirOrtho

    @property
    def slipPlaneLabel(self):
        slipPlane = self.slipPlaneMiller
        if self.crystalSym == "hexagonal":
            return "({:d}{:d}{:d}{:d})".format(*slipPlane)
        else:
            return "({:d}{:d}{:d})".format(*slipPlane)

    @property
    def slipDirLabel(self):
        slipDir = self.slipDirMiller
        if self.crystalSym == "hexagonal":
            return "[{:d}{:d}{:d}{:d}]".format(*slipDir)
        else:
            return "[{:d}{:d}{:d}]".format(*slipDir)

    @staticmethod
    def loadSlipSystems(name, crystalSym, cOverA=None):
        """Load in slip systems from file. 3 integers for slip plane
        normal and 3 for slip direction. Returns a list of list of slip
        systems grouped by slip plane.

        Args:
            name (string): name of the slip system file (without file
            extension) stored in the defdap install dir or path to a file
            crystalSym (string): The crystal symmetry ("cubic" or "hexagonal")
            cOverA (float, optional): c over a ratio for hexagonal crystals

        Returns:
            list(list(SlipSystem)): A list of list of slip systems
            grouped slip plane.

        Raises:
            IOError: Raised if not 6/8 integers per line
        """
        # try and load from package dir first
        try:
            fileExt = ".txt"
            packageDir, _ = os.path.split(__file__)
            filepath = "{:}/slip_systems/{:}{:}".format(
                packageDir, name, fileExt
            )

            slipSystemFile = open(filepath)

        except FileNotFoundError:
            # if it doesn't exist in the package dir try and load the path
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

        ssData = np.loadtxt(filepath, delimiter='\t', skiprows=2, dtype=int)
        if ssData.shape[1] != 2 * vectSize:
            raise IOError("Slip system file not valid")

        # Create list of slip system objects
        slipSystems = []
        for row in ssData:
            slipSystems.append(SlipSystem(
                row[0:vectSize], row[vectSize:2 * vectSize],
                crystalSym, cOverA=cOverA
            ))

        # Group slip sytems by slip plane
        groupedSlipSystems = SlipSystem.groupSlipSystems(slipSystems)

        return groupedSlipSystems, slipTraceColours

    @staticmethod
    def groupSlipSystems(slipSystems):
        """Groups slip systems by there slip plane.

        Args:
            slipSytems (list(SlipSystem)): A list of slip systems

        Returns:
            list(list(SlipSystem)): A list of list of slip systems
            grouped slip plane.
        """
        distSlipSystems = [slipSystems[0]]
        groupedSlipSystems = [[slipSystems[0]]]

        for slipSystem in slipSystems[1:]:

            for i, distSlipSystem in enumerate(distSlipSystems):
                if slipSystem == distSlipSystem:
                    groupedSlipSystems[i].append(slipSystem)
                    break
            else:
                distSlipSystems.append(slipSystem)
                groupedSlipSystems.append([slipSystem])

        return groupedSlipSystems

