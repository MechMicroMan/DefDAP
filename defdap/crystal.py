# Copyright 2020 Mechanics of Microstructures Group
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
        # Currently only for cubic
        self.crystalSym = crystalSym    # symmetry of material e.g. "cubic", "hexagonal"

        # Stored as Miller indicies (Miller-Bravais for hexagonal)
        self.slipPlaneMiller = slipPlane
        self.slipDirMiller = slipDir

        # Stored as vectors in a cartesian basis
        if crystalSym == "cubic":
            self.slipPlaneOrtho = slipPlane / np.sqrt(np.dot(slipPlane, slipPlane))
            self.slipDirOrtho = slipDir / np.sqrt(np.dot(slipDir, slipDir))
        elif crystalSym == "hexagonal":
            if cOverA is None:
                raise Exception("No c over a ratio given")
            self.cOverA = cOverA

            # Convert plane and dir from Miller-Bravais to Miller
            slipPlaneM = slipPlane[[0, 1, 3]]
            slipDirM = slipDir[[0, 1, 3]]
            slipDirM[[0, 1]] -= slipDir[2]

            # Create L matrix. Transformation from crystal to orthonormal coords
            lMatrix = SlipSystem.lMatrix(1, 1, cOverA, np.pi / 2, np.pi / 2, np.pi * 2 / 3)

            # Create Q matrix fro transforming planes
            qMatrix = SlipSystem.qMatrix(lMatrix)

            # Transform into orthonormal basis and then normalise
            self.slipPlaneOrtho = np.matmul(qMatrix, slipPlaneM)
            self.slipDirOrtho = np.matmul(lMatrix, slipDirM)
            self.slipPlaneOrtho /= np.sqrt(np.dot(self.slipPlaneOrtho, self.slipPlaneOrtho))
            self.slipDirOrtho /= np.sqrt(np.dot(self.slipDirOrtho, self.slipDirOrtho))
        else:
            raise Exception("Only cubic and hexagonal currently supported.")

    # overload ==. Two slip systems are equal if they have the same slip
    # plane in miller
    def __eq__(self, right):
        return np.all(self.slipPlaneMiller == right.slipPlaneMiller)

    @property
    def slipPlane(self):
        """Return the slip plane as an array. For example [0, 0, 1].

        Returns
        -------
        numpy.ndarray
            Slip plane.

        """
        return self.slipPlaneOrtho

    @property
    def slipDir(self):
        """Return the slip direction as an array. For example [0, 0, 1].

        Returns
        -------
        numpy.ndarray
            Slip direction.

        """
        return self.slipDirOrtho

    @property
    def slipPlaneLabel(self):
        """Return the slip plane label. For example '(111)'.

        Returns
        -------
        str
            Slip plane label.

        """
        slipPlane = self.slipPlaneMiller
        if self.crystalSym == "hexagonal":
            return "({:d}{:d}{:d}{:d})".format(slipPlane[0], slipPlane[1], slipPlane[2], slipPlane[3])
        else:
            return "({:d}{:d}{:d})".format(slipPlane[0], slipPlane[1], slipPlane[2])

    @property
    def slipDirLabel(self):
        """Returns the slip direction label. For example '[110]'.

        Returns
        -------
        str
            Slip direction label.

        """
        slipDir = self.slipDirMiller
        if self.crystalSym == "hexagonal":
            return "[{:d}{:d}{:d}{:d}]".format(slipDir[0], slipDir[1], slipDir[2], slipDir[3])
        else:
            return "[{:d}{:d}{:d}]".format(slipDir[0], slipDir[1], slipDir[2])

    @staticmethod
    def loadSlipSystems(name, crystalSym, cOverA=None):
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

        Returns
        -------
        list(list(SlipSystem))
            A list of list of slip systems grouped slip plane.

        Raises
        -------
        IOError
            Raised if not 6/8 integers per line.

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

        # Group slip systems by slip plane
        groupedSlipSystems = SlipSystem.groupSlipSystems(slipSystems)

        return groupedSlipSystems, slipTraceColours

    @staticmethod
    def groupSlipSystems(slipSystems):
        """
        Groups slip systems by their slip plane.

        Parameters
        ----------
        slipSystems : (list(SlipSystem))
            A list of slip systems.

        Returns
        ----------
        list(list(SlipSystem))
            A list of list of slip systems grouped slip plane.

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

    @staticmethod
    def printSlipSystemDirectory():
        """
        Prints the location where slip system definition files are stored.

        """
        packageDir, _ = os.path.split(__file__)
        print("Slip system definition files are stored in directory:")
        print("{:}/slip_systems/".format(packageDir))

    @staticmethod
    def lMatrix(a, b, c, alpha, beta, gamma):
        """Construct l matrix.

        Parameters
        ----------
        a : float
        b : float
        c : float
        alpha : float
        beta : float
        gamma : float

        Returns
        -------
        numpy.ndarray
            l matrix.

        References
        -------
        based on Page 22 of
        Randle and Engle - Introduction To Texture Analysis

        """
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

        lMatrix[2, 2] = c * np.sqrt(1 + 2 * cosAlpha * cosBeta * cosGamma -
                                    cosAlpha**2 - cosBeta**2 - cosGamma**2) / sinGamma

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
        """Construct matrix of reciprocal lattice vectors to transform plane normals.

        Parameters
        ----------
        lMatrix : numpy.ndarray
            l matrix.

        Returns
        -------
        numpy.ndarray
            q matrix.

        References
        -------
        C. T. Young and J. L. Lytton, J. Appl. Phys., vol. 43, no. 4, pp. 1408–1417, 1972.

        """
        a = lMatrix[:, 0]
        b = lMatrix[:, 1]
        c = lMatrix[:, 2]

        volume = abs(np.dot(a, np.cross(b, c)))
        aStar = np.cross(b, c) / volume
        bStar = np.cross(c, a) / volume
        cStar = np.cross(a, b) / volume

        qMatrix = np.stack((aStar, bStar, cStar), axis=1)

        return qMatrix
