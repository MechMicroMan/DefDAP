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

from defdap import plotting

from typing import Union, Tuple, List, Optional


class Quat(object):
    """Class used to define and perform operations on quaternions. These
    are interpreted in the passive sense.

    """
    __slots__ = ['quatCoef']

    def __init__(self, *args, allow_southern: Optional[bool] = False) -> None:
        """
        Construct a Quat object from 4 quat coefficients or an array of
        quat coefficients.

        Parameters
        ----------
        *args
            Variable length argument list.
        allow_southern
            if False, move quat to northern hemisphere.

        """
        # construct with array of quat coefficients
        if len(args) == 1:
            if len(args[0]) != 4:
                raise TypeError("Arrays input must have 4 elements")
            self.quatCoef = np.array(args[0], dtype=float)

        # construct with quat coefficients
        elif len(args) == 4:
            self.quatCoef = np.array(args, dtype=float)

        else:
            raise TypeError("Incorrect argument length. Input should be "
                            "an array of quat coefficients or idividual "
                            "quat coefficients")

        # move to northern hemisphere
        if not allow_southern and self.quatCoef[0] < 0:
            self.quatCoef = self.quatCoef * -1

    @classmethod
    def fromEulerAngles(cls, ph1: float, phi: float, ph2: float) -> 'Quat':
        """Create a quat object from 3 Bunge euler angles.

        Parameters
        ----------
        ph1
            First Euler angle, rotation around Z in radians.
        phi
            Second Euler angle, rotation around new X in radians.
        ph2
            Third Euler angle, rotation around new Z in radians.

        Returns
        -------
        defdap.quat.Quat
            Initialised Quat object.

        """
        # calculate quat coefficients
        quatCoef = np.array([
            np.cos(phi / 2.0) * np.cos((ph1 + ph2) / 2.0),
            -np.sin(phi / 2.0) * np.cos((ph1 - ph2) / 2.0),
            -np.sin(phi / 2.0) * np.sin((ph1 - ph2) / 2.0),
            -np.cos(phi / 2.0) * np.sin((ph1 + ph2) / 2.0)
        ], dtype=float)

        # call constructor
        return cls(quatCoef)

    @classmethod
    def fromAxisAngle(cls, axis: np.ndarray, angle: float) -> 'Quat':
        """Create a quat object from a rotation around an axis. This
        creates a quaternion to represent the passive rotation (-ve axis).

        Parameters
        ----------
        axis
            Axis that the rotation is applied around.
        angle
            Magnitude of rotation in radians.

        Returns
        -------
        defdap.quat.Quat
            Initialised Quat object.

        """
        # normalise the axis vector
        axis = np.array(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        # calculate quat coefficients
        quatCoef = np.zeros(4, dtype=float)
        quatCoef[0] = np.cos(angle / 2)
        quatCoef[1:4] = -np.sin(angle / 2) * axis

        # call constructor
        return cls(quatCoef)

    def eulerAngles(self) -> np.ndarray:
        """Calculate the Euler angle representation for this rotation.

        Returns
        -------
        eulers : numpy.ndarray, shape 3
            Bunge euler angles (in radians).

        References
        ----------
            Melcher A. et al., 'Conversion of EBSD data by a quaternion
            based algorithm to be used for grain structure simulations',
            Technische Mechanik, 30(4)401 – 413

            Rowenhorst D. et al., 'Consistent representations of and
            conversions between 3D rotations',
            Model. Simul. Mater. Sci. Eng., 23(8)

        """
        eulers = np.empty(3, dtype=float)

        q = self.quatCoef
        q03 = q[0]**2 + q[3]**2
        q12 = q[1]**2 + q[2]**2
        chi = np.sqrt(q03 * q12)

        if chi == 0 and q12 == 0:
            eulers[0] = np.arctan2(-2 * q[0] * q[3], q[0]**2 - q[3]**2)
            eulers[1] = 0
            eulers[2] = 0

        elif chi == 0 and q03 == 0:
            eulers[0] = np.arctan2(2 * q[1] * q[2], q[1]**2 - q[2]**2)
            eulers[1] = np.pi
            eulers[2] = 0

        else:
            cosPh1 = (-q[0] * q[1] - q[2] * q[3]) / chi
            sinPh1 = (-q[0] * q[2] + q[1] * q[3]) / chi

            cosPhi = q[0]**2 + q[3]**2 - q[1]**2 - q[2]**2
            sinPhi = 2 * chi

            cosPh2 = (-q[0] * q[1] + q[2] * q[3]) / chi
            sinPh2 = (q[1] * q[3] + q[0] * q[2]) / chi

            eulers[0] = np.arctan2(sinPh1, cosPh1)
            eulers[1] = np.arctan2(sinPhi, cosPhi)
            eulers[2] = np.arctan2(sinPh2, cosPh2)

        if eulers[0] < 0:
            eulers[0] += 2 * np.pi
        if eulers[2] < 0:
            eulers[2] += 2 * np.pi

        return eulers

    def rotMatrix(self) -> np.ndarray:
        """Calculate the rotation matrix representation for this rotation.

        Returns
        -------
        rotMatrix : numpy.ndarray, shape (3, 3)
            Rotation matrix.

        References
        ----------
            Melcher A. et al., 'Conversion of EBSD data by a quaternion
            based algorithm to be used for grain structure simulations',
            Technische Mechanik, 30(4)401 – 413

            Rowenhorst D. et al., 'Consistent representations of and
            conversions between 3D rotations',
            Model. Simul. Mater. Sci. Eng., 23(8)

        """
        rotMatrix = np.empty((3, 3), dtype=float)

        q = self.quatCoef
        qbar = q[0]**2 - q[1]**2 - q[2]**2 - q[3]**2

        rotMatrix[0, 0] = qbar + 2 * q[1]**2
        rotMatrix[0, 1] = 2 * (q[1] * q[2] - q[0] * q[3])
        rotMatrix[0, 2] = 2 * (q[1] * q[3] + q[0] * q[2])

        rotMatrix[1, 0] = 2 * (q[1] * q[2] + q[0] * q[3])
        rotMatrix[1, 1] = qbar + 2 * q[2]**2
        rotMatrix[1, 2] = 2 * (q[2] * q[3] - q[0] * q[1])

        rotMatrix[2, 0] = 2 * (q[1] * q[3] - q[0] * q[2])
        rotMatrix[2, 1] = 2 * (q[2] * q[3] + q[0] * q[1])
        rotMatrix[2, 2] = qbar + 2 * q[3]**2

        return rotMatrix

    # show components when the quat is printed
    def __repr__(self) -> str:
        return "[{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*self.quatCoef)

    def __str__(self) -> str:
        return self.__repr__()

    def _plotIPF(
        self,
        direction: np.ndarray,
        symGroup: str,
        **kwargs
    ) -> 'plotting.PolePlot':
        Quat.plotIPF([self], direction, symGroup, **kwargs)

    # overload * operator for quaternion product and vector product
    def __mul__(self, right: 'Quat', allow_southern: bool = False) -> 'Quat':
        if isinstance(right, type(self)):   # another quat
            newQuatCoef = np.zeros(4, dtype=float)
            newQuatCoef[0] = (
                    self.quatCoef[0] * right.quatCoef[0] -
                    np.dot(self.quatCoef[1:4], right.quatCoef[1:4])
            )
            newQuatCoef[1:4] = (
                    self.quatCoef[0] * right.quatCoef[1:4] +
                    right.quatCoef[0] * self.quatCoef[1:4] +
                    np.cross(self.quatCoef[1:4], right.quatCoef[1:4])
            )
            return Quat(newQuatCoef, allow_southern=allow_southern)

        raise TypeError("{:} - {:}".format(type(self), type(right)))

    def dot(self, right: 'Quat') -> float:
        """ Calculate dot product between two quaternions.

        Parameters
        ----------
        right
            Right hand quaternion.

        Returns
        -------
        float
            Dot product.

        """
        if isinstance(right, type(self)):
            return np.dot(self.quatCoef, right.quatCoef)
        raise TypeError()

    # overload + operator
    def __add__(self, right: 'Quat') -> 'Quat':
        if isinstance(right, type(self)):
            return Quat(self.quatCoef + right.quatCoef)
        raise TypeError()

    # overload += operator
    def __iadd__(self, right: 'Quat') -> 'Quat':
        if isinstance(right, type(self)):
            self.quatCoef += right.quatCoef
            return self
        raise TypeError()

    # allow array like setting/getting of components
    def __getitem__(self, key: int) -> float:
        return self.quatCoef[key]

    def __setitem__(self, key: int, value: float) -> None:
        self.quatCoef[key] = value

    def norm(self) -> float:
        """Calculate the norm of the quaternion.

        Returns
        -------
        float
            Norm of the quaternion.

        """
        return np.sqrt(np.dot(self.quatCoef[0:4], self.quatCoef[0:4]))

    def normalise(self) -> 'Quat':
        """ Normalise the quaternion (turn it into an unit quaternion).

        Returns
        -------
        defdap.quat.Quat
            Normalised quaternion.

        """
        self.quatCoef /= self.norm()
        return

    # also the inverse if this is a unit quaternion
    @property
    def conjugate(self) -> 'Quat':
        """Calculate the conjugate of the quaternion.

        Returns
        -------
        defdap.quat.Quat
            Conjugate of quaternion.

        """
        return Quat(self[0], -self[1], -self[2], -self[3])

    def transformVector(
        self,
        vector: Union[Tuple, List, np.ndarray]
    ) -> np.ndarray:
        """
        Transforms vector by the quaternion. For passive EBSD quaterions
        this is a transformation from sample space to crystal space.
        Perform on conjugate of quaternion for crystal to sample. For a
        quaternion representing a passive rotation from CS1 to CS2 and a
        fixed vector V defined in CS1, this gives the coordinates
        of V in CS2.

        Parameters
        ----------
        vector : numpy.ndarray, shape 3 or equivalent
            Vector to transform.

        Returns
        -------
        numpy.ndarray, shape 3
            Transformed vector.

        """
        if not isinstance(vector, (np.ndarray, list, tuple)):
            raise TypeError("Vector must be a tuple, list or numpy array.")
        if np.array(vector).shape != (3,):
            raise TypeError("Vector must be length 3.")

        vectorQuat = Quat(0, *vector)
        vectorQuatTransformed = self.__mul__(
            vectorQuat.__mul__(self.conjugate, allow_southern=True),
            allow_southern=True
        )
        return vectorQuatTransformed.quatCoef[1:4]

    def misOri(
        self,
        right: 'Quat',
        symGroup: str,
        returnQuat: Optional[int] = 0
    ) -> Tuple[float, 'Quat']:
        """
        Calculate misorientation angle between 2 orientations taking
        into account the symmetries of the crystal structure.
        Angle is 2*arccos(output).

        Parameters
        ----------
        right
            Orientation to find misorientation to.
        symGroup
            Crystal type (cubic, hexagonal).
        returnQuat
            What to return: 0 for minimum misorientation, 1 for
            symmetric equivalent with minimum misorientation, 2 for both.

        Returns
        -------
        float
            Minimum misorientation.
        defdap.quat.Quat
            Symmetric equivalent orientation with minimum misorientation.

        """
        if isinstance(right, type(self)):
            # looking for max of this as it is cos of misorientation angle
            minMisOri = 0
            # loop over symmetrically equivalent orientations
            for sym in Quat.symEqv(symGroup):
                quatSym = sym * right
                currentMisOri = abs(self.dot(quatSym))
                if currentMisOri > minMisOri:   # keep if misorientation lower
                    minMisOri = currentMisOri
                    minQuatSym = quatSym

            if returnQuat == 1:
                return minQuatSym
            elif returnQuat == 2:
                return minMisOri, minQuatSym
            else:
                return minMisOri
        raise TypeError("Input must be a quaternion.")

    def misOriAxis(self, right: 'Quat') -> np.ndarray:
        """
        Calculate misorientation axis between 2 orientations. This
        does not consider symmetries of the crystal structure.

        Parameters
        ----------
        right : defdap.quat.Quat
            Orientation to find misorientation axis to.

        Returns
        -------
        numpy.ndarray, shape 3
            Axis of misorientation.

        """
        if isinstance(right, type(self)):
            Dq = right * self.conjugate
            Dq = Dq.quatCoef
            misOriAxis = 2 * Dq[1:4] * np.arccos(Dq[0]) / np.sqrt(1 - Dq[0]**2)
            return misOriAxis
        raise TypeError("Input must be a quaternion.")

    def plotIPF(
        self,
        direction: np.ndarray,
        symGroup: str,
        projection:  Optional[str] = None,
        plot: Optional['plotting.Plot'] = None,
        fig: Optional['matplotlib.figure.Figure'] = None,
        ax: Optional['matplotlib.axes.Axes'] = None,
        plotColourBar: Optional[bool] = False,
        clabel: Optional[str] = "",
        makeInteractive: Optional[bool] = False,
        markerColour: Optional[Union[List[str], str]] = None,
        markerSize: Optional[float] = 40,
        **kwargs
    ) -> 'plotting.PolePlot':
        """
        Plot IPF of orientation, with relation to specified sample direction.

        Parameters
        ----------
        direction
            Sample reference direction for IPF.
        symGroup
            Crystal type (cubic, hexagonal).
        projection
             Projection to use. Either string (stereographic or lambert)
             or a function.
        plot
            Defdap plot to plot on.
        fig
            Figure to plot on, if not provided the current
            active axis is used.
        ax
            Axis to plot on, if not provided the current
            active axis is used.
        makeInteractive
            If true, make the plot interactive.
        plotColourBar : bool
            If true, plot a colour bar next to the map.
        clabel : str
            Label for the colour bar.
        markerColour: str or list of str
            Colour of markers (only used for half and half colouring,
            otherwise use argument c).
        markerSize
            Size of markers (only used for half and half colouring,
            otherwise use argument s).
        kwargs
            All other arguments are passed to :func:`defdap.plotting.PolePlot.addPoints`.

        """
        plotParams = {'marker': '+'}
        plotParams.update(kwargs)

        # Works as an instance or static method on a list of Quats
        if isinstance(self, Quat):
            quats = [self]
        else:
            quats = self

        alphaFund, betaFund = Quat.calcFundDirs(quats, direction, symGroup)

        if plot is None:
            plot = plotting.PolePlot(
                "IPF", symGroup, projection=projection,
                ax=ax, fig=fig, makeInteractive=makeInteractive
            )
        plot.addPoints(
            alphaFund, betaFund,
            markerColour=markerColour, markerSize=markerSize,
            **plotParams
        )

        if plotColourBar:
            plot.addColourBar(clabel)

        return plot

    def plotUnitCell(
        self,
        crystalStructure: 'defdap.crystal.CrystalStructure',
        OI: Optional[bool] = True,
        plot: Optional['plotting.CrystalPlot'] = None,
        fig: Optional['matplotlib.figure.Figure'] = None,
        ax: Optional['matplotlib.axes.Axes'] = None,
        makeInteractive: Optional[bool] = False,
        **kwargs
    ) -> 'plotting.CrystalPlot':
        """Plots a unit cell.

        Parameters
        ----------
        crystalStructure
            Crystal structure.
        OI
            True if using oxford instruments system.
        plot
            Plot object to plot to.
        fig
            Figure to plot on, if not provided the current active axis is used.
        ax
            Axis to plot on, if not provided the current active axis is used.
        makeInteractive
            True to make the plot interactive.
        kwargs
            All other arguments are passed to :func:`defdap.plotting.CrystalPlot.addVerts`.

        """
        # Set default plot parameters then update with any input
        plotParams = {}
        plotParams.update(kwargs)

        # TODO: most of this should be moved to either the crystal or
        #  plotting module

        vert = crystalStructure.vertices
        faces = crystalStructure.faces

        if crystalStructure.name == 'hexagonal':
            szFac = 0.18
            if OI:
                # Add 30 degrees to phi2 for OI
                eulerAngles = self.eulerAngles()
                eulerAngles[2] += np.pi / 6
                gg = Quat.fromEulerAngles(*eulerAngles).rotMatrix().T
            else:
                gg = self.rotMatrix().T

        elif crystalStructure.name == 'cubic':
            szFac = 0.25
            gg = self.rotMatrix().T

        # Rotate the lattice cell points
        pts = np.matmul(gg, vert.T).T * szFac

        # Plot unit cell
        planes = []
        for face in faces:
            planes.append(pts[face, :])

        if plot is None:
            plot = plotting.CrystalPlot(
                ax=ax, fig=fig, makeInteractive=makeInteractive
            )

        plot.ax.set_xlim3d(-0.15, 0.15)
        plot.ax.set_ylim3d(-0.15, 0.15)
        plot.ax.set_zlim3d(-0.15, 0.15)
        plot.ax.view_init(azim=270, elev=90)
        plot.ax._axis3don = False

        plot.addVerts(planes, **plotParams)

        return plot

# Static methods

    @staticmethod
    def createManyQuats(eulerArray: np.ndarray) -> np.ndarray:
        """Create a an array of quats from an array of Euler angles.

        Parameters
        ----------
        eulerArray
            Array of Bunge Euler angles of shape 3 x n x ... x m.

        Returns
        -------
        quats : numpy.ndarray(defdap.quat.Quat)
            Array of quat objects of shape n x ... x m.

        """
        ph1 = eulerArray[0]
        phi = eulerArray[1]
        ph2 = eulerArray[2]
        oriShape = eulerArray.shape[1:]

        quatComps = np.zeros((4,) + oriShape, dtype=float)

        quatComps[0] = np.cos(phi / 2.0) * np.cos((ph1 + ph2) / 2.0)
        quatComps[1] = -np.sin(phi / 2.0) * np.cos((ph1 - ph2) / 2.0)
        quatComps[2] = -np.sin(phi / 2.0) * np.sin((ph1 - ph2) / 2.0)
        quatComps[3] = -np.cos(phi / 2.0) * np.sin((ph1 + ph2) / 2.0)

        quats = np.empty(oriShape, dtype=Quat)

        for i, idx in enumerate(np.ndindex(oriShape)):
            quats[idx] = Quat(quatComps[(slice(None),) + idx])

        return quats

    @staticmethod
    def multiplyManyQuats(quats: List['Quat'], right: 'Quat') -> List['Quat']:
        """ Multiply all quats in a list of quats, by a single quat.

        Parameters
        ----------
        quats
            List of quats to be operated on.
        right
            Single quaternion to multiply with the list of quats.

        Returns
        -------
        list(defdap.quat.Quat)

        """
        quatArray = np.array([q.quatCoef for q in quats])

        tempArray = np.zeros((len(quatArray),4), dtype=float)
        tempArray[...,0] = ((quatArray[...,0] * right.quatCoef[0]) -
                             np.dot(quatArray[...,1:4], right.quatCoef[1:4]))

        tempArray[...,1:4] = ((quatArray[...,0,None] * right.quatCoef[None,1:4]) +
                              (right.quatCoef[0] * quatArray[...,1:4]) +
                              np.cross(quatArray[...,1:4], right.quatCoef[1:4]))

        return [Quat(coefs) for coefs in tempArray]

    @staticmethod
    def extract_quat_comps(quats: np.ndarray) -> np.ndarray:
        """Return a NumPy array of the provided quaternion components

        Input quaternions may be given as a list of Quat objects or any iterable
        whose items have 4 components which map to the quaternion.

        Parameters
        ----------
        quats : numpy.ndarray(defdap.quat.Quat)
            A list of Quat objects to return the components of

        Returns
        -------
        numpy.ndarray
            Array of quaternion components, shape (4, ..)

        """
        quats = np.array(quats)
        quat_comps = np.empty((4,) + quats.shape)
        for idx in np.ndindex(quats.shape):
            quat_comps[(slice(None),) + idx] = quats[idx].quatCoef

        return quat_comps

    @staticmethod
    def calcSymEqvs(
        quats: np.ndarray,
        symGroup: str,
        dtype: Optional[type] = np.float
    ) -> np.ndarray:
        """Calculate all symmetrically equivalent quaternions of given quaternions.

        Parameters
        ----------
        quats : numpy.ndarray(defdap.quat.Quat)
            Array of quat objects.
        symGroup
            Crystal type (cubic, hexagonal).
        dtype
            Data type used for calculation, defaults to np.float.

        Returns
        -------
        quatComps: numpy.ndarray, shape: (numSym x 4 x numQuats)
            Array containing all symmetrically equivalent quaternion components of input quaternions.

        """
        syms = Quat.symEqv(symGroup)
        quatComps = np.empty((len(syms), 4, len(quats)), dtype=dtype)

        # store quat components in array
        quatComps[0] = Quat.extract_quat_comps(quats)

        # calculate symmetrical equivalents
        for i, sym in enumerate(syms[1:], start=1):
            # sym[i] * quat for all points (* is quaternion product)
            quatComps[i, 0, :] = (
                quatComps[0, 0, :] * sym[0] - quatComps[0, 1, :] * sym[1] -
                quatComps[0, 2, :] * sym[2] - quatComps[0, 3, :] * sym[3])
            quatComps[i, 1, :] = (
                quatComps[0, 0, :] * sym[1] + quatComps[0, 1, :] * sym[0] -
                quatComps[0, 2, :] * sym[3] + quatComps[0, 3, :] * sym[2])
            quatComps[i, 2, :] = (
                quatComps[0, 0, :] * sym[2] + quatComps[0, 2, :] * sym[0] -
                quatComps[0, 3, :] * sym[1] + quatComps[0, 1, :] * sym[3])
            quatComps[i, 3, :] = (
                quatComps[0, 0, :] * sym[3] + quatComps[0, 3, :] * sym[0] -
                quatComps[0, 1, :] * sym[2] + quatComps[0, 2, :] * sym[1])

            # swap into positive hemisphere if required
            quatComps[i, :, quatComps[i, 0, :] < 0] *= -1

        return quatComps

    @staticmethod
    def calcAverageOri(
        quatComps: np.ndarray
    ) -> 'Quat':
        """Calculate the average orientation of given quats.

        Parameters
        ----------
        quatComps : numpy.ndarray
            Array containing all symmetrically equivalent quaternion components of given quaternions
            (shape: numSym x 4 x numQuats), can be calculated with :func:`Quat.calcSymEqvs`.

        Returns
        -------
        avOri : defdap.quat.Quat
            Average orientation of input quaternions.

        """
        avOri = np.copy(quatComps[0, :, 0])
        currMisOris = np.empty(quatComps.shape[0])

        for i in range(1, quatComps.shape[2]):
            # calculate misorientation between current average and all
            # symmetrical equivalents. Dot product of each symm quat in
            # quatComps with refOri for point i
            currMisOris[:] = abs(np.einsum(
                "ij,j->i", quatComps[:, :, i], avOri
            ))

            # find min misorientation with current average then add to it
            maxIdx = np.argmax(currMisOris[:])
            avOri += quatComps[maxIdx, :, i]

        # Convert components back to a quat and normalise
        avOri = Quat(avOri)
        avOri.normalise()

        return avOri

    @staticmethod
    def calcMisOri(
        quatComps: np.ndarray,
        refOri: 'Quat'
    ) -> Tuple[np.ndarray, 'Quat']:
        """Calculate the misorientation between the quaternions and a reference quaternion.

        Parameters
        ----------
        quatComps
            Array containing all symmetrically equivalent quaternion components of given quaternions
            (shape: numSym x 4 x numQuats), can be calculated from quats with :func:`Quat.calcSymEqvs` .
        refOri
            Reference orientation.

        Returns
        -------
        minMisOris : numpy.ndarray, len numQuats
            Minimum misorientation between quats and reference orientation.
        minQuatComps : defdap.quat.Quat
            Quaternion components describing minimum misorientation between quats and reference orientation.

        """
        misOris = np.empty((quatComps.shape[0], quatComps.shape[2]))

        # Dot product of each quat in quatComps with refOri
        misOris[:, :] = abs(np.einsum("ijk,j->ik", quatComps, refOri.quatCoef))

        maxIdxs0 = np.argmax(misOris, axis=0)
        maxIdxs1 = np.arange(misOris.shape[1])

        minMisOris = misOris[maxIdxs0, maxIdxs1]

        minQuatComps = quatComps[maxIdxs0, :, maxIdxs1].transpose()

        minMisOris[minMisOris > 1] = 1

        return minMisOris, minQuatComps

    @staticmethod
    def polarAngles(x: np.ndarray, y: np.ndarray, z: np.ndarray):
        """Convert Cartesian coordinates to polar coordinates, for an
        unit vector.

        Parameters
        ----------
        x : numpy.ndarray(float)
            x coordinate.
        y : numpy.ndarray(float)
            y coordinate.
        z : numpy.ndarray(float)
            z coordinate.

        Returns
        -------
        float, float
            inclination angle and azimuthal angle (around z axis from x
            in anticlockwise as per ISO).

        """
        mod = np.sqrt(x**2 + y**2 + z**2)
        x = x / mod
        y = y / mod
        z = z / mod

        alpha = np.arccos(z)
        beta = np.arctan2(y, x)

        return alpha, beta

    @staticmethod
    def calcIPFcolours(
        quats: np.ndarray,
        direction: np.ndarray,
        symGroup: str,
        dtype: Optional[type] = np.float32
    ) -> np.ndarray:
        """
        Calculate the RGB colours, based on the location of the given quats
        on the fundamental region of the IPF for the sample direction specified.

        Parameters
        ----------
        quats : numpy.ndarray(defdap.quat.Quat)
            Array of quat objects.
        direction
            Direction in sample space.
        symGroup
            Crystal type (cubic, hexagonal).
        dtype
            Data type to use for calculation.

        Returns
        -------
        numpy.ndarray, shape (3, numQuats)
            Array of rgb colours for each quat.

        References
        -------
        Stephen Cluff (BYU) - IPF_rgbcalc.m subroutine in OpenXY
        https://github.com/BYU-MicrostructureOfMaterials/OpenXY/blob/master/Code/PlotIPF.m

        """
        numQuats = len(quats)

        alphaFund, betaFund = Quat.calcFundDirs(
            quats, direction, symGroup, dtype=dtype
        )

        # revert to cartesians
        dirvec = np.empty((3, numQuats), dtype=dtype)
        dirvec[0, :] = np.sin(alphaFund) * np.cos(betaFund)
        dirvec[1, :] = np.sin(alphaFund) * np.sin(betaFund)
        dirvec[2, :] = np.cos(alphaFund)

        if symGroup == 'cubic':
            poleDirections = np.array([[0, 0, 1], 
                                        [1, 0, 1]/np.sqrt(2), 
                                        [1, 1, 1]/np.sqrt(3)], dtype=dtype)
        if symGroup == 'hexagonal':
            poleDirections = np.array([[0, 0, 1], 
                                        [np.sqrt(3), 1, 0]/np.sqrt(4), 
                                        [1, 0, 0]], dtype=dtype)

        rvect = np.broadcast_to(poleDirections[0].reshape((-1, 1)), (3, numQuats))
        gvect = np.broadcast_to(poleDirections[1].reshape((-1, 1)), (3, numQuats))
        bvect = np.broadcast_to(poleDirections[2].reshape((-1, 1)), (3, numQuats))

        rgb = np.zeros((3, numQuats), dtype=dtype)

        # Red Component
        RDirPlane = np.cross(dirvec, rvect, axis=0)
        GBplane = np.cross(bvect, gvect, axis=0)
        Rintersect = np.cross(RDirPlane, GBplane, axis=0)
        NORM = np.linalg.norm(Rintersect, axis=0, keepdims=True)
        NORM[NORM == 0] = 1       #Prevent division by zero
        Rintersect /= NORM

        temp = np.arccos(np.clip(np.einsum("ij,ij->j", dirvec, Rintersect), -1, 1))
        Rintersect[:, temp > (np.pi / 2)] *= -1
        rgb[0, :] = np.divide(
            np.arccos(np.clip(np.einsum("ij,ij->j", dirvec, Rintersect), -1, 1)),
            np.arccos(np.clip(np.einsum("ij,ij->j", rvect, Rintersect), -1, 1))
        )

        # Green Component
        GDirPlane = np.cross(dirvec, gvect, axis=0)
        RBplane = np.cross(rvect, bvect, axis=0)
        Gintersect = np.cross(GDirPlane, RBplane, axis=0)
        NORM = np.linalg.norm(Gintersect, axis=0, keepdims=True)
        NORM[NORM == 0] = 1       #Prevent division by zero
        Gintersect /= NORM

        temp = np.arccos(np.clip(np.einsum("ij,ij->j", dirvec, Gintersect), -1, 1))
        Gintersect[:, temp > (np.pi / 2)] *= -1
        rgb[1, :] = np.divide(
            np.arccos(np.clip(np.einsum("ij,ij->j", dirvec, Gintersect), -1, 1)),
            np.arccos(np.clip(np.einsum("ij,ij->j", gvect, Gintersect), -1, 1))
        )

        # Blue Component
        BDirPlane = np.cross(dirvec, bvect, axis=0)
        RGplane = np.cross(gvect, rvect, axis=0)
        Bintersect = np.cross(BDirPlane, RGplane, axis=0)
        NORM = np.linalg.norm(Bintersect, axis=0, keepdims=True)
        NORM[NORM == 0] = 1       #Prevent division by zero
        Bintersect /= NORM

        temp = np.arccos(np.clip(np.einsum("ij,ij->j", dirvec, Bintersect), -1, 1))
        Bintersect[:, temp > (np.pi / 2)] *= -1
        rgb[2, :] = np.divide(
            np.arccos(np.clip(np.einsum("ij,ij->j", dirvec, Bintersect), -1, 1)),
            np.arccos(np.clip(np.einsum("ij,ij->j", bvect, Bintersect), -1, 1))
        )
        rgb /= np.amax(rgb, axis=0)

        return rgb

    @staticmethod
    def calcFundDirs(
        quats: np.ndarray,
        direction: np.ndarray,
        symGroup: str,
        dtype: Optional[type] = np.float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform the sample direction to crystal coords based on the quats
        and find the ones in the fundamental sector of the IPF.

        Parameters
        ----------
        quats: array_like(defdap.quat.Quat)
            Array of quat objects.
        direction
            Direction in sample space.
        symGroup
            Crystal type (cubic, hexagonal).
        dtype
            Data type to use for calculation.

        Returns
        -------
        float, float
            inclination angle and azimuthal angle (around z axis from x in anticlockwise).

        """
        # convert direction to float array
        direction = np.array(direction, dtype=dtype)

        # get array of symmetry operations. shape - (numSym, 4, numQuats)
        quatCompsSym = Quat.calcSymEqvs(quats, symGroup, dtype=dtype)

        # array to store crystal directions for all orientations and symmetries
        directionCrystal = np.empty(
            (3, quatCompsSym.shape[0], quatCompsSym.shape[2]), dtype=dtype
        )

        # temp variables to use below
        quatDotVec = (quatCompsSym[:, 1, :] * direction[0] +
                      quatCompsSym[:, 2, :] * direction[1] +
                      quatCompsSym[:, 3, :] * direction[2])
        temp = (np.square(quatCompsSym[:, 0, :]) -
                np.square(quatCompsSym[:, 1, :]) -
                np.square(quatCompsSym[:, 2, :]) -
                np.square(quatCompsSym[:, 3, :]))

        # transform the pole direction to crystal coords for all
        # orientations and symmetries
        # (quatCompsSym * vectorQuat) * quatCompsSym.conjugate
        directionCrystal[0, :, :] = (
                2 * quatDotVec * quatCompsSym[:, 1, :] +
                temp * direction[0] +
                2 * quatCompsSym[:, 0, :] * (
                        quatCompsSym[:, 2, :] * direction[2] -
                        quatCompsSym[:, 3, :] * direction[1]
                )
        )
        directionCrystal[1, :, :] = (
                2 * quatDotVec * quatCompsSym[:, 2, :] +
                temp * direction[1] +
                2 * quatCompsSym[:, 0, :] * (
                        quatCompsSym[:, 3, :] * direction[0] -
                        quatCompsSym[:, 1, :] * direction[2]
                )
        )
        directionCrystal[2, :, :] = (
                2 * quatDotVec * quatCompsSym[:, 3, :] +
                temp * direction[2] +
                2 * quatCompsSym[:, 0, :] * (
                        quatCompsSym[:, 1, :] * direction[1] -
                        quatCompsSym[:, 2, :] * direction[0]
                )
        )

        # normalise vectors
        directionCrystal /= np.sqrt(np.einsum(
            'ijk,ijk->jk', directionCrystal, directionCrystal
        ))

        # move all vectors into north hemisphere
        directionCrystal[:, directionCrystal[2, :, :] < 0] *= -1

        # convert to spherical coordinates
        alpha, beta = Quat.polarAngles(
            directionCrystal[0], directionCrystal[1], directionCrystal[2]
        )

        # find the poles in the fundamental triangle
        if symGroup == "cubic":
            # first beta should be between 0 and 45 deg leaving 3
            # symmetric equivalents per orientation
            trialPoles = np.logical_and(beta >= 0, beta <= np.pi / 4)

            # if less than 3 left need to expand search slighly to
            # catch edge cases
            if np.any(np.sum(trialPoles, axis=0) < 3):
                deltaBeta = 1e-8
                trialPoles = np.logical_and(beta >= -deltaBeta,
                                            beta <= np.pi / 4 + deltaBeta)

            # now of symmetric equivalents left we want the one with
            # minimum alpha
            min_alpha_idx = np.nanargmin(np.where(trialPoles==False, np.nan, alpha), axis=0)
            betaFund = beta[min_alpha_idx, np.arange(len(min_alpha_idx))]
            alphaFund = alpha[min_alpha_idx, np.arange(len(min_alpha_idx))]

        elif symGroup == "hexagonal":
            # first beta should be between 0 and 30 deg leaving 1
            # symmetric equivalent per orientation
            trialPoles = np.logical_and(beta >= 0, beta <= np.pi / 6)
            # if less than 1 left need to expand search slighly to
            # catch edge cases
            if np.any(np.sum(trialPoles, axis=0) < 1):
                deltaBeta = 1e-8
                trialPoles = np.logical_and(beta >= -deltaBeta,
                                            beta <= np.pi / 6 + deltaBeta)

            # non-indexed points cause more than 1 symmetric equivalent, use this
            # to pick one and filter non-indexed points later
            first_idx = (trialPoles==True).argmax(axis=0)
            betaFund = beta[first_idx, np.arange(len(first_idx))]
            alphaFund = alpha[first_idx, np.arange(len(first_idx))]

        else:
            raise Exception("symGroup must be cubic or hexagonal")

        return alphaFund, betaFund

    @staticmethod
    def symEqv(symGroup: str) -> List['Quat']:
        """Returns all symmetric equivalents for a given crystal type.
        LEGACY: move to use symmetries defined in crystal structures

        Parameters
        ----------
        symGroup : str
            Crystal type (cubic, hexagonal).

        Returns
        -------
        list of defdap.quat.Quat
            Symmetrically equivalent quats.

        """
        # Dirty fix to stop circular dependency
        from defdap.crystal import crystalStructures
        try:
            return crystalStructures[symGroup].symmetries
        except KeyError:
            # return just identity if unknown structure
            return [Quat(1.0, 0.0, 0.0, 0.0)]
