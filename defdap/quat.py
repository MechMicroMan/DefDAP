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

import numpy as np

from defdap import plotting


class Quat(object):
    def __init__(self, *args, **kwargs):
        """
        Create a quat object from 3 Bunge euler angles, 4 quat coefficients or an array of 4 quat coefficients
        """
        # construct with Bunge euler angles (radians, ZXZ)
        if len(args) == 3:
            ph1 = args[0]
            phi = args[1]
            ph2 = args[2]

            self.quatCoef = np.array([np.cos(phi / 2.0) * np.cos((ph1 + ph2) / 2.0),
                                      -np.sin(phi / 2.0) * np.cos((ph1 - ph2) / 2.0),
                                      -np.sin(phi / 2.0) * np.sin((ph1 - ph2) / 2.0),
                                      -np.cos(phi / 2.0) * np.sin((ph1 + ph2) / 2.0)],
                                     dtype=float)

        # construct with array of quat coefficients
        elif len(args) == 1:
            if len(args[0]) == 4:
                self.quatCoef = np.array(args[0], dtype=float)
            else:
                raise Exception("Arrays should have 4 elements, use 3 arguments for Euler angles")

        # construct with quat coefficients
        elif len(args) == 4:
            self.quatCoef = np.array([args[0], args[1], args[2], args[3]], dtype=float)
        elif len(args) > 4 or len(args) ==2:
            raise Exception("Incorrect argument length\n"
                            "Allowable inputs are 3 Bunge euler angles, array of quat coeffs or quat coeffs")

        if self.quatCoef[0] < 0:
            self.quatCoef = self.quatCoef * -1

        # overload static method with instance method of same name in object
        self.plotIPF = self._plotIPF

    @classmethod
    def fromAxisAngle(cls, axis, angle):
        """Create a quat object from an axis angle pair

        Args:
            axis (np.array size 3): Axis of rotation
            angle (float): Rotation around axis (radians)

        Returns:
            Quat: Initialised Quat object
        """

        # normalise the axis vector
        axis = axis / np.sqrt(np.dot(axis, axis))
        # calculate quat coefficients
        quatCoef = np.zeros(4, dtype=float)
        quatCoef[0] = np.cos(angle / 2)
        quatCoef[1:4] = np.sin(angle / 2) * axis

        # call constructor
        return cls(quatCoef)

    def eulerAngles(self):
        """Calculate euler angles for quat

        Returns:
            eulers (np.array size 3): Bunge euler angles (in radians)

        References:
            Melcher A. et al., 'Conversion of EBSD data by a quaternion based algorithm
            to be used for grain structure simulations', Technische Mechanik, 30(4)401 – 413
            Rowenhorst D. et al., 'Consistent representations of and conversions between
            3D rotations', Model. Simul. Mater. Sci. Eng., 23(8) 
        """

        eulers = np.empty(3, dtype=float)

        q = self.quatCoef
        q03 = q[0]**2 + q[3]**2
        q12 = q[1]**2 + q[2]**2
        chi = np.sqrt(q03 * q12)

        if (chi == 0 and q12 == 0):
            eulers[0] = np.arctan2(-2 * q[0] * q[3],
                                   q[0]**2 - q[3]**2)
            eulers[1] = 0
            eulers[2] = 0

        elif (chi == 0 and q03 == 0):
            eulers[0] = np.arctan2(2 * q[1] * q[2],
                                   q[1]**2 - q[2]**2)
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

    def rotMatrix(self):
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
    def __repr__(self):
        return "[{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*self.quatCoef)

    def __str__(self):
        return self.__repr__()

    def _plotIPF(self, direction, symGroup, **kwargs):
        Quat.plotIPF([self], direction, symGroup, **kwargs)

    # overload * operator for quaterion product and vector product
    def __mul__(self, right):
        if isinstance(right, type(self)):   # another quat
            newQuatCoef = np.zeros(4, dtype=float)
            newQuatCoef[0] = (self.quatCoef[0] * right.quatCoef[0] -
                              np.dot(self.quatCoef[1:4], right.quatCoef[1:4]))
            newQuatCoef[1:4] = (self.quatCoef[0] * right.quatCoef[1:4] +
                                right.quatCoef[0] * self.quatCoef[1:4] +
                                np.cross(self.quatCoef[1:4], right.quatCoef[1:4]))
            return Quat(newQuatCoef)
        raise TypeError()

    # # overload % operator for dot product
    # def __mod__(self, right):
    def dot(self, right):
        if isinstance(right, type(self)):
            return np.dot(self.quatCoef, right.quatCoef)
        raise TypeError()

    # overload + operator
    def __add__(self, right):
        if isinstance(right, type(self)):
            return Quat(self.quatCoef + right.quatCoef)
        raise TypeError()

    # overload += operator
    def __iadd__(self, right):
        if isinstance(right, type(self)):
            self.quatCoef += right.quatCoef
            return self
        raise TypeError()

    # allow array like setting/getting of components
    def __getitem__(self, key):
        return self.quatCoef[key]

    def __setitem__(self, key, value):
        self.quatCoef[key] = value
        return

    def norm(self):
        return np.sqrt(np.dot(self.quatCoef[0:4], self.quatCoef[0:4]))

    def normalise(self):
        self.quatCoef /= self.norm()
        return

    # also the inverse if this is a unit quaterion
    @property
    def conjugate(self):
        return Quat(self[0], -self[1], -self[2], -self[3])

    def transformVector(self, vector):
        """Transforms vector by the quaternion. For EBSD quaterions this
        is a transformation from sample space to crystal space. Perform
        on conjugate of quaternion for crystal to sample.

        Args:
            vector (numpy.ndarray): Vector to transform

        Returns:
            numpy.ndarray: Transformed vector
        """

        if isinstance(vector, np.ndarray) and vector.shape == (3,):
            vectorQuat = Quat(0, vector[0], vector[1], vector[2])
            vectorQuatTransformed = self * (vectorQuat * self.conjugate)
            vectorTransformed = vectorQuatTransformed.quatCoef[1:4]
            return vectorTransformed

        raise TypeError("Vector must be a size 3 numpy array.")

    def misOri(self, right, symGroup, returnQuat=0):
        """Calculate misorientation angle between 2 orientations taking
        into account the symmetries of the crystal structure.
        Angle is 2*arccos(output).

        Args:
            rigth (quat): Orientation to find misorientation to
            symGroup (str): Crystal type (cubic, hexagonal)
            returnQuat (int): What to return

        Returns:
            various:    returnQuat = 0 - misorientation
                        returnQuat = 1 - symmetric equivalent with min misorientation
                        returnQuat = 2 - both
        """
        if isinstance(right, type(self)):
            minMisOri = 0   # actually looking for max of this as it is cos of misoriention angle
            for sym in Quat.symEqv(symGroup):   # loop over symmetrically equivelent orienations
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

    def misOriAxis(self, right):
        """Calculate misorientation axis between 2 orientations.
        This does not consider symmetries of the crystal structure.

        Args:
            rigth (quat): Orientation to find misorientation axis to

        Returns:
            numpy.ndarray: axis of misorientation
        """
        if isinstance(right, type(self)):
            Dq = right * self.conjugate
            Dq = Dq.quatCoef
            misOriAxis = (2 * Dq[1:4] * np.arccos(Dq[0])) / np.sqrt(1 - np.power(Dq[0], 2))

            return misOriAxis
        raise TypeError("Input must be a quaternion.")

    def plotUnitCell(self, fig=None, ax=None, symGroup=None, cOverA=None, OI=True):
        """ Plots an unit cell
        Args:
            symGroup (str): symmetry group, hexagonal or cubic
            cOverA (float): c over a ratio, for hexagonal
            OI (boolean): true if using oxford instruments system
            szFac (float): size factor for plot
            ax: the axis on which to plot the unit cell
        """
        
        if symGroup == None:
            raise TypeError("symGroup must be specified")
            
        if symGroup == 'hexagonal':
        
            if cOverA == None:
                raise TypeError("cOverA must be specified for hcp")
            
            szFac = 0.2

            vert = np.matrix([
                [0,0,0],
                [1,0,0],
                [np.cos(np.deg2rad(60)), np.sin(np.deg2rad(60)), 0],
                [-np.cos(np.deg2rad(60)), np.sin(np.deg2rad(60)), 0],
                [-1, 0, 0],
                [-np.cos(np.deg2rad(60)), -np.sin(np.deg2rad(60)), 0],
                [np.cos(np.deg2rad(60)), -np.sin(np.deg2rad(60)), 0],
                [1/2., 1/3.*np.sin(np.deg2rad(60)), cOverA/2.],
                [-1/2., 1/3.*np.sin(np.deg2rad(60)), cOverA/2.],
                [0, -2.*1/3.*np.sin(np.deg2rad(60)), cOverA/2.],
                [0, 0, cOverA],
                [1, 0, cOverA],
                [np.cos(np.deg2rad(60)), np.sin(np.deg2rad(60)), cOverA],
                [-np.cos(np.deg2rad(60)), np.sin(np.deg2rad(60)), cOverA],
                [-1, 0, cOverA],
                [-np.cos(np.deg2rad(60)), -np.sin(np.deg2rad(60)), cOverA],
                [np.cos(np.deg2rad(60)), -np.sin(np.deg2rad(60)), cOverA],
                [0, 0, 3.*cOverA],
                [0, 1/3.*np.sin(np.deg2rad(60))+((3.**0.5)/2.), cOverA/2.]
            ])

            vert[:,2] = vert[:,2]-cOverA/2.;

            faces = [[2, 3, 4, 5, 6, 7], [12, 13, 14, 15, 16, 17],
                     [2, 12, 13, 3], [3, 13, 14, 4],
                     [4, 14, 15, 5], [5, 15, 16, 6],
                     [6, 16, 17, 7], [7, 17, 12, 2]]

            if OI:
                # Add 30 degrees to phi2 for OI
                quatObj = Quat(self.eulerAngles()[0], 
                                    self.eulerAngles()[1], 
                                    self.eulerAngles()[2]+np.pi/6)

        elif symGroup == 'cubic':

            szFac = 0.3
            
            vert = np.matrix([
                [-1/2,   -1/2,    -1/2],
                [1/2,   -1/2,    -1/2],
                [1/2,    1/2,    -1/2],
                [-1/2,    1/2,    -1/2],
                [-1/2,   -1/2,     1/2],
                [1/2,   -1/2,     1/2],
                [1/2,    1/2,     1/2],
                [-1/2,    1/2,     1/2]])
            
            faces = [[1, 2, 3, 4], [5, 6, 7, 8],
                [1, 2, 6, 5], [2, 3, 7, 6],
                [3, 4, 8, 7], [4, 1, 5, 8]]
                
            quatObj = self
            
        else:
            print('Only cubic and hexagonal supported')
        
        # Rotate the lattice cell points
        g_glob=np.matrix(quatObj.rotMatrix()).H
        gg = np.matrix(g_glob);
        pts = (gg*(vert.T)).T*szFac

        # Plot unit cell    
        planes = []
        for face in faces:
            x = []; y = []; z = [];
            for idx in face:
                x.append(pts[idx-1].item(0))
                y.append(pts[idx-1].item(1))
                z.append(pts[idx-1].item(2))
            planes.append([list(zip(x, y, z))])  
            
        plot = plotting.crystalPlot()
        plot.addVerts(planes)

# Static methods

    @staticmethod
    def createManyQuats(eulerArray):
        """Create a an array of quats from an array of Euler angles

        Args:
            eulerArray (array): Size 3 x n x ... x m
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
            # quatComps[(slice(None),) + idx] is equivalent to quatComps[:, idx[0], ..., idx[n]]

        return quats

    @staticmethod
    def calcSymEqvs(quats, symGroup, dtype=np.float):
        syms = Quat.symEqv(symGroup)
        quatComps = np.empty((len(syms), 4, len(quats)), dtype=dtype)

        # store quat components in array
        for i, quat in enumerate(quats):
            quatComps[0, :, i] = quat.quatCoef

        # calculate symmetrical equivalents
        for i, sym in enumerate(syms[1:], start=1):
            # sym[i] * quat for all points (* is quaternion product)
            quatComps[i, 0, :] = (quatComps[0, 0, :] * sym[0] - quatComps[0, 1, :] * sym[1] -
                                  quatComps[0, 2, :] * sym[2] - quatComps[0, 3, :] * sym[3])
            quatComps[i, 1, :] = (quatComps[0, 0, :] * sym[1] + quatComps[0, 1, :] * sym[0] -
                                  quatComps[0, 2, :] * sym[3] + quatComps[0, 3, :] * sym[2])
            quatComps[i, 2, :] = (quatComps[0, 0, :] * sym[2] + quatComps[0, 2, :] * sym[0] -
                                  quatComps[0, 3, :] * sym[1] + quatComps[0, 1, :] * sym[3])
            quatComps[i, 3, :] = (quatComps[0, 0, :] * sym[3] + quatComps[0, 3, :] * sym[0] -
                                  quatComps[0, 1, :] * sym[2] + quatComps[0, 2, :] * sym[1])

            # swap into positve hemisphere if required
            quatComps[i, :, quatComps[i, 0, :] < 0] = -quatComps[i, :, quatComps[i, 0, :] < 0]

        return quatComps

    @staticmethod
    def calcAverageOri(quatComps):
        avOri = np.copy(quatComps[0, :, 0])
        currMisOris = np.empty(quatComps.shape[0])

        for i in range(1, quatComps.shape[2]):
            # calculate misorientation between current average and all symmetrical equivalents
            # Dot product of each symm quat in quatComps with refOri for point i
            currMisOris[:] = abs(np.einsum("ij,j->i", quatComps[:, :, i], avOri))

            # find min misorientation with current average then add to it
            maxIdx = np.argmax(currMisOris[:])
            avOri += quatComps[maxIdx, :, i]

        # Convert components back to a quat and normalise
        avOri = Quat(avOri)
        avOri.normalise()

        return avOri

    @staticmethod
    def calcMisOri(quatComps, refOri):
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
    def polarAngles(x, y, z):      # spherical coordinates as per Wikipedia
        mod = np.sqrt(x**2 + y**2 + z**2)
        x = x / mod
        y = y / mod
        z = z / mod

        # alpha - angle with z axis
        alpha = np.arccos(z)
        # beta - angle around z axis from x in anticlockwise as per ISO
        beta = np.arctan2(y, x)

        return alpha, beta

    @staticmethod
    def plotIPF(quats, direction, symGroup, projection=None,
                plot=None, fig=None, ax=None, makeInteractive=False,
                plotColourBar=False, cLabel="",
                markerColour=None, markerSize=40, **kwargs):
        """
        Plot IPF of orientations for specified sample diection.

        Parameters
        ----------
        quats : enumerable(quat)
            Quats to plot on the IPF
        direction : np.array
            Sample reference direction for IPF
        symGroup : string
            Crystal type (cubic, hexagonal)
        projection
             Projection to use. Either string (stereographic or lambert) or a function
        ax
            matplotlib axis to plot on, if not provided the current active axis is used
        markerColour : string
            Colour of markers (only used for half and half colouring, otherwise us arguemnt c)
        markerSize : int
            Size of markers (only used for half and half colouring, otherwise us arguemnt s)
        kwargs
            All other arguments are passed to the matplotlib scatter call
        """
        plotParams = {'marker': '+'}
        plotParams.update(kwargs)

        alphaFund, betaFund = Quat.calcFundDirs(quats, direction, symGroup)

        if plot is None:
            plot = plotting.PolePlot(
                "IPF", symGroup, projection=projection,
                ax=ax, fig=fig, makeInteractive=makeInteractive
            )
        plot.addPoints(alphaFund, betaFund,
                       markerColour=markerColour, markerSize=markerSize,
                       **plotParams)

        if plotColourBar:
            plot.addColourBar(cLabel)

        return plot

    @staticmethod
    def calcIPFcolours(quats, direction, symGroup):
        if symGroup != "cubic":
            raise NotImplementedError("Only available for cubic currently")

        numQuats = len(quats)

        # Calculating as float32 seems to speed this up
        alphaFund, betaFund = Quat.calcFundDirs(quats, direction, symGroup, dtype=np.float32)

        # revert to cartesians
        # at some this should be changed to have the quats dimention
        # last to fit with numpys row major storage direction vector in
        # cartesians. Changes this to float32 causes errors in arccos,
        # so leave to default to 64
        dirvec = np.empty((numQuats, 3))
        dirvec[:, 0] = np.sin(alphaFund) * np.cos(betaFund)
        dirvec[:, 1] = np.sin(alphaFund) * np.sin(betaFund)
        dirvec[:, 2] = np.cos(alphaFund)

        rvect = np.repeat(np.array([0., 0., 1.])[np.newaxis, :], numQuats, axis=0)
        gvect = np.repeat(np.array([1., 0., 1.])[np.newaxis, :] / np.sqrt(2), numQuats, axis=0)
        bvect = np.repeat(np.array([1., 1., 1.])[np.newaxis, :] / np.sqrt(3), numQuats, axis=0)
        rgb = np.zeros((numQuats, 3))

        # Red Component; these subroutines are converted from
        # Stephen Cluff's IPF_rgbcalc.m (BYU)
        RDirPlane = np.cross(dirvec, rvect)
        GBplane = np.cross(bvect, gvect)
        Rintersect = np.cross(RDirPlane, GBplane)
        NORM = np.sqrt(np.power(Rintersect[:, 0], 2) + np.power(Rintersect[:, 1], 2) + np.power(Rintersect[:, 2], 2))
        Rintersect[NORM != 0, :] = np.divide(Rintersect[NORM != 0, :], np.repeat(NORM[NORM != 0][:, np.newaxis], 3, axis=1))

        temp = np.arccos(np.einsum("ij,ij->i", dirvec, Rintersect))
        Rintersect[temp > (np.pi / 2), :] = Rintersect[temp > (np.pi / 2), :] * -1
        rgb[:, 0] = np.divide(np.arccos(np.einsum("ij,ij->i", dirvec, Rintersect)), np.arccos(np.einsum("ij,ij->i", rvect, Rintersect)))

        # Green Component
        GDirPlane = np.cross(dirvec, gvect)
        RBplane = np.cross(rvect, bvect)
        Gintersect = np.cross(GDirPlane, RBplane)
        NORM = np.sqrt(np.power(Gintersect[:, 0], 2) + np.power(Gintersect[:, 1], 2) + np.power(Gintersect[:, 2], 2))
        Gintersect[NORM != 0, :] = np.divide(Gintersect[NORM != 0, :], np.repeat(NORM[NORM != 0][:, np.newaxis], 3, axis=1))

        temp = np.arccos(np.einsum("ij,ij->i", dirvec, Gintersect))
        Gintersect[temp > (np.pi / 2), :] = Gintersect[temp > (np.pi / 2), :] * -1
        rgb[:, 1] = np.divide(np.arccos(np.einsum("ij,ij->i", dirvec, Gintersect)), np.arccos(np.einsum("ij,ij->i", gvect, Gintersect)))

        # Blue Component
        BDirPlane = np.cross(dirvec, bvect)
        RGplane = np.cross(gvect, rvect)
        Bintersect = np.cross(BDirPlane, RGplane)
        NORM = np.sqrt(np.power(Bintersect[:, 0], 2) + np.power(Bintersect[:, 1], 2) + np.power(Bintersect[:, 2], 2))
        Bintersect[NORM != 0, :] = np.divide(Bintersect[NORM != 0, :], np.repeat(NORM[NORM != 0][:, np.newaxis], 3, axis=1))

        temp = np.arccos(np.einsum("ij,ij->i", dirvec, Bintersect))
        Bintersect[temp > (np.pi / 2), :] = Bintersect[temp > (np.pi / 2), :] * -1
        rgb[:, 2] = np.divide(np.arccos(np.einsum("ij,ij->i", dirvec, Bintersect)), np.arccos(np.einsum("ij,ij->i", bvect, Bintersect)))
        rgb = np.divide(rgb, np.repeat(np.amax(rgb, axis=1)[:, np.newaxis], 3, axis=1))

        return rgb

    @staticmethod
    def calcFundDirs(quats, direction, symGroup, dtype=np.float):
        # convert direction to float array
        direction = np.array(direction, dtype=dtype)

        # get array of symmetry operations. shape - (numSym, 4, numQuats)
        quatCompsSym = Quat.calcSymEqvs(quats, symGroup, dtype=dtype)

        # array to store crytal directions for all orientations and symmetries
        directionCrystal = np.empty((3, quatCompsSym.shape[0], quatCompsSym.shape[2]), dtype=dtype)

        # temp variables to use bleow
        quatDotVec = (quatCompsSym[:, 1, :] * direction[0] +
                      quatCompsSym[:, 2, :] * direction[1] +
                      quatCompsSym[:, 3, :] * direction[2])
        temp = (np.square(quatCompsSym[:, 0, :]) - np.square(quatCompsSym[:, 1, :]) -
                np.square(quatCompsSym[:, 2, :]) - np.square(quatCompsSym[:, 3, :]))

        # transform the pole direction to crystal coords for all orientations and symmetries
        # (quatCompsSym * vectorQuat) * quatCompsSym.conjugate
        directionCrystal[0, :, :] = (2 * quatDotVec * quatCompsSym[:, 1, :] +
                                     temp * direction[0] +
                                     2 * quatCompsSym[:, 0, :] * (quatCompsSym[:, 2, :] * direction[2] -
                                                                  quatCompsSym[:, 3, :] * direction[1]))
        directionCrystal[1, :, :] = (2 * quatDotVec * quatCompsSym[:, 2, :] +
                                     temp * direction[1] +
                                     2 * quatCompsSym[:, 0, :] * (quatCompsSym[:, 3, :] * direction[0] -
                                                                  quatCompsSym[:, 1, :] * direction[2]))
        directionCrystal[2, :, :] = (2 * quatDotVec * quatCompsSym[:, 3, :] +
                                     temp * direction[2] +
                                     2 * quatCompsSym[:, 0, :] * (quatCompsSym[:, 1, :] * direction[1] -
                                                                  quatCompsSym[:, 2, :] * direction[0]))

        # normalise vectors
        directionCrystal /= np.sqrt(np.einsum('ijk,ijk->jk', directionCrystal, directionCrystal))

        # move all vectors into north hemisphere
        directionCrystal[:, directionCrystal[2, :, :] < 0] *= -1

        # convert to spherical coordinates
        alpha, beta = Quat.polarAngles(directionCrystal[0], directionCrystal[1], directionCrystal[2])

        # find the poles in the fundamental triangle
        if symGroup == "cubic":
            # first beta should be between 0 and 45 deg leaving 3 symmetric equivalents per orientation
            trialPoles = np.logical_and(beta >= 0, beta <= np.pi / 4)

            # if less than 3 left need to expand search slighly to catch edge cases
            if np.sum(np.sum(trialPoles, axis=0) < 3) > 0:
                deltaBeta = 1e-8
                trialPoles = np.logical_and(beta >= -deltaBeta, beta <= np.pi / 4 + deltaBeta)

            # create array to store angles of pols in fundermental triangle
            alphaFund, betaFund = np.empty((quatCompsSym.shape[2])), np.empty((quatCompsSym.shape[2]))

            # now of symmetric equivalents left we want the one with minimum alpha
            # loop over different orientations
            # this seems quite slow so might be worth finding a different way to do it
            for i in range(trialPoles.shape[1]):
                # create array of indexes of poles kept in previous step
                trialPoleIdxs = np.arange(trialPoles.shape[0])[trialPoles[:, i]]

                # find pole with minimum alpha of those kept in previous step
                # then use trialPoleIdxs to get its index in original arrays
                poleIdx = trialPoleIdxs[np.argmin(alpha[trialPoles[:, i], i])]

                # add to final array of poles
                alphaFund[i] = alpha[poleIdx, i]
                betaFund[i] = beta[poleIdx, i]

        elif symGroup == "hexagonal":
            # first beta should be between 0 and 30 deg leaving 1 symmetric equivalent per orientation
            trialPoles = np.logical_and(beta >= 0, beta <= np.pi / 6)

            # if less than 1 left need to expand search slighly to catch edge cases
            if np.sum(np.sum(trialPoles, axis=0) < 1) > 0:
                deltaBeta = 1e-8
                trialPoles = np.logical_and(beta >= -deltaBeta, beta <= np.pi / 6 + deltaBeta)

            alphaFund = alpha[trialPoles]
            betaFund = beta[trialPoles]

        else:
            raise Exception("symGroup must be cubic or hexagonal")

        return alphaFund, betaFund

    @staticmethod
    def symEqv(group):
        overRoot2 = np.sqrt(2) / 2
        sqrt3over2 = np.sqrt(3) / 2
        qsym = []
        # identity - this should always be returned as the first symmetry
        qsym.append(Quat(np.array([1.0, 0.0, 0.0, 0.0])))

        # from Pete Bate's fspl_orir.f90 code
        # checked for consistency with mtex
        # cubic tetrads(100)
        qsym.append(Quat(np.array([overRoot2, overRoot2, 0.0, 0.0])))
        qsym.append(Quat(np.array([0.0, 1.0, 0.0, 0.0])))
        qsym.append(Quat(np.array([overRoot2, -overRoot2, 0.0, 0.0])))

        qsym.append(Quat(np.array([overRoot2, 0.0, overRoot2, 0.0])))
        qsym.append(Quat(np.array([0.0, 0.0, 1.0, 0.0])))
        qsym.append(Quat(np.array([overRoot2, 0.0, -overRoot2, 0.0])))

        qsym.append(Quat(np.array([overRoot2, 0.0, 0.0, overRoot2])))
        qsym.append(Quat(np.array([0.0, 0.0, 0.0, 1.0])))
        qsym.append(Quat(np.array([overRoot2, 0.0, 0.0, -overRoot2])))

        # cubic dyads (110)
        qsym.append(Quat(np.array([0.0, overRoot2, overRoot2, 0.0])))
        qsym.append(Quat(np.array([0.0, -overRoot2, overRoot2, 0.0])))

        qsym.append(Quat(np.array([0.0, overRoot2, 0.0, overRoot2])))
        qsym.append(Quat(np.array([0.0, -overRoot2, 0.0, overRoot2])))

        qsym.append(Quat(np.array([0.0, 0.0, overRoot2, overRoot2])))
        qsym.append(Quat(np.array([0.0, 0.0, -overRoot2, overRoot2])))

        # cubic triads (111)
        qsym.append(Quat(np.array([0.5, 0.5, 0.5, 0.5])))
        qsym.append(Quat(np.array([0.5, -0.5, -0.5, -0.5])))

        qsym.append(Quat(np.array([0.5, -0.5, 0.5, 0.5])))
        qsym.append(Quat(np.array([0.5, 0.5, -0.5, -0.5])))

        qsym.append(Quat(np.array([0.5, 0.5, -0.5, 0.5])))
        qsym.append(Quat(np.array([0.5, -0.5, 0.5, -0.5])))

        qsym.append(Quat(np.array([0.5, 0.5, 0.5, -0.5])))
        qsym.append(Quat(np.array([0.5, -0.5, -0.5, 0.5])))

        # hexagonal hexads
        qsym.append(Quat(np.array([sqrt3over2, 0.0, 0.0, 0.5])))
        qsym.append(Quat(np.array([0.5, 0.0, 0.0, sqrt3over2])))
        qsym.append(Quat(np.array([0.5, 0.0, 0.0, -sqrt3over2])))
        qsym.append(Quat(np.array([sqrt3over2, 0.0, 0.0, -0.5])))

        # hexagonal diads
        qsym.append(Quat(np.array([0.0, -0.5, -sqrt3over2, 0.0])))
        qsym.append(Quat(np.array([0.0, 0.5, -sqrt3over2, 0.0])))
        qsym.append(Quat(np.array([0.0, sqrt3over2, -0.5, 0.0])))
        qsym.append(Quat(np.array([0.0, -sqrt3over2, -0.5, 0.0])))

        if group == 'cubic':
            return qsym[0:24]
        elif group == 'hexagonal':
            return [qsym[0], qsym[2], qsym[5], qsym[8]] + qsym[-8:32]
        else:
            return [qsym[0]]
