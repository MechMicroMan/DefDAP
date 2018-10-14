import numpy as np
import matplotlib.pyplot as plt


# be careful with deep and shallow copies
class Quat(object):
    def __init__(self, *args, **kwargs):
        self.quatCoef = np.zeros(4, dtype=float)
        # construt with Bunge euler angles (radians, ZXZ)
        if len(args) == 3:
            ph1 = args[0]
            phi = args[1]
            ph2 = args[2]

            self.quatCoef[0] = np.cos(phi / 2.0) * np.cos((ph1 + ph2) / 2.0)
            self.quatCoef[1] = -np.sin(phi / 2.0) * np.cos((ph1 - ph2) / 2.0)
            self.quatCoef[2] = -np.sin(phi / 2.0) * np.sin((ph1 - ph2) / 2.0)
            self.quatCoef[3] = -np.cos(phi / 2.0) * np.sin((ph1 + ph2) / 2.0)
        # construt with array of quat coefficients
        elif len(args) == 1:
            self.quatCoef = args[0]
        # construt with quat coefficients
        elif len(args) == 4:
            self.quatCoef[0] = args[0]
            self.quatCoef[1] = args[1]
            self.quatCoef[2] = args[2]
            self.quatCoef[3] = args[3]

        if (self.quatCoef[0] < 0):
            self.quatCoef = self.quatCoef * -1

        # overload static method with instance method of same name in object
        self.plotIPF = self._plotIPF

    @classmethod
    def fromAxisAngle(cls, axis, angle):
        """Create a quat object from an axis angle pair

        Args:
            axis (np.array size 3): Axis of rotation
            angle (float): Rotation arround axis (radians)

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
        # See Melcher, a. Unser, A. Reichhardt, M. Nestler, B. Conversion of EBSD data by a
        # quaternion based algorithm to be used for grain structure simulations
        # or
        # Rowenhorst, D et al. Consistent representations of and conversions between 3D rotations
        # P = +1

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
        return "[%.4f, %.4f, %.4f, %.4f]" % (self.quatCoef[0], self.quatCoef[1], self.quatCoef[2], self.quatCoef[3])

    def __str__(self):
        return "[%.4f, %.4f, %.4f, %.4f]" % (self.quatCoef[0], self.quatCoef[1], self.quatCoef[2], self.quatCoef[3])

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
        return Quat(self.quatCoef[0], -self.quatCoef[1], -self.quatCoef[2], -self.quatCoef[3])

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
            vectorQuatTransformed = (self * vectorQuat) * self.conjugate
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

        for idx in np.ndindex(oriShape):
            quats[idx] = Quat(quatComps[(slice(None),) + idx])
            # quatComps[(slice(None),) + idx] is equivalent to quatComps[:, idx[0], ..., idx[n]]

        return quats

    @staticmethod
    def calcSymEqvs(quats, symGroup):
        syms = Quat.symEqv(symGroup)
        quatComps = np.empty((len(syms), 4, len(quats)))

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
    def polarAngles(x, y, z):
        mod = np.sqrt(x**2 + y**2 + z**2)
        x = x / mod
        y = y / mod
        z = z / mod

        # alpha - angle with z axis
        alpha = np.arccos(z)
        # beta - angle around z axis
        beta = np.arctan2(y, x)

        return alpha, beta

    @staticmethod
    def stereoProject(*args):
        if len(args) == 3:
            alpha, beta = Quat.polarAngles(args[0], args[1], args[2])
        elif len(args) == 2:
            alpha, beta = args
        else:
            raise Exception("3 arguments for pole directions and 2 for polar angles.")

        alphaComp = np.tan(alpha / 2)
        xp = alphaComp * np.cos(beta)
        yp = alphaComp * np.sin(beta)

        return xp, yp

    @staticmethod
    def plotLine(startPoint, endPoint, plotSymmetries=False, symGroup=None, res=100, projection=None, ax=None, **kwargs):
        if projection is None:
            projection = Quat.stereoProject
        if ax is None:
            ax = plt.gca()

        lines = []
        lines.append((startPoint, endPoint))
        if plotSymmetries:
            if symGroup is None:
                raise Exception("Please provide a symGroup")
            for symm in Quat.symEqv(symGroup)[1:]:
                startPointSymm = symm.transformVector(startPoint).astype(int)
                endPointSymm = symm.transformVector(endPoint).astype(int)

                if startPointSymm[2] < 0:
                    startPointSymm *= -1
                if endPointSymm[2] < 0:
                    endPointSymm *= -1

                lines.append((startPointSymm, endPointSymm))

        linePoints = np.zeros((3, res), dtype=float)
        for line in lines:
            for i in range(3):
                if line[0][i] == line[1][i]:
                    linePoints[i] = np.full(res, line[0][i])
                else:
                    linePoints[i] = np.linspace(line[0][i], line[1][i], res)

            xp, yp = projection(linePoints[0], linePoints[1], linePoints[2])
            ax.plot(xp, yp, **kwargs)

    @staticmethod
    def labelPoint(point, label, projection=None, ax=None, padX=0, padY=0, **kwargs):
        if projection is None:
            projection = Quat.stereoProject
        if ax is None:
            ax = plt.gca()

        xp, yp = projection(point[0], point[1], point[2])
        ax.text(xp + padX, yp + padY, label, **kwargs)

    @staticmethod
    def plotPoleAxis(plotType, symGroup, ax=None):
        if ax is None:
            ax = plt.gca()

        if plotType == "IPF" and symGroup == "cubic":
            # line between [001] and [111]
            Quat.plotLine(np.array([0, 0, 1]), np.array([1, 1, 1]), ax=ax, c='k', lw=2)

            # line between [001] and [101]
            Quat.plotLine(np.array([0, 0, 1]), np.array([1, 0, 1]), ax=ax, c='k', lw=2)

            # line between [101] and [111]
            Quat.plotLine(np.array([1, 0, 1]), np.array([1, 1, 1]), ax=ax, c='k', lw=2)

            # label poles
            Quat.labelPoint(np.array([0, 0, 1]), '001', ax=ax, padY=-0.005, va='top', ha='center')
            Quat.labelPoint(np.array([1, 0, 1]), '101', ax=ax, padY=-0.005, va='top', ha='center')
            Quat.labelPoint(np.array([1, 1, 1]), '111', ax=ax, padY=0.005, va='bottom', ha='center')

            ax.axis('equal')
            ax.axis('off')

        else:
            print("Only works for cubic")

    @staticmethod
    def plotIPF(quats, direction, symGroup, ax=None, **kwargs):
        plotParams = {'marker': '+', 'c': 'r'}
        plotParams.update(kwargs)
        if ax is None:
            ax = plt.gca()

        if symGroup == "hexagonal":
            raise Exception("Have fun with that")

        # Plot IPF axis
        # plt.figure()
        Quat.plotPoleAxis("IPF", symGroup, ax=ax)

        # get array of symmetry operations. shape - (numSym, 4, numQuats)
        quatCompsSym = Quat.calcSymEqvs(quats, symGroup)

        # array to store crytal directions for all orientations and symmetries
        directionCrystal = np.empty((3, quatCompsSym.shape[0], quatCompsSym.shape[2]))

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
            for i in range(trialPoles.shape[1]):
                # create array of indexes of poles kept in previous step
                trialPoleIdxs = np.arange(trialPoles.shape[0])[trialPoles[:, i]]

                # find pole with minimum alpha of those kept in previous step
                # then use trialPoleIdxs to get its index in original arrays
                poleIdx = trialPoleIdxs[np.argmin(alpha[trialPoles[:, i], i])]

                # add to final array of poles
                alphaFund[i] = alpha[poleIdx, i]
                betaFund[i] = beta[poleIdx, i]
        else:
            print("Only works for cubic")

        # project onto equatorial plane
        xp, yp = Quat.stereoProject(alphaFund, betaFund)

        # plot poles
        ax.scatter(xp, yp, **plotParams)

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

        if (group == 'cubic'):
            return qsym[0:24]
        elif (group == 'hexagonal'):
            return [qsym[0], qsym[2], qsym[5], qsym[8]] + qsym[-8:32]
        else:
            return [qsym[0]]
