import numpy as np


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
            self.quatCoef[1] = np.sin(phi / 2.0) * np.cos((ph1 - ph2) / 2.0)
            self.quatCoef[2] = np.sin(phi / 2.0) * np.sin((ph1 - ph2) / 2.0)
            self.quatCoef[3] = np.cos(phi / 2.0) * np.sin((ph1 + ph2) / 2.0)
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

    def eulerAngles(self):
        # See Melcher, a. Unser, A. Reichhardt, M. Nestler, B. Conversion of EBSD data by a
        # quaternion based algorithm to be used for grain structure simulations

        eulers = np.empty(3, dtype=float)

        chi = np.sqrt((self.quatCoef[0]**2 + self.quatCoef[3]**2) * (self.quatCoef[1]**2 + self.quatCoef[2]**2))

        cosPh1 = (self.quatCoef[0] * self.quatCoef[1] - self.quatCoef[2] * self.quatCoef[3]) / (2 * chi)
        sinPh1 = (self.quatCoef[0] * self.quatCoef[2] + self.quatCoef[1] * self.quatCoef[3]) / (2 * chi)

        cosPhi = self.quatCoef[0]**2 + self.quatCoef[3]**2 - self.quatCoef[1]**2 - self.quatCoef[2]**2
        sinPhi = 2 * chi

        cosPh2 = (self.quatCoef[0] * self.quatCoef[1] + self.quatCoef[2] * self.quatCoef[3]) / (2 * chi)
        sinPh2 = (self.quatCoef[1] * self.quatCoef[3] - self.quatCoef[0] * self.quatCoef[2]) / (2 * chi)

        # eulers[0] = np.arctan(sinPh1/cosPh1)
        # eulers[1] = np.arctan(sinPhi/cosPhi)
        # eulers[2] = np.arctan(sinPh2/cosPh2)

        eulers[0] = np.arctan2(sinPh1, cosPh1)
        eulers[1] = np.arctan2(sinPhi, cosPhi)
        eulers[2] = np.arctan2(sinPh2, cosPh2)

        if eulers[0] < 0:
            eulers[0] += 2 * np.pi
        if eulers[2] < 0:
            eulers[2] += 2 * np.pi

        return eulers

    # show components when the quat is printed
    def __repr__(self):
        return "[%.4f, %.4f, %.4f, %.4f]" % (self.quatCoef[0], self.quatCoef[1], self.quatCoef[2], self.quatCoef[3])

    def __str__(self):
        return "[%.4f, %.4f, %.4f, %.4f]" % (self.quatCoef[0], self.quatCoef[1], self.quatCoef[2], self.quatCoef[3])

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
            vectorQuatTransformed = (self.conjugate * vectorQuat) * self
            vectorTransformed = vectorQuatTransformed.quatCoef[1:4]
            return vectorTransformed

        raise TypeError("Vector must be a size 3 numpy array.")

    def misOri(self, right, symGroup, returnQuat=0):
        if isinstance(right, type(self)):
            minMisOri = 0   # actually looking for max of this as it is cos of misoriention angle
            for sym in Quat.symEqv(symGroup):   # loop over symmetrically equivelent orienations
                quatSym = right * sym
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

    @staticmethod
    def calcSymEqvs(quats, symGroup):
        syms = Quat.symEqv(symGroup)
        quatComps = np.empty((len(syms), 4, len(quats)))

        # store quat components in array
        for i, quat in enumerate(quats):
            quatComps[0, :, i] = quat.quatCoef

        # calculate symmetrical equivalents
        for i, sym in enumerate(syms[1:], start=1):
            # quat * sym[i] for all points (* is quaternion product)
            quatComps[i, 0, :] = (quatComps[0, 0, :] * sym[0] - quatComps[0, 1, :] * sym[1] -
                                  quatComps[0, 2, :] * sym[2] - quatComps[0, 3, :] * sym[3])
            quatComps[i, 1, :] = (quatComps[0, 0, :] * sym[1] + quatComps[0, 1, :] * sym[0] +
                                  quatComps[0, 2, :] * sym[3] - quatComps[0, 3, :] * sym[2])
            quatComps[i, 2, :] = (quatComps[0, 0, :] * sym[2] + quatComps[0, 2, :] * sym[0] +
                                  quatComps[0, 3, :] * sym[1] - quatComps[0, 1, :] * sym[3])
            quatComps[i, 3, :] = (quatComps[0, 0, :] * sym[3] + quatComps[0, 3, :] * sym[0] +
                                  quatComps[0, 1, :] * sym[2] - quatComps[0, 2, :] * sym[1])

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
    def symEqv(group):
        qsym = []
        qsym.append(Quat(np.array([1.0, 0.0, 0.0, 0.0])))

        # from Pete Bate's fspl_orir.f90 code
        # cubic tetrads(100)
        qsym.append(Quat(np.array([0.7071068, 0.7071068, 0.0, 0.0])))
        qsym.append(Quat(np.array([0.0, 1.0, 0.0, 0.0])))
        qsym.append(Quat(np.array([0.7071068, -0.7071068, 0.0, 0.0])))

        qsym.append(Quat(np.array([0.7071068, 0.0, 0.7071068, 0.0])))
        qsym.append(Quat(np.array([0.0, 0.0, 1.0, 0.0])))
        qsym.append(Quat(np.array([0.7071068, 0.0, -0.7071068, 0.0])))

        qsym.append(Quat(np.array([0.7071068, 0.0, 0.0, 0.7071068])))
        qsym.append(Quat(np.array([0.0, 0.0, 0.0, 1.0])))
        qsym.append(Quat(np.array([0.7071068, 0.0, 0.0, -0.7071068])))

        # cubic dyads (110)
        qsym.append(Quat(np.array([0.0, 0.7071068, 0.7071068, 0.0])))
        qsym.append(Quat(np.array([0.0, -0.7071068, 0.7071068, 0.0])))

        qsym.append(Quat(np.array([0.0, 0.7071068, 0.0, 0.7071068])))
        qsym.append(Quat(np.array([0.0, -0.7071068, 0.0, 0.7071068])))

        qsym.append(Quat(np.array([0.0, 0.0, 0.7071068, 0.7071068])))
        qsym.append(Quat(np.array([0.0, 0.0, -0.7071068, 0.7071068])))

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
        qsym.append(Quat(np.array([0.866254, 0.0, 0.0, 0.5])))
        qsym.append(Quat(np.array([0.5, 0.0, 0.0, 0.866254])))
        qsym.append(Quat(np.array([0.5, 0.0, 0.0, -0.866254])))
        qsym.append(Quat(np.array([0.866254, 0.0, 0.0, -0.5])))

        # hexagonal diads
        qsym.append(Quat(np.array([0.0, -0.5, 0.866254, 0.0])))
        qsym.append(Quat(np.array([0.0, -0.5, -0.866254, 0.0])))

        if (group == 'cubic'):
            return qsym[0:24]
        elif (group == 'hexagonal'):
            return [qsym[0], qsym[2], qsym[8]] + qsym[-6:30]
        else:
            return qsym
