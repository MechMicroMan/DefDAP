import pytest
import numpy as np

import defdap.quat

## Initialisation
# Quat initialisation should raise a DimensionError if not length 1, 3, 4
@pytest.mark.parametrize('inputLength',
                         [2, 5, 6])
def testInitDimension(inputLength):
    with pytest.raises(Exception):
        defdap.quat.Quat(tuple(range(inputLength)))

# Check quatCoef is correct after initialisation with Eulers
@pytest.mark.parametrize('ph1, phi, ph2, expectedOutput', [
    (np.pi, np.pi, np.pi, [0, 1., 0, 0]),
    (np.pi/2., np.pi, np.pi, [0, np.cos(np.pi/4.), -np.cos(np.pi/4.), 0]),
    (np.pi/2., np.pi, np.pi/4., [0, -np.sin(np.pi/2.)*np.cos(np.pi/8.), -np.sin(np.pi/2.)*np.sin(np.pi/8.), 0]),
    (np.pi / 2., 5.*np.pi, np.pi / 4.,
                    [0, -np.sin(np.pi / 2.) * np.cos(np.pi / 8.), -np.sin(np.pi / 2.) * np.sin(np.pi / 8.), 0]),
    (np.pi / 2, np.pi / 4, -np.pi, [np.cos(np.pi/8)*np.cos(-np.pi/4),
     np.sin(np.pi/8)*np.cos(-np.pi/4), -np.sin(np.pi/8)*np.cos(-np.pi/4), np.cos(np.pi/8)*np.cos(-np.pi/4)]),
    (0, 0, 0, [1., 0, 0, 0]),
    (-np.pi, -np.pi, -np.pi, [0, -1., 0, 0]),
])
def testInitEuler(ph1, phi, ph2, expectedOutput):
    returnedQuat = defdap.quat.Quat(ph1, phi, ph2).quatCoef
    assert np.allclose(returnedQuat, expectedOutput, atol=1e-4)

# Check quatCoef is correct after initialisation with quat array
@pytest.mark.parametrize('testValues, expectedOutput', [
    ([0, 0, 0, 0], [0, 0, 0, 0]),
    ([1., 2., 3., 4.], [1., 2., 3., 4.]),
    ([-0.5, -0.5, 1, 2], [0.5, 0.5, -1., -2.]),
])
def testInitArray(testValues, expectedOutput):
    returnedQuat = defdap.quat.Quat(testValues).quatCoef
    assert np.allclose(returnedQuat, expectedOutput, atol=1e-4)

# Check quatCoef is correct after initialisation with quat coeffs
@pytest.mark.parametrize('a1, a2, a3, a4, expectedOutput', [
    (0, 0, 0, 0, [0, 0, 0, 0]),
    (1, 2, 3, 4, [1, 2, 3, 4]),
    (-0.5, -0.5, 1, 2, [0.5, 0.5, -1, -2]),
])
def testInitCoeffs(a1, a2, a3, a4, expectedOutput):
    returnedQuat = defdap.quat.Quat(a1, a2, a3, a4).quatCoef
    assert np.allclose(returnedQuat, expectedOutput, atol=1e-4)

# Check quat initialisation with an array that's too short
@pytest.mark.parametrize('input', [2, 3, 4])
def testShortArray(input):
    with pytest.raises(Exception):
        defdap.quat.Quat(input)

# Check for error if quat not initialised with numbers
@pytest.mark.parametrize('input', [[1, 3, 1, 'l']])
def testInitStrArray(input):
    with pytest.raises(ValueError):
        defdap.quat.Quat(input)

# Check for error if quat not initialised with numbers
@pytest.mark.parametrize('a1, a2, a3, a4', [(1, 3, 1, 'l')])
def testInitStrCoeff(a1, a2, a3, a4):
    with pytest.raises(ValueError):
        defdap.quat.Quat(a1, a2, a3, a4)

# Check for error if quat not initialised with numbers
@pytest.mark.parametrize('ph1, phi, ph2', [(1,2,'l')])
def testInitStrEul(ph1, phi, ph2):
    with pytest.raises(TypeError):
        defdap.quat.Quat(ph1, phi, ph2)

## fromAxisAngle
# Check quatCoef is correct for given axis and angle
@pytest.mark.parametrize('axis, angle, expectedOutput', [
    ([1, 0, 0], np.pi, [0, 1, 0, 0]),
    ([1, 1, 0], -np.pi/2, [np.sin(np.pi/4), -0.5, -0.5, 0]),
    ([1, -1, 0], -np.pi/2, [np.sin(np.pi/4), -0.5, 0.5, 0]),
    ([1, -1, -1], -np.pi/2, [np.sin(np.pi/4), -0.4082483, 0.4082483, 0.4082483]),

])
def testFromAxisAngle(axis, angle, expectedOutput):
    returnedQuat = defdap.quat.Quat.fromAxisAngle(axis, angle).quatCoef
    assert np.allclose(returnedQuat, expectedOutput, atol=1e-4)

# String in axis should give error
@pytest.mark.parametrize('axis, angle,', [
    ([1, 0, 'l'], 20)
])
def testFromAxisAngleStr(axis, angle):
    with pytest.raises(ValueError):
        defdap.quat.Quat.fromAxisAngle(axis, angle)

# String in angle should give error
@pytest.mark.parametrize('axis, angle,', [
    ([1, 0, 1], 'l')
])
def testFromAxisAngleStr2(axis, angle):
    with pytest.raises(TypeError):
        defdap.quat.Quat.fromAxisAngle(axis, angle)



''' Functions left to test
eulerAngles(self):
rotMatrix(self):
__repr__(self):
__str__(self):
_plotIPF(self, direction, symGroup, **kwargs):
__mul__(self, right):
dot(self, right)
 __add__(self, right)
__iadd__(self, right)
__getitem__(self, key)
__setitem__(self, key, value)
norm(self)
normalise(self)
conjugate(self)
transformVector(self, vector)
misOri(self, right, symGroup, returnQuat=0)
misOriAxis(self, right)
createManyQuats(eulerArray)
calcSymEqvs(quats, symGroup, dtype=np.float)
calcAverageOri(quatComps)
calcMisOri(quatComps, refOri)
polarAngles(x, y, z)
stereoProject(*args)
lambertProject(*args)
_validateProjection(projectionIn, validateDefault=False)
plotLine(startPoint, endPoint, plotSymmetries=False, symGroup=None,
labelPoint(point, label, projection=None, ax=None, padX=0, padY=0, **kwargs)
plotPoleAxis(plotType, symGroup, projection=None, ax=None)
plotIPF(quats, direction, symGroup, projection=None, ax=None, markerColour=None, markerSize=40, **kwargs)
plotIPFmap(quatArray, direction, symGroup, **kwargs)
calcIPFcolours(quats, direction, symGroup)
calcFundDirs(quats, direction, symGroup, dtype=np.float)
symEqv(group)
'''