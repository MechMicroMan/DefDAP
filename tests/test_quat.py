import pytest
from pytest import approx
import numpy as np

from defdap.quat import Quat


# Initialisation tests


@pytest.mark.parametrize('inputLength',
                         [2, 5, 6])
def testInitDimension(inputLength):
    """Quat initialisation should raise a DimensionError if not length
    1, 3, 4"""
    with pytest.raises(Exception):
        Quat(tuple(range(inputLength)))


@pytest.mark.parametrize('ph1, phi, ph2, expectedOutput', [
    (np.pi, np.pi, np.pi, [0, 1., 0, 0]),
    (np.pi/2., np.pi, np.pi,
     [0, 0.7071067811865476, -0.7071067811865476, 0]),
    (np.pi/2., np.pi, np.pi/4.,
     [0, -0.9238795325112867, -0.3826834323650898, 0]),
    (np.pi / 2., 5.*np.pi, np.pi / 4.,
     [0, -0.9238795325112867, -0.3826834323650898, 0]),
    (np.pi / 2, np.pi / 4, -np.pi,
     [0.6532814824381883, 0.2705980500730985, -0.2705980500730985,
      0.6532814824381883]),
    (0, 0, 0, [1., 0, 0, 0]),
    (-np.pi, -np.pi, -np.pi, [0, -1., 0, 0]),
])
def testInitEuler(ph1, phi, ph2, expectedOutput):
    """Check quatCoef is correct after initialisation with Eulers"""
    returnedQuat = Quat.fromEulerAngles(ph1, phi, ph2)
    assert np.allclose(returnedQuat.quatCoef, expectedOutput, atol=1e-4)


# Check quatCoef is correct after initialisation with quat array
@pytest.mark.parametrize('testValues, expectedOutput', [
    ([0, 0, 0, 0], [0, 0, 0, 0]),
    ([1., 2., 3., 4.], [1., 2., 3., 4.])
])
def testInitArray(testValues, expectedOutput):
    returnedQuat = Quat(testValues).quatCoef
    assert np.allclose(returnedQuat, expectedOutput, atol=1e-4)


# Check quatCoef is correct after initialisation with quat coeffs
@pytest.mark.parametrize('a1, a2, a3, a4, expectedOutput', [
    (0, 0, 0, 0, [0, 0, 0, 0]),
    (1, 2, 3, 4, [1, 2, 3, 4])
])
def testInitCoeffs(a1, a2, a3, a4, expectedOutput):
    returnedQuat = Quat(a1, a2, a3, a4)
    assert np.allclose(returnedQuat.quatCoef, expectedOutput, atol=1e-4)


def testFlipToNorthernHemisphere():
    expectedOutput = [0.5, 0.5, -0.5, 0.5]
    returnedQuat = Quat(-0.5, -0.5, 0.5, -0.5)
    assert np.allclose(returnedQuat.quatCoef, expectedOutput, atol=1e-4)


# Check quat initialisation with an array that's too short
@pytest.mark.parametrize('input', [2, 3, 4])
def testShortArray(input):
    with pytest.raises(Exception):
        Quat(input)


# Check for error if quat not initialised with numbers
@pytest.mark.parametrize('input', [[1, 3, 1, 'l']])
def testInitStrArray(input):
    with pytest.raises(ValueError):
        Quat(input)


# Check for error if quat not initialised with numbers
@pytest.mark.parametrize('a1, a2, a3, a4', [(1, 3, 1, 'l')])
def testInitStrCoeff(a1, a2, a3, a4):
    with pytest.raises(ValueError):
        Quat(a1, a2, a3, a4)


# Check for error if quat not initialised with numbers
@pytest.mark.parametrize('ph1, phi, ph2', [(1,2,'l')])
def testInitStrEul(ph1, phi, ph2):
    with pytest.raises(TypeError):
        Quat(ph1, phi, ph2)


## fromAxisAngle
# Check quatCoef is correct for given axis and angle
@pytest.mark.parametrize('axis, angle, expectedOutput', [
    ([1, 0, 0], np.pi, [0, 1, 0, 0]),
    ([1, 1, 0], -np.pi/2, [np.sin(np.pi/4), -0.5, -0.5, 0]),
    ([1, -1, 0], -np.pi/2, [np.sin(np.pi/4), -0.5, 0.5, 0]),
    ([1, -1, -1], -np.pi/2, [np.sin(np.pi/4), -0.4082483, 0.4082483, 0.4082483])
])
def testFromAxisAngle(axis, angle, expectedOutput):
    returnedQuat = Quat.fromAxisAngle(axis, angle).quatCoef
    assert np.allclose(returnedQuat, expectedOutput, atol=1e-4)


# String in axis should give error
@pytest.mark.parametrize('axis, angle,', [
    ([1, 0, 'l'], 20)
])
def testFromAxisAngleStr(axis, angle):
    with pytest.raises(ValueError):
        Quat.fromAxisAngle(axis, angle)


# String in angle should give error
@pytest.mark.parametrize('axis, angle,', [
    ([1, 0, 1], 'l')
])
def testFromAxisAngleStr2(axis, angle):
    with pytest.raises(TypeError):
        Quat.fromAxisAngle(axis, angle)


# Test conversions


@pytest.fixture
def single_quat():
    """From Eulers (20, 10, 40)"""
    return Quat(0.86272992, -0.08583165,  0.01513444, -0.49809735)


@pytest.fixture
def single_quat2():
    """From Eulers (110, 70, 160)"""
    return Quat(0.57922797,  0.51983679, -0.24240388,  0.57922797)


def test_eulerAngles_return_type(single_quat):
    returnEulers = single_quat.eulerAngles()

    assert type(returnEulers) is np.ndarray
    assert returnEulers.shape == (3, )


def test_eulerAngles_calc(single_quat):
    returnEulers = single_quat.eulerAngles()

    assert np.allclose(returnEulers*180/np.pi, [20., 10., 40.])


def test_eulerAngles_calc_chi_q12_0():
    in_quat = Quat(0.70710678, 0., 0., 0.70710678)
    returnEulers = in_quat.eulerAngles()

    assert np.allclose(returnEulers, [4.71238898, 0., 0.])


def test_eulerAngles_calc_chi_q03_0():
    in_quat = Quat(0., 0.70710678, 0.70710678, 0.)
    returnEulers = in_quat.eulerAngles()

    assert np.allclose(returnEulers, [1.57079633, 3.14159265, 0.])


def test_rotMatrix_return_type(single_quat):
    returnMatrix = single_quat.rotMatrix()

    assert type(returnMatrix) is np.ndarray
    assert returnMatrix.shape == (3, 3)


def test_rotMatrix_calc(single_quat):
    returnMatrix = single_quat.rotMatrix()

    expectedMatrix = np.array([
        [ 0.50333996,  0.85684894,  0.1116189 ],
        [-0.862045  ,  0.48906392,  0.13302222],
        [ 0.05939117, -0.16317591,  0.98480775]
    ])

    assert np.allclose(returnMatrix, expectedMatrix)


# Test arithmetic


def test_mul_return_type(single_quat, single_quat2):
    result = single_quat * single_quat2

    assert type(result) is Quat


def test_mul_calc(single_quat, single_quat2):
    result = single_quat * single_quat2

    assert np.allclose(result.quatCoef,
                       [0.8365163, 0.28678822, -0.40957602, 0.22414387])


def test_dot_return_type(single_quat, single_quat2):
    result = single_quat.dot(single_quat2)

    assert type(result) is np.float64


def test_dot_calc(single_quat, single_quat2):
    result = single_quat.dot(single_quat2)

    assert result == approx(0.16291828363609984)

''' Functions left to test
__repr__(self):
__str__(self):
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