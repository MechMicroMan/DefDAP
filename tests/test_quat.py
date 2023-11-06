import pytest
from pytest import approx
from pytest_cases import parametrize, parametrize_with_cases

import numpy as np
from defdap.quat import Quat


# Initialisation tests


@pytest.mark.parametrize('inputLength', [2, 5, 6])
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
    """Check quat_coef is correct after initialisation with Eulers"""
    returnedQuat = Quat.from_euler_angles(ph1, phi, ph2)
    assert np.allclose(returnedQuat.quat_coef, expectedOutput, atol=1e-4)


# Check quat_coef is correct after initialisation with quat array
@pytest.mark.parametrize('testValues, expectedOutput', [
    ([0, 0, 0, 0], [0, 0, 0, 0]),
    ([1., 2., 3., 4.], [1., 2., 3., 4.])
])
def testInitArray(testValues, expectedOutput):
    returnedQuat = Quat(testValues).quat_coef
    assert np.allclose(returnedQuat, expectedOutput, atol=1e-4)


# Check quat_coef is correct after initialisation with quat coeffs
@pytest.mark.parametrize('a1, a2, a3, a4, expectedOutput', [
    (0, 0, 0, 0, [0, 0, 0, 0]),
    (1, 2, 3, 4, [1, 2, 3, 4])
])
def testInitCoeffs(a1, a2, a3, a4, expectedOutput):
    returnedQuat = Quat(a1, a2, a3, a4)
    assert np.allclose(returnedQuat.quat_coef, expectedOutput, atol=1e-4)


def testFlipToNorthernHemisphere():
    expectedOutput = [0.5, 0.5, -0.5, 0.5]
    returnedQuat = Quat(-0.5, -0.5, 0.5, -0.5)
    assert np.allclose(returnedQuat.quat_coef, expectedOutput, atol=1e-4)


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


## from_axis_angle
# Check quat_coef is correct for given axis and angle
@pytest.mark.parametrize('axis, angle, expectedOutput', [
    ([1, 0, 0], np.pi, [0, -1, 0, 0]),
    ([1, 1, 0], -np.pi/2, [np.sin(np.pi/4), 0.5, 0.5, 0]),
    ([1, -1, 0], -np.pi/2, [np.sin(np.pi/4), 0.5, -0.5, 0]),
    ([1, -1, -1], -np.pi/2, [np.sin(np.pi/4), 0.4082483, -0.4082483, -0.4082483])
])
def testFromAxisAngle(axis, angle, expectedOutput):
    returnedQuat = Quat.from_axis_angle(axis, angle).quat_coef
    assert np.allclose(returnedQuat, expectedOutput, atol=1e-4)


# String in axis should give error
@pytest.mark.parametrize('axis, angle,', [
    ([1, 0, 'l'], 20)
])
def testFromAxisAngleStr(axis, angle):
    with pytest.raises(ValueError):
        Quat.from_axis_angle(axis, angle)


# String in angle should give error
@pytest.mark.parametrize('axis, angle,', [
    ([1, 0, 1], 'l')
])
def testFromAxisAngleStr2(axis, angle):
    with pytest.raises(TypeError):
        Quat.from_axis_angle(axis, angle)


@pytest.fixture
def single_quat() -> Quat:
    """From Eulers (20, 10, 40)"""
    return Quat(0.86272992, -0.08583165,  0.01513444, -0.49809735)


@pytest.fixture
def single_quat2() -> Quat:
    """From Eulers (110, 70, 160)"""
    return Quat(0.57922797,  0.51983679, -0.24240388,  0.57922797)


@pytest.fixture
def single_quat_not_unit() -> Quat:
    """From Eulers (110, 70, 160)"""
    return Quat(1., 2., -3., 4.)


@pytest.fixture
def ori_quat_list_valid():
    """A 1d array of sample quaternion representing the orientations."""
    return np.array([
        Quat(0.22484510, 0.45464871, -0.70807342, 0.49129550),
        Quat(0.36520321, 0.25903472, -0.40342268, 0.79798357)
    ])


# Test conversions


class TestEulerAngles:

    @staticmethod
    def test_return_type(single_quat):
        returnEulers = single_quat.euler_angles()

        assert type(returnEulers) is np.ndarray
        assert returnEulers.shape == (3, )

    @staticmethod
    def test_calc(single_quat):
        returnEulers = single_quat.euler_angles()

        assert np.allclose(returnEulers*180/np.pi, [20., 10., 40.])

    @staticmethod
    def test_calc_chi_q12_0():
        in_quat = Quat(0.70710678, 0., 0., 0.70710678)
        returnEulers = in_quat.euler_angles()

        assert np.allclose(returnEulers, [4.71238898, 0., 0.])

    @staticmethod
    def test_calc_chi_q03_0():
        in_quat = Quat(0., 0.70710678, 0.70710678, 0.)
        returnEulers = in_quat.euler_angles()

        assert np.allclose(returnEulers, [1.57079633, 3.14159265, 0.])


class TestRotMatrix:

    @staticmethod
    def test_return_type(single_quat):
        returnMatrix = single_quat.rot_matrix()

        assert type(returnMatrix) is np.ndarray
        assert returnMatrix.shape == (3, 3)

    @staticmethod
    def test_calc(single_quat):
        returnMatrix = single_quat.rot_matrix()

        expectedMatrix = np.array([
            [ 0.50333996,  0.85684894,  0.1116189 ],
            [-0.862045  ,  0.48906392,  0.13302222],
            [ 0.05939117, -0.16317591,  0.98480775]
        ])

        assert np.allclose(returnMatrix, expectedMatrix)


# Test arithmetic


class TestMul:

    @staticmethod
    def test_return_type(single_quat, single_quat2):
        result = single_quat * single_quat2

        assert type(result) is Quat
        assert result is not single_quat and result is not single_quat2

    @staticmethod
    def test_calc(single_quat, single_quat2):
        result = single_quat * single_quat2

        assert np.allclose(result.quat_coef,
                           [0.8365163, 0.28678822, -0.40957602, 0.22414387])

    @staticmethod
    def test_bad_in_type(single_quat):
        with pytest.raises(TypeError):
            single_quat * 4


class TestDot:

    @staticmethod
    def test_return_type(single_quat, single_quat2):
        result = single_quat.dot(single_quat2)

        assert type(result) is np.float64

    @staticmethod
    def test_calc(single_quat, single_quat2):
        result = single_quat.dot(single_quat2)

        assert result == approx(0.16291828363609984)

    @staticmethod
    def test_bad_in_type(single_quat):
        with pytest.raises(TypeError):
            single_quat.dot(4)


class TestAdd:

    @staticmethod
    def test_return_type(single_quat, single_quat2):
        result = single_quat + single_quat2

        assert type(result) is Quat
        assert result is not single_quat and result is not single_quat2

    @staticmethod
    def test__calc(single_quat, single_quat2):
        result = single_quat + single_quat2

        assert np.allclose(result.quat_coef,
                           [1.44195788, 0.43400514, -0.22726944, 0.08113062])

    @staticmethod
    def test_bad_in_type(single_quat):
        with pytest.raises(TypeError):
            single_quat + 4


class TestIadd:

    @staticmethod
    def test_return_type(single_quat, single_quat2):
        single_quat_in = single_quat
        single_quat += single_quat2

        assert type(single_quat) is Quat
        assert single_quat is single_quat_in

    @staticmethod
    def test_calc(single_quat, single_quat2):
        single_quat += single_quat2

        assert np.allclose(single_quat.quat_coef,
                           [1.44195788, 0.43400514, -0.22726944, 0.08113062])

    @staticmethod
    def test_bad_in_type(single_quat):
        with pytest.raises(TypeError):
            single_quat += 4


class TestGetitem:

    @staticmethod
    def test_return_type(single_quat):
        for i in range(4):
            assert type(single_quat[i]) is np.float64

    @staticmethod
    def test_val(single_quat):
        assert single_quat[0] == approx(0.86272992)
        assert single_quat[1] == approx(-0.08583165)
        assert single_quat[2] == approx(0.01513444)
        assert single_quat[3] == approx(-0.49809735)


class TestSetitem:

    @staticmethod
    def test_val(single_quat):
        single_quat[0] = 0.1
        assert np.allclose(single_quat.quat_coef,
                           [0.1, -0.08583165, 0.01513444, -0.49809735])

        single_quat[1] = 0.2
        assert np.allclose(single_quat.quat_coef,
                           [0.1, 0.2, 0.01513444, -0.49809735])

        single_quat[2] = 0.3
        assert np.allclose(single_quat.quat_coef,
                           [0.1, 0.2, 0.3, -0.49809735])

        single_quat[3] = 0.4
        assert np.allclose(single_quat.quat_coef, [0.1, 0.2, 0.3, 0.4])


class TestNorm:

    @staticmethod
    def test_return_type(single_quat_not_unit):
        result = single_quat_not_unit.norm()

        assert type(result) is np.float64

    @staticmethod
    def test_calc(single_quat_not_unit):
        result = single_quat_not_unit.norm()

        assert result == approx(5.477225575051661)


class TestNormalise:

    @staticmethod
    def test_calc(single_quat_not_unit):
        single_quat_not_unit.normalise()

        assert np.allclose(single_quat_not_unit.quat_coef,
                           [0.18257419, 0.36514837, -0.54772256, 0.73029674])


class TestConjugate:

    @staticmethod
    def test_return_type(single_quat):
        result = single_quat.conjugate

        assert type(result) is Quat
        assert result is not single_quat

    @staticmethod
    def test_calc(single_quat):
        result = single_quat.conjugate

        assert np.allclose(result.quat_coef,
                           [0.86272992, 0.08583165, -0.01513444, 0.49809735])


class TestTransformVector:

    @staticmethod
    def test_return_type(single_quat):
        result = single_quat.transform_vector(np.array([1., 2., 3.]))

        assert type(result) is np.ndarray
        assert result.shape == (3,)

    @staticmethod
    def test_calc(single_quat):
        result = single_quat.transform_vector(np.array([1., 2., 3.]))

        assert np.allclose(result, [2.55189453, 0.5151495, 2.68746261])

    @staticmethod
    def test_bad_in_type(single_quat):
        with pytest.raises(TypeError):
            single_quat.transform_vector(10)

        with pytest.raises(TypeError):
            single_quat.transform_vector(np.array([1., 2., 3., 4.]))


class TestMisOriCases:

    @staticmethod
    @parametrize(rtn_quat=[0, 1, 2, 'potato'])
    def case_cubic(rtn_quat, single_quat, single_quat2):

        # TODO: Would be better to get quaternions from a fixture but
        #  not currently supported by pytest-cases
        ins = (single_quat, single_quat2, "cubic", rtn_quat)

        outs = (0.8887075008823285,
                [0.96034831, 0.13871646, 0.19810764, -0.13871646])

        return ins, outs

    @staticmethod
    @parametrize(rtn_quat=[0, 1, 2, 'potato'])
    def case_hexagonal(rtn_quat, single_quat, single_quat2):

        ins = (single_quat, single_quat2, "hexagonal", rtn_quat)

        outs = (0.8011677034014963,
                [0.57922797, -0.24240388, -0.51983679, -0.57922797])

        return ins, outs

    @staticmethod
    @parametrize(rtn_quat=[0, 1, 2, 'potato'])
    def case_null(rtn_quat, single_quat, single_quat2):

        ins = (single_quat, single_quat2, "potato", rtn_quat)

        outs = (0.16291828692295218,
                [0.57922797,  0.51983679, -0.24240388,  0.57922797])

        return ins, outs


class TestMisOri:

    @staticmethod
    @parametrize_with_cases("ins, outs", cases=TestMisOriCases)
    def test_return_type(ins, outs):
        result = ins[0].mis_ori(*ins[1:])

        if ins[3] == 1:
            assert type(result) is Quat
        elif ins[3] == 2:
            assert type(result) is tuple
            assert len(result) == 2
            assert type(result[0]) is np.float64
            assert type(result[1]) is Quat
        else:
            assert type(result) is np.float64

    @staticmethod
    @parametrize_with_cases("ins, outs", cases=TestMisOriCases)
    def test_calc(ins, outs):
        result = ins[0].mis_ori(*ins[1:])

        if ins[3] == 1:
            assert np.allclose(result.quat_coef, outs[1])
        elif ins[3] == 2:
            assert result[0] == approx(outs[0])
            assert np.allclose(result[1].quat_coef, outs[1])
        else:
            assert result == approx(outs[0])

    @staticmethod
    def test_bad_in_type(single_quat):
        with pytest.raises(TypeError):
            single_quat.mis_ori(4, "blah")


class TestMisOriAxis:

    @staticmethod
    def test_return_type(single_quat, single_quat2):
        result = single_quat.mis_ori_axis(single_quat2)

        assert type(result) is np.ndarray
        assert result.shape == (3,)

    @staticmethod
    def test_calc(single_quat, single_quat2):
        result = single_quat.mis_ori_axis(single_quat2)

        assert np.allclose(result, [1.10165762, -1.21828737, 2.285256])

    @staticmethod
    def test_bad_in_type(single_quat):
        with pytest.raises(TypeError):
            single_quat.mis_ori_axis(4)


class TestExtractQuatComps:
    """Test the method that returns a NumPy array from a list of Quats."""
    @staticmethod
    def test_return_type(ori_quat_list_valid):
        quat_comps = Quat.extract_quat_comps(ori_quat_list_valid)

        assert type(quat_comps) is np.ndarray
        assert quat_comps.shape == (4, len(ori_quat_list_valid))

    @staticmethod
    def test_calc(ori_quat_list_valid):
        quat_comps = Quat.extract_quat_comps(ori_quat_list_valid)

        expected_comps = np.array([
            [0.22484510, 0.45464871, -0.70807342, 0.49129550],
            [0.36520321, 0.25903472, -0.40342268, 0.79798357]
        ]).T

        assert np.allclose(quat_comps, expected_comps)


class TestSymEqvCases:

    @staticmethod
    def case_cubic():
        ins = ('cubic',)
        outs = (
            [[1.0, 0.0, 0.0, 0.0],
             [0.7071067811865476, 0.7071067811865476, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.7071067811865476, -0.7071067811865476, 0.0, 0.0],
             [0.7071067811865476, 0.0, 0.7071067811865476, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.7071067811865476, 0.0, -0.7071067811865476, 0.0],
             [0.7071067811865476, 0.0, 0.0, 0.7071067811865476],
             [0.0, 0.0, 0.0, 1.0],
             [0.7071067811865476, 0.0, 0.0, -0.7071067811865476],
             [0.0, 0.7071067811865476, 0.7071067811865476, 0.0],
             [0.0, -0.7071067811865476, 0.7071067811865476, 0.0],
             [0.0, 0.7071067811865476, 0.0, 0.7071067811865476],
             [0.0, -0.7071067811865476, 0.0, 0.7071067811865476],
             [0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
             [0.0, 0.0, -0.7071067811865476, 0.7071067811865476],
             [0.5, 0.5, 0.5, 0.5],
             [0.5, -0.5, -0.5, -0.5],
             [0.5, -0.5, 0.5, 0.5],
             [0.5, 0.5, -0.5, -0.5],
             [0.5, 0.5, -0.5, 0.5],
             [0.5, -0.5, 0.5, -0.5],
             [0.5, 0.5, 0.5, -0.5],
             [0.5, -0.5, -0.5, 0.5]],
        )

        return ins, outs

    @staticmethod
    def case_hexagonal():
        ins = ('hexagonal',)
        outs = (
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0],
             [0.8660254037844386, 0.0, 0.0, 0.5],
             [0.5, 0.0, 0.0, 0.8660254037844386],
             [0.5, 0.0, 0.0, -0.8660254037844386],
             [0.8660254037844386, 0.0, 0.0, -0.5],
             [0.0, -0.5, -0.8660254037844386, 0.0],
             [0.0, 0.5, -0.8660254037844386, 0.0],
             [0.0, 0.8660254037844386, -0.5, 0.0],
             [0.0, -0.8660254037844386, -0.5, 0.0]],
        )

        return ins, outs

    @staticmethod
    def case_null():
        ins = ('potato',)
        outs = ([[1.0, 0.0, 0.0, 0.0]],)

        return ins, outs


class TestSymEqv:

    @staticmethod
    @parametrize_with_cases("ins, outs", cases=TestSymEqvCases)
    def test_return_type(ins, outs):
        syms = Quat.sym_eqv(*ins)

        assert type(syms) is list
        assert len(syms) == len(outs[0])
        assert all([type(sym) is Quat for sym in syms])

    @staticmethod
    @parametrize_with_cases("ins, outs", cases=TestSymEqvCases)
    def test_calc(ins, outs):
        syms = Quat.sym_eqv(*ins)

        assert all([np.allclose(sym.quat_coef, row) for sym, row
                    in zip(syms, outs[0])])







''' Functions left to test
__repr__(self):
__str__(self):
plot_ipf
plotUnitCell

create_many_quats(eulerArray)
calc_sym_eqvs(quats, symGroup, dtype=np.float)
calc_average_ori(quatComps)
calcMisOri(quatComps, refOri)
polar_angles(x, y, z)
calc_ipf_colours(quats, direction, symGroup)
calc_fund_dirs(quats, direction, symGroup, dtype=np.float)
'''



