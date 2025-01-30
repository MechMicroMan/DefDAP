# Copyright 2024 Mechanics of Microstructures Group
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

from functools import partial
import numpy as np

from defdap import defaults

__all__ = [
    'create_l_matrix', 
    'create_q_matrix', 
    'convert_idc', 
    'equavlent_indicies', 
    'project_to_orth', 
    'pos_idc', 
    'reduce_idc', 
    'safe_int_cast', 
    'idc_to_string',
    'str_idx',
]


def create_l_matrix(a, b, c, alpha, beta, gamma, convention=None):
    """ Construct L matrix based on Page 22 of
    Randle and Engle - Introduction to texture analysis"""
    l_matrix = np.zeros((3, 3))

    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    cos_gamma = np.cos(gamma)

    sin_gamma = np.sin(gamma)

    l_matrix[0, 0] = a
    l_matrix[0, 1] = b * cos_gamma
    l_matrix[0, 2] = c * cos_beta

    l_matrix[1, 1] = b * sin_gamma
    l_matrix[1, 2] = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma

    l_matrix[2, 2] = c * np.sqrt(
        1 + 2 * cos_alpha * cos_beta * cos_gamma -
        cos_alpha**2 - cos_beta**2 - cos_gamma**2
    ) / sin_gamma

    # OI/HKL convention - x // [10-10],     y // a2 [-12-10]
    # TSL    convention - x // a1 [2-1-10], y // [01-10]
    if convention is None:
        convention = defaults['crystal_ortho_conv']

    if convention.lower() in ['hkl', 'oi']:
        # Swap 00 with 11 and 01 with 10 due to how OI orthonormalises
        # From Brad Wynne
        t1 = l_matrix[0, 0]
        t2 = l_matrix[1, 0]

        l_matrix[0, 0] = l_matrix[1, 1]
        l_matrix[1, 0] = l_matrix[0, 1]

        l_matrix[1, 1] = t1
        l_matrix[0, 1] = t2

    elif convention.lower() != 'tsl':
        raise ValueError(
            f"Unknown convention '{convention}' for orthonormalisation of "
            f"crystal structure, can be 'hkl' or 'tsl'"
        )

    # Set small components to 0
    l_matrix[np.abs(l_matrix) < 1e-10] = 0

    return l_matrix


def create_q_matrix(l_matrix):
    """ Construct matrix of reciprocal lattice vectors to transform
    plane normals See C. T. Young and J. L. Lytton, J. Appl. Phys.,
    vol. 43, no. 4, pp. 1408–1417, 1972."""
    a = l_matrix[:, 0]
    b = l_matrix[:, 1]
    c = l_matrix[:, 2]

    volume = abs(np.dot(a, np.cross(b, c)))
    a_star = np.cross(b, c) / volume
    b_star = np.cross(c, a) / volume
    c_star = np.cross(a, b) / volume

    q_matrix = np.stack((a_star, b_star, c_star), axis=1)

    return q_matrix


def check_len(val, length):
    if len(val) != length:
        raise ValueError(f"Vector must have {length} values.")


def convert_idc(in_type, *, dir=None, plane=None):
    """
    Convert between Miller and Miller-Bravais indices.

    Parameters
    ----------
    in_type : str {'m', 'mb'}
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

    if in_type.lower() == 'm':
        if dir is None:
            # plane M->MB
            check_len(plane, 3)
            out = np.array(plane)[[0, 1, 0, 2]]
            out[2] += plane[1]
            out[2] *= -1

        else:
            # direction M->MB
            check_len(dir, 3)
            u, v, w = dir
            out = np.array([2*u-v, 2*v-u, -u-v, 3*w]) / 3
            try:
                # Attempt to cast to integers
                out = safe_int_cast(out)
            except ValueError:
                pass

    elif in_type.lower() == 'mb':
        if dir is None:
            # plane MB->M
            check_len(plane, 4)
            out = np.array(plane)[[0, 1, 3]]

        else:
            # direction MB->M
            check_len(dir, 4)
            out = np.array(dir)[[0, 1, 3]]
            out[[0, 1]] -= dir[2]

    else:
        raise ValueError("`inType` must be either 'm' or 'mb'.")

    return tuple(out)


def equavlent_indicies(
    crystal_symm,
    symmetries,
    *, 
    dir=None, 
    plane=None, 
    c_over_a=None, 
    in_type=None
):
    if dir is None and plane is None:
        raise ValueError("One of either `direction` or `plane` must be "
                         "provided.")
    if dir is not None and plane is not None:
        raise ValueError("One of either `direction` or `plane` must be "
                         "provided, not both.")
    if in_type is None:
        in_type = 'mb' if crystal_symm == 'hexagonal' else 'm'
    
    planes = []
    dirs = []
    
    if in_type == 'mb':
        if dir is None:
            check_len(plane, 4)
            plane = convert_idc('mb', plane=plane)
        if plane is None:
            check_len(dir, 4)
            dir = convert_idc('mb', dir=dir)
    elif in_type != 'm':
        raise ValueError("`inType` must be either 'm' or 'mb'.")
    
    if dir is None:
        check_len(plane, 3)
        rtn = planes
    else:
        check_len(dir, 3)
        rtn = dirs

    if crystal_symm == 'hexagonal':
        # L matrix for transforming directions
        l_matrix = create_l_matrix(
            1, 1, c_over_a, np.pi / 2, np.pi / 2, np.pi * 2 / 3
        )
        # Q matrix for transforming planes
        q_matrix = create_q_matrix(l_matrix)
        
        if dir is None:
            plane = np.matmul(q_matrix, plane)
        else:
            dir = np.matmul(l_matrix, dir)

    for i, symm in enumerate(symmetries):
        if dir is None:
            plane_symm = symm.transform_vector(plane)
            if plane_symm[2] < 0:
                plane_symm *= -1
            if crystal_symm == 'hexagonal':
                # q_matrix inverse is equal to l_matrix transposed and vice-versa
                plane_symm = reduce_idc(convert_idc(
                    'm', plane=safe_int_cast(np.matmul(l_matrix.T, plane_symm))
                ))
            planes.append(safe_int_cast(plane_symm))
        else:
            dir_symm = symm.transform_vector(dir)
            if dir_symm[2] < 0:
                dir_symm *= -1
            if crystal_symm == 'hexagonal':
                dir_symm = reduce_idc(convert_idc(
                    'm', dir=safe_int_cast(np.matmul(q_matrix.T, dir_symm))
                ))
            dirs.append(safe_int_cast(dir_symm))
    
    return rtn


def project_to_orth(c_over_a, *, dir=None, plane=None, in_type='mb'):
    """
    Project from crystal aligned coordinates to an orthogonal set.

    Parameters
    ----------
    in_type : str {'m', 'mb'}
        Type of indices provided
    dir : tuple of int or equivalent, optional
        Direction to convert. This OR `plane` must me provided.
    plane : tuple of int or equivalent, optional
        Plane to convert. This OR `direction` must me provided.

    Returns
    -------


    """
    if dir is None and plane is None:
        raise ValueError("One of either `direction` or `plane` must be "
                         "provided.")
    if dir is not None and plane is not None:
        raise ValueError("One of either `direction` or `plane` must be "
                         "provided, not both.")
    
    if in_type == 'mb':
        if dir is None:
            check_len(plane, 4)
            plane = convert_idc('mb', plane=plane)
        if plane is None:
            check_len(dir, 4)
            dir = convert_idc('mb', dir=dir)
    elif in_type != 'm':
        raise ValueError("`inType` must be either 'm' or 'mb'.")
    
    # L matrix for transforming directions
    l_matrix = create_l_matrix(
        1, 1, c_over_a, np.pi / 2, np.pi / 2, np.pi * 2 / 3
    )
    
    if dir is None:
        check_len(plane, 3)
        # Q matrix for transforming planes
        q_matrix = create_q_matrix(l_matrix)
        return np.matmul(q_matrix, plane)
    else:
        check_len(dir, 3)
        return np.matmul(l_matrix, dir)


def pos_idc(vec):
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


def reduce_idc(vec):
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


def safe_int_cast(vec, tol=1e-3):
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


def idc_to_string(idc, brackets=None, str_type='unicode'):
    """
    String representation of a set of indicies.

    Parameters
    ----------
    idc : collection of int
    brackets : str
        String of opening and closing brackets eg '()'
    str_type : str {'unicode', 'tex'}

    Returns
    -------
    str

    """
    text = ''.join(map(partial(str_idx, str_type=str_type), idc))
    if brackets is not None:
        text = brackets[0] + text + brackets[1]
    return text


def str_idx(idx, str_type='unicode'):
    """
    String representation of an index with overbars.

    Parameters
    ----------
    idx : int
    str_type : str {'unicode', 'tex'}

    Returns
    -------
    str

    """
    if not isinstance(idx, (int, np.integer)):
        raise ValueError("Index must be an integer.")
        
    pre, post = (r'$\bar{', r'}$') if str_type == 'tex' else ('', u'\u0305')
    return str(idx) if idx >= 0 else pre + str(-idx) + post