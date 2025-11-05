# Copyright 2025 Mechanics of Microstructures Group
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
from defdap import defaults

from typing import Union, Tuple, List, Optional


class Quat(object):
    """Class used to define and perform operations on quaternions. These
    are interpreted in the passive sense.

    """
    __slots__ = ['quat_coef']

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
            self.quat_coef = np.array(args[0], dtype=float)

        # construct with quat coefficients
        elif len(args) == 4:
            self.quat_coef = np.array(args, dtype=float)

        else:
            raise TypeError("Incorrect argument length. Input should be "
                            "an array of quat coefficients or idividual "
                            "quat coefficients")

        # move to northern hemisphere
        if not allow_southern and self.quat_coef[0] < 0:
            self.quat_coef = self.quat_coef * -1

    @classmethod
    def from_euler_angles(cls, ph1: float, phi: float, ph2: float) -> 'Quat':
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
        quat_coef = np.array([
            np.cos(phi / 2.0) * np.cos((ph1 + ph2) / 2.0),
            -np.sin(phi / 2.0) * np.cos((ph1 - ph2) / 2.0),
            -np.sin(phi / 2.0) * np.sin((ph1 - ph2) / 2.0),
            -np.cos(phi / 2.0) * np.sin((ph1 + ph2) / 2.0)
        ], dtype=float)

        # call constructor
        return cls(quat_coef)

    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle: float) -> 'Quat':
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
        quat_coef = np.zeros(4, dtype=float)
        quat_coef[0] = np.cos(angle / 2)
        quat_coef[1:4] = -np.sin(angle / 2) * axis

        # call constructor
        return cls(quat_coef)

    def euler_angles(self) -> np.ndarray:
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

        q = self.quat_coef
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
            cos_ph1 = (-q[0] * q[1] - q[2] * q[3]) / chi
            sin_ph1 = (-q[0] * q[2] + q[1] * q[3]) / chi

            cos_phi = q[0]**2 + q[3]**2 - q[1]**2 - q[2]**2
            sin_phi = 2 * chi

            cos_ph2 = (-q[0] * q[1] + q[2] * q[3]) / chi
            sin_ph2 = (q[1] * q[3] + q[0] * q[2]) / chi

            eulers[0] = np.arctan2(sin_ph1, cos_ph1)
            eulers[1] = np.arctan2(sin_phi, cos_phi)
            eulers[2] = np.arctan2(sin_ph2, cos_ph2)

        if eulers[0] < 0:
            eulers[0] += 2 * np.pi
        if eulers[2] < 0:
            eulers[2] += 2 * np.pi

        return eulers

    def rot_matrix(self) -> np.ndarray:
        """Calculate the rotation matrix representation for this rotation.

        Returns
        -------
        rot_matrix : numpy.ndarray, shape (3, 3)
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
        rot_matrix = np.empty((3, 3), dtype=float)

        q = self.quat_coef
        qbar = q[0]**2 - q[1]**2 - q[2]**2 - q[3]**2

        rot_matrix[0, 0] = qbar + 2 * q[1]**2
        rot_matrix[0, 1] = 2 * (q[1] * q[2] - q[0] * q[3])
        rot_matrix[0, 2] = 2 * (q[1] * q[3] + q[0] * q[2])

        rot_matrix[1, 0] = 2 * (q[1] * q[2] + q[0] * q[3])
        rot_matrix[1, 1] = qbar + 2 * q[2]**2
        rot_matrix[1, 2] = 2 * (q[2] * q[3] - q[0] * q[1])

        rot_matrix[2, 0] = 2 * (q[1] * q[3] - q[0] * q[2])
        rot_matrix[2, 1] = 2 * (q[2] * q[3] + q[0] * q[1])
        rot_matrix[2, 2] = qbar + 2 * q[3]**2

        return rot_matrix

    # show components when the quat is printed
    def __repr__(self) -> str:
        return "[{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(*self.quat_coef)

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, right: 'Quat') -> bool:
        return (isinstance(right, type(self)) and 
            self.quat_coef.tolist() == right.quat_coef.tolist())

    def __hash__(self) -> int:
        return hash(tuple(self.quat_coef.tolist()))

    def _plotIPF(
        self,
        direction: np.ndarray,
        sym_group: str,
        **kwargs
    ) -> 'plotting.PolePlot':
        Quat.plot_ipf([self], direction, sym_group, **kwargs)

    # overload * operator for quaternion product and vector product
    def __mul__(self, right: 'Quat', allow_southern: bool = False) -> 'Quat':
        if isinstance(right, type(self)):   # another quat
            new_quat_coef = np.zeros(4, dtype=float)
            new_quat_coef[0] = (
                    self.quat_coef[0] * right.quat_coef[0] -
                    np.dot(self.quat_coef[1:4], right.quat_coef[1:4])
            )
            new_quat_coef[1:4] = (
                    self.quat_coef[0] * right.quat_coef[1:4] +
                    right.quat_coef[0] * self.quat_coef[1:4] +
                    np.cross(self.quat_coef[1:4], right.quat_coef[1:4])
            )
            return Quat(new_quat_coef, allow_southern=allow_southern)

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
            return np.dot(self.quat_coef, right.quat_coef)
        raise TypeError()

    # overload + operator
    def __add__(self, right: 'Quat') -> 'Quat':
        if isinstance(right, type(self)):
            return Quat(self.quat_coef + right.quat_coef)
        raise TypeError()

    # overload += operator
    def __iadd__(self, right: 'Quat') -> 'Quat':
        if isinstance(right, type(self)):
            self.quat_coef += right.quat_coef
            return self
        raise TypeError()

    # allow array like setting/getting of components
    def __getitem__(self, key: int) -> float:
        return self.quat_coef[key]

    def __setitem__(self, key: int, value: float) -> None:
        self.quat_coef[key] = value

    def norm(self) -> float:
        """Calculate the norm of the quaternion.

        Returns
        -------
        float
            Norm of the quaternion.

        """
        return np.sqrt(np.dot(self.quat_coef[0:4], self.quat_coef[0:4]))

    def normalise(self) -> 'Quat':
        """ Normalise the quaternion (turn it into an unit quaternion).

        Returns
        -------
        defdap.quat.Quat
            Normalised quaternion.

        """
        self.quat_coef /= self.norm()
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

    def transform_vector(
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

        vector_quat = Quat(0, *vector)
        vector_quat_transformed = self.__mul__(
            vector_quat.__mul__(self.conjugate, allow_southern=True),
            allow_southern=True
        )
        return vector_quat_transformed.quat_coef[1:4]

    def mis_ori(
        self,
        right: 'Quat',
        sym_group: str,
        return_quat: Optional[int] = 0
    ) -> Tuple[float, 'Quat']:
        """
        Calculate misorientation angle between 2 orientations taking
        into account the symmetries of the crystal structure.
        Angle is 2*arccos(output).

        Parameters
        ----------
        right
            Orientation to find misorientation to.
        sym_group
            Crystal type (cubic, hexagonal).
        return_quat
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
            min_mis_ori = 0
            # loop over symmetrically equivalent orientations
            for sym in Quat.sym_eqv(sym_group):
                quat_sym = sym * right
                current_mis_ori = abs(self.dot(quat_sym))
                if current_mis_ori > min_mis_ori:   # keep if misorientation lower
                    min_mis_ori = current_mis_ori
                    min_quat_sym = quat_sym

            if return_quat == 1:
                return min_quat_sym
            elif return_quat == 2:
                return min_mis_ori, min_quat_sym
            else:
                return min_mis_ori
        raise TypeError("Input must be a quaternion.")

    def mis_ori_axis(self, right: 'Quat') -> np.ndarray:
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
            Dq = Dq.quat_coef
            mis_ori_axis = 2 * Dq[1:4] * np.arccos(Dq[0]) / np.sqrt(1 - Dq[0]**2)
            return mis_ori_axis
        raise TypeError("Input must be a quaternion.")

    def plot_ipf(
        self,
        direction: np.ndarray,
        sym_group: str,
        projection:  Optional[str] = None,
        plot: Optional['plotting.Plot'] = None,
        fig: Optional['matplotlib.figure.Figure'] = None,
        ax: Optional['matplotlib.axes.Axes'] = None,
        plot_colour_bar: Optional[bool] = False,
        clabel: Optional[str] = "",
        make_interactive: Optional[bool] = False,
        marker_colour: Optional[Union[List[str], str]] = None,
        marker_size: Optional[float] = 40,
        **kwargs
    ) -> 'plotting.PolePlot':
        """
        Plot IPF of orientation, with relation to specified sample direction.

        Parameters
        ----------
        direction
            Sample reference direction for IPF.
        sym_group
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
        make_interactive
            If true, make the plot interactive.
        plot_colour_bar : bool
            If true, plot a colour bar next to the map.
        clabel : str
            Label for the colour bar.
        marker_colour: str or list of str
            Colour of markers (only used for half and half colouring,
            otherwise use argument c).
        marker_size
            Size of markers (only used for half and half colouring,
            otherwise use argument s).
        kwargs
            All other arguments are passed to :func:`defdap.plotting.PolePlot.add_points`.

        """
        plot_params = {'marker': '+'}
        plot_params.update(kwargs)

        # Works as an instance or static method on a list of Quats
        if isinstance(self, Quat):
            quats = [self]
        else:
            quats = self

        alpha_fund, beta_fund = Quat.calc_fund_dirs(quats, direction, sym_group)

        if plot is None:
            plot = plotting.PolePlot(
                "IPF", sym_group, projection=projection,
                ax=ax, fig=fig, make_interactive=make_interactive
            )
        plot.add_points(
            alpha_fund, beta_fund,
            marker_colour=marker_colour, marker_size=marker_size,
            **plot_params
        )

        if plot_colour_bar:
            plot.add_colour_bar(clabel)

        return plot

    def plot_unit_cell(
        self,
        crystal_structure: 'defdap.crystal.CrystalStructure',
        OI: Optional[bool] = True,
        plot: Optional['plotting.CrystalPlot'] = None,
        fig: Optional['matplotlib.figure.Figure'] = None,
        ax: Optional['matplotlib.axes.Axes'] = None,
        make_interactive: Optional[bool] = False,
        **kwargs
    ) -> 'plotting.CrystalPlot':
        """Plots a unit cell.

        Parameters
        ----------
        crystal_structure
            Crystal structure.
        OI
            True if using oxford instruments system.
        plot
            Plot object to plot to.
        fig
            Figure to plot on, if not provided the current active axis is used.
        ax
            Axis to plot on, if not provided the current active axis is used.
        make_interactive
            True to make the plot interactive.
        kwargs
            All other arguments are passed to :func:`defdap.plotting.CrystalPlot.add_verts`.

        """
        # Set default plot parameters then update with any input
        plot_params = {}
        plot_params.update(kwargs)

        # TODO: most of this should be moved to either the crystal or
        #  plotting module

        vert = crystal_structure.vertices
        faces = crystal_structure.faces

        if crystal_structure.name == 'hexagonal':
            sz_fac = 0.18
            if OI:
                # Add 30 degrees to phi2 for OI
                eulerAngles = self.euler_angles()
                eulerAngles[2] += np.pi / 6
                gg = Quat.from_euler_angles(*eulerAngles).rot_matrix().T
            else:
                gg = self.rot_matrix().T

        elif crystal_structure.name == 'cubic':
            sz_fac = 0.25
            gg = self.rot_matrix().T

        # Rotate the lattice cell points
        pts = np.matmul(gg, vert.T).T * sz_fac

        # Plot unit cell
        planes = []
        for face in faces:
            planes.append(pts[face, :])

        if plot is None:
            plot = plotting.CrystalPlot(
                ax=ax, fig=fig, make_interactive=make_interactive
            )

        plot.ax.set_xlim3d(-0.15, 0.15)
        plot.ax.set_ylim3d(-0.15, 0.15)
        plot.ax.set_zlim3d(-0.15, 0.15)
        plot.ax.view_init(azim=270, elev=90)
        plot.ax._axis3don = False

        plot.add_verts(planes, **plot_params)

        return plot

# Static methods

    @staticmethod
    def create_many_quats(eulerArray: np.ndarray) -> np.ndarray:
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
        ori_shape = eulerArray.shape[1:]

        quat_comps = np.zeros((4,) + ori_shape, dtype=float)

        quat_comps[0] = np.cos(phi / 2.0) * np.cos((ph1 + ph2) / 2.0)
        quat_comps[1] = -np.sin(phi / 2.0) * np.cos((ph1 - ph2) / 2.0)
        quat_comps[2] = -np.sin(phi / 2.0) * np.sin((ph1 - ph2) / 2.0)
        quat_comps[3] = -np.cos(phi / 2.0) * np.sin((ph1 + ph2) / 2.0)

        quats = np.empty(ori_shape, dtype=Quat)

        for i, idx in enumerate(np.ndindex(ori_shape)):
            quats[idx] = Quat(quat_comps[(slice(None),) + idx])

        return quats

    @staticmethod
    def multiply_many_quats(quats: List['Quat'], right: 'Quat') -> List['Quat']:
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
        quat_array = np.array([q.quat_coef for q in quats])

        temp_array = np.zeros((len(quat_array),4), dtype=float)
        temp_array[...,0] = ((quat_array[...,0] * right.quat_coef[0]) -
                            np.dot(quat_array[...,1:4], right.quat_coef[1:4]))

        temp_array[...,1:4] = ((quat_array[...,0,None] * right.quat_coef[None, 1:4]) +
                              (right.quat_coef[0] * quat_array[..., 1:4]) +
                              np.cross(quat_array[...,1:4], right.quat_coef[1:4]))

        return [Quat(coefs) for coefs in temp_array]

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
            quat_comps[(slice(None),) + idx] = quats[idx].quat_coef

        return quat_comps

    @staticmethod
    def calc_sym_eqvs(
        quats: np.ndarray,
        sym_group: str,
        dtype: Optional[type] = float
    ) -> np.ndarray:
        """Calculate all symmetrically equivalent quaternions of given quaternions.

        Parameters
        ----------
        quats : numpy.ndarray(defdap.quat.Quat)
            Array of quat objects.
        sym_group
            Crystal type (cubic, hexagonal).
        dtype
            Datatype used for calculation, defaults to `float`.

        Returns
        -------
        quat_comps: numpy.ndarray, shape: (numSym x 4 x numQuats)
            Array containing all symmetrically equivalent quaternion components of input quaternions.

        """
        syms = Quat.sym_eqv(sym_group)
        quat_comps = np.empty((len(syms), 4, len(quats)), dtype=dtype)

        # store quat components in array
        quat_comps[0] = Quat.extract_quat_comps(quats)

        # calculate symmetrical equivalents
        for i, sym in enumerate(syms[1:], start=1):
            # sym[i] * quat for all points (* is quaternion product)
            quat_comps[i, 0, :] = (
                quat_comps[0, 0, :] * sym[0] - quat_comps[0, 1, :] * sym[1] -
                quat_comps[0, 2, :] * sym[2] - quat_comps[0, 3, :] * sym[3])
            quat_comps[i, 1, :] = (
                quat_comps[0, 0, :] * sym[1] + quat_comps[0, 1, :] * sym[0] -
                quat_comps[0, 2, :] * sym[3] + quat_comps[0, 3, :] * sym[2])
            quat_comps[i, 2, :] = (
                quat_comps[0, 0, :] * sym[2] + quat_comps[0, 2, :] * sym[0] -
                quat_comps[0, 3, :] * sym[1] + quat_comps[0, 1, :] * sym[3])
            quat_comps[i, 3, :] = (
                quat_comps[0, 0, :] * sym[3] + quat_comps[0, 3, :] * sym[0] -
                quat_comps[0, 1, :] * sym[2] + quat_comps[0, 2, :] * sym[1])

            # swap into positive hemisphere if required
            quat_comps[i, :, quat_comps[i, 0, :] < 0] *= -1

        return quat_comps

    @staticmethod
    def calc_average_ori(
        quat_comps: np.ndarray
    ) -> 'Quat':
        """Calculate the average orientation of given quats.

        Parameters
        ----------
        quat_comps : numpy.ndarray
            Array containing all symmetrically equivalent quaternion components of given quaternions
            (shape: numSym x 4 x numQuats), can be calculated with :func:`Quat.calc_sym_eqvs`.

        Returns
        -------
        av_ori : defdap.quat.Quat
            Average orientation of input quaternions.

        """
        av_ori = np.copy(quat_comps[0, :, 0])
        curr_mis_oris = np.empty(quat_comps.shape[0])

        for i in range(1, quat_comps.shape[2]):
            # calculate misorientation between current average and all
            # symmetrical equivalents. Dot product of each symm quat in
            # quatComps with refOri for point i
            curr_mis_oris[:] = abs(np.einsum(
                "ij,j->i", quat_comps[:, :, i], av_ori
            ))

            # find min misorientation with current average then add to it
            max_idx = np.argmax(curr_mis_oris[:])
            av_ori += quat_comps[max_idx, :, i]

        # Convert components back to a quat and normalise
        av_ori = Quat(av_ori)
        av_ori.normalise()

        return av_ori

    @staticmethod
    def calcMisOri(
        quat_comps: np.ndarray,
        ref_ori: 'Quat'
    ) -> Tuple[np.ndarray, 'Quat']:
        """Calculate the misorientation between the quaternions and a reference quaternion.

        Parameters
        ----------
        quat_comps
            Array containing all symmetrically equivalent quaternion components of given quaternions
            (shape: numSym x 4 x numQuats), can be calculated from quats with :func:`Quat.calc_sym_eqvs` .
        ref_ori
            Reference orientation.

        Returns
        -------
        min_mis_oris : numpy.ndarray, len numQuats
            Minimum misorientation between quats and reference orientation.
        min_quat_comps : defdap.quat.Quat
            Quaternion components describing minimum misorientation between quats and reference orientation.

        """
        mis_oris = np.empty((quat_comps.shape[0], quat_comps.shape[2]))

        # Dot product of each quat in quatComps with refOri
        mis_oris[:, :] = abs(np.einsum("ijk,j->ik", quat_comps, ref_ori.quat_coef))

        max_idxs0 = np.argmax(mis_oris, axis=0)
        max_idxs1 = np.arange(mis_oris.shape[1])

        min_mis_oris = mis_oris[max_idxs0, max_idxs1]

        min_quat_comps = quat_comps[max_idxs0, :, max_idxs1].transpose()

        min_mis_oris[min_mis_oris > 1] = 1

        return min_mis_oris, min_quat_comps

    @staticmethod
    def polar_angles(x: np.ndarray, y: np.ndarray, z: np.ndarray):
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
    def calc_ipf_colours(
        quats: np.ndarray,
        direction: np.ndarray,
        sym_group: str,
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
        sym_group
            Crystal type (cubic, hexagonal).
        dtype
            Data type to use for calculation.

        Returns
        -------
        numpy.ndarray, shape (3, num_quats)
            Array of rgb colours for each quat.

        References
        -------
        Stephen Cluff (BYU) - IPF_rgbcalc.m subroutine in OpenXY
        https://github.com/BYU-MicrostructureOfMaterials/OpenXY/blob/master/Code/PlotIPF.m

        """
        num_quats = len(quats)

        alpha_fund, beta_fund = Quat.calc_fund_dirs(
            quats, direction, sym_group, triangle='up', dtype=dtype
        )

        # revert to cartesians
        dirvec = np.empty((3, num_quats), dtype=dtype)
        dirvec[0, :] = np.sin(alpha_fund) * np.cos(beta_fund)
        dirvec[1, :] = np.sin(alpha_fund) * np.sin(beta_fund)
        dirvec[2, :] = np.cos(alpha_fund)

        if sym_group == 'cubic':
            pole_directions = np.array([
                [0, 0, 1],
                [0, 1, 1]/np.sqrt(2),
                [-1, 1, 1]/np.sqrt(3)
            ], dtype=dtype)
        elif sym_group == 'hexagonal':
            pole_directions = np.array([
                [0, 0, 1],
                [0, 1, 0],
                [-0.5, np.sqrt(3)/2, 0]
            ], dtype=dtype)
        else:
            raise ValueError(f'Unknown sym_group `{sym_group}`')

        rvect = np.broadcast_to(pole_directions[0].reshape((-1, 1)), (3, num_quats))
        gvect = np.broadcast_to(pole_directions[1].reshape((-1, 1)), (3, num_quats))
        bvect = np.broadcast_to(pole_directions[2].reshape((-1, 1)), (3, num_quats))

        rgb = np.zeros((3, num_quats), dtype=dtype)

        # Red Component
        r_dir_plane = np.cross(dirvec, rvect, axis=0)
        gb_plane = np.cross(bvect, gvect, axis=0)
        r_intersect = np.cross(r_dir_plane, gb_plane, axis=0)
        r_norm = np.linalg.norm(r_intersect, axis=0, keepdims=True)
        r_norm[r_norm == 0] = 1       #Prevent division by zero
        r_intersect /= r_norm

        temp = np.arccos(np.clip(np.einsum("ij,ij->j", dirvec, r_intersect), -1, 1))
        r_intersect[:, temp > (np.pi / 2)] *= -1
        rgb[0, :] = np.divide(
            np.arccos(np.clip(np.einsum("ij,ij->j", dirvec, r_intersect), -1, 1)),
            np.arccos(np.clip(np.einsum("ij,ij->j", rvect, r_intersect), -1, 1))
        )

        # Green Component
        g_dir_plane = np.cross(dirvec, gvect, axis=0)
        rb_plane = np.cross(rvect, bvect, axis=0)
        g_intersect = np.cross(g_dir_plane, rb_plane, axis=0)
        g_norm = np.linalg.norm(g_intersect, axis=0, keepdims=True)
        g_norm[g_norm == 0] = 1       #Prevent division by zero
        g_intersect /= g_norm

        temp = np.arccos(np.clip(np.einsum("ij,ij->j", dirvec, g_intersect), -1, 1))
        g_intersect[:, temp > (np.pi / 2)] *= -1
        rgb[1, :] = np.divide(
            np.arccos(np.clip(np.einsum("ij,ij->j", dirvec, g_intersect), -1, 1)),
            np.arccos(np.clip(np.einsum("ij,ij->j", gvect, g_intersect), -1, 1))
        )

        # Blue Component
        b_dir_plane = np.cross(dirvec, bvect, axis=0)
        rg_plane = np.cross(gvect, rvect, axis=0)
        b_intersect = np.cross(b_dir_plane, rg_plane, axis=0)
        b_norm = np.linalg.norm(b_intersect, axis=0, keepdims=True)
        b_norm[b_norm == 0] = 1       #Prevent division by zero
        b_intersect /= b_norm

        temp = np.arccos(np.clip(np.einsum("ij,ij->j", dirvec, b_intersect), -1, 1))
        b_intersect[:, temp > (np.pi / 2)] *= -1
        rgb[2, :] = np.divide(
            np.arccos(np.clip(np.einsum("ij,ij->j", dirvec, b_intersect), -1, 1)),
            np.arccos(np.clip(np.einsum("ij,ij->j", bvect, b_intersect), -1, 1))
        )
        rgb /= np.amax(rgb, axis=0)

        return rgb

    @staticmethod
    def calc_fund_dirs(
        quats: np.ndarray,
        direction: np.ndarray,
        sym_group: str,
        dtype: Optional[type] = float,
        triangle: Optional[str] = None,
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
        sym_group
            Crystal type (cubic, hexagonal).
        dtype
            Data type to use for calculation.
        triangle:  str, optional
            Triangle convention to use for hexagonal symmetry (up, down). If None,
            defaults to the value in `defaults['ipf_triangle_convention']`.

        Returns
        -------
        float, float
            inclination angle and azimuthal angle (around z axis from x in anticlockwise).

        """
        # convert direction to float array
        direction = np.array(direction, dtype=dtype)

        # get array of symmetry operations. shape - (numSym, 4, numQuats)
        quat_comps_sym = Quat.calc_sym_eqvs(quats, sym_group, dtype=dtype)

        # array to store crystal directions for all orientations and symmetries
        direction_crystal = np.empty(
            (3, quat_comps_sym.shape[0], quat_comps_sym.shape[2]), dtype=dtype
        )

        # temp variables to use below
        quat_dot_vec = (quat_comps_sym[:, 1, :] * direction[0] +
                      quat_comps_sym[:, 2, :] * direction[1] +
                      quat_comps_sym[:, 3, :] * direction[2])
        temp = (np.square(quat_comps_sym[:, 0, :]) -
                np.square(quat_comps_sym[:, 1, :]) -
                np.square(quat_comps_sym[:, 2, :]) -
                np.square(quat_comps_sym[:, 3, :]))

        # transform the pole direction to crystal coords for all
        # orientations and symmetries
        # (quat_comps_sym * vectorQuat) * quat_comps_sym.conjugate
        direction_crystal[0, :, :] = (
                2 * quat_dot_vec * quat_comps_sym[:, 1, :] +
                temp * direction[0] +
                2 * quat_comps_sym[:, 0, :] * (
                        quat_comps_sym[:, 2, :] * direction[2] -
                        quat_comps_sym[:, 3, :] * direction[1]
                )
        )
        direction_crystal[1, :, :] = (
                2 * quat_dot_vec * quat_comps_sym[:, 2, :] +
                temp * direction[1] +
                2 * quat_comps_sym[:, 0, :] * (
                        quat_comps_sym[:, 3, :] * direction[0] -
                        quat_comps_sym[:, 1, :] * direction[2]
                )
        )
        direction_crystal[2, :, :] = (
                2 * quat_dot_vec * quat_comps_sym[:, 3, :] +
                temp * direction[2] +
                2 * quat_comps_sym[:, 0, :] * (
                        quat_comps_sym[:, 1, :] * direction[1] -
                        quat_comps_sym[:, 2, :] * direction[0]
                )
        )

        # normalise vectors
        direction_crystal /= np.sqrt(np.einsum(
            'ijk,ijk->jk', direction_crystal, direction_crystal
        ))

        # move all vectors into north hemisphere
        direction_crystal[:, direction_crystal[2, :, :] < 0] *= -1

        # convert to spherical coordinates
        alpha, beta = Quat.polar_angles(
            direction_crystal[0], direction_crystal[1], direction_crystal[2]
        )

        # find the poles in the fundamental triangle
        if sym_group == "cubic":
            beta_range = (np.pi / 2, 3/4 * np.pi, 3)

        elif sym_group == "hexagonal":
            if triangle is None:
                triangle = defaults['ipf_triangle_convention']

            if triangle == 'up':
                beta_range = (np.pi / 2, 2/3 * np.pi, 1)
            elif triangle == 'down':
                beta_range = (1/3 * np.pi, np.pi / 2, 1)
            else:
                ValueError("`triangle` must be 'up' or 'down'")
        else:
            raise ValueError("sym_group must be cubic or hexagonal")
        
        trial_poles = np.logical_and(beta >= beta_range[0], beta <= beta_range[1])
        # expand search slightly to catch edge cases if needed
        if np.any(np.sum(trial_poles, axis=0) < beta_range[2]):
            delta_beta = 1e-8
            trial_poles = np.logical_and(
                beta >= beta_range[0] - delta_beta,
                beta <= beta_range[1] + delta_beta
            )

        if sym_group == "cubic":
            # now of symmetric equivalents left we want the one with
            # minimum alpha
            fund_idx = np.nanargmin(np.where(trial_poles, alpha, np.nan), axis=0)
        else:
            # non-indexed points cause more than 1 symmetric equivalent, use this
            # to pick one and filter non-indexed points later
            fund_idx = trial_poles.argmax(axis=0)

        fund_idx = (fund_idx, range(len(fund_idx)))
        return alpha[fund_idx], beta[fund_idx]

    @staticmethod
    def sym_eqv(symGroup: str) -> List['Quat']:
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
