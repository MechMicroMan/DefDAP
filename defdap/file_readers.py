# Copyright 2023 Mechanics of Microstructures Group
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
from numpy.lib.recfunctions import structured_to_unstructured
import pandas as pd
import pathlib
import re

from typing import TextIO, Dict, List, Callable, Any, Type, Optional

from defdap.crystal import Phase
from defdap.quat import Quat
from defdap.utils import Datastore


class EBSDDataLoader(object):
    """Class containing methods for loading and checking EBSD data

    """
    def __init__(self) -> None:
        # required metadata
        self.loaded_metadata = {
            'shape': (0, 0),
            'step_size': 0.,
            'acquisition_rotation': Quat(1.0, 0.0, 0.0, 0.0),
            'phases': [],
            'edx': {'Count': 0},
        }
        # required data
        self.loaded_data = Datastore()
        self.loaded_data.add(
            'phase', None, unit='', type='map', order=0,
            comment='1-based, 0 is non-indexed points',
            plot_params={
                'vmin': 0,
            }
        )
        self.loaded_data.add(
            'euler_angle', None, unit='rad', type='map', order=1,
            default_component='all_euler'
        )
        self.data_format = None

    @staticmethod
    def get_loader(data_type: str) -> 'Type[EBSDDataLoader]':
        if data_type is None:
            data_type = "OxfordBinary"

        data_type = data_type.lower()
        if data_type == "oxfordbinary":
            return OxfordBinaryLoader()
        elif data_type == "oxfordtext":
            return OxfordTextLoader()
        elif data_type == "edaxang":
            return EdaxAngLoader()
        elif data_type == "pythondict":
            return PythonDictLoader()
        else:
            raise ValueError(f"No loader for EBSD data of type {data_type}.")

    def check_metadata(self) -> None:
        """
        Checks that the number of phases from metadata matches
        the amount of phases loaded.

        """
        for phase in self.loaded_metadata['phases']:
            assert type(phase) is Phase

    def check_data(self) -> None:
        shape = self.loaded_metadata['shape']

        assert self.loaded_data.phase.shape == shape
        assert self.loaded_data.euler_angle.shape == (3,) + shape
        # assert self.loaded_data['bandContrast'].shape == mapShape


class OxfordTextLoader(EBSDDataLoader):
    def load(
        self,
        file_name: str,
        file_dir: str = ""
    ) -> None:
        """ Read an Oxford Instruments .ctf file, which is a HKL single
        orientation file.

        Parameters
        ----------
        file_name
            File name.
        file_dir
            Path to file.

        """
        # open data file and read in metadata
        file_name = "{}.ctf".format(file_name)
        file_path = pathlib.Path(file_dir) / pathlib.Path(file_name)
        if not file_path.is_file():
            raise FileNotFoundError("Cannot open file {}".format(file_path))

        def parse_phase() -> Phase:
            line_split = line.split('\t')
            dims = line_split[0].split(';')
            dims = tuple(round(float(s), 3) for s in dims)
            angles = line_split[1].split(';')
            angles = tuple(round(float(s), 3) * np.pi / 180 for s in angles)
            lattice_params = dims + angles
            phase = Phase(
                line_split[2],
                int(line_split[3]),
                int(line_split[4]),
                lattice_params
            )
            return phase

        # default values for acquisition rotation in case missing in in file
        acq_eulers = [0., 0., 0.]
        with open(str(file_path), 'r') as ctf_file:
            for i, line in enumerate(ctf_file):
                if 'XCells' in line:
                    x_dim = int(line.split()[-1])
                elif 'YCells' in line:
                    y_dim = int(line.split()[-1])
                elif 'XStep' in line:
                    self.loaded_metadata['step_size'] = float(line.split()[-1])
                elif 'AcqE1' in line:
                    acq_eulers[0] = float(line.split()[-1])
                elif 'AcqE2' in line:
                    acq_eulers[1] = float(line.split()[-1])
                elif 'AcqE3' in line:
                    acq_eulers[2] = float(line.split()[-1])
                elif 'Phases' in line:
                    num_phases = int(line.split()[-1])
                    self.loaded_data['phase', 'plot_params']['vmax'] = num_phases
                    for j in range(num_phases):
                        line = next(ctf_file)
                        self.loaded_metadata['phases'].append(parse_phase())
                    # phases are last in the header, so read the column
                    # headings then break out the loop
                    header_text = next(ctf_file)
                    num_header_lines = i + j + 3
                    break

        shape = (y_dim, x_dim)
        self.loaded_metadata['shape'] = shape
        self.loaded_metadata['acquisition_rotation'] = Quat.from_euler_angles(
            *(np.array(acq_eulers) * np.pi / 180)
        )

        self.check_metadata()

        # Construct data format from table header
        field_lookup = {
            'Phase': ('phase', 'uint8'),
            'X': ('x', 'float32'),
            'Y': ('y', 'float32'),
            'Bands': ('numBands', 'uint8'),
            'Error': ('error', 'uint8'),
            'Euler1': ('ph1', 'float32'),
            'Euler2': ('phi', 'float32'),
            'Euler3': ('ph2', 'float32'),
            'MAD': ('MAD', 'float32'),  # Mean Angular Deviation
            'BC': ('BC', 'uint8'),      # Band Contrast
            'BS': ('BS', 'uint8'),      # Band Slope
        }

        keep_col_names = ('phase', 'ph1', 'phi', 'ph2', 'BC', 'BS', 'MAD')
        data_format = []
        load_cols = []
        try:
            for i, col_title in enumerate(header_text.split()):
                if field_lookup[col_title][0] in keep_col_names:
                    data_format.append(field_lookup[col_title])
                    load_cols.append(i)
        except KeyError:
            raise TypeError("Unknown data in EBSD file.")
        self.data_format = np.dtype(data_format)

        # now read the data from file
        data = np.loadtxt(
            str(file_path), dtype=self.data_format, usecols=load_cols,
            skiprows=num_header_lines
        )

        self.loaded_data.add(
            'band_contrast', data['BC'].reshape(shape),
            unit='', type='map', order=0,
            plot_params={
                'plot_colour_bar': True,
                'cmap': 'gray',
                'clabel': 'Band contrast',
            }

        )
        self.loaded_data.add(
            'band_slope', data['BS'].reshape(shape),
            unit='', type='map', order=0,
            plot_params={
                'plot_colour_bar': True,
                'cmap': 'gray',
                'clabel': 'Band slope',
            }
        )
        self.loaded_data.add(
            'mean_angular_deviation', data['MAD'].reshape(shape),
            unit='', type='map', order=0,
            plot_params={
                'plot_colour_bar': True,
                'clabel': 'Mean angular deviation',
            }
        )
        self.loaded_data.phase = data['phase'].reshape(shape)

        euler_angle = structured_to_unstructured(
            data[['ph1', 'phi', 'ph2']].reshape(shape)).transpose((2, 0, 1))
        euler_angle *= np.pi / 180
        self.loaded_data.euler_angle = euler_angle

        self.check_data()


class EdaxAngLoader(EBSDDataLoader):
    def load(
        self,
        file_name: str,
        file_dir: str = ""
    ) -> None:
        """ Read an EDAX .ang file.

        Parameters
        ----------
        file_name
            File name.
        file_dir
            Path to file.

        """
        # open data file and read in metadata
        file_name = f'{file_name}.ang'
        file_path = pathlib.Path(file_dir) / pathlib.Path(file_name)
        if not file_path.is_file():
            raise FileNotFoundError("Cannot open file {}".format(file_path))

        i_phase = 1
        # parse header lines (starting with #)
        with open(str(file_path), 'r') as ang_file:
            while True:
                line = ang_file.readline()

                if not line.startswith('#'):
                    # end of header
                    break
                # remove #
                line = line[1:].strip()

                if line.startswith('Phase'):
                    if int(line.split()[1]) != i_phase:
                        raise ValueError('Phases not sequential in file?')

                    phase_lines = read_until_string(
                        ang_file, '#', exact=True,
                        line_process=lambda l: l[1:].strip()
                    )
                    self.loaded_metadata['phases'].append(
                        EdaxAngLoader.parse_phase(phase_lines)
                    )
                    i_phase += 1

                elif line.startswith('GRID'):
                    if line.split()[-1] != 'SqrGrid':
                        raise ValueError('Only square grids supported')
                elif line.startswith('XSTEP'):
                    self.loaded_metadata['step_size'] = float(line.split()[-1])
                elif line.startswith('NCOLS_ODD'):
                    xdim = int(line.split()[-1])
                elif line.startswith('NROWS'):
                    ydim = int(line.split()[-1])

        shape = (ydim, xdim)
        self.loaded_metadata['shape'] = shape

        self.check_metadata()

        # Construct fixed data format
        self.data_format = np.dtype([
            ('ph1', 'float32'),
            ('phi', 'float32'),
            ('ph2', 'float32'),
            # ('x', 'float32'),
            # ('y', 'float32'),
            ('IQ', 'float32'),
            ('CI', 'float32'),
            ('phase', 'uint8'),
            # ('SE_signal', 'float32'),
            ('FF', 'float32'),
        ])
        load_cols = (0, 1, 2, 5, 6, 7, 8, 9)

        # now read the data from file
        data = np.loadtxt(
            str(file_path), dtype=self.data_format, comments='#',
            usecols=load_cols
        )

        self.loaded_data.add(
            'image_quality', data['IQ'].reshape(shape),
            unit='', type='map', order=0,
            plot_params={
                'plot_colour_bar': True,
                'clabel': 'Image quality',
            }
        )
        self.loaded_data.add(
            'confidence_index', data['CI'].reshape(shape),
            unit='', type='map', order=0,
            plot_params={
                'plot_colour_bar': True,
                'clabel': 'Confidence index',
            }
        )
        self.loaded_data.add(
            'fit_factor', data['FF'].reshape(shape),
            unit='', type='map', order=0,
            plot_params={
                'plot_colour_bar': True,
                'clabel': 'Fit factor',
            }
        )
        self.loaded_data.phase = data['phase'].reshape(shape) + 1
        self.loaded_data['phase', 'plot_params']['vmax'] = len(self.loaded_metadata['phases'])

        # flatten the structured dtype
        euler_angle = structured_to_unstructured(
            data[['ph1', 'phi', 'ph2']].reshape(shape)).transpose((2, 0, 1))
        euler_angle[0] -= np.pi / 2
        euler_angle[0, euler_angle[0] < 0.] += 2 * np.pi
        self.loaded_data.euler_angle = euler_angle

        self.check_data()

    @staticmethod
    def parse_phase(lines) -> Phase:
        for line in lines:
            line = line.split()

            if line[0] == 'MaterialName':
                name = line[1]
            if line[0] == 'Symmetry':
                point_group = line[1]
                if point_group in ('43', 'm3m'):
                    # cubic high
                    laue_group = 11
                    # can't determine but set to BCC for now
                    space_group = 229
                elif point_group == '6/mmm':
                    # hex high
                    laue_group = 9
                    space_group = None
                else:
                    raise ValueError(f'Unknown crystal symmetry {point_group}')
            elif line[0] == 'LatticeConstants':
                dims = line[1:4]
                dims = tuple(round(float(s), 3) for s in dims)
                angles = line[4:7]
                angles = tuple(round(float(s), 3) * np.pi / 180
                               for s in angles)
                lattice_params = dims + angles

        return Phase(name, laue_group, space_group, lattice_params)


class OxfordBinaryLoader(EBSDDataLoader):
    def load(
        self,
        fileName: str,
        file_dir: str = ""
    ) -> None:
        """Read Oxford Instruments .cpr/.crc file pair.

        Parameters
        ----------
        fileName
            File name.
        file_dir
            Path to file.

        """
        self.load_oxford_cpr(fileName, file_dir=file_dir)
        self.load_oxford_crc(fileName, file_dir=file_dir)

    def load_oxford_cpr(self, file_name: str, file_dir: str = "") -> None:
        """
        Read an Oxford Instruments .cpr file, which is a metadata file
        describing EBSD data.

        Parameters
        ----------
        file_name
            File name.
        file_dir
            Path to file.

        """
        comment_char = ';'

        file_name = "{}.cpr".format(file_name)
        file_path = pathlib.Path(file_dir) / pathlib.Path(file_name)
        if not file_path.is_file():
            raise FileNotFoundError("Cannot open file {}".format(file_path))

        # CPR file is split into groups, load each group into a
        # hierarchical dict

        metadata = dict()
        group_pat = re.compile(r"\[(.+)\]")

        def parse_line(line: str, group_dict: Dict) -> None:
            try:
                key, val = line.strip().split('=')
                group_dict[key] = val
            except ValueError:
                pass

        with open(str(file_path), 'r') as cpr_file:
            while True:
                line = cpr_file.readline()
                if not line:
                    # End of file
                    break
                if line.strip() == '' or line.strip()[0] == comment_char:
                    # Skip comment or empty line
                    continue

                group_name = group_pat.match(line.strip()).group(1)
                group_dict = dict()
                read_until_string(cpr_file, '[', comment_char=comment_char,
                                  line_process=lambda l: parse_line(l, group_dict))
                metadata[group_name] = group_dict

        # Create phase objects and move metadata to object metadata dict

        x_dim = int(metadata['Job']['xCells'])
        y_dim = int(metadata['Job']['yCells'])
        self.loaded_metadata['shape'] = (y_dim, x_dim)
        self.loaded_metadata['step_size'] = float(metadata['Job']['GridDistX'])
        self.loaded_metadata['acquisition_rotation'] = Quat.from_euler_angles(
            float(metadata['Acquisition Surface']['Euler1']) * np.pi / 180.,
            float(metadata['Acquisition Surface']['Euler2']) * np.pi / 180.,
            float(metadata['Acquisition Surface']['Euler3']) * np.pi / 180.
        )
        num_phases = int(metadata['Phases']['Count'])

        for i in range(num_phases):
            phase_metadata = metadata['Phase{:}'.format(i + 1)]
            self.loaded_metadata['phases'].append(Phase(
                phase_metadata['StructureName'],
                int(phase_metadata['LaueGroup']),
                int(phase_metadata.get('SpaceGroup', 0)),
                (
                    round(float(phase_metadata['a']), 3),
                    round(float(phase_metadata['b']), 3),
                    round(float(phase_metadata['c']), 3),
                    round(float(phase_metadata['alpha']), 3) * np.pi / 180,
                    round(float(phase_metadata['beta']), 3) * np.pi / 180,
                    round(float(phase_metadata['gamma']), 3) * np.pi / 180
                )
            ))
        self.loaded_data['phase', 'plot_params']['vmax'] = num_phases

        # Deal with EDX data
        edx_fields = {}
        if 'EDX Windows' in metadata:
            self.loaded_metadata['edx'] = metadata['EDX Windows']
            count = int(self.loaded_metadata['edx']['Count'])
            self.loaded_metadata['edx']['Count'] = count
            for i in range(1, count + 1):
                name = self.loaded_metadata['edx'][f"Window{i}"]
                edx_fields[100+i] = (f'EDX {name}', 'float32')

        self.check_metadata()

        # Construct binary data format from listed fields
        unknown_field_count = 0
        data_format = [('phase', 'uint8')]
        field_lookup = {
            3: ('ph1', 'float32'),
            4: ('phi', 'float32'),
            5: ('ph2', 'float32'),
            6: ('MAD', 'float32'),  # Mean Angular Deviation
            7: ('BC', 'uint8'),  # Band Contrast
            8: ('BS', 'uint8'),  # Band Slope
            10: ('numBands', 'uint8'),
            11: ('AFI', 'uint8'),  # Advanced Fit index. legacy
            12: ('IB6', 'float32')  # ?
        }
        field_lookup.update(edx_fields)
        try:
            for i in range(int(metadata['Fields']['Count'])):
                field_id = int(metadata['Fields']['Field{:}'.format(i + 1)])
                data_format.append(field_lookup[field_id])
        except KeyError:
            print(f'\nUnknown field in file with key {field_id}. '
                  f'Assuming float32 data.')
            unknown_field_count += 1
            data_format.append((f'unknown_{unknown_field_count}', 'float32'))

        self.data_format = np.dtype(data_format)

    def load_oxford_crc(self, file_name: str, file_dir: str = "") -> None:
        """Read binary EBSD data from an Oxford Instruments .crc file

        Parameters
        ----------
        file_name
            File name.
        file_dir
            Path to file.

        """
        shape = self.loaded_metadata['shape']

        file_name = "{}.crc".format(file_name)
        file_path = pathlib.Path(file_dir) / pathlib.Path(file_name)
        if not file_path.is_file():
            raise FileNotFoundError("Cannot open file {}".format(file_path))

        # load binary data from file
        data = np.fromfile(str(file_path), self.data_format, count=-1)

        self.loaded_data.add(
            'band_contrast', data['BC'].reshape(shape),
            unit='', type='map', order=0,
            plot_params={
                'plot_colour_bar': True,
                'cmap': 'gray',
                'clabel': 'Band contrast',
            }
        )
        self.loaded_data.add(
            'band_slope', data['BS'].reshape(shape),
            unit='', type='map', order=0,
            plot_params={
                'plot_colour_bar': True,
                'cmap': 'gray',
                'clabel': 'Band slope',
            }
        )
        self.loaded_data.add(
            'mean_angular_deviation',
            data['MAD'].reshape(shape),
            unit='', type='map', order=0,
            plot_params={
                'plot_colour_bar': True,
                'clabel': 'Mean angular deviation',
            }
        )
        self.loaded_data.phase = data['phase'].reshape(shape)

        # flatten the structured dtype
        self.loaded_data.euler_angle = structured_to_unstructured(
            data[['ph1', 'phi', 'ph2']].reshape(shape)).transpose((2, 0, 1))

        if self.loaded_metadata['edx']['Count'] > 0:
            EDXFields = [key for key in data.dtype.fields.keys() if key.startswith('EDX')]
            for field in EDXFields:
                self.loaded_data.add(
                    field,
                    data[field].reshape(shape),
                    unit='counts', type='map', order=0,
                    plot_params={
                        'plot_colour_bar': True,
                        'clabel': field + ' counts',
                    }
            )

        self.check_data()


class PythonDictLoader(EBSDDataLoader):
    def load(self, data_dict: Dict[str, Any]) -> None:
        """Construct EBSD data from a python dictionary.

        Parameters
        ----------
        data_dict
            Dictionary with keys:
                'step_size'
                'phases'
                'phase'
                'euler_angle'
                'band_contrast'

        """
        self.loaded_metadata['shape'] = data_dict['phase'].shape
        self.loaded_metadata['step_size'] = data_dict['step_size']
        assert type(data_dict['phases']) is list
        self.loaded_metadata['phases'] = data_dict['phases']
        self.check_metadata()

        self.loaded_data.add(
            'band_contrast', data_dict['band_contrast'],
            unit='', type='map', order=0
        )
        self.loaded_data.phase = data_dict['phase']
        self.loaded_data['phase', 'plot_params']['vmax'] = len(self.loaded_metadata['phases'])
        self.loaded_data.euler_angle = data_dict['euler_angle']
        self.check_data()


class DICDataLoader(object):
    """Class containing methods for loading and checking HRDIC data

    """
    def __init__(self) -> None:
        self.loaded_metadata = {
            'format': '',
            'version': '',
            'binning': '',
            'shape': (0, 0),
        }
        # required data
        self.loaded_data = Datastore()
        self.loaded_data.add(
            'coordinate', None, unit='px', type='map', order=1,
            default_component='magnitude',
            plot_params={
                'plot_colour_bar': True,
                'clabel': 'Coordinate',
            }
        )
        self.loaded_data.add(
            'displacement', None, unit='px', type='map', order=1,
            default_component='magnitude',
            plot_params={
                'plot_colour_bar': True,
                'clabel': 'Displacement',
            }
        )

    @staticmethod
    def get_loader(data_type: str) -> 'Type[DICDataLoader]':
        if data_type is None:
            data_type = "Davis"

        data_type = data_type.lower()
        if data_type == "davis":
            return DavisLoader()
        elif data_type == "openpiv":
            return OpenPivLoader()
        else:
            raise ValueError(f"No loader for EBSD data of type {data_type}.")

    def checkMetadata(self) -> None:
        return

    def check_data(self) -> None:
        """ Calculate size of map from loaded data and check it matches
        values from metadata.

        """
        # check binning
        binning = self.loaded_metadata['binning']
        binning_x = min(abs(np.diff(self.loaded_data.coordinate[0].flat)))
        binning_y = max(abs(np.diff(self.loaded_data.coordinate[1].flat)))
        if not (binning_x == binning_y == binning):
            raise ValueError(
                f'Binning of data and header do not match `{binning_x}`, '
                f'`{binning_y}`, `{binning}`'
            )

        # check shape
        coord = self.loaded_data.coordinate
        shape = (coord.max(axis=(1, 2)) - coord.min(axis=(1, 2))) / binning + 1
        shape = tuple(shape[::-1].astype(int))
        if shape != self.loaded_metadata['shape']:
            raise ValueError(
                f'Dimensions of data and header do not match `{shape}, '
                f'`{self.loaded_metadata["shape"]}`'
            )


class DavisLoader(DICDataLoader):
    def load(
        self,
        file_name: str,
        file_dir: str = ""
    ) -> None:
        """ Load from Davis .txt file.

        Parameters
        ----------
        file_name
            File name.
        file_dir
            Path to file.

        """
        file_path = pathlib.Path(file_dir) / pathlib.Path(file_name)
        if not file_path.is_file():
            raise FileNotFoundError("Cannot open file {}".format(file_path))

        with open(str(file_path), 'r') as f:
            header = f.readline()
        metadata = header.split()

        # Software name and version
        self.loaded_metadata['format'] = metadata[0].strip('#')
        self.loaded_metadata['version'] = metadata[1]
        # Sub-window width in pixels
        self.loaded_metadata['binning'] = int(metadata[3])
        # shape of map (from header)
        self.loaded_metadata['shape'] = (int(metadata[4]), int(metadata[5]))

        self.checkMetadata()

        data = pd.read_table(str(file_path), delimiter='\t', skiprows=1,
                             header=None).values
        data = data.reshape(self.loaded_metadata['shape'] + (-1,))
        data = data.transpose((2, 0, 1))

        self.loaded_data.coordinate = data[:2]
        self.loaded_data.displacement = data[2:]

        self.check_data()

    @staticmethod
    def loadDavisImageData(file_name: str, file_dir: str = "") -> np.ndarray:
        """ A .txt file from DaVis containing a 2D image

        Parameters
        ----------
        file_name
            File name.
        file_dir
            Path to file.

        Returns
        -------
        np.ndarray
            Array of data.

        """
        file_path = pathlib.Path(file_dir) / pathlib.Path(file_name)
        if not file_path.is_file():
            raise FileNotFoundError("Cannot open file {}".format(file_path))

        data = pd.read_table(str(file_path), delimiter='\t', skiprows=1,
                             header=None)
       
        # x and y coordinates
        loaded_data = np.array(data)

        return loaded_data


class OpenPivLoader(DICDataLoader):
    def load(
            self,
            file_name: str,
            file_dir: str = ""
    ) -> None:
        """ Load from Open PIV .txt file.

        Parameters
        ----------
        file_name
            File name.
        file_dir
            Path to file.

        """
        file_path = pathlib.Path(file_dir) / pathlib.Path(file_name)
        if not file_path.is_file():
            raise FileNotFoundError(f"Cannot open file {file_path}")

        with open(str(file_path), 'r') as f:
            header = f.readline()[1:].split()
            data = np.loadtxt(f)
        col = {
            'x': 0,
            'y': 1,
            'u': 2,
            'v': 3,
        }

        # Software name and version
        self.loaded_metadata['format'] = 'OpenPIV'
        self.loaded_metadata['version'] = 'n/a'

        # Sub-window width in pixels
        binning_x = int(np.min(np.abs(np.diff(data[:, col['x']]))))
        binning_y = int(np.max(np.abs(np.diff(data[:, col['y']]))))
        assert binning_x == binning_y
        binning = binning_x
        self.loaded_metadata['binning'] = binning

        # shape of map (from header)
        shape = data[:, [col['y'], col['x']]].max(axis=0) + binning / 2
        assert np.allclose(shape % binning, 0.)
        shape = tuple((shape / binning).astype(int).tolist())
        self.loaded_metadata['shape'] = shape

        self.checkMetadata()

        data = data.reshape(shape + (-1,))[::-1].transpose((2, 0, 1))

        self.loaded_data.coordinate = data[[col['x'], col['y']]]
        self.loaded_data.displacement = data[[col['u'], col['v']]]

        self.check_data()


def read_until_string(
    file: TextIO,
    term_string: str,
    comment_char: str = '*',
    line_process: Optional[Callable[[str], Any]] = None,
    exact: bool = False
) -> List[Any]:
    """Read lines in a file until a line starting with the `termString`
    is encountered. The file position is returned before the line starting
    with the `termString` when found. Comment and empty lines are ignored.

    Parameters
    ----------
    file
        An open python text file object.
    term_string
        String to terminate reading.
    comment_char
        Character at start of a comment line to ignore.
    line_process
        Function to apply to each line when loaded.
    exact
        A line must exactly match `termString` to stop.

    Returns
    -------
    list
        List of processed lines loaded from file.

    """
    lines = []
    while True:
        curr_pos = file.tell()  # save position in file
        line = file.readline()
        if (not line
                or (exact and line.strip() == term_string)
                or (not exact and line.strip().startswith(term_string))):
            file.seek(curr_pos)  # return to before prev line
            break
        if line.strip() == '' or line.strip()[0] == comment_char:
            # Skip comment or empty line
            continue
        if line_process is not None:
            line = line_process(line)
        lines.append(line)
    return lines
