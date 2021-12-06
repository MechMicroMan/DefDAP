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
        self.loadedMetadata = {
            'shape': (0, 0),
            'step_size': 0.,
            'acquisition_rotation': Quat(1.0, 0.0, 0.0, 0.0),
            'phases': [],
            'edx': {'Count': 0},
        }
        # required data
        self.loadedData = Datastore()
        self.loadedData.add(
            'phase', None, unit='', type='map', order=0,
            comment='1-based, 0 is non-indexed points',
            plot_params={
                'plotColourBar': True,
                'clabel': 'Phase',
            }
        )
        self.loadedData.add(
            'euler_angle', None, unit='rad', type='map', order=1,
            plot_params={
                'plotColourBar': True,
                'clabel': 'Euler angle',
            }
        )
        self.dataFormat = None

    @staticmethod
    def getLoader(dataType: str) -> 'Type[EBSDDataLoader]':
        if dataType is None:
            dataType = "OxfordBinary"

        if dataType == "OxfordBinary":
            return OxfordBinaryLoader()
        elif dataType == "OxfordText":
            return OxfordTextLoader()
        elif dataType == "EdaxAng":
            return EdaxAngLoader()
        elif dataType == "PythonDict":
            return PythonDictLoader()
        else:
            raise ValueError(f"No loader for EBSD data of type {dataType}.")

    def checkMetadata(self) -> None:
        """
        Checks that the number of phases from metadata matches
        the amount of phases loaded.

        """
        for phase in self.loadedMetadata['phases']:
            assert type(phase) is Phase

    def checkData(self) -> None:
        shape = self.loadedMetadata['shape']

        assert self.loadedData.phase.shape == shape
        assert self.loadedData.euler_angle.shape == (3,) + shape
        # assert self.loadedData['bandContrast'].shape == mapShape


class OxfordTextLoader(EBSDDataLoader):
    def load(
        self,
        fileName: str,
        fileDir: str = ""
    ) -> None:
        """ Read an Oxford Instruments .ctf file, which is a HKL single
        orientation file.

        Parameters
        ----------
        fileName
            File name.
        fileDir
            Path to file.

        """
        # open data file and read in metadata
        fileName = "{}.ctf".format(fileName)
        filePath = pathlib.Path(fileDir) / pathlib.Path(fileName)
        if not filePath.is_file():
            raise FileNotFoundError("Cannot open file {}".format(filePath))

        def parsePhase() -> Phase:
            lineSplit = line.split('\t')
            dims = lineSplit[0].split(';')
            dims = tuple(round(float(s), 3) for s in dims)
            angles = lineSplit[1].split(';')
            angles = tuple(round(float(s), 3) * np.pi / 180 for s in angles)
            latticeParams = dims + angles
            phase = Phase(
                lineSplit[2],
                int(lineSplit[3]),
                int(lineSplit[4]),
                latticeParams
            )
            return phase

        # default values for acquisition rotation in case missing in in file
        acqEulers = [0., 0., 0.]
        with open(str(filePath), 'r') as ctfFile:
            for i, line in enumerate(ctfFile):
                if 'XCells' in line:
                    xDim = int(line.split()[-1])
                elif 'YCells' in line:
                    yDim = int(line.split()[-1])
                elif 'XStep' in line:
                    self.loadedMetadata['step_size'] = float(line.split()[-1])
                elif 'AcqE1' in line:
                    acqEulers[0] = float(line.split()[-1])
                elif 'AcqE2' in line:
                    acqEulers[1] = float(line.split()[-1])
                elif 'AcqE3' in line:
                    acqEulers[2] = float(line.split()[-1])
                elif 'Phases' in line:
                    numPhases = int(line.split()[-1])
                    for j in range(numPhases):
                        line = next(ctfFile)
                        self.loadedMetadata['phases'].append(parsePhase())
                    # phases are last in the header, so read the column
                    # headings then break out the loop
                    headerText = next(ctfFile)
                    numHeaderLines = i + j + 3
                    break

        shape = (yDim, xDim)
        self.loadedMetadata['shape'] = shape
        self.loadedMetadata['acquisition_rotation'] = Quat.fromEulerAngles(
            *(np.array(acqEulers) * np.pi / 180)
        )

        self.checkMetadata()

        # Construct data format from table header
        fieldLookup = {
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

        keepColNames = ('phase', 'ph1', 'phi', 'ph2', 'BC', 'BS', 'MAD')
        dataFormat = []
        loadCols = []
        try:
            for i, colTitle in enumerate(headerText.split()):
                if fieldLookup[colTitle][0] in keepColNames:
                    dataFormat.append(fieldLookup[colTitle])
                    loadCols.append(i)
        except KeyError:
            raise TypeError("Unknown data in EBSD file.")
        self.dataFormat = np.dtype(dataFormat)

        # now read the data from file
        data = np.loadtxt(
            str(filePath), dtype=self.dataFormat, usecols=loadCols,
            skiprows=numHeaderLines
        )

        self.loadedData.add(
            'band_contrast', data['BC'].reshape(shape),
            unit='', type='map', order=0,
            plot_params={
                'plotColourBar': True,
                'cmap': 'gray',
                'clabel': 'Band contrast',
            }

        )
        self.loadedData.add(
            'band_slope', data['BS'].reshape(shape),
            unit='', type='map', order=0,
            plot_params={
                'plotColourBar': True,
                'cmap': 'gray',
                'clabel': 'Band slope',
            }
        )
        self.loadedData.add(
            'mean_angular_deviation', data['MAD'].reshape(shape),
            unit='', type='map', order=0,
            plot_params={
                'plotColourBar': True,
                'clabel': 'Mean angular deviation',
            }
        )
        self.loadedData.phase = data['phase'].reshape(shape)

        euler_angle = structured_to_unstructured(
            data[['ph1', 'phi', 'ph2']].reshape(shape)).transpose((2, 0, 1))
        euler_angle *= np.pi / 180
        self.loadedData.euler_angle = euler_angle

        self.checkData()


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

                    phase_lines = readUntilString(
                        ang_file, '#', exact=True,
                        lineProcess=lambda l: l[1:].strip()
                    )
                    self.loadedMetadata['phases'].append(
                        EdaxAngLoader.parse_phase(phase_lines)
                    )
                    i_phase += 1

                elif line.startswith('GRID'):
                    if line.split()[-1] != 'SqrGrid':
                        raise ValueError('Only square grids supported')
                elif line.startswith('XSTEP'):
                    self.loadedMetadata['step_size'] = float(line.split()[-1])
                elif line.startswith('NCOLS_ODD'):
                    xdim = int(line.split()[-1])
                elif line.startswith('NROWS'):
                    ydim = int(line.split()[-1])

        shape = (ydim, xdim)
        self.loadedMetadata['shape'] = shape

        self.checkMetadata()

        # Construct fixed data format
        self.dataFormat = np.dtype([
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
            str(file_path), dtype=self.dataFormat, comments='#',
            usecols=load_cols
        )

        self.loadedData.add(
            'image_quality', data['IQ'].reshape(shape),
            unit='', type='map', order=0,
            plot_params={
                'plotColourBar': True,
                'clabel': 'Image quality',
            }
        )
        self.loadedData.add(
            'confidence_index', data['CI'].reshape(shape),
            unit='', type='map', order=0,
            plot_params={
                'plotColourBar': True,
                'clabel': 'Confidence index',
            }
        )
        self.loadedData.add(
            'fit_factor', data['FF'].reshape(shape),
            unit='', type='map', order=0,
            plot_params={
                'plotColourBar': True,
                'clabel': 'Fit factor',
            }
        )
        self.loadedData.phase = data['phase'].reshape(shape) + 1

        # flatten the structured dtype
        euler_angle = structured_to_unstructured(
            data[['ph1', 'phi', 'ph2']].reshape(shape)).transpose((2, 0, 1))
        euler_angle[0] -= np.pi / 2
        euler_angle[0, euler_angle[0] < 0.] += 2 * np.pi
        self.loadedData.euler_angle = euler_angle

        self.checkData()

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
                    laue_group = 11
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
        fileDir: str = ""
    ) -> None:
        """Read Oxford Instruments .cpr/.crc file pair.

        Parameters
        ----------
        fileName
            File name.
        fileDir
            Path to file.

        """
        self.loadOxfordCPR(fileName, fileDir=fileDir)
        self.loadOxfordCRC(fileName, fileDir=fileDir)

    def loadOxfordCPR(self, fileName: str, fileDir: str = "") -> None:
        """
        Read an Oxford Instruments .cpr file, which is a metadata file
        describing EBSD data.

        Parameters
        ----------
        fileName
            File name.
        fileDir
            Path to file.

        """
        commentChar = ';'

        fileName = "{}.cpr".format(fileName)
        filePath = pathlib.Path(fileDir) / pathlib.Path(fileName)
        if not filePath.is_file():
            raise FileNotFoundError("Cannot open file {}".format(filePath))

        # CPR file is split into groups, load each group into a
        # hierarchical dict

        metadata = dict()
        groupPat = re.compile("\[(.+)\]")

        def parseLine(line: str, groupDict: Dict) -> None:
            try:
                key, val = line.strip().split('=')
                groupDict[key] = val
            except ValueError:
                pass

        with open(str(filePath), 'r') as cprFile:
            while True:
                line = cprFile.readline()
                if not line:
                    # End of file
                    break
                if line.strip() == '' or line.strip()[0] == commentChar:
                    # Skip comment or empty line
                    continue

                groupName = groupPat.match(line.strip()).group(1)
                groupDict = dict()
                readUntilString(cprFile, '[', commentChar=commentChar,
                                lineProcess=lambda l: parseLine(l, groupDict))
                metadata[groupName] = groupDict

        # Create phase objects and move metadata to object metadata dict

        xDim = int(metadata['Job']['xCells'])
        yDim = int(metadata['Job']['yCells'])
        self.loadedMetadata['shape'] = (yDim, xDim)
        self.loadedMetadata['step_size'] = float(metadata['Job']['GridDistX'])
        self.loadedMetadata['acquisition_rotation'] = Quat.fromEulerAngles(
            float(metadata['Acquisition Surface']['Euler1']) * np.pi / 180.,
            float(metadata['Acquisition Surface']['Euler2']) * np.pi / 180.,
            float(metadata['Acquisition Surface']['Euler3']) * np.pi / 180.
        )
        numPhases = int(metadata['Phases']['Count'])

        for i in range(numPhases):
            phaseMetadata = metadata['Phase{:}'.format(i + 1)]
            self.loadedMetadata['phases'].append(Phase(
                phaseMetadata['StructureName'],
                int(phaseMetadata['LaueGroup']),
                int(phaseMetadata['SpaceGroup']),
                (
                    round(float(phaseMetadata['a']), 3),
                    round(float(phaseMetadata['b']), 3),
                    round(float(phaseMetadata['c']), 3),
                    round(float(phaseMetadata['alpha']), 3) * np.pi / 180,
                    round(float(phaseMetadata['beta']), 3) * np.pi / 180,
                    round(float(phaseMetadata['gamma']), 3) * np.pi / 180
                )
            ))

        # Deal with EDX data
        edx_fields = {}
        if 'EDX Windows' in metadata:
            self.loadedMetadata['edx'] = metadata['EDX Windows']
            count = int(self.loadedMetadata['edx']['Count'])
            self.loadedMetadata['edx']['Count'] = count
            for i in range(1, count + 1):
                name = self.loadedMetadata['edx'][f"Window{i}"]
                edx_fields[100+i] = (f'EDX {name}', 'float32')

        self.checkMetadata()

        # Construct binary data format from listed fields
        dataFormat = [('phase', 'uint8')]
        fieldLookup = {
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
        fieldLookup.update(edx_fields)
        try:
            for i in range(int(metadata['Fields']['Count'])):
                fieldID = int(metadata['Fields']['Field{:}'.format(i + 1)])
                dataFormat.append(fieldLookup[fieldID])
        except KeyError:
            raise TypeError("Unknown data in EBSD file.")

        self.dataFormat = np.dtype(dataFormat)

    def loadOxfordCRC(self, fileName: str, fileDir: str = "") -> None:
        """Read binary EBSD data from an Oxford Instruments .crc file

        Parameters
        ----------
        fileName
            File name.
        fileDir
            Path to file.

        """
        shape = self.loadedMetadata['shape']

        fileName = "{}.crc".format(fileName)
        filePath = pathlib.Path(fileDir) / pathlib.Path(fileName)
        if not filePath.is_file():
            raise FileNotFoundError("Cannot open file {}".format(filePath))

        # load binary data from file
        data = np.fromfile(str(filePath), self.dataFormat, count=-1)

        self.loadedData.add(
            'band_contrast', data['BC'].reshape(shape),
            unit='', type='map', order=0,
            plot_params={
                'plotColourBar': True,
                'cmap': 'gray',
                'clabel': 'Band contrast',
            }
        )
        self.loadedData.add(
            'band_slope', data['BS'].reshape(shape),
            unit='', type='map', order=0,
            plot_params={
                'plotColourBar': True,
                'cmap': 'gray',
                'clabel': 'Band slope',
            }
        )
        self.loadedData.add(
            'mean_angular_deviation',
            data['MAD'].reshape(shape),
            unit='', type='map', order=0,
            plot_params={
                'plotColourBar': True,
                'clabel': 'Mean angular deviation',
            }
        )
        self.loadedData.phase = data['phase'].reshape(shape)

        # flatten the structured dtype
        self.loadedData.euler_angle = structured_to_unstructured(
            data[['ph1', 'phi', 'ph2']].reshape(shape)).transpose((2, 0, 1))

        if self.loadedMetadata['edx']['Count'] > 0:
            EDXFields = [key for key in data.dtype.fields.keys() if key.startswith('EDX')]
            for field in EDXFields:
                self.loadedData.add(
                    field,
                    data[field].reshape(shape),
                    unit='counts', type='map', order=0,
                    plot_params={
                        'plotColourBar': True,
                        'clabel': field + ' counts',
                    }
            )

        self.checkData()


class PythonDictLoader(EBSDDataLoader):
    def load(self, dataDict: Dict[str, Any]) -> None:
        """Construct EBSD data from a python dictionary.

        Parameters
        ----------
        dataDict
            Dictionary with keys:
                'step_size'
                'phases'
                'phase'
                'euler_angle'
                'band_contrast'

        """
        self.loadedMetadata['shape'] = dataDict['phase'].shape
        self.loadedMetadata['step_size'] = dataDict['step_size']
        assert type(dataDict['phases']) is list
        self.loadedMetadata['phases'] = dataDict['phases']
        self.checkMetadata()

        self.loadedData.add(
            'band_contrast', dataDict['band_contrast'],
            unit='', type='map', order=0
        )
        self.loadedData.phase = dataDict['phase']
        self.loadedData.euler_angle = dataDict['euler_angle']
        self.checkData()


class DICDataLoader(object):
    """Class containing methods for loading and checking HRDIC data

    """
    def __init__(self) -> None:
        self.loadedMetadata = {
            'format': '',
            'version': '',
            'binning': '',
            'shape': (0, 0),
        }
        # required data
        self.loadedData = Datastore()
        self.loadedData.add(
            'coordinate', None, unit='px', type='map', order=1,
            plot_params={
                'plotColourBar': True,
                'clabel': 'Coordinate',
            }
        )
        self.loadedData.add(
            'displacement', None, unit='px', type='map', order=1,
            plot_params={
                'plotColourBar': True,
                'clabel': 'Displacement',
            }
        )

    def checkMetadata(self) -> None:
        return

    def checkData(self) -> None:
        """ Calculate size of map from loaded data and check it matches
        values from metadata.

        """
        # check binning
        binning = self.loadedMetadata['binning']
        binning_x = min(abs(np.diff(self.loadedData.coordinate[0].flat)))
        binning_y = max(abs(np.diff(self.loadedData.coordinate[1].flat)))
        if not (binning_x == binning_y == binning):
            raise ValueError('Binning of data and header do not match')

        # check shape
        coord = self.loadedData.coordinate
        shape = (coord.max(axis=(1, 2)) - coord.min(axis=(1, 2))) / binning + 1
        shape = tuple(shape[::-1].astype(int))
        if shape != self.loadedMetadata['shape']:
            raise ValueError('Dimensions of data and header do not match')

    def loadDavisMetadata(
        self,
        fileName: str,
        fileDir: str = ""
    ) -> Dict[str, Any]:
        """ Load DaVis metadata from Davis .txt file.

        Parameters
        ----------
        fileName
            File name.
        fileDir
            Path to file.

        Returns
        -------
        dict
            Davis metadata.

        """
        filePath = pathlib.Path(fileDir) / pathlib.Path(fileName)
        if not filePath.is_file():
            raise FileNotFoundError("Cannot open file {}".format(filePath))

        with open(str(filePath), 'r') as f:
            header = f.readline()
        metadata = header.split()

        # Software name and version
        self.loadedMetadata['format'] = metadata[0].strip('#')
        self.loadedMetadata['version'] = metadata[1]
        # Sub-window width in pixels
        self.loadedMetadata['binning'] = int(metadata[3])
        # shape of map (from header)
        self.loadedMetadata['shape'] = (int(metadata[4]), int(metadata[5]))

        self.checkMetadata()

        return self.loadedMetadata

    def loadDavisData(
        self,
        fileName: str,
        fileDir: str = ""
    ) -> Dict[str, Any]:
        """Load displacement data from Davis .txt file containing x and
        y coordinates and x and y displacements for each coordinate.

        Parameters
        ----------
        fileName
            File name.
        fileDir
            Path to file.

        Returns
        -------
        dict
            Coordinates and displacements.

        """
        filePath = pathlib.Path(fileDir) / pathlib.Path(fileName)
        if not filePath.is_file():
            raise FileNotFoundError("Cannot open file {}".format(filePath))

        data = pd.read_table(str(filePath), delimiter='\t', skiprows=1,
                             header=None).values
        data = data.reshape(self.loadedMetadata['shape'] + (-1, ))
        data = data.transpose((2, 0, 1))

        self.loadedData.coordinate = data[:2]
        self.loadedData.displacement = data[2:]

        self.checkData()

        return self.loadedData

    @staticmethod
    def loadDavisImageData(fileName: str, fileDir: str = "") -> np.ndarray:
        """ A .txt file from DaVis containing a 2D image

        Parameters
        ----------
        fileName
            File name.
        fileDir
            Path to file.

        Returns
        -------
        np.ndarray
            Array of data.

        """
        filePath = pathlib.Path(fileDir) / pathlib.Path(fileName)
        if not filePath.is_file():
            raise FileNotFoundError("Cannot open file {}".format(filePath))

        data = pd.read_table(str(filePath), delimiter='\t', skiprows=1,
                             header=None)
       
        # x and y coordinates
        loadedData = np.array(data)

        return loadedData


def readUntilString(
    file: TextIO,
    termString: str,
    commentChar: str = '*',
    lineProcess: Optional[Callable[[str], Any]] = None,
    exact: bool = False
) -> List[Any]:
    """Read lines in a file until a line starting with the `termString`
    is encountered. The file position is returned before the line starting
    with the `termString` when found. Comment and empty lines are ignored.

    Parameters
    ----------
    file
        An open python text file object.
    termString
        String to terminate reading.
    commentChar
        Character at start of a comment line to ignore.
    lineProcess
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
        currPos = file.tell()  # save position in file
        line = file.readline()
        if (not line
                or (exact and line.strip() == termString)
                or (not exact and line.strip().startswith(termString))):
            file.seek(currPos)  # return to before prev line
            break
        if line.strip() == '' or line.strip()[0] == commentChar:
            # Skip comment or empty line
            continue
        if lineProcess is not None:
            line = lineProcess(line)
        lines.append(line)
    return lines
