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
import pandas as pd
import pathlib
import re

from typing import TextIO, Dict, List, Callable, Any, Type, Optional

from defdap.crystal import Phase
from defdap.quat import Quat


class EBSDDataLoader(object):
    """Class containing methods for loading and checking EBSD data

    """
    def __init__(self) -> None:
        self.loadedMetadata = {
            'xDim': 0,
            'yDim': 0,
            'stepSize': 0.,
            'acquisitionRotation': Quat(1.0, 0.0, 0.0, 0.0),
            'phases': []
        }
        self.loadedData = {
            'phase': None,
            'eulerAngle': None,
            'bandContrast': None
        }
        self.dataFormat = None

    @staticmethod
    def getLoader(dataType: str) -> 'Type[EBSDDataLoader]':
        if dataType is None:
            dataType = "OxfordBinary"

        if dataType == "OxfordBinary":
            return OxfordBinaryLoader()
        elif dataType == "OxfordText":
            return OxfordTextLoader()
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
        mapShape = (self.loadedMetadata['yDim'], self.loadedMetadata['xDim'])

        assert self.loadedData['phase'].shape == mapShape
        assert self.loadedData['eulerAngle'].shape == (3,) + mapShape
        assert self.loadedData['bandContrast'].shape == mapShape


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

        Returns
        -------
        dict, dict
            EBSD metadata and EBSD data.

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
                latticeParams
            )
            return phase

        # default values for acquisition rotation in case missing in in file
        acqEulers = [0., 0., 0.]
        with open(str(filePath), 'r') as ctfFile:
            for i, line in enumerate(ctfFile):
                if 'XCells' in line:
                    xDim = int(line.split()[-1])
                    self.loadedMetadata['xDim'] = xDim
                elif 'YCells' in line:
                    yDim = int(line.split()[-1])
                    self.loadedMetadata['yDim'] = yDim
                elif 'XStep' in line:
                    self.loadedMetadata['stepSize'] = float(line.split()[-1])
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

        self.loadedMetadata['acquisitionRotation'] = Quat.fromEulerAngles(
            *(np.array(acqEulers) * np.pi / 180)
        )

        # TODO: Load EDX data from .ctf file, if it's accesible
        self.loadedMetadata['EDX Windows'] = {'Count': int(0)}

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
        binData = np.loadtxt(
            str(filePath), self.dataFormat, usecols=loadCols,
            skiprows=numHeaderLines
        )

        self.loadedData['bandContrast'] = np.reshape(
            binData['BC'], (yDim, xDim)
        )
        self.loadedData['bandSlope'] = np.reshape(
            binData['BS'], (yDim, xDim)
        )
        self.loadedData['meanAngularDeviation'] = np.reshape(
            binData['MAD'], (yDim, xDim)
        )
        self.loadedData['phase'] = np.reshape(
            binData['phase'], (yDim, xDim)
        )
        eulerAngles = np.reshape(
            binData[['ph1', 'phi', 'ph2']], (yDim, xDim)
        )
        # flatten the structures so that the Euler angles are stored
        # into a normal array
        eulerAngles = np.array(eulerAngles.tolist()).transpose((2, 0, 1))
        self.loadedData['eulerAngle'] = eulerAngles * np.pi / 180.

        self.checkData()


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

        Returns
        -------
        dict, dict
            EBSD metadata and EBSD data.

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

        self.loadedMetadata['xDim'] = int(metadata['Job']['xCells'])
        self.loadedMetadata['yDim'] = int(metadata['Job']['yCells'])
        self.loadedMetadata['stepSize'] = float(metadata['Job']['GridDistX'])
        self.loadedMetadata['acquisitionRotation'] = Quat.fromEulerAngles(
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
            self.loadedMetadata['EDX Windows'] = metadata['EDX Windows']
            edx_fields = {}
            for i in range(1, int(self.loadedMetadata['EDX Windows']['Count']) + 1):
                name = self.loadedMetadata['EDX Windows'][f"Window{i}"]
                edx_fields[100+i] = (f'EDX {name}', 'float32')
        else:
            self.loadedMetadata['EDX Windows'] = {'Count': int(0)}

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
        xDim = self.loadedMetadata['xDim']
        yDim = self.loadedMetadata['yDim']

        fileName = "{}.crc".format(fileName)
        filePath = pathlib.Path(fileDir) / pathlib.Path(fileName)
        if not filePath.is_file():
            raise FileNotFoundError("Cannot open file {}".format(filePath))

        # load binary data from file
        binData = np.fromfile(str(filePath), self.dataFormat, count=-1)

        self.loadedData['bandContrast'] = np.reshape(
            binData['BC'], (yDim, xDim)
        )
        self.loadedData['bandSlope'] = np.reshape(
            binData['BS'], (yDim, xDim)
        )
        self.loadedData['meanAngularDeviation'] = np.reshape(
            binData['MAD'], (yDim, xDim)
        )
        self.loadedData['phase'] = np.reshape(
            binData['phase'], (yDim, xDim)
        )
        eulerAngles = np.reshape(
            binData[['ph1', 'phi', 'ph2']], (yDim, xDim)
        )

        # Load EDX data into a dict
        if int(self.loadedMetadata['EDX Windows']['Count']) > 0:
            EDXFields = [key for key in binData.dtype.fields.keys() if key.startswith('EDX')]        
            self.loadedData['EDXDict'] = dict(
                [(field[4:], np.reshape(binData[field], (yDim, xDim))) for field in EDXFields]
                )

        # flatten the structures so that the Euler angles are stored
        # into a normal array
        eulerAngles = np.array(eulerAngles.tolist()).transpose((2, 0, 1))
        self.loadedData['eulerAngle'] = eulerAngles

        self.checkData()


class PythonDictLoader(EBSDDataLoader):
    def load(self, dataDict: Dict[str, Any]) -> None:
        """Construct EBSD data from a python dictionary.

        Parameters
        ----------
        dataDict
            Dictionary with keys:
                'stepSize'
                'phases'
                'phase'
                'eulerAngle'
                'bandContrast'

        """
        self.loadedMetadata['xDim'] = dataDict['phase'].shape[1]
        self.loadedMetadata['yDim'] = dataDict['phase'].shape[0]
        self.loadedMetadata['stepSize'] = dataDict['stepSize']
        assert type(dataDict['phases']) is list
        self.loadedMetadata['phases'] = dataDict['phases']
        self.loadedMetadata['EDX Windows'] = {'Count': int(0)}

        self.checkMetadata()

        self.loadedData['phase'] = dataDict['phase']
        self.loadedData['eulerAngle'] = dataDict['eulerAngle']
        self.loadedData['bandContrast'] = dataDict['bandContrast']

        self.checkData()


class DICDataLoader(object):
    """Class containing methods for loading and checking HRDIC data

    """
    def __init__(self) -> None:
        self.loadedMetadata = {
            'format': "",
            'version': "",
            'binning': "",
            'xDim': 0,
            'yDim': 0
        }
        self.loadedData = {
            'xc': None,
            'yc': None,
            'xd': None,
            'yd': None
        }

    def checkMetadata(self) -> None:
        return

    def checkData(self) -> None:
        """ Calculate size of map from loaded data and check it matches
        values from metadata.

        """
        coords = self.loadedData['xc']
        xdim = int(
            (coords.max() - coords.min()) / min(abs(np.diff(coords))) + 1
        )

        coords = self.loadedData['yc']
        ydim = int(
            (coords.max() - coords.min()) / max(abs(np.diff(coords))) + 1
        )

        assert xdim == self.loadedMetadata['xDim'], "Dimensions of data and header do not match"
        assert ydim == self.loadedMetadata['yDim'], "Dimensions of data and header do not match"

    def loadDavisMetadata(self,
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
        # size of map along x and y (from header)
        self.loadedMetadata['xDim'] = int(metadata[5])
        self.loadedMetadata['yDim'] = int(metadata[4])

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
                             header=None)
        # x and y coordinates
        self.loadedData['xc'] = data.values[:, 0]
        self.loadedData['yc'] = data.values[:, 1]
        # x and y displacement
        self.loadedData['xd'] = data.values[:, 2]
        self.loadedData['yd'] = data.values[:, 3]

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
    lineProcess: Optional[Callable[[str], Any]] = None
) -> List[Any]:
    """Read lines in a file until a line starting with the `termString`
    is encounted. The file position is returned before the line starting
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

    Returns
    -------
    list
        List of processed lines loaded from file.

    """
    lines = []
    while True:
        currPos = file.tell()  # save position in file
        line = file.readline()
        if not line or line.strip().startswith(termString):
            file.seek(currPos)  # return to before prev line
            break
        if line.strip() == '' or line.strip()[0] == commentChar:
            # Skip comment or empty line
            continue
        if lineProcess is not None:
            line = lineProcess(line)
        lines.append(line)
    return lines
