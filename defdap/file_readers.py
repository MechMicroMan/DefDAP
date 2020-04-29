# Copyright 2019 Mechanics of Microstructures Group
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

from defdap.crystal import Phase, crystalStructures
from defdap.quat import Quat


class EBSDDataLoader(object):

    def __init__(self):
        self.loadedMetadata = {
            'xDim': 0,
            'yDim': 0,
            'stepSize': 0.,
            'acquisitionRotation': Quat(1.0, 0.0, 0.0, 0.0),
            'numPhases': 0,
            'phases': []
        }
        self.loadedData = {
            'eulerAngle': None,
            'bandContrast': None,
            'phase': None
        }
        self.dataFormat = None

    def checkMetadata(self):
        """ Checks that the number of phases from metadata matches
        the amount of phase names."""
        if len(self.loadedMetadata['phases']) != self.loadedMetadata['numPhases']:
            raise ValueError("Number of phases mismatch.")

    def checkData(self, binData):
        return

    def loadOxfordCPR(self, fileName, fileDir=""):
        """ A .cpr file is a metadata file describing EBSD data.
        This function opens the cpr file, reading in the x and y
        dimensions and phase names."""
        fileName = "{}.cpr".format(fileName)
        filePath = pathlib.Path(fileDir) / pathlib.Path(fileName)
        if not filePath.is_file():
            raise FileNotFoundError("Cannot open file {}".format(filePath))

        metadata = dict()
        groupPat = re.compile("\[(.+)\]")

        def parseLine(line):
            try:
                key, val = line.strip().split('=')
                groupDict[key] = val
            except ValueError:
                pass

        with open(str(filePath), 'r') as cprFile:
            while True:
                line = cprFile.readline()
                if not line:
                    break

                groupName = groupPat.match(line.strip()).group(1)
                groupDict = dict()
                readUntilComment(cprFile, commentChar='[',
                                 lineProcess=parseLine)
                metadata[groupName] = groupDict

        self.loadedMetadata['xDim'] = int(metadata['Job']['xCells'])
        self.loadedMetadata['yDim'] = int(metadata['Job']['yCells'])
        self.loadedMetadata['stepSize'] = float(metadata['Job']['GridDistX'])
        self.loadedMetadata['acquisitionRotation'] = Quat.fromEulerAngles(
            float(metadata['Acquisition Surface']['Euler1']) * np.pi / 180.,
            float(metadata['Acquisition Surface']['Euler2']) * np.pi / 180.,
            float(metadata['Acquisition Surface']['Euler3']) * np.pi / 180.
        )
        self.loadedMetadata['numPhases'] = int(metadata['Phases']['Count'])

        for i in range(self.loadedMetadata['numPhases']):
            phaseMetadata = metadata['Phase{:}'.format(i+1)]
            self.loadedMetadata['phases'].append(Phase(
                phaseMetadata['StructureName'],
                EBSDDataLoader.laueGroupLookup(int(phaseMetadata['LaueGroup'])),
                (
                    float(phaseMetadata['a']),
                    float(phaseMetadata['b']),
                    float(phaseMetadata['c']),
                    float(phaseMetadata['alpha']),
                    float(phaseMetadata['beta']),
                    float(phaseMetadata['gamma'])
                )
            ))

        self.checkMetadata()

        # Construct binary data format from listed fields
        dataFormat = [('phase', 'uint8')]

        fieldLookup = {
            3: ('ph1', 'float32'),
            4: ('phi', 'float32'),
            5: ('ph2', 'float32'),
            6: ('MAD', 'float32'),  # Mean Angular Deviation
            7: ('BC', 'uint8'),     # Band Contrast
            8: ('BS', 'uint8'),     # Band Slope
            10: ('numBands', 'uint8'),
            11: ('AFI', 'uint8'),   # Advanced Fit index. legacy
            12: ('IB6', 'float32')  # ?
        }
        try:
            for i in range(int(metadata['Fields']['Count'])):
                fieldID = int(metadata['Fields']['Field{:}'.format(i+1)])
                dataFormat.append(fieldLookup[fieldID])
        except KeyError:
            raise TypeError("Unknown data in EBSD file.")

        self.dataFormat = np.dtype(dataFormat)

        return self.loadedMetadata

    def loadOxfordCRC(self, fileName, fileDir=""):
        """Read binary EBSD data from a .crc file"""
        xDim = self.loadedMetadata['xDim']
        yDim = self.loadedMetadata['yDim']

        fileName = "{}.crc".format(fileName)
        filePath = pathlib.Path(fileDir) / pathlib.Path(fileName)
        if not filePath.is_file():
            raise FileNotFoundError("Cannot open file {}".format(filePath))

        # laod binary data from file
        binData = np.fromfile(str(filePath), self.dataFormat, count=-1)

        self.checkData(binData)

        self.loadedData['bandContrast'] = np.reshape(
            binData['BC'], (yDim, xDim)
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
        self.loadedData['eulerAngle'] = eulerAngles

        return self.loadedData

    def loadOxfordCTF(self, fileName, fileDir=""):
        """ A .ctf file is a HKL single orientation file. This is a
        data file generated by the Oxford EBSD instrument."""

        # open data file and read in metadata
        fileName = "{}.ctf".format(fileName)
        filePath = pathlib.Path(fileDir) / pathlib.Path(fileName)
        if not filePath.is_file():
            raise FileNotFoundError("Cannot open file {}".format(filePath))

        def parsePhase():
            lineSplit = line.split('\t')
            latticeParams = lineSplit[0].split(';') + lineSplit[1].split(';')
            latticeParams = tuple(float(val) for val in latticeParams)
            phase = Phase(
                lineSplit[2],
                EBSDDataLoader.laueGroupLookup(int(lineSplit[3])),
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
                    self.loadedMetadata['numPhases'] = numPhases
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

        keepColNames = ('phase', 'ph1', 'phi', 'ph2', 'BC')
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

        self.checkData(binData)

        self.loadedData['bandContrast'] = np.reshape(
            binData['BC'], (yDim, xDim)
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

        return self.loadedMetadata, self.loadedData

    @staticmethod
    def laueGroupLookup(laueGroup):
        if laueGroup == 11:
            return crystalStructures['cubic']
        elif laueGroup == 9:
            return crystalStructures['hexagonal']

        raise ValueError("Only cubic and hexagonal crystal structures "
                         "are currently supported.")


class DICDataLoader(object):

    def __init__(self):
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

    def checkMetadata(self):
        return

    def checkData(self):
        """ Calculate size of map from loaded data and check it matches
        values from metadata"""
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

    def loadDavisMetadata(self, fileName, fileDir=""):
        """ Load DaVis metadata"""
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

    def loadDavisData(self, fileName, fileDir=""):
        """ A .txt file from DaVis contains x and y coordinates
        and x and y displacements for each coordinate"""
        filePath = pathlib.Path(fileDir) / pathlib.Path(fileName)
        if not filePath.is_file():
            raise FileNotFoundError("Cannot open file {}".format(filePath))

        data = pd.read_table(str(filePath), delimiter='\t', skiprows=1, header=None)
        # x and y coordinates
        self.loadedData['xc'] = data.values[:, 0]
        self.loadedData['yc'] = data.values[:, 1]
        # x and y displacement
        self.loadedData['xd'] = data.values[:, 2]
        self.loadedData['yd'] = data.values[:, 3]

        self.checkData()

        return self.loadedData
        
    def loadDavisImageData(self, fileName, fileDir=""):
        """ A .txt file from DaVis containing a 2D image
        """
        filePath = pathlib.Path(fileDir) / pathlib.Path(fileName)
        if not filePath.is_file():
            raise FileNotFoundError("Cannot open file {}".format(filePath))

        data = pd.read_table(str(filePath), delimiter='\t', skiprows=1, header=None)
       
       # x and y coordinates
        loadedData = np.array(data)

        return loadedData


def readUntilComment(file, commentChar='*', lineProcess=None):
    lines = []
    while True:
        currPos = file.tell()  # save position in file
        line = file.readline()
        if not line or line[0] == commentChar:
            file.seek(currPos)  # return to before prev line
            break
        if lineProcess is not None:
            line = lineProcess(line)
        lines.append(line)
    return lines
