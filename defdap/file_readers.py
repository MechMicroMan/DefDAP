# Copyright 2020 Mechanics of Microstructures Group
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


class EBSDDataLoader(object):
    """Class containing methods for loading and checking EBSD data

    """
    def __init__(self):
        self.loadedMetadata = {
            'xDim': 0,
            'yDim': 0,
            'stepSize': 0.,
            'numPhases': 0,
            'phaseNames': []
        }
        self.loadedData = {
            'eulerAngle': None,
            'bandContrast': None,
            'phase': None
        }

    def checkMetadata(self):
        """
        Checks that the number of phases from metadata matches
        the amount of phase names.

        """
        if len(self.loadedMetadata['phaseNames']) != self.loadedMetadata['numPhases']:
            print("Number of phases mismatch.")
            raise AssertionError

    def checkData(self, binData):
        return

    def loadOxfordCPR(self, fileName, fileDir=""):
        """
        Read an Oxford Instruments .cpr file, which is a metadata file
        describing EBSD data.

        Parameters
        ----------
        fileName : str
            File name.
        fileDir : str
            Path to file.

        Returns
        -------
        dict
            Metadata for EBSD data, including dimensions and number of phases.

        """
        fileName = "{}.cpr".format(fileName)
        filePath = pathlib.Path(fileDir) / pathlib.Path(fileName)
        if not filePath.is_file():
            raise FileNotFoundError("Cannot open file {}".format(filePath))

        cprFile = open(str(filePath), 'r')

        for line in cprFile:
            if 'xCells' in line:
                self.loadedMetadata['xDim'] = int(line.split("=")[-1])
            elif 'yCells' in line:
                self.loadedMetadata['yDim'] = int(line.split("=")[-1])
            elif 'GridDistX' in line:
                self.loadedMetadata['stepSize'] = float(line.split("=")[-1])
            elif '[Phases]' in line:
                self.loadedMetadata['numPhases'] = int(next(cprFile).split("=")[-1])
            elif '[Phase' in line:
                phaseName = next(cprFile).split("=")[-1].strip('\n')
                self.loadedMetadata['phaseNames'].append(phaseName)

        cprFile.close()

        self.checkMetadata()

        return self.loadedMetadata

    def loadOxfordCRC(self, fileName, fileDir=""):
        """Read binary EBSD data from an Oxford Instruments .crc file

        Parameters
        ----------
        fileName : str
            File name.
        fileDir : str
            Path to file.

        Returns
        -------
        dict
            EBSD data including phase, MAD and euler angle arrays.

        """
        xDim = self.loadedMetadata['xDim']
        yDim = self.loadedMetadata['yDim']

        fileName = "{}.crc".format(fileName)
        filePath = pathlib.Path(fileDir) / pathlib.Path(fileName)
        if not filePath.is_file():
            raise FileNotFoundError("Cannot open file {}".format(filePath))

        dataFormat = np.dtype([
            ('Phase', 'b'),
            ('Eulers', [('ph1', 'f'), ('phi', 'f'), ('ph2', 'f')]),
            ('MAD', 'f'),
            ('BC', 'uint8'),
            ('IB3', 'uint8'),
            ('IB4', 'uint8'),
            ('IB5', 'uint8'),
            ('IB6', 'f')
        ])
        binData = np.fromfile(str(filePath), dataFormat, count=-1)

        self.checkData(binData)

        self.loadedData['bandContrast'] = np.reshape(
            binData['BC'], (yDim, xDim)
        )
        self.loadedData['phase'] = np.reshape(
            binData['Phase'], (yDim, xDim)
        )
        eulerAngles = np.reshape(
            binData['Eulers'], (yDim, xDim)
        )
        # flatten the structures so that the Euler angles are stored
        # into a normal array
        eulerAngles = np.array(eulerAngles.tolist()).transpose((2, 0, 1))
        self.loadedData['eulerAngle'] = eulerAngles

        return self.loadedData

    def loadOxfordCTF(self, fileName, fileDir=""):
        """ Read an Oxford Instruments .ctf file, which is a HKL single orientation file.

        Parameters
        ----------
        fileName : str
            File name.
        fileDir : str
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

        ctfFile = open(str(filePath), 'r')

        for i, line in enumerate(ctfFile):
            if 'XCells' in line:
                xDim = int(line.split()[-1])
                self.loadedMetadata['xDim'] = xDim
            elif 'YCells' in line:
                yDim = int(line.split()[-1])
                self.loadedMetadata['yDim'] = yDim
            elif 'XStep' in line:
                self.loadedMetadata['stepSize'] = float(line.split()[-1])
            elif 'Phases' in line:
                numPhases = int(line.split()[-1])
                self.loadedMetadata['numPhases'] = numPhases
                for j in range(numPhases):
                    self.loadedMetadata['phaseNames'].append(
                        next(ctfFile).split()[2]
                    )
                numHeaderLines = i + j + 3
                # phases are last in the header so break out the loop
                break

        ctfFile.close()

        self.checkMetadata()

        # now read the data from file
        dataFormat = np.dtype([
            ('Phase', 'b'),
            ('Eulers', [('ph1', 'f'), ('phi', 'f'), ('ph2', 'f')]),
            ('MAD', 'f'),
            ('BC', 'uint8')
        ])
        binData = np.loadtxt(
            str(filePath), dataFormat,
            skiprows=numHeaderLines, usecols=(0, 5, 6, 7, 8, 9)
        )

        self.checkData(binData)

        self.loadedData['bandContrast'] = np.reshape(
            binData['BC'], (yDim, xDim)
        )
        self.loadedData['phase'] = np.reshape(
            binData['Phase'], (yDim, xDim)
        )
        eulerAngles = np.reshape(
            binData['Eulers'], (yDim, xDim)
        )
        # flatten the structures the Euler angles are stored into a
        # normal array
        eulerAngles = np.array(eulerAngles.tolist()).transpose((2, 0, 1))
        self.loadedData['eulerAngle'] = eulerAngles * np.pi / 180.

        return self.loadedMetadata, self.loadedData


class DICDataLoader(object):
    """Class containing methods for loading and checking HRDIC data

    """
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

    def loadDavisMetadata(self, fileName, fileDir=""):
        """ Load DaVis metadata from Davis .txt file.

        Parameters
        ----------
        fileName : str
            File name.
        fileDir : str
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

    def loadDavisData(self, fileName, fileDir=""):
        """ Load displacement data from Davis .txt file containing x and y coordinates
        and x and y displacements for each coordinate.

        Parameters
        ----------
        fileName : str
            File name.
        fileDir : str
            Path to file.

        Returns
        -------
        dict
            Coordinates and displacements.

        """
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

        Parameters
        ----------
        fileName : str
            File name.
        fileDir : str
            Path to file.

        """
        filePath = pathlib.Path(fileDir) / pathlib.Path(fileName)
        if not filePath.is_file():
            raise FileNotFoundError("Cannot open file {}".format(filePath))

        data = pd.read_table(str(filePath), delimiter='\t', skiprows=1, header=None)
       
       # x and y coordinates
        loadedData = np.array(data)

        return loadedData