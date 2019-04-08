import numpy as np
import pandas as pd
import pathlib


class EBSDDataLoader(object):

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
        if len(self.loadedMetadata['phaseNames']) != self.loadedMetadata['numPhases']:
            print("Number of phases mismatch.")
            raise AssertionError

    def checkData(self, binData):
        return

    def loadOxfordCPR(self, file_path):
        """ A .cpr file is a metadata file describing EBSD data.
        This function opens the cpr file, reading in the x and y dimensions and phase names."""

        cpr_path = "{}.cpr".format(file_path)
        if not pathlib.Path(cpr_path).is_file():
            raise FileNotFoundError("Cannot open file {}".format(cpr_path))

        cpr_file = open(cpr_path, 'r')

        for line in cpr_file:
            if 'xCells' in line:
                xDim = int(line.split("=")[-1])
                self.loadedMetadata['xDim'] = xDim
            elif 'yCells' in line:
                yDim = int(line.split("=")[-1])
                self.loadedMetadata['yDim'] = yDim
            elif 'GridDistX' in line:
                self.loadedMetadata['stepSize'] = float(line.split("=")[-1])
            elif '[Phases]' in line:
                self.loadedMetadata['numPhases'] = int(next(cpr_file).split("=")[-1])
            elif '[Phase' in line:
                phase_name = next(cpr_file).split("=")[-1].strip('\n')
                self.loadedMetadata['phaseNames'].append(phase_name)

        cpr_file.close()

        self.checkMetadata()

        return self.loadedMetadata

    def read_crc(self, file_path):
        xDim = self.loadedMetadata['xDim']
        yDim = self.loadedMetadata['yDim']

        # now read the binary .crc file
        crc_path = "{}.crc".format(file_path)
        if not pathlib.Path(crc_path).is_file():
            raise FileNotFoundError("Cannot open file {}".format(crc_path))

        fmt_np = np.dtype([('Phase', 'b'),
                           ('Eulers', [('ph1', 'f'), ('phi', 'f'), ('ph2', 'f')]),
                           ('MAD', 'f'),
                           ('BC', 'uint8'),
                           ('IB3', 'uint8'),
                           ('IB4', 'uint8'),
                           ('IB5', 'uint8'),
                           ('IB6', 'f')])
        binData = np.fromfile(crc_path, fmt_np, count=-1)
        self.checkData(binData)
        self.loadedData['bandContrast'] = np.reshape(binData['BC'], (yDim, xDim))
        self.loadedData['phase'] = np.reshape(binData['Phase'], (yDim, xDim))
        eulerAngles = np.reshape(binData['Eulers'], (yDim, xDim))
        # flatten the structures so that the Euler angles are stored into a normal array
        eulerAngles = np.array(eulerAngles.tolist()).transpose((2, 0, 1))
        self.loadedData['eulerAngle'] = eulerAngles
        return self.loadedData

    def loadOxfordCTF(self, fileName, fileDir=""):
        """ A .ctf file is a HKL single orientation file. This is a data file generated
        by the Oxford EBSD instrument."""

        # open data file and read in metadata
        filePathBase = "{:}{:}".format(fileDir, fileName)
        filePath = "{:}.ctf".format(filePathBase)
        dataFile = open(filePath, 'r')

        for i, line in enumerate(dataFile):
            if line[:6] == 'XCells':
                xDim = int(line[7:])
                self.loadedMetadata['xDim'] = xDim
            elif line[:6] == 'YCells':
                yDim = int(line[7:])
                self.loadedMetadata['yDim'] = yDim
            elif line[:5] == 'XStep':
                self.loadedMetadata['stepSize'] = float(line[6:])
            elif line[:6] == 'Phases':
                numPhases = int(line[7:])
                self.loadedMetadata['numPhases'] = numPhases
                for j in range(numPhases):
                    self.loadedMetadata['phaseNames'].append(
                        next(dataFile).split()[2]
                    )
                numHeaderLines = i + j + 3
                # phases are last in the header so break out the loop
                break

        dataFile.close()

        self.checkMetadata()

        # now read the data from file
        fmt_np = np.dtype([
            ('Phase', 'b'),
            ('Eulers', [('ph1', 'f'),
                        ('phi', 'f'),
                        ('ph2', 'f')]),
            ('MAD', 'f'),
            ('BC', 'uint8')
        ])
        binData = np.loadtxt(
            filePath, fmt_np, delimiter='\t',
            skiprows=numHeaderLines, usecols=(0, 5, 6, 7, 8, 9)
        )

        self.checkData(binData)

        self.loadedData['bandContrast'] = np.reshape(
            binData['BC'],
            (yDim, xDim)
        )

        self.loadedData['phase'] = np.reshape(
            binData['Phase'],
            (yDim, xDim)
        )

        eulerAngles = np.reshape(
            binData['Eulers'],
            (yDim, xDim)
        )
        # flatten the structures the Euler angles are stored into a
        # normal array
        eulerAngles = np.array(eulerAngles.tolist()).transpose((2, 0, 1))
        self.loadedData['eulerAngle'] = eulerAngles * np.pi / 180.

        return self.loadedMetadata, self.loadedData


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
        # Calculate size of map from loaded data and check it matches
        # values from metadata
        coords = self.loadedData['xc']
        xdim = int(
            (coords.max() - coords.min()) / min(abs(np.diff(coords))) + 1
        )

        coords = self.loadedData['yc']
        ydim = int(
            (coords.max() - coords.min()) / max(abs(np.diff(coords))) + 1
        )

        if ((xdim != self.loadedMetadata['xDim']) or
                (ydim != self.loadedMetadata['yDim'])):
            raise Exception("Dimensions of data and header do not match")

    def loadDavisTXT(self, fileName, fileDir=""):
        # Load metadata
        filePath = "{:}{:}".format(fileDir, fileName)
        with open(filePath, 'r') as f:
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

        # Load data

        data = pd.read_table(filePath, delimiter='\t', skiprows=1, header=None)
        # x and y coordinates
        self.loadedData['xc'] = data.values[:, 0]
        self.loadedData['yc'] = data.values[:, 1]
        # x and y displacement
        self.loadedData['xd'] = data.values[:, 2]
        self.loadedData['yd'] = data.values[:, 3]

        self.checkData()

        return self.loadedMetadata, self.loadedData
