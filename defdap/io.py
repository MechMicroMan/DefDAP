import numpy as np


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
            'phase': None,
        }

    def checkMetadata(self):
        if len(self.loadedMetadata['phaseNames']) != self.loadedMetadata['numPhases']:
            print("Number of phases mismatch.")

    def checkData(self):


    def loadOxfordCPR(self, fileName):
        # open meta data file and read in x and y dimensions and phase names
        metadataFile = open(fileName + ".cpr", 'r')
        for line in metadataFile:
            if line[:6] == 'xCells':
                xDim = int(line[7:])
                self.loadedMetadata['xDim'] = xDim
            elif line[:6] == 'yCells':
                yDim = int(line[7:])
                self.loadedMetadata['yDim'] = yDim
            elif line[:9] == 'GridDistX':
                self.loadedMetadata['stepSize'] = float(line[10:])
            elif line[:8] == '[Phases]':
                self.loadedMetadata['numPhases'] = int(next(metadataFile)[6:])
            elif line[:6] == '[Phase':
                self.loadedMetadata['phaseNames'].append(next(metadataFile)[14:].strip('\n'))

        metadataFile.close()
        self.checkMetadata()

        # now read the binary .crc file
        fmt_np = np.dtype([('Phase', 'b'),
                           ('Eulers', [('ph1', 'f'),
                                       ('phi', 'f'),
                                       ('ph2', 'f')]),
                           ('MAD', 'f'),
                           ('BC',  'uint8'),
                           ('IB3', 'uint8'),
                           ('IB4', 'uint8'),
                           ('IB5', 'uint8'),
                           ('IB6', 'f')])
        binData = np.fromfile(fileName + ".crc", fmt_np, count=-1)

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
        # flatten the structures the Euler angles are stored into a normal array
        eulerAngles = np.array(eulerAngles.tolist()).transpose((2, 0, 1))
        self.loadedData['eulerAngle'] = eulerAngles

        self.checkData()

        return self.loadedMetadata, self.loadedData
