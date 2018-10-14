import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from skimage import morphology as mph

import copy

from quat import Quat
import base


class Map(base.Map):
    """Summary

    Attributes:
        averageSchmidFactor (TYPE): Description
        binData (TYPE): imported binary data
        boundaries (TYPE): Description
        cacheEulerMap (TYPE): Description
        crystalSym (TYPE): Description
        currGrainId (TYPE): Description
        grainList (list): Description
        grains (TYPE): Description
        homogPoints (TYPE): Description
        misOri (TYPE): Description
        misOriAxis (TYPE): Description
        origin (tuple): Description
        plotDefault (TYPE): Description
        quatArray (TYPE): Description
        slipSystems (TYPE): Description
        stepSize (TYPE): Description
        xDim (int): x dimension of map
        yDim (int): y dimension of map
    """

    def __init__(self, fileName, crystalSym):
        """Initialise class and import DIC data from file

        Args:
            filename(str): Path to file, including name, excluding extension
            crystalSym(str): Crystal sturcture, 'cubic' or 'hexagonal'
        """

        # Call base class constructor
        super(Map, self).__init__()

        print("Loading EBSD data...", end="")

        self.crystalSym = None              # (str) symmetry of material e.g. "cubic", "hexagonal"
        self.xDim = None                    # (int) dimensions of maps
        self.yDim = None
        self.stepSize = None                # (float) step size
        self.binData = None                 # imported binary data
        self.quatArray = None               # (array) array of quaterions for each point of map
        self.numPhases = None               # (int) number of phases
        self.phaseArray = None              # (array) map of phase ids
        self.phaseNames = []                # (array) array of phase names
        self.boundaries = None              # (array) map of boundaries. -1 for a boundary, 0 otherwise
        self.phaseBoundaries = None         # (array) map of phase boundaries. -1 for boundary, 0 otherwise
        self.grains = None                  # (array) map of grains
        self.grainList = None               # (list) list of grains
        self.misOri = None                  # (array) map of misorientation
        self.misOriAxis = None              # (list of arrays) map of misorientation axis components
        self.kam = None                     # (array) map of kam
        self.averageSchmidFactor = None     # (array) map of average Schmid factor
        self.slipSystems = None             # (list(list(slipSystems))) slip systems grouped by slip plane
        self.slipTraceColours = None        # (list) colours used when plotting slip traces
        self.currGrainId = None             # (int) ID of last selected grain
        self.origin = (0, 0)                # Map origin (y, x). Used by linker class where origin is a
                                            # homologue point of the maps

        self.plotHomog = self.plotEulerMap  # Use euler map for defining homologous points
        self.highlightAlpha = 1

        self.loadData(fileName, crystalSym)
        return

    def plotDefault(self, *args, **kwargs):
        self.plotEulerMap(*args, **kwargs)

    def loadData(self, fileName, crystalSym):
        # open meta data file and read in x and y dimensions and phase names
        metaFile = open(fileName + ".cpr", 'r')
        for line in metaFile:
            if line[:6] == 'xCells':
                self.xDim = int(line[7:])
            elif line[:6] == 'yCells':
                self.yDim = int(line[7:])
            elif line[:9] == 'GridDistX':
                self.stepSize = float(line[10:])
            elif line[:8] == '[Phases]':
                self.numPhases = int(next(metaFile)[6:])
            elif line[:6] == '[Phase':
                self.phaseNames.append(next(metaFile)[14:].strip('\n'))

        if len(self.phaseNames) != self.numPhases:
            print("Error with cpr file. Number of phases mismatch.")

        metaFile.close()
        # now read the binary .crc file
        fmt_np = np.dtype([('Phase', 'b'),
                           ('Eulers', [('ph1', 'f'),
                                       ('phi', 'f'),
                                       ('ph2', 'f')]),
                           ('mad', 'f'),
                           ('IB2', 'uint8'),
                           ('IB3', 'uint8'),
                           ('IB4', 'uint8'),
                           ('IB5', 'uint8'),
                           ('IB6', 'f')])
        # for ctf files that have been converted using channel 5
        # CHANGE BACK!!!!!!!
        # fmt_np = np.dtype([('Phase', 'b'),
        #                    ('Eulers', [('ph1', 'f'),
        #                                ('phi', 'f'),
        #                                ('ph2', 'f')]),
        #                    ('mad', 'f'),
        #                    ('IB2', 'uint8'),
        #                    ('IB3', 'uint8'),
        #                    ('IB4', 'uint8'),
        #                    ('IB5', 'uint8')])
        self.binData = np.fromfile(fileName + ".crc", fmt_np, count=-1)
        self.crystalSym = crystalSym
        self.phaseArray = np.reshape(self.binData[('Phase')], (self.yDim, self.xDim))

        print("\rLoaded EBSD data (dimensions: {0} x {1} pixels, step size: {2} um)".
              format(self.xDim, self.yDim, self.stepSize))

        return

    def plotBandContrastMap(self):
        self.checkDataLoaded()

        bcmap = np.reshape(self.binData[('IB2')], (self.yDim, self.xDim))
        plt.imshow(bcmap, cmap='gray')
        plt.colorbar()
        return

    def plotEulerMap(self, updateCurrent=False, highlightGrains=None, highlightColours=None):
        """Plots an orientation map in Euler colouring

        Args:
            updateCurrent (bool, optional): Description
            highlightGrains (List int, optional): Grain ids of grains to highlight
        """
        self.checkDataLoaded()

        if not updateCurrent:
            emap = np.transpose(np.array([self.binData['Eulers']['ph1'],
                                          self.binData['Eulers']['phi'],
                                          self.binData['Eulers']['ph2']]))
            # this is the normalization for the
            norm = np.tile(np.array([2 * np.pi, np.pi / 2, np.pi / 2]), (self.yDim, self.xDim))
            norm = np.reshape(norm, (self.yDim, self.xDim, 3))
            eumap = np.reshape(emap, (self.yDim, self.xDim, 3))
            # make non-indexed points green
            eumap = np.where(eumap != [0., 0., 0.], eumap, [0., 1., 0.])

            self.cacheEulerMap = eumap / norm
            self.fig, self.ax = plt.subplots()

        self.ax.imshow(self.cacheEulerMap, aspect='equal')

        if highlightGrains is not None:
            self.highlightGrains(highlightGrains, highlightColours)

        return

    def plotPhaseMap(self, cmap='viridis'):
        """Plots a phase map

        Args:
            cmap(str, optional): Colour map
        """
        values = [-1] + list(range(1, self.numPhases + 1))
        names = ["Non-indexed"] + self.phaseNames

        plt.figure(figsize=(10, 6))
        im = plt.imshow(self.phaseArray, cmap=cmap, vmin=-1, vmax=self.numPhases)

        # Find colour values for phases
        colors = [im.cmap(im.norm(value)) for value in values]

        # Get colour patches for each phase and make legend
        patches = [mpl.patches.Patch(color=colors[i], label=names[i]) for i in range(len(values))]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.show()

        return

    def calcKam(self):
        """Calculates Kernel Average Misorientaion (KAM) for the EBSD map. Crystal symmetric
           equivalences are not considered. Stores result in self.kam.
        """
        quatComps = np.empty((4, self.yDim, self.xDim))

        for i, row in enumerate(self.quatArray):
            for j, quat in enumerate(row):
                quatComps[:, i, j] = quat.quatCoef

        self.kam = np.empty((self.yDim, self.xDim))

        # Start with rows. Caluculate misorientation with neigbouring rows.
        # First and last row only in one direction
        self.kam[0, :] = abs(np.einsum("ij,ij->j", quatComps[:, 0, :], quatComps[:, 1, :]))
        self.kam[-1, :] = abs(np.einsum("ij,ij->j", quatComps[:, -1, :], quatComps[:, -2, :]))
        for i in range(1, self.yDim - 1):
            self.kam[i, :] = (abs(np.einsum("ij,ij->j", quatComps[:, i, :], quatComps[:, i + 1, :])) +
                              abs(np.einsum("ij,ij->j", quatComps[:, i, :], quatComps[:, i - 1, :]))) / 2

        self.kam[self.kam > 1] = 1

        # Do the same for columns
        self.kam[:, 0] += abs(np.einsum("ij,ij->j", quatComps[:, :, 0], quatComps[:, :, 1]))
        self.kam[:, -1] += abs(np.einsum("ij,ij->j", quatComps[:, :, -1], quatComps[:, :, -2]))
        for i in range(1, self.xDim - 1):
            self.kam[:, i] += (abs(np.einsum("ij,ij->j", quatComps[:, :, i], quatComps[:, :, i + 1])) +
                               abs(np.einsum("ij,ij->j", quatComps[:, :, i], quatComps[:, :, i - 1]))) / 2

        self.kam /= 2
        self.kam[self.kam > 1] = 1

    def plotKamMap(self, vmin=None, vmax=None, cmap="viridis"):
        """Plots Kernel Average Misorientaion (KAM) for the EBSD map.

        Args:
            vmin (float, optional): Minimum of colour scale.
            vmax (float, optional): Maximum of colour scale.
            cmap (str, optional): Colourmap to show data with.
        """
        self.calcKam()
        # Convert to degrees and plot
        kam = 2 * np.arccos(self.kam) * 180 / np.pi
        plt.figure()
        plt.imshow(kam, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar()

    def checkDataLoaded(self):
        if self.binData is None:
            raise Exception("Data not loaded")
        return True

    def buildQuatArray(self):
        print("Building quaternion array...", end="")

        self.checkDataLoaded()

        if self.quatArray is None:
            eulerArray = self.binData[('Eulers')]
            # reshape to map dimensions
            eulerArray = eulerArray.reshape((self.yDim, self.xDim))
            # this flattens the structures the Euler angles are stored into a normal array
            eulerArray = np.array(eulerArray.tolist())
            eulerArray = eulerArray.transpose((2, 0, 1))
            # create the array of quat objects
            self.quatArray = Quat.createManyQuats(eulerArray)

        print("\r", end="")
        return

    def findBoundaries(self, boundDef=10):
        self.buildQuatArray()
        print("Finding boundaries...", end="")

        syms = Quat.symEqv(self.crystalSym)
        numSyms = len(syms)

        # array to store quat components of initial and symmetric equivalents
        quatComps = np.empty((numSyms, 4, self.yDim, self.xDim))

        # populate with initial quat components
        for i, row in enumerate(self.quatArray):
            for j, quat in enumerate(row):
                quatComps[0, :, i, j] = quat.quatCoef

        # loop of over symmetries and apply to initial quat components
        # (excluding first symmetry as this is the identity transformation)
        for i, sym in enumerate(syms[1:], start=1):
            # sym[i] * quat for all points (* is quaternion product)
            quatComps[i, 0, :, :] = (quatComps[0, 0, :, :] * sym[0] - quatComps[0, 1, :, :] * sym[1] -
                                     quatComps[0, 2, :, :] * sym[2] - quatComps[0, 3, :, :] * sym[3])
            quatComps[i, 1, :, :] = (quatComps[0, 0, :, :] * sym[1] + quatComps[0, 1, :, :] * sym[0] -
                                     quatComps[0, 2, :, :] * sym[3] + quatComps[0, 3, :, :] * sym[2])
            quatComps[i, 2, :, :] = (quatComps[0, 0, :, :] * sym[2] + quatComps[0, 2, :, :] * sym[0] -
                                     quatComps[0, 3, :, :] * sym[1] + quatComps[0, 1, :, :] * sym[3])
            quatComps[i, 3, :, :] = (quatComps[0, 0, :, :] * sym[3] + quatComps[0, 3, :, :] * sym[0] -
                                     quatComps[0, 1, :, :] * sym[2] + quatComps[0, 2, :, :] * sym[1])

            # swap into positve hemisphere if required
            quatComps[i, :, quatComps[i, 0, :, :] < 0] = -quatComps[i, :, quatComps[i, 0, :, :] < 0]

        # Arrays to store neigbour misorientation in positive x and y direction
        misOrix = np.zeros((numSyms, self.yDim, self.xDim))
        misOriy = np.zeros((numSyms, self.yDim, self.xDim))

        # loop over symmetries calculating misorientation to initial
        for i in range(numSyms):
            for j in range(self.xDim - 1):
                misOrix[i, :, j] = abs(np.einsum("ij,ij->j", quatComps[0, :, :, j], quatComps[i, :, :, j + 1]))

            for j in range(self.yDim - 1):
                misOriy[i, j, :] = abs(np.einsum("ij,ij->j", quatComps[0, :, j, :], quatComps[i, :, j + 1, :]))

        misOrix[misOrix > 1] = 1
        misOriy[misOriy > 1] = 1

        # find min misorientation (max here as misorientaion is cos of this)
        misOrix = np.max(misOrix, axis=0)
        misOriy = np.max(misOriy, axis=0)

        # convert to misorientation in degrees
        misOrix = 360 * np.arccos(misOrix) / np.pi
        misOriy = 360 * np.arccos(misOriy) / np.pi

        # set boundary locations where misOrix or misOriy are greater than set value
        self.boundaries = np.zeros((self.yDim, self.xDim), dtype=int)

        for i in range(self.xDim):
            for j in range(self.yDim):
                if (misOrix[j, i] > boundDef) or (misOriy[j, i] > boundDef):
                    self.boundaries[j, i] = -1

        print("\r", end="")
        return

    def findPhaseBoundaries(self):
        """Finds boundaries in the phase map
        """

        print("Finding phase boundaries...", end="")

        # make new array shifted by one to left and up
        phaseArrayShifted = np.full((self.yDim, self.xDim), -3)
        phaseArrayShifted[:-1, :-1] = self.phaseArray[1:, 1:]

        # where shifted array not equal to starting array, set to -1
        self.phaseBoundaries = np.zeros((self.yDim, self.xDim))
        self.phaseBoundaries = np.where(np.not_equal(self.phaseArray, phaseArrayShifted), -1, 0)

        print("\r", end="")

    def plotPhaseBoundaryMap(self, dilate=False):
        """Plots phase boundary map
        """

        plt.figure()

        boundariesImage = -self.phaseBoundaries

        if dilate:
            boundariesImage = mph.binary_dilation(-self.phaseBoundaries)

        plt.imshow(boundariesImage, vmax=1, cmap='gray')
        plt.colorbar()

    def plotBoundaryMap(self, dilate=False):
        """Plots grain boundary map
        """
        plt.figure()

        boundariesImage = -self.boundaries

        if dilate:
            boundariesImage = mph.binary_dilation(-self.boundaries)

        plt.imshow(boundariesImage, vmax=1, cmap='gray')
        plt.colorbar()

    def findGrains(self, minGrainSize=10):
        print("Finding grains...", end="")

        # Initialise the grain map
        self.grains = np.copy(self.boundaries)

        self.grainList = []

        # List of points where no grain has be set yet
        unknownPoints = np.where(self.grains == 0)
        # Start counter for grains
        grainIndex = 1

        # Loop until all points (except boundaries) have been assigned to a grain or ignored
        while unknownPoints[0].shape[0] > 0:
            # Flood fill first unknown point and return grain object
            currentGrain = self.floodFill(unknownPoints[1][0], unknownPoints[0][0], grainIndex)

            grainSize = len(currentGrain)
            if grainSize < minGrainSize:
                # if grain size less than minimum, ignore grain and set values in grain map to -2
                for coord in currentGrain.coordList:
                    self.grains[coord[1], coord[0]] = -2
            else:
                # add grain and size to lists and increment grain label
                self.grainList.append(currentGrain)
                grainIndex += 1

            # update unknown points
            unknownPoints = np.where(self.grains == 0)
        print("\r", end="")
        return

    def plotGrainMap(self):
        plt.figure()
        plt.imshow(self.grains)
        plt.colorbar()
        return

    def locateGrainID(self, clickEvent=None):
        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        # reset current selected grain and plot euler map with click handler
        self.currGrainId = None
        self.plotEulerMap()
        if clickEvent is None:
            # default click handler which highlights grain and prints id
            self.fig.canvas.mpl_connect('button_press_event', self.clickGrainId)
        else:
            # click handler loaded from linker classs. Pass current ebsd map to it.
            self.fig.canvas.mpl_connect('button_press_event', lambda x: clickEvent(x, self))

    def clickGrainId(self, event):
        if event.inaxes is not None:
            # grain id of selected grain
            self.currGrainId = int(self.grains[int(event.ydata), int(event.xdata)] - 1)
            print("Grain ID: {}".format(self.currGrainId))

            # clear current axis and redraw euler map with highlighted grain overlay
            self.ax.clear()
            self.plotEulerMap(updateCurrent=True, highlightGrains=[self.currGrainId])
            self.fig.canvas.draw()

    def floodFill(self, x, y, grainIndex):
        currentGrain = Grain(self)

        currentGrain.addPoint((x, y), self.quatArray[y, x])

        edge = [(x, y)]
        grain = [(x, y)]

        self.grains[y, x] = grainIndex
        while edge:
            newedge = []

            for (x, y) in edge:
                moves = np.array([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])

                movesIndexShift = 0
                if x <= 0:
                    moves = np.delete(moves, 1, 0)
                    movesIndexShift = 1
                elif x >= self.xDim - 1:
                    moves = np.delete(moves, 0, 0)
                    movesIndexShift = 1

                if y <= 0:
                    moves = np.delete(moves, 3 - movesIndexShift, 0)
                elif y >= self.yDim - 1:
                    moves = np.delete(moves, 2 - movesIndexShift, 0)

                for (s, t) in moves:
                    if self.grains[t, s] == 0:
                        currentGrain.addPoint((s, t), self.quatArray[t, s])
                        newedge.append((s, t))
                        grain.append((s, t))
                        self.grains[t, s] = grainIndex
                    elif self.grains[t, s] == -1 and (s > x or t > y):
                        currentGrain.addPoint((s, t), self.quatArray[t, s])
                        grain.append((s, t))
                        self.grains[t, s] = grainIndex

            if newedge == []:
                return currentGrain
            else:
                edge = newedge

    def calcGrainAvOris(self):
        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        for grain in self.grainList:
            grain.calcAverageOri()

    def calcGrainMisOri(self, calcAxis=False):
        print("Calculating grain misorientations...", end="")

        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        for grain in self.grainList:
            grain.buildMisOriList(calcAxis=calcAxis)

        print("\r", end="")
        return

    def plotMisOriMap(self, component=0, plotGBs=False, boundaryColour='black', vmin=None, vmax=None,
                      cmap="viridis", cBarLabel="ROD (degrees)"):
        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        self.misOri = np.ones([self.yDim, self.xDim])

        plt.figure()

        if component in [1, 2, 3]:
            for grain in self.grainList:
                for coord, misOriAxis in zip(grain.coordList, np.array(grain.misOriAxisList)):
                    self.misOri[coord[1], coord[0]] = misOriAxis[component - 1]

            plt.imshow(self.misOri * 180 / np.pi, interpolation='None', vmin=vmin, vmax=vmax, cmap=cmap)

        else:
            for grain in self.grainList:
                for coord, misOri in zip(grain.coordList, grain.misOriList):
                    self.misOri[coord[1], coord[0]] = misOri

            plt.imshow(np.arccos(self.misOri) * 360 / np.pi, interpolation='None',
                       vmin=vmin, vmax=vmax, cmap=cmap)

        plt.colorbar(label=cBarLabel)

        if plotGBs:
            self.plotGBs(colour=boundaryColour)

        return

    def loadSlipSystems(self, filepath, cOverA=None):
        self.slipSystems, self.slipTraceColours = base.SlipSystem.loadSlipSystems(filepath, self.crystalSym, cOverA=cOverA)

        if self.grainList is not None:
            for grain in self.grainList:
                grain.slipSystems = self.slipSystems

    def calcAverageGrainSchmidFactors(self, loadVector=np.array([0, 0, 1]), slipSystems=None):
        print("Calculating grain average Schmid factors...", end="")

        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        for grain in self.grainList:
            grain.calcAverageSchmidFactors(loadVector=loadVector, slipSystems=slipSystems)

        print("\r", end="")

    def plotAverageGrainSchmidFactorsMap(self, plotGBs=True):
        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        self.averageSchmidFactor = np.zeros([self.yDim, self.xDim])

        for grain in self.grainList:
            # max Schmid factor
            currentSchmidFactor = np.array(grain.averageSchmidFactors).flatten().max()
            # currentSchmidFactor = grain.averageSchmidFactors[0][0]
            for coord in grain.coordList:
                self.averageSchmidFactor[coord[1], coord[0]] = currentSchmidFactor

        self.averageSchmidFactor[self.averageSchmidFactor == 0] = 0.5

        plt.figure()
        plt.imshow(self.averageSchmidFactor, interpolation='none', cmap='gray', vmin=0, vmax=0.5)
        plt.colorbar(label="Schmid factor")

        if plotGBs:
            self.plotGBs()

        return


class Grain(base.Grain):

    def __init__(self, ebsdMap):
        # Call base class constructor
        super(Grain, self).__init__()

        self.crystalSym = ebsdMap.crystalSym    # symmetry of material e.g. "cubic", "hexagonal"
        self.slipSystems = ebsdMap.slipSystems
        self.ebsdMap = ebsdMap                  # ebsd map this grain is a member of
        self.quatList = []                      # list of quats
        self.misOriList = None                  # list of misOri at each point in grain
        self.misOriAxisList = None              # list of misOri axes at each point in grain
        self.refOri = None                      # (quat) average ori of grain
        self.averageMisOri = None               # average misOri of grain

        self.averageSchmidFactors = None        # list of list Schmid factors (grouped by slip plane)
        self.slipTraceAngles = None             # list of slip trace angles
        self.slipTraceInclinations = None

    # quat is a quaterion and coord is a tuple (x, y)
    def addPoint(self, coord, quat):
        self.coordList.append(coord)
        self.quatList.append(quat)

    def calcAverageOri(self):
        quatCompsSym = Quat.calcSymEqvs(self.quatList, self.crystalSym)

        self.refOri = Quat.calcAverageOri(quatCompsSym)

    def buildMisOriList(self, calcAxis=False):
        quatCompsSym = Quat.calcSymEqvs(self.quatList, self.crystalSym)

        if self.refOri is None:
            self.refOri = Quat.calcAverageOri(quatCompsSym)

        misOriArray, minQuatComps = Quat.calcMisOri(quatCompsSym, self.refOri)

        self.averageMisOri = misOriArray.mean()
        self.misOriList = list(misOriArray)

        if calcAxis:
            # Now for axis calulation
            refOriInv = self.refOri.conjugate

            misOriAxis = np.empty((3, minQuatComps.shape[1]))
            Dq = np.empty((4, minQuatComps.shape[1]))

            # refOriInv * minQuat for all points (* is quaternion product)
            # change to minQuat * refOriInv
            Dq[0, :] = (refOriInv[0] * minQuatComps[0, :] - refOriInv[1] * minQuatComps[1, :] -
                        refOriInv[2] * minQuatComps[2, :] - refOriInv[3] * minQuatComps[3, :])

            Dq[1, :] = (refOriInv[1] * minQuatComps[0, :] + refOriInv[0] * minQuatComps[1, :] +
                        refOriInv[3] * minQuatComps[2, :] - refOriInv[2] * minQuatComps[3, :])

            Dq[2, :] = (refOriInv[2] * minQuatComps[0, :] + refOriInv[0] * minQuatComps[2, :] +
                        refOriInv[1] * minQuatComps[3, :] - refOriInv[3] * minQuatComps[1, :])

            Dq[3, :] = (refOriInv[3] * minQuatComps[0, :] + refOriInv[0] * minQuatComps[3, :] +
                        refOriInv[2] * minQuatComps[1, :] - refOriInv[1] * minQuatComps[2, :])

            Dq[:, Dq[0] < 0] = -Dq[:, Dq[0] < 0]

            # numpy broadcasting taking care of different array sizes
            misOriAxis[:, :] = (2 * Dq[1:4, :] * np.arccos(Dq[0, :])) / np.sqrt(1 - np.power(Dq[0, :], 2))

            # hack it back into a list. Need to change self.*List to be arrays, it was a bad decision to
            # make them lists in the beginning
            self.misOriAxisList = []
            for row in misOriAxis.transpose():
                self.misOriAxisList.append(row)

    def plotRefOri(self, direction=np.array([0, 0, 1]), marker='+'):
        Quat.plotIPF([self.refOri], direction, self.crystalSym, marker=marker)

    def plotOriSpread(self, direction=np.array([0, 0, 1]), marker='.'):
        Quat.plotIPF(self.quatList, direction, self.crystalSym, marker=marker)

    # component
    # 0 = misOri
    # {1-3} = misOri axis {1-3}
    # 4 = all
    # 5 = all axis
    def plotMisOri(self, component=0, vmin=None, vmax=None, vRange=[None, None, None],
                   cmap=["viridis", "bwr"], plotSlipTraces=False):
        component = int(component)

        x0, y0, xmax, ymax = self.extremeCoords

        if component in [4, 5]:
            # subplots
            grainMisOri = np.full((4, ymax - y0 + 1, xmax - x0 + 1), np.nan, dtype=float)

            for coord, misOri, misOriAxis in zip(self.coordList,
                                                 np.arccos(self.misOriList) * 360 / np.pi,
                                                 np.array(self.misOriAxisList) * 180 / np.pi):
                grainMisOri[0, coord[1] - y0, coord[0] - x0] = misOri
                grainMisOri[1:4, coord[1] - y0, coord[0] - x0] = misOriAxis

            f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

            img = ax1.imshow(grainMisOri[0], interpolation='none', cmap=cmap[0], vmin=vmin, vmax=vmax)
            plt.colorbar(img, ax=ax1, label="Grain misorientation ($^\circ$)")
            vmin = None if vRange[0] is None else -vRange[0]
            img = ax2.imshow(grainMisOri[1], interpolation='none', cmap=cmap[1], vmin=vmin, vmax=vRange[0])
            plt.colorbar(img, ax=ax2, label="x rotation ($^\circ$)")
            vmin = None if vRange[0] is None else -vRange[1]
            img = ax3.imshow(grainMisOri[2], interpolation='none', cmap=cmap[1], vmin=vmin, vmax=vRange[1])
            plt.colorbar(img, ax=ax3, label="y rotation ($^\circ$)")
            vmin = None if vRange[0] is None else -vRange[2]
            img = ax4.imshow(grainMisOri[3], interpolation='none', cmap=cmap[1], vmin=vmin, vmax=vRange[2])
            plt.colorbar(img, ax=ax4, label="z rotation ($^\circ$)")

            for ax in (ax1, ax2, ax3, ax4):
                ax.set_xticks([])
                ax.set_yticks([])

        else:
            # single plot
            # initialise array with nans so area not in grain displays white
            grainMisOri = np.full((ymax - y0 + 1, xmax - x0 + 1), np.nan, dtype=float)

            if component in [1, 2, 3]:
                plotData = np.array(self.misOriAxisList)[:, component - 1] * 180 / np.pi
            else:
                plotData = np.arccos(self.misOriList) * 360 / np.pi

            for coord, misOri in zip(self.coordList, plotData):
                grainMisOri[coord[1] - y0, coord[0] - x0] = misOri

            plt.figure()
            plt.imshow(grainMisOri, interpolation='none', vmin=vmin, vmax=vmax, cmap=cmap[0])

            plt.colorbar(label="ROD (degrees)")
            plt.xticks([])
            plt.yticks([])

            if plotSlipTraces:
                self.plotSlipTraces()

        return

    # define load axis as unit vector
    def calcAverageSchmidFactors(self, loadVector=np.array([0, 0, 1]), slipSystems=None):
        if slipSystems is None:
            slipSystems = self.slipSystems
        if self.refOri is None:
            self.calcAverageOri()

        # orientation of grain
        grainAvOri = self.refOri

        # Transform the load vector into crystal coordinates
        loadVectorCrystal = grainAvOri.transformVector(loadVector)

        self.averageSchmidFactors = []
        # flatten list of lists
        # slipSystems = chain.from_iterable(slipSystems)

        # Loop over groups of slip systems with same slip plane
        for i, slipSystemGroup in enumerate(slipSystems):
            self.averageSchmidFactors.append([])
            # Then loop over individual slip systems
            for slipSystem in slipSystemGroup:
                schmidFactor = abs(np.dot(loadVectorCrystal, slipSystem.slipPlane) *
                                   np.dot(loadVectorCrystal, slipSystem.slipDir))
                self.averageSchmidFactors[i].append(schmidFactor)

        return

    @property
    def slipTraces(self):
        if self.slipTraceAngles is None:
            self.calcSlipTraces()

        return self.slipTraceAngles

    def calcSlipTraces(self, slipSystems=None):
        if slipSystems is None:
            slipSystems = self.slipSystems
        if self.refOri is None:
            self.calcAverageOri()

        screenPlaneNorm = np.array((0, 0, 1))   # in sample frame

        grainAvOri = self.refOri   # orientation of grain

        screenPlaneNormCrystal = grainAvOri.transformVector(screenPlaneNorm)

        self.slipTraceAngles = []
        self.slipTraceInclinations = []
        # Loop over each group of slip systems
        for slipSystemGroup in slipSystems:
            # Take slip plane from first in group
            slipPlaneNorm = slipSystemGroup[0].slipPlane
            # planeLabel = slipSystemGroup[0].slipPlaneLabel

            # Calculate intersection of slip plane with plane of screen
            intersectionCrystal = np.cross(screenPlaneNormCrystal, slipPlaneNorm)

            # Calculate angle between slip plane and screen plane
            inclination = np.arccos(np.dot(screenPlaneNormCrystal, slipPlaneNorm))
            if inclination > np.pi / 2:
                inclination = np.pi - inclination
            # print("{} inclination: {:.1f}".format(planeLabel, inclination * 180 / np.pi))

            # Transform intersection back into sample coordinates
            intersection = grainAvOri.conjugate.transformVector(intersectionCrystal)
            intersection = intersection / np.sqrt(np.dot(intersection, intersection))  # normalise

            # Calculate trace angle. Starting vertical and proceeding counter clockwise
            if intersection[0] > 0:
                intersection *= -1
            traceAngle = np.arccos(np.dot(intersection, np.array([0, 1.0, 0])))

            # Append to list
            self.slipTraceAngles.append(traceAngle)
            self.slipTraceInclinations.append(inclination)


class Linker(object):
    """Class for linking multiple ebsd maps of the same region for analysis of deformation

    Attributes:
        ebsdMaps (list(ebsd.Map)): List of ebsd.Map objects that are linked
        links (list): List of grain link. Each link is stored as a tuple of grain IDs (one from each
                      map stored in same order of maps)
        numMaps (TYPE): Number of linked maps
    """

    def __init__(self, maps):
        self.ebsdMaps = maps
        self.numMaps = len(maps)
        self.links = []
        return

    def setOrigin(self):
        for ebsdMap in self.ebsdMaps:
            ebsdMap.locateGrainID(clickEvent=self.clickSetOrigin)

    def clickSetOrigin(self, event, currentEbsdMap):
        currentEbsdMap.origin = (int(event.ydata), int(event.xdata))
        print("Origin set to ({:}, {:})".format(currentEbsdMap.origin[0], currentEbsdMap.origin[1]))

    def startLinking(self):
        for ebsdMap in self.ebsdMaps:
            ebsdMap.locateGrainID(clickEvent=self.clickGrainGuess)

            # Add make link button to axes
            btnAx = ebsdMap.fig.add_axes([0.8, 0.0, 0.1, 0.07])
            Button(btnAx, 'Make link', color='0.85', hovercolor='0.95')

    def clickGrainGuess(self, event, currentEbsdMap):
        # self is cuurent linker instance even if run as click event handler from map class
        if event.inaxes is currentEbsdMap.fig.axes[0]:
            # axis 0 then is a click on the map

            if currentEbsdMap is self.ebsdMaps[0]:
                # clicked on 'master' map so highlight and guess grain on other maps
                for ebsdMap in self.ebsdMaps:
                    if ebsdMap is currentEbsdMap:
                        # set current grain in ebsd map that clicked
                        ebsdMap.clickGrainId(event)
                    else:
                        # Guess at grain in other maps
                        # Calculated position relative to set origin of the map, scaled from step size of maps
                        y0m = currentEbsdMap.origin[0]
                        x0m = currentEbsdMap.origin[1]
                        y0 = ebsdMap.origin[0]
                        x0 = ebsdMap.origin[1]
                        scaling = currentEbsdMap.stepSize / ebsdMap.stepSize

                        x = int((event.xdata - x0m) * scaling + x0)
                        y = int((event.ydata - y0m) * scaling + y0)

                        ebsdMap.currGrainId = int(ebsdMap.grains[y, x]) - 1
                        print(ebsdMap.currGrainId)

                        # clear current axis and redraw euler map with highlighted grain overlay
                        ebsdMap.ax.clear()
                        ebsdMap.plotEulerMap(updateCurrent=True, highlightGrains=[ebsdMap.currGrainId])
                        ebsdMap.fig.canvas.draw()
            else:
                # clicked on other map so correct guessed selected grain
                currentEbsdMap.clickGrainId(event)

        elif event.inaxes is currentEbsdMap.fig.axes[1]:
            # axis 1 then is a click on the button
            self.makeLink()

    def makeLink(self):
        # create empty list for link
        currLink = []

        for i, ebsdMap in enumerate(self.ebsdMaps):
            if ebsdMap.currGrainId is not None:
                currLink.append(ebsdMap.currGrainId)
            else:
                raise Exception("No grain setected in map {:d}.".format(i + 1))

        self.links.append(tuple(currLink))

        print("Link added " + str(tuple(currLink)))

    def resetLinks(self):
        self.links = []

#   Analysis routines

    def setAvOriFromInitial(self):
        masterMap = self.ebsdMaps[0]

        # loop over each map (not first/refernece) and each link. Set refOri of linked grains
        # to refOri of grain in first map
        for i, ebsdMap in enumerate(self.ebsdMaps[1:], start=1):
            for link in self.links:
                ebsdMap.grainList[link[i]].refOri = copy.deepcopy(masterMap.grainList[link[0]].refOri)

        return

    def updateMisOri(self, calcAxis=False):
        # recalculate misorientation for linked grain (not for first map)
        for i, ebsdMap in enumerate(self.ebsdMaps[1:], start=1):
            for link in self.links:
                ebsdMap.grainList[link[i]].buildMisOriList(calcAxis=calcAxis)

        return
