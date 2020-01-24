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
from matplotlib.widgets import Button
from skimage import morphology as mph

import copy
import warnings

from defdap.file_readers import EBSDDataLoader
from defdap.quat import Quat
from defdap.crystal import SlipSystem
from defdap import base

from defdap.plotting import MapPlot, GrainPlot
from defdap.utils import reportProgress


class Map(base.Map):
    """
    Class to encapsulate EBSD data and useful analysis and plotting
    methods.

    Attributes
    ----------
    crystalSym : str
        symmetry of material e.g. "cubic", "hexagonal"
    xDim : int
        size of map in x direction
    yDim : int
        size of map in y direction
    stepSize : float
        step size
    eulerAngleArray
    bandContrastArray
    quatArray : numpy.ndarray
        array of quaterions for each point of map
    numPhases : int
        number of phases
    phaseArray : numpy.ndarray
        map of phase ids
    phaseNames : list(str)
        list of phase names
    boundaries : numpy.ndarray
        map of boundaries. -1 for a boundary, 0 otherwise
    phaseBoundaries : numpy.ndarray
        map of phase boundaries. -1 for boundary, 0 otherwise
    cacheEulerMap
    grains : numpy.ndarray
        map of grains. Grain numbers start at 1 here but everywhere else
        grainID starts at 0. Regions that are smaller than the minimum
        grain size are given value -2.
    grainList : list(defdap.ebsd.Grain)
        list of grains
    misOri : numpy.ndarray
        map of misorientation
    misOriAxis : list(numpy.ndarray)
        map of misorientation axis components
    kam : numpy.ndarray
        map of kam
    averageSchmidFactor : numpy.ndarray
        map of average Schmid factor
    slipSystems : list(list(slipSystems))
        slip systems grouped by slip plane
    slipTraceColours list(str)
        colours used when plotting slip traces
    currGrainId : int
        ID of last selected grain
    origin : tuple(int)
        Map origin (y, x). Used by linker class where origin is a
        homologue point of the maps
    GND
        GND scalar map
    Nye
        3x3 Nye tensor at each point
    fig
    ax
    """

    def __init__(self, fileName, crystalSym, cOverA=None, dataType=None):
        """
        Initialise class and load EBSD data

        Parameters
        ----------
        fileName : str
            Path to EBSD file, including name, excluding extension
        crystalSym : str, {'cubic', 'hexagonal'}
            Crystal structure
        dataType : str, {'OxfordBinary', 'OxfordText'}
            Format of EBSD data file
        """
        # Call base class constructor
        super(Map, self).__init__()

        self.crystalSym = None
        self.cOverA = None
        self.xDim = None
        self.yDim = None
        self.stepSize = None
        self.eulerAngleArray = None
        self.bandContrastArray = None
        self.quatArray = None
        self.numPhases = None
        self.phaseArray = None
        self.phaseNames = []
        self.boundaries = None
        self.phaseBoundaries = None
        self.cacheEulerMap = None
        self.grains = None
        self.misOri = None
        self.misOriAxis = None
        self.kam = None
        self.averageSchmidFactor = None
        self.slipSystems = None
        self.slipTraceColours = None
        self.currGrainId = None
        self.origin = (0, 0)
        self.GND = None
        self.Nye = None

        # Use euler map for defining homologous points
        self.plotHomog = self.plotEulerMap
        self.highlightAlpha = 1

        self.loadData(fileName, crystalSym, cOverA, dataType=dataType)

    @property
    def plotDefault(self):
        # return self.plotEulerMap(*args, **kwargs)
        return lambda *args, **kwargs: self.plotEulerMap(*args, **kwargs)

    @reportProgress("loading EBSD data")
    def loadData(self, fileName, crystalSym, cOverA, dataType=None):
        """
        Load in EBSD data

        Parameters
        ----------
        fileName : str
            Path to EBSD file, including name, excluding extension
        crystalSym : str, {'cubic', 'hexagonal'}
            Crystal structure
        dataType : str, {'OxfordBinary', 'OxfordText'}
            Format of EBSD data file
        """
        if dataType is None:
            dataType = "OxfordBinary"

        dataLoader = EBSDDataLoader()
        if dataType == "OxfordBinary":
            metadataDict = dataLoader.loadOxfordCPR(fileName)
            dataDict = dataLoader.loadOxfordCRC(fileName)
        elif dataType == "OxfordText":
            metadataDict, dataDict = dataLoader.loadOxfordCTF(fileName)
        else:
            raise Exception("No loader found for this EBSD data.")

        self.xDim = metadataDict['xDim']
        self.yDim = metadataDict['yDim']
        self.stepSize = metadataDict['stepSize']
        self.numPhases = metadataDict['numPhases']
        self.phaseNames = metadataDict['phaseNames']

        self.eulerAngleArray = dataDict['eulerAngle']
        self.bandContrastArray = dataDict['bandContrast']
        self.phaseArray = dataDict['phase']

        self.crystalSym = crystalSym
        
        if self.crystalSym == 'hexagonal':
            if cOverA is None:
                warnings.warn("No c/a ratio given. Using ideal ratio 1.633")
                cOverA = 1.633
            self.cOverA = cOverA

        # write final status
        yield "Loaded EBSD data (dimensions: {:} x {:} pixels, step " \
              "size: {:} um)".format(self.xDim, self.yDim, self.stepSize)

    @property
    def scale(self):
        return self.stepSize

    @reportProgress("transforming EBSD data")
    def transformData(self):
        """
        Rotate map by 180 degrees and transform quats
        """
        self.eulerAngleArray = self.eulerAngleArray[:, ::-1, ::-1]
        self.bandContrastArray = self.bandContrastArray[::-1, ::-1]
        self.phaseArray = self.phaseArray[::-1, ::-1]
        self.buildQuatArray()
        
        transformQuat = Quat.fromAxisAngle(np.array([0, 0, 1]), np.pi)
        for i in range(self.xDim):
            for j in range(self.yDim):
                self.quatArray[j, i] = self.quatArray[j, i] * transformQuat

            # report progress
            yield i / self.xDim

    def plotBandContrastMap(self, **kwargs):
        """
        Plot band contrast map
        """
        self.checkDataLoaded()

        # Set default plot parameters then update with any input
        plotParams = {
            'plotColourBar': True,
            'cmap:': 'grey',
            'cLabel': "Band contrast"
        }
        plotParams.update(kwargs)

        plot = MapPlot.create(self, self.bandContrastArray, **plotParams)

        return plot

    def plotEulerMap(self, **kwargs):
        """
        Plot an orientation map in Euler colouring

        Parameters
        ----------
        ax
        makeInteractive
        plotGBs
        dilateBoundaries
        boundaryColour
        plotScaleBar
        kwargs
        updateCurrent : bool, optional

        highlightGrains : iterable(int), optional
            List of grain ids to highlight
        highlightColours : str, optional
            Colour of list of colours to highlight grains. If less
            colours are given than grains, then the final colour is
            used for the remaining grains.
        """
        self.checkDataLoaded()

        # Set default plot parameters then update with any input
        plotParams = {}
        plotParams.update(kwargs)

        eulerMap = np.transpose(self.eulerAngleArray, axes=(1, 2, 0))
        # this is the normalisation - different foreach crystal symmetry!
        if self.crystalSym == 'cubic':
            norm = np.tile(np.array([2 * np.pi, np.pi / 2, np.pi / 2]),
                           (self.yDim, self.xDim))
            norm = np.reshape(norm, (self.yDim, self.xDim, 3))
        elif self.crystalSym == 'hexagonal':
            norm = np.tile(np.array([np.pi, np.pi, np.pi / 3]),
                           (self.yDim, self.xDim))
            norm = np.reshape(norm, (self.yDim, self.xDim, 3))
        else:
            Exception("Only hexagonal and cubic symGroup supported")
        # make non-indexed points green
        eulerMap = np.where(eulerMap != [0., 0., 0.], eulerMap, [0., 1., 0.])
        eulerMap /= norm

        plot = MapPlot.create(self, eulerMap, **plotParams)

        return plot

    def plotIPFMap(self, direction, **kwargs):
        # Set default plot parameters then update with any input
        plotParams = {}
        plotParams.update(kwargs)

        # calculate IPF colours
        IPFcolours = Quat.calcIPFcolours(
            self.quatArray.flatten(),
            direction,
            self.crystalSym
        )
        # reshape back to map shape array
        IPFcolours = np.reshape(IPFcolours, (self.yDim, self.xDim, 3))

        plot = MapPlot.create(self, IPFcolours, **plotParams)

        return plot

    def plotPhaseMap(self, **kwargs):
        """
        Plot a phase map.

        Parameters
        ----------
        cmap : str, optional
            Colour scale to plot with.
        """
        # Set default plot parameters then update with any input
        plotParams = {
            'vmin': -1,
            'vmax': self.numPhases
        }
        plotParams.update(kwargs)

        plot = MapPlot.create(self, self.phaseArray, **plotParams)

        # add a legend to the plot
        phaseIDs = [-1] + list(range(1, self.numPhases + 1))
        phaseNames = ["Non-indexed"] + self.phaseNames
        plot.addLegend(phaseIDs, phaseNames,
                       loc=2, borderaxespad=0.)

        return plot

    def calcKam(self):
        """
        Calculates Kernel Average Misorientaion (KAM) for the EBSD map.
        Crystal symmetric equivalences are not considered. Stores
        result in self.kam.
        """
        quatComps = np.empty((4, self.yDim, self.xDim))

        for i, row in enumerate(self.quatArray):
            for j, quat in enumerate(row):
                quatComps[:, i, j] = quat.quatCoef

        self.kam = np.empty((self.yDim, self.xDim))

        # Start with rows. Calculate misorientation with neighbouring rows.
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

    def plotKamMap(self, **kwargs):
        """
        Plot Kernel Average Misorientaion (KAM) for the EBSD map.

        Parameters
        ----------
        vmin : float, optional
            Minimum of colour scale
        vmax : float, optional
            Maximum of colour scale
        cmap : str, optional
            Colour scale to plot with.
        """
        # Set default plot parameters then update with any input
        plotParams = {
            'plotColourBar': True,
            'cLabel': "Kernel average misorientation (KAM) ($^\circ$)"
        }
        plotParams.update(kwargs)

        self.calcKam()
        # Convert to degrees and plot
        kam = 2 * np.arccos(self.kam) * 180 / np.pi

        plot = MapPlot.create(self, kam, **plotParams)

        return plot

    @reportProgress("calculating Nye tensor")
    def calcNye(self):
        """
        Calculates Nye tensor and related GND density for the EBSD map.
        Stores result in self.Nye and self.GND.
        """
        self.buildQuatArray()
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
            quatComps[i, 0] = (quatComps[0, 0] * sym[0] - quatComps[0, 1] * sym[1] -
                               quatComps[0, 2] * sym[2] - quatComps[0, 3] * sym[3])
            quatComps[i, 1] = (quatComps[0, 0] * sym[1] + quatComps[0, 1] * sym[0] -
                               quatComps[0, 2] * sym[3] + quatComps[0, 3] * sym[2])
            quatComps[i, 2] = (quatComps[0, 0] * sym[2] + quatComps[0, 2] * sym[0] -
                               quatComps[0, 3] * sym[1] + quatComps[0, 1] * sym[3])
            quatComps[i, 3] = (quatComps[0, 0] * sym[3] + quatComps[0, 3] * sym[0] -
                               quatComps[0, 1] * sym[2] + quatComps[0, 2] * sym[1])

            # swap into positve hemisphere if required
            quatComps[i, :, quatComps[i, 0] < 0] *= -1

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
        argmisOrix = np.argmax(misOrix, axis=0)
        argmisOriy = np.argmax(misOriy, axis=0)
        misOrix = np.max(misOrix, axis=0)
        misOriy = np.max(misOriy, axis=0)

        # convert to misorientation in degrees
        misOrix = 360 * np.arccos(misOrix) / np.pi
        misOriy = 360 * np.arccos(misOriy) / np.pi

        # calculate relative elastic distortion tensors at each point in the two directions
        betaderx = np.zeros((3, 3, self.yDim, self.xDim))
        betadery = betaderx
        for i in range(self.xDim - 1):
            for j in range(self.yDim - 1):
                q0x = Quat(quatComps[0, 0, j, i], quatComps[0, 1, j, i],
                           quatComps[0, 2, j, i], quatComps[0, 3, j, i])
                qix = Quat(quatComps[argmisOrix[j, i], 0, j, i + 1],
                           quatComps[argmisOrix[j, i], 1, j, i + 1],
                           quatComps[argmisOrix[j, i], 2, j, i + 1],
                           quatComps[argmisOrix[j, i], 3, j, i + 1])
                misoquatx = qix.conjugate * q0x
                # change stepsize to meters
                betaderx[:, :, j, i] = (Quat.rotMatrix(misoquatx) - np.eye(3)) / self.stepSize / 1e-6
                q0y = Quat(quatComps[0, 0, j, i], quatComps[0, 1, j, i],
                           quatComps[0, 2, j, i], quatComps[0, 3, j, i])
                qiy = Quat(quatComps[argmisOriy[j, i], 0, j + 1, i],
                           quatComps[argmisOriy[j, i], 1, j + 1, i],
                           quatComps[argmisOriy[j, i], 2, j + 1, i],
                           quatComps[argmisOriy[j, i], 3, j + 1, i])
                misoquaty = qiy.conjugate * q0y
                # change stepsize to meters
                betadery[:, :, j, i] = (Quat.rotMatrix(misoquaty) - np.eye(3)) / self.stepSize / 1e-6

        # Calculate the Nye Tensor
        alpha = np.empty((3, 3, self.yDim, self.xDim))
        bavg = 1.4e-10  # Burgers vector
        alpha[0, 2] = (betadery[0, 0] - betaderx[0, 1]) / bavg
        alpha[1, 2] = (betadery[1, 0] - betaderx[1, 1]) / bavg
        alpha[2, 2] = (betadery[2, 0] - betaderx[2, 1]) / bavg
        alpha[:, 1] = betaderx[:, 2] / bavg
        alpha[:, 0] = -1 * betadery[:, 2] / bavg

        # Calculate 3 possible L1 norms of Nye tensor for total
        # disloction density
        alpha_total3 = np.empty((self.yDim, self.xDim))
        alpha_total5 = np.empty((self.yDim, self.xDim))
        alpha_total9 = np.empty((self.yDim, self.xDim))
        alpha_total3 = 30 / 10. *(
                abs(alpha[0, 2]) + abs(alpha[1, 2]) +
                abs(alpha[2, 2])
        )
        alpha_total5 = 30 / 14. * (
                abs(alpha[0, 2]) + abs(alpha[1, 2]) + abs(alpha[2, 2]) +
                abs(alpha[1, 0]) + abs(alpha[0, 1])
        )
        alpha_total9 = 30 / 20. * (
                abs(alpha[0, 2]) + abs(alpha[1, 2]) + abs(alpha[2, 2]) +
                abs(alpha[0, 0]) + abs(alpha[1, 0]) + abs(alpha[2, 0]) +
                abs(alpha[0, 1]) + abs(alpha[1, 1]) + abs(alpha[2, 1])
        )
        alpha_total3[abs(alpha_total3) < 1] = 1e12
        alpha_total5[abs(alpha_total3) < 1] = 1e12
        alpha_total9[abs(alpha_total3) < 1] = 1e12

        # choose from the different alpha_totals according to preference;
        # see Ruggles GND density paper
        self.GND = alpha_total9
        self.Nye = alpha

        yield 1.

    def plotGNDMap(self, **kwargs):
        # Set default plot parameters then update with any input
        plotParams = {
            'plotColourBar': True,
            'cLabel': "Geometrically necessary dislocation (GND) content"
        }
        plotParams.update(kwargs)

        self.calcNye()

        plot = MapPlot.create(self, np.log10(self.GND), **plotParams)

        return plot

    def checkDataLoaded(self):
        """ Checks if EBSD data is loaded

        :return: True if data loaded
        """
        if self.eulerAngleArray is None:
            raise Exception("Data not loaded")
        return True

    @reportProgress("building quaternion array")
    def buildQuatArray(self):
        """
        Build quaternion array
        """
        self.checkDataLoaded()

        if self.quatArray is None:
            # create the array of quat objects
            self.quatArray = Quat.createManyQuats(self.eulerAngleArray)

        yield 1.

    @reportProgress("finding grain boundaries")
    def findBoundaries(self, boundDef=10):
        """
        Find grain boundaries

        :param boundDef: critical misorientation
        :type boundDef: float
        """
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
            quatComps[i, 0] = (quatComps[0, 0] * sym[0] - quatComps[0, 1] * sym[1] -
                               quatComps[0, 2] * sym[2] - quatComps[0, 3] * sym[3])
            quatComps[i, 1] = (quatComps[0, 0] * sym[1] + quatComps[0, 1] * sym[0] -
                               quatComps[0, 2] * sym[3] + quatComps[0, 3] * sym[2])
            quatComps[i, 2] = (quatComps[0, 0] * sym[2] + quatComps[0, 2] * sym[0] -
                               quatComps[0, 3] * sym[1] + quatComps[0, 1] * sym[3])
            quatComps[i, 3] = (quatComps[0, 0] * sym[3] + quatComps[0, 3] * sym[0] -
                               quatComps[0, 1] * sym[2] + quatComps[0, 2] * sym[1])

            # swap into positive hemisphere if required
            quatComps[i, :, quatComps[i, 0] < 0] *= -1

        # Arrays to store neighbour misorientation in positive x and y
        # directions
        misOrix = np.ones((numSyms, self.yDim, self.xDim))
        misOriy = np.ones((numSyms, self.yDim, self.xDim))

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
        misOrix = 2 * np.arccos(misOrix) * 180 / np.pi
        misOriy = 2 * np.arccos(misOriy) * 180 / np.pi

        # set boundary locations where misOrix or misOriy are greater
        # than set value
        self.boundariesX = misOrix > boundDef
        self.boundariesY = misOriy > boundDef
        self.misOriX = misOrix
        self.misOriY = misOriy
        self.boundaries = np.logical_or(self.boundariesX, self.boundariesY)
        self.boundaries = -self.boundaries.astype(int)

        yield 1.

    @reportProgress("finding phase boundaries")
    def findPhaseBoundaries(self, treatNonIndexedAs=None):
        """Finds boundaries in the phase map

        :param treatNonIndexedAs: value to assign to non-indexed points, defaults to -1
        """
        # make new array shifted by one to left and up
        phaseArrayShifted = np.full((self.yDim, self.xDim), -3)
        phaseArrayShifted[:-1, :-1] = self.phaseArray[1:, 1:]

        if treatNonIndexedAs:
            self.phaseArray[self.phaseArray == -1] = treatNonIndexedAs
            phaseArrayShifted[phaseArrayShifted == -1] = treatNonIndexedAs

        # where shifted array not equal to starting array, set to -1
        self.phaseBoundaries = np.zeros((self.yDim, self.xDim))
        self.phaseBoundaries = np.where(np.not_equal(self.phaseArray, phaseArrayShifted), -1, 0)

        yield 1.

    def plotPhaseBoundaryMap(self, dilate=False, **kwargs):
        """Plot phase boundary map

        :param dilate: Dilate boundary by one pixel
        """
        # Set default plot parameters then update with any input
        plotParams = {
            'vmax': 1,
            'plotColourBar': True,
            'cmap': 'grey'
        }
        plotParams.update(kwargs)

        boundariesImage = -self.phaseBoundaries
        if dilate:
            boundariesImage = mph.binary_dilation(boundariesImage)

        plot = MapPlot.create(self, boundariesImage, **plotParams)

        return plot

    def plotBoundaryMap(self, **kwargs):
        """Plot grain boundary map

        :param dilate: Dilate boundary by one pixel
        """
        # Set default plot parameters then update with any input
        plotParams = {
            'plotGBs': True,
            'boundaryColour': 'black'
        }
        plotParams.update(kwargs)

        plot = MapPlot.create(self, None, **plotParams)

        return plot

    @reportProgress("finding grains")
    def findGrains(self, minGrainSize=10):
        """
        Find grains and assign ids

        :param minGrainSize: Minimum grain area in pixels
        """
        # Initialise the grain map
        self.grains = np.copy(self.boundaries)

        self.grainList = []

        # List of points where no grain has be set yet
        unknownPoints = np.where(self.grains == 0)
        numPoints = unknownPoints[0].shape[0]
        totalPoints = numPoints
        # Start counter for grains
        grainIndex = 1

        # Loop until all points (except boundaries) have been assigned
        # to a grain or ignored
        while numPoints > 0:
            # report progress
            yield 1. - numPoints / totalPoints

            # Flood fill first unknown point and return grain object
            currentGrain = self.floodFill(unknownPoints[1][0], unknownPoints[0][0], grainIndex)

            grainSize = len(currentGrain)
            if grainSize < minGrainSize:
                # if grain size less than minimum, ignore grain and set
                # values in grain map to -2
                for coord in currentGrain.coordList:
                    self.grains[coord[1], coord[0]] = -2
            else:
                # add grain and size to lists and increment grain label
                self.grainList.append(currentGrain)
                grainIndex += 1

            # update unknown points
            unknownPoints = np.where(self.grains == 0)
            numPoints = unknownPoints[0].shape[0]

    def plotGrainMap(self, **kwargs):
        """
        Plot a map with grains coloured

        :return: Figure
        """
        # Set default plot parameters then update with any input
        plotParams = {
            'cLabel': "Grain number"
        }
        plotParams.update(kwargs)

        plot = MapPlot.create(self, self.grains, **plotParams)

        return plot

    def floodFill(self, x, y, grainIndex):
        """Flood fill algorithm that uses the x and y boundary arrays to
        fill a connected area around the seed point. The points are inserted
        into a grain object and the grain map array is updated.

        Parameters
        ----------
        x : int
            Seed point x for flood fill
        y : int
            Seed point y for flood fill
        grainIndex : int
            Value to fill in grain map

        Returns
        -------
        currentGrain : defdap.ebsd.Grain
            New grain object with points added
        """
        # create new grain
        currentGrain = Grain(self)

        # add first point to the grain
        currentGrain.addPoint((x, y), self.quatArray[y, x])
        self.grains[y, x] = grainIndex
        edge = [(x, y)]

        while edge:
            x, y = edge.pop(0)

            moves = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

            # get rid of any that go out of the map area
            if x <= 0:
                moves.pop(1)
            elif x >= self.xDim - 1:
                moves.pop(0)
            if y <= 0:
                moves.pop(-1)
            elif y >= self.yDim - 1:
                moves.pop(-2)

            for (s, t) in moves:
                if self.grains[t, s] > 0:
                    continue

                addPoint = False

                if t == y:
                    # moving horizontally
                    if s > x:
                        # moving right
                        addPoint = not self.boundariesX[y, x]
                    else:
                        # moving left
                        addPoint = not self.boundariesX[t, s]
                else:
                    # moving vertically
                    if t > y:
                        # moving down
                        addPoint = not self.boundariesY[y, x]
                    else:
                        # moving up
                        addPoint = not self.boundariesY[t, s]

                if addPoint:
                    currentGrain.addPoint((s, t), self.quatArray[t, s])
                    self.grains[t, s] = grainIndex
                    edge.append((s, t))

        return currentGrain

    @reportProgress("calculating grain mean orientations")
    def calcGrainAvOris(self):
        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        numGrains = len(self)
        for iGrain, grain in enumerate(self):
            grain.calcAverageOri()

            # report progress
            yield (iGrain + 1) / numGrains

    @reportProgress("calculating grain misorientations")
    def calcGrainMisOri(self, calcAxis=False):
        """
        Calculate grain misorientation

        :param calcAxis: Calculate the misorientation axis also
        :return:
        """
        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        numGrains = len(self)
        for iGrain, grain in enumerate(self):
            grain.buildMisOriList(calcAxis=calcAxis)

            # report progress
            yield (iGrain + 1) / numGrains

    def plotMisOriMap(self, component=0, **kwargs):
        """
        Plot misorientation map

        :param component: 0: misorientation, 1, 2, 3: rotation about x, y, z
        :param plotGBs: Plot grain boundaries
        :param boundaryColour: Colour of grain boundary
        :param vmin: Minimum of colour scale (optional)
        :type vmin: float
        :param vmax: Maximum of colour scale (optional)
        :type vmax: float
        :param cmap: Colour map (optional)
        :type cmap: str
        :param cBarLabel: Label for colour bar
        :return: Figure
        """

        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        self.misOri = np.ones([self.yDim, self.xDim])

        if component in [1, 2, 3]:
            for grain in self.grainList:
                for coord, misOriAxis in zip(grain.coordList, np.array(grain.misOriAxisList)):
                    self.misOri[coord[1], coord[0]] = misOriAxis[component - 1]

            misOri = self.misOri * 180 / np.pi
            cLabel = "Rotation around {:} axis ($^\circ$)".format(
                ['X', 'Y', 'Z'][component-1]
            )
        else:
            for grain in self.grainList:
                for coord, misOri in zip(grain.coordList, grain.misOriList):
                    self.misOri[coord[1], coord[0]] = misOri

            misOri = np.arccos(self.misOri) * 360 / np.pi
            cLabel = "Grain reference orienation deviation (GROD) ($^\circ$)"

        # Set default plot parameters then update with any input
        plotParams = {
            'plotColourBar': True,
            'cLabel': cLabel
        }
        plotParams.update(kwargs)

        plot = MapPlot.create(self, misOri, **plotParams)

        return plot

    def loadSlipSystems(self, name):
        """
        Load slip system definitions from file

        :param name: name of the slip system file (without file
        extension) stored in the defdap install dir or path to a file
        """
        self.slipSystems, self.slipTraceColours = SlipSystem.loadSlipSystems(
            name, self.crystalSym, cOverA=self.cOverA
        )

        if self.grainList is not None:
            for grain in self.grainList:
                grain.slipSystems = self.slipSystems

    def printSlipSystems(self):
        """
        Print a list of slip planes (with colours) and slip directions
        """
        for i, (ssGroup, colour) in enumerate(zip(self.slipSystems,
                                                  self.slipTraceColours)):
            print('Plane {0}: {1}\tColour: {2}'.format(
                i, ssGroup[0].slipPlaneLabel, colour
            ))
            for j, ss in enumerate(ssGroup):
                print('  Direction {0}: {1}'.format(j, ss.slipDirLabel))

    @reportProgress("calculating grain average Schmid factors")
    def calcAverageGrainSchmidFactors(self, loadVector, slipSystems=None):
        """
        Calculates Schmid factors for all slip systems, for all grains,
        based on average grain orientation

        :param loadVector: Loading vector, e.g. [1, 0, 0]
        :param slipSystems: Slip systems
        """
        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        numGrains = len(self)
        for iGrain, grain in enumerate(self.grainList):
            grain.calcAverageSchmidFactors(loadVector, slipSystems=slipSystems)

            # report progress
            yield (iGrain + 1) / numGrains

    def plotAverageGrainSchmidFactorsMap(self, planes=None, directions=None,
                                         **kwargs):
        """
        Plot maximum Schmid factor map, based on average grain
        orientation (for all slip systems unless specified)

        :param planes: Plane ID(s) to consider (optional)
        :type planes: list
        :param directions: Direction ID(s) to consider (optional)
        :type directions: list
        :param plotGBs: Plots grain boundaries if True
        :param boundaryColour:  Colour of grain boundaries
        :param dilateBoundaries: Dilates grain boundaries if True
        :type boundaryColour: string
        :return:
        """
        # Set default plot parameters then update with any input
        plotParams = {
            'vmin': 0,
            'vmax': 0.5,
            'cmap': 'gray',
            'plotColourBar': True,
            'cLabel': "Schmid factor"
        }
        plotParams.update(kwargs)

        # Check that grains have been detected in the map
        self.checkGrainsDetected()
        self.averageSchmidFactor = np.zeros([self.yDim, self.xDim])

        if self[0].averageSchmidFactors is None:
            raise Exception("Run 'calcAverageGrainSchmidFactors' first")

        for grain in self.grainList:
            currentSchmidFactor = []

            if planes is not None:
                # Error catching
                if np.max(planes) > len(self.slipSystems) - 1:
                    raise Exception("Check plane IDs exists, IDs range from 0 "
                                    "to {0}".format(len(self.slipSystems) - 1))

                for plane in planes:
                    if directions is not None:
                        for direction in directions:
                            currentSchmidFactor.append(grain.averageSchmidFactors[plane][direction])
                    else:
                        currentSchmidFactor.append(grain.averageSchmidFactors[plane])
                # TODO: what is this doing?
                currentSchmidFactor = [max(s) for s in zip(*currentSchmidFactor)]
            else:
                currentSchmidFactor = [max(s) for s in zip(*grain.averageSchmidFactors)]

            # Fill grain with colour
            for coord in grain.coordList:
                self.averageSchmidFactor[coord[1], coord[0]] = currentSchmidFactor[0]

        self.averageSchmidFactor[self.averageSchmidFactor == 0] = 0.5

        plot = MapPlot.create(self, self.averageSchmidFactor, **plotParams)

        return plot


class Grain(base.Grain):

    def __init__(self, ebsdMap):
        # Call base class constructor
        super(Grain, self).__init__()

        self.crystalSym = ebsdMap.crystalSym    # symmetry of material e.g. "cubic", "hexagonal"
        self.slipSystems = ebsdMap.slipSystems
        self.ebsdMap = ebsdMap                  # ebsd map this grain is a member of
        self.ownerMap = ebsdMap
        self.quatList = []                      # list of quats
        self.misOriList = None                  # list of misOri at each point in grain
        self.misOriAxisList = None              # list of misOri axes at each point in grain
        self.refOri = None                      # (quat) average ori of grain
        self.averageMisOri = None               # average misOri of grain

        self.averageSchmidFactors = None        # list of list Schmid factors (grouped by slip plane)
        self.slipTraceAngles = None             # list of slip trace angles
        self.slipTraceInclinations = None

    # quat is a quaternion and coord is a tuple (x, y)
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

    def plotRefOri(self, direction=np.array([0, 0, 1]), **kwargs):
        plotParams = {'marker': '+'}
        plotParams.update(kwargs)
        return Quat.plotIPF([self.refOri], direction, self.crystalSym,
                            **plotParams)

    def plotOriSpread(self, direction=np.array([0, 0, 1]), **kwargs):
        plotParams = {'marker': '.'}
        plotParams.update(kwargs)
        return Quat.plotIPF(self.quatList, direction, self.crystalSym,
                            **plotParams)
                            
    def plotUnitCell(self, fig=None, ax=None):
        Quat.plotUnitCell(self.refOri, fig=fig, ax=ax, symGroup=self.crystalSym, cOverA=self.ebsdMap.cOverA)

    # component
    # 0 = misOri
    # {1-3} = misOri axis {1-3}
    def plotMisOri(self, component=0, **kwargs):
        component = int(component)

        # Set default plot parameters then update with any input
        plotParams = {
            'plotColourBar': True
        }
        if component == 0:
            plotParams['cLabel'] = "Grain reference orientation " \
                                   "deviation (GROD) ($^\circ$)"
            plotData = 2 * np.arccos(self.misOriList)

        elif 0 < component < 4:
            plotParams['cLabel'] = "Rotation around {:} ($^\circ$)".format(
                ['X', 'Y', 'Z'][component-1]
            )
            plotData = np.array(self.misOriAxisList)[:, component-1]

        else:
            raise ValueError("Component must between 0 and 3")
        plotParams.update(kwargs)

        plotData *= 180 / np.pi
        plot = self.plotGrainData(grainData=plotData, **plotParams)

        return plot

    # def plotMisOriMulti(self, vmin=None, vmax=None, vRange=(None, None, None),
        # cmap=("viridis", "bwr")):
        #
        # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        #
        # # TODO: Update plot grain misori method
        #
        # # subplots
        # grainMisOri = np.full((4, ymax - y0 + 1, xmax - x0 + 1), np.nan, dtype=float)
        #
        # for coord, misOri, misOriAxis in zip(self.coordList,
        #                                      np.arccos(self.misOriList) * 360 / np.pi,
        #                                      np.array(self.misOriAxisList) * 180 / np.pi):
        #     grainMisOri[0, coord[1] - y0, coord[0] - x0] = misOri
        #     grainMisOri[1:4, coord[1] - y0, coord[0] - x0] = misOriAxis
        #
        # f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        #
        # img = ax1.imshow(grainMisOri[0], interpolation='none', cmap=cmap[0], vmin=vmin, vmax=vmax)
        # plt.colorbar(img, ax=ax1, label="Grain misorientation ($^\circ$)")
        # vmin = None if vRange[0] is None else -vRange[0]
        # img = ax2.imshow(grainMisOri[1], interpolation='none', cmap=cmap[1], vmin=vmin, vmax=vRange[0])
        # plt.colorbar(img, ax=ax2, label="x rotation ($^\circ$)")
        # vmin = None if vRange[0] is None else -vRange[1]
        # img = ax3.imshow(grainMisOri[2], interpolation='none', cmap=cmap[1], vmin=vmin, vmax=vRange[1])
        # plt.colorbar(img, ax=ax3, label="y rotation ($^\circ$)")
        # vmin = None if vRange[0] is None else -vRange[2]
        # img = ax4.imshow(grainMisOri[3], interpolation='none', cmap=cmap[1], vmin=vmin, vmax=vRange[2])
        # plt.colorbar(img, ax=ax4, label="z rotation ($^\circ$)")
        #
        # for ax in (ax1, ax2, ax3, ax4):
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #
        # return

    # define load axis as unit vector
    def calcAverageSchmidFactors(self, loadVector, slipSystems=None):
        """
        Calculate Schmid factors for grain, using average orientation

        :param loadVector: Loading vector, i.e. [1, 0, 0]
        :param slipSystems: Slip systems
        """
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

    def printSlipTraces(self):
        """
        Print a list of slip planes (with colours) and slip directions
        """

        self.calcSlipTraces()

        if self.averageSchmidFactors is None:
            raise Exception("Run 'calcAverageGrainSchmidFactors' on the EBSD map first")

        for ssGroup, colour, sfGroup, slipTrace in zip(self.slipSystems, self.ebsdMap.slipTraceColours,
                                                       self.averageSchmidFactors, self.slipTraces):
            print('{0}\tColour: {1}\tAngle: {2:.2f}'.format(ssGroup[0].slipPlaneLabel, colour, slipTrace * 180 / np.pi))
            for ss, sf in zip(ssGroup, sfGroup):
                print('  {0}   SF: {1:.3f}'.format(ss.slipDirLabel, sf))

    def calcSlipTraces(self, slipSystems=None):
        if slipSystems is None:
            slipSystems = self.slipSystems
        if self.refOri is None:
            self.calcAverageOri()

        screenPlaneNorm = np.array((0, 0, 1))   # in sample orientation frame

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

            # Transform intersection back into sample coordinates and normalise
            intersection = grainAvOri.conjugate.transformVector(intersectionCrystal)
            intersection = intersection / np.sqrt(np.dot(intersection, intersection))

            # Calculate trace angle. Starting vertical and proceeding
            # counter clockwise
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
                        ebsdMap.clickGrainID(event)
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
                currentEbsdMap.clickGrainID(event)

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
