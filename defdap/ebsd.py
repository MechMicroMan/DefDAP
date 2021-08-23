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
from matplotlib.widgets import Button
from skimage import morphology as mph
import networkx as nx

import copy
from warnings import warn

from defdap.file_readers import EBSDDataLoader
from defdap.file_writers import EBSDDataWriter
from defdap.quat import Quat
from defdap.crystal import SlipSystem
from defdap import base

from defdap import defaults
from defdap.plotting import MapPlot
from defdap.utils import reportProgress


class Map(base.Map):
    """
    Class to encapsulate an EBSD map and useful analysis and plotting
    methods.

    Attributes
    ----------
    xDim : int
        Size of map in x direction.
    yDim : int
        Size of map in y direction.
    stepSize : float
        Step size in micron.
    eulerAngleArray : numpy.ndarray
        Euler angles for eaxh point of the map. Shape (3, yDim, xDim).
    bandContrastArray : numpy.ndarray
        Band contrast for each point of map. Shape (yDim, xDim).
    quatArray : numpy.ndarray of defdap.quat.Quat
        Quaterions for each point of map. Shape (yDim, xDim).
    numPhases : int
        Number of phases.
    phaseArray : numpy.ndarray
        Map of phase ids. 1-based, 0 is non-indexed points
    phases : list of defdap.crystal.Phase
        List of phases.
    boundaries : numpy.ndarray
        Map of boundaries. -1 for a boundary, 0 otherwise.
    phaseBoundaries : numpy.ndarray
        Map of phase boundaries. -1 for boundary, 0 otherwise.
    grains : numpy.ndarray
        Map of grains. Grain numbers start at 1 here but everywhere else
        grainID starts at 0. Regions that are smaller than the minimum
        grain size are given value -2. Remnant boundary points are -1.
    misOri : numpy.ndarray
        Map of misorientation.
    misOriAxis : list of numpy.ndarray
        Map of misorientation axis components.
    kam : numpy.ndarray
        Map of KAM.
    slipSystems : list of list of defdap.crystal.SlipSystem
        Slip systems grouped by slip plane.
    slipTraceColours list(str)
        Colours used when plotting slip traces.
    origin : tuple(int)
        Map origin (y, x). Used by linker class where origin is a
        homologue point of the maps.
    GND : numpy.ndarray
        GND scalar map.
    Nye : numpy.ndarray
        3x3 Nye tensor at each point.

    """

    def __init__(self, fileName, dataType=None):
        """
        Initialise class and load EBSD data.

        Parameters
        ----------
        fileName : str
            Path to EBSD file, including name, excluding extension.
        dataType : str, {'OxfordBinary', 'OxfordText'}
            Format of EBSD data file.

        """
        # Call base class constructor
        super(Map, self).__init__()

        self.xDim = None
        self.yDim = None
        self.stepSize = None
        self.eulerAngleArray = None
        self.bandContrastArray = None
        self.quatArray = None
        self.phaseArray = None
        self.phases = []
        self.boundaries = None
        self.boundariesX = None
        self.boundariesY = None
        self.boundaryLines = None
        self.phaseBoundariesX = None
        self.phaseBoundariesY = None
        self.phaseBoundaryLines = None
        self.phaseBoundaries = None
        self.grains = None
        self.misOri = None
        self.misOriAxis = None
        self.kam = None
        self.origin = (0, 0)
        self.GND = None
        self.Nye = None
        self.slipSystems = None
        self.slipTraceColours = None

        # Phase used for the maps crystal structure and cOverA. So old
        # functions still work for the 'main' phase in the map. 0-based
        self.primaryPhaseID = 0

        # Use euler map for defining homologous points
        self.plotHomog = self.plotEulerMap
        self.plotDefault = self.plotEulerMap
        self.highlightAlpha = 1

        self.loadData(fileName, dataType=dataType)

    @reportProgress("loading EBSD data")
    def loadData(self, fileName, dataType=None):
        """Load in EBSD data from file.

        Parameters
        ----------
        fileName : str
            Path to EBSD file, including name, excluding extension.
        dataType : str, {'OxfordBinary', 'OxfordText'}
            Format of EBSD data file.

        """
        dataLoader = EBSDDataLoader.getLoader(dataType)
        dataLoader.load(fileName)

        metadataDict = dataLoader.loadedMetadata
        self.xDim = metadataDict['xDim']
        self.yDim = metadataDict['yDim']
        self.stepSize = metadataDict['stepSize']
        self.phases = metadataDict['phases']

        dataDict = dataLoader.loadedData
        self.eulerAngleArray = dataDict['eulerAngle']
        self.bandContrastArray = dataDict['bandContrast']
        self.bandSlopeArray = dataDict['bandSlope']
        self.meanAngularDeviationArray = dataDict['meanAngularDeviation']
        self.phaseArray = dataDict['phase']
        if int(metadataDict['EDX Windows']['Count']) > 0:
            self.EDX = dataDict['EDXDict']

        # write final status
        yield "Loaded EBSD data (dimensions: {:} x {:} pixels, step " \
              "size: {:} um)".format(self.xDim, self.yDim, self.stepSize)

    def save(self, file_name, data_type=None, file_dir=""):
        """Save EBSD map to file.

        Parameters
        ----------
        file_name : str
            Name of file to save to, it must not already exist.
        data_type : str, {'OxfordText'}
            Format of EBSD data file to save.
        file_dir : str
            Directory to save the file to.

        """
        data_writer = EBSDDataWriter.get_writer(data_type)

        data_writer.metadata['shape'] = self.shape
        data_writer.metadata['step_size'] = self.stepSize
        data_writer.metadata['phases'] = self.phases

        data_writer.data['phase'] = self.phaseArray
        data_writer.data['quat'] = self.quatArray
        data_writer.data['band_contrast'] = self.bandContrastArray

        data_writer.write(file_name, file_dir=file_dir)

    @property
    def crystalSym(self):
        """Crystal symmetry of the primary phase.

        Returns
        -------
        str
            Crystal symmetry

        """
        return self.primaryPhase.crystalStructure.name

    @property
    def cOverA(self):
        """C over A ratio of the primary phase

        Returns
        -------
        float or None
            C over A ratio if hexagonal crystal structure otherwise None

        """
        return self.primaryPhase.cOverA

    @property
    def numPhases(self):
        return len(self.phases) or None

    @property
    def primaryPhase(self):
        """Primary phase of the EBSD map.

        Returns
        -------
        defdap.crystal.Phase
            Primary phase

        """
        return self.phases[self.primaryPhaseID]

    @property
    def scale(self):
        return self.stepSize

    @reportProgress("rotating EBSD data")
    def rotateData(self):
        """Rotate map by 180 degrees and transform quats accordingly.

        """
        self.eulerAngleArray = self.eulerAngleArray[:, ::-1, ::-1]
        self.bandContrastArray = self.bandContrastArray[::-1, ::-1]
        self.phaseArray = self.phaseArray[::-1, ::-1]
        self.buildQuatArray(force=True)     # Force rebuild quat array

        # Rotation from old coord system to new
        transformQuat = Quat.fromAxisAngle(np.array([0, 0, 1]), np.pi).conjugate

        # Perform vectorised multiplication
        quats = Quat.multiplyManyQuats(self.quatArray.flatten(), transformQuat)
        self.quatArray = np.array(quats).reshape(self.yDim, self.xDim)

        yield 1.

    def plotBandContrastMap(self, **kwargs):
        """Plot band contrast map

        kwargs
            All arguments are passed to :func:`defdap.plotting.MapPlot.create`.

        Returns
        -------
        defdap.plotting.MapPlot

        """
        self.checkDataLoaded()

        # Set default plot parameters then update with any input
        plotParams = {
            'plotColourBar': True,
            'cmap': 'gray',
            'clabel': "Band contrast"
        }
        plotParams.update(kwargs)

        plot = MapPlot.create(self, self.bandContrastArray, **plotParams)

        return plot

    def plotEulerMap(self, phases=None, **kwargs):
        """Plot an orientation map in Euler colouring

        Parameters
        ----------
        phases : list of int
            Which phases to plot for
        kwargs
            All arguments are passed to :func:`defdap.plotting.MapPlot.create`.

        Returns
        -------
        defdap.plotting.MapPlot

        """
        self.checkDataLoaded()

        # Set default plot parameters then update with any input
        plot_params = {}
        plot_params.update(kwargs)

        if phases is None:
            phases = self.phases
            phase_ids = range(len(phases))
        else:
            phase_ids = phases
            phases = [self.phases[i] for i in phase_ids]

        map_colours = np.zeros(self.shape + (3,))

        for phase, phase_id in zip(phases, phase_ids):
            if phase.crystalStructure.name == 'cubic':
                norm = np.array([2 * np.pi, np.pi / 2, np.pi / 2])
            elif phase.crystalStructure.name == 'hexagonal':
                norm = np.array([np.pi, np.pi, np.pi / 3])
            else:
                ValueError("Only hexagonal and cubic symGroup supported")

            # Apply normalisation for each phase
            phase_mask = self.phaseArray == phase_id + 1
            map_colours[phase_mask] = self.eulerAngleArray[:, phase_mask].T / norm

        return MapPlot.create(self, map_colours, **plot_params)

    def plotIPFMap(self, direction, backgroundColour = [0., 0., 0.], phases=None, **kwargs):
        """
        Plot a map with points coloured in IPF colouring,
        with respect to a given sample direction.

        Parameters
        ----------
        direction : np.array len 3
            Sample directiom.
        backgroundColour : np.array len 3
            Colour of background (i.e. for phases not plotted).
        phases : list of int
            Which phases to plot IPF data for.
        kwargs
            Other arguments passed to :func:`defdap.plotting.MapPlot.create`.

        Returns
        -------
        defdap.plotting.MapPlot

        """
        # Set default plot parameters then update with any input
        plot_params = {}
        plot_params.update(kwargs)

        if phases is None:
            phases = self.phases
            phase_ids = range(len(phases))
        else:
            phase_ids = phases
            phases = [self.phases[i] for i in phase_ids]

        map_colours = np.tile(np.array(backgroundColour), self.shape + (1,))

        for phase, phase_id in zip(phases, phase_ids):
            # calculate IPF colours for phase
            phase_mask = self.phaseArray == phase_id + 1
            map_colours[phase_mask] = Quat.calcIPFcolours(
                self.quatArray[phase_mask],
                direction,
                phase.crystalStructure.name
            ).T

        return MapPlot.create(self, map_colours, **plot_params)

    def plotPhaseMap(self, **kwargs):
        """Plot a phase map.

        Parameters
        ----------
        kwargs
            All arguments passed to :func:`defdap.plotting.MapPlot.create`.

        Returns
        -------
        defdap.plotting.MapPlot

        """
        # Set default plot parameters then update with any input
        plotParams = {
            'vmin': 0,
            'vmax': self.numPhases
        }
        plotParams.update(kwargs)

        plot = MapPlot.create(self, self.phaseArray, **plotParams)

        # add a legend to the plot
        phaseIDs = list(range(0, self.numPhases + 1))
        phaseNames = ["Non-indexed"] + [phase.name for phase in self.phases]
        plot.addLegend(phaseIDs, phaseNames, loc=2, borderaxespad=0.)

        return plot

    def calcKam(self):
        """
        Calculates Kernel Average Misorientaion (KAM) for the EBSD map,
        based on a 3x3 kernel. Crystal symmetric equivalences are not
        considered. Stores result in self.kam.

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
        """Plot Kernel Average Misorientaion (KAM) for the EBSD map.

        Parameters
        ----------
        kwargs
            All arguments are passed to :func:`defdap.plotting.MapPlot.create`.

        Returns
        -------
        defdap.plotting.MapPlot

        """
        # Set default plot parameters then update with any input
        plotParams = {
            'plotColourBar': True,
            'clabel': "Kernel average misorientation (KAM) ($^\circ$)"
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
        Stores result in self.Nye and self.GND. Uses the crystal
        symmetry of the primary phase.

        """
        self.buildQuatArray()
        syms = self.primaryPhase.crystalStructure.symmetries
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
        """Plots a map of geometrically necessary dislocation (GND) density

        Parameters
        ----------
        kwargs
            All arguments are passed to :func:`defdap.plotting.MapPlot.create`.

        Returns
        -------
        defdap.plotting.MapPlot

        """
        # Set default plot parameters then update with any input
        plotParams = {
            'plotColourBar': True,
            'clabel': "Geometrically necessary dislocation (GND) content"
        }
        plotParams.update(kwargs)

        self.calcNye()

        plot = MapPlot.create(self, np.log10(self.GND), **plotParams)

        return plot

    def checkDataLoaded(self):
        """ Checks if EBSD data is loaded

        Returns
        -------
        bool
            True if data loaded

        """
        if self.eulerAngleArray is None:
            raise Exception("Data not loaded")
        return True

    @reportProgress("building quaternion array")
    def buildQuatArray(self, force = False):
        """Build quaternion array

        Parameters
        ----------
        force, optional
            If true, re-build quaternion array
        """
        self.checkDataLoaded()

        if force == False:
            if self.quatArray is None:
                # create the array of quat objects
                self.quatArray = Quat.createManyQuats(self.eulerAngleArray)
        if force == True:
            self.quatArray = Quat.createManyQuats(self.eulerAngleArray)

        yield 1.

    def filterData(self, misOriTol=5):
        # Kuwahara filter
        print("8 quadrants")
        misOriTol *= np.pi / 180
        misOriTol = np.cos(misOriTol / 2)

        # store quat components in array
        quatComps = np.empty((4,) + self.shape)
        for idx in np.ndindex(self.shape):
            quatComps[(slice(None),) + idx] = self.quatArray[idx].quatCoef

        # misorientation in each quadrant surrounding a point
        misOris = np.zeros((8,) + self.shape)

        for i in range(2, self.shape[0] - 2):
            for j in range(2, self.shape[1] - 2):

                refQuat = quatComps[:, i, j]
                quadrants = [
                    quatComps[:, i - 2:i + 1, j - 2:j + 1],   # UL
                    quatComps[:, i - 2:i + 1, j - 1:j + 2],   # UC
                    quatComps[:, i - 2:i + 1, j:j + 3],       # UR
                    quatComps[:, i - 1:i + 2, j:j + 3],       # MR
                    quatComps[:, i:i + 3, j:j + 3],           # LR
                    quatComps[:, i:i + 3, j - 1:j + 2],       # LC
                    quatComps[:, i:i + 3, j - 2:j + 1],       # LL
                    quatComps[:, i - 1:i + 2, j - 2:j + 1]    # ML
                ]

                for k, quats in enumerate(quadrants):
                    misOrisQuad = np.abs(
                        np.einsum("ijk,i->jk", quats, refQuat)
                    )
                    misOrisQuad = misOrisQuad[misOrisQuad > misOriTol]
                    misOris[k, i, j] = misOrisQuad.mean()

        minMisOriQuadrant = np.argmax(misOris, axis=0)
        # minMisOris = np.max(misOris, axis=0)
        # minMisOris[minMisOris > 1.] = 1.
        # minMisOris = 2 * np.arccos(minMisOris)

        quatCompsNew = np.copy(quatComps)

        for i in range(2, self.shape[0] - 2):
            for j in range(2, self.shape[1] - 2):
                # if minMisOris[i, j] < misOriTol:
                #     continue

                refQuat = quatComps[:, i, j]
                quadrants = [
                    quatComps[:, i - 2:i + 1, j - 2:j + 1],   # UL
                    quatComps[:, i - 2:i + 1, j - 1:j + 2],   # UC
                    quatComps[:, i - 2:i + 1, j:j + 3],       # UR
                    quatComps[:, i - 1:i + 2, j:j + 3],       # MR
                    quatComps[:, i:i + 3, j:j + 3],           # LR
                    quatComps[:, i:i + 3, j - 1:j + 2],       # LC
                    quatComps[:, i:i + 3, j - 2:j + 1],       # LL
                    quatComps[:, i - 1:i + 2, j - 2:j + 1]    # ML
                ]
                quats = quadrants[minMisOriQuadrant[i, j]]

                misOrisQuad = np.abs(
                    np.einsum("ijk,i->jk", quats, refQuat)
                )
                quats = quats[:, misOrisQuad > misOriTol]

                avOri = np.einsum("ij->i", quats)
                # avOri /= np.sqrt(np.dot(avOri, avOri))

                quatCompsNew[:, i, j] = avOri

        quatCompsNew /= np.sqrt(np.einsum("ijk,ijk->jk", quatCompsNew, quatCompsNew))

        quatArrayNew = np.empty(self.shape, dtype=Quat)

        for idx in np.ndindex(self.shape):
            quatArrayNew[idx] = Quat(quatCompsNew[(slice(None),) + idx])

        self.quatArray = quatArrayNew

        return quats

    @reportProgress("finding grain boundaries")
    def findBoundaries(self, boundDef=10):
        """Find grain and phase boundaries

        Parameters
        ----------
        boundDef : float
            Critical misorientation.

        """
        # TODO: what happens with non-indexed points
        # TODO: grain boundaries should be calculated per crystal structure
        syms = self.primaryPhase.crystalStructure.symmetries
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
        misOriX = np.ones((numSyms, self.yDim, self.xDim))
        misOriY = np.ones((numSyms, self.yDim, self.xDim))

        # loop over symmetries calculating misorientation to initial
        for i in range(numSyms):
            for j in range(self.xDim - 1):
                misOriX[i, :, j] = abs(
                    np.einsum("ij,ij->j", quatComps[0, :, :, j], quatComps[i, :, :, j + 1]))

            for j in range(self.yDim - 1):
                misOriY[i, j, :] = abs(np.einsum("ij,ij->j", quatComps[0, :, j, :], quatComps[i, :, j + 1, :]))

        misOriX[misOriX > 1] = 1
        misOriY[misOriY > 1] = 1

        # find min misorientation (max here as misorientaion is cos of this)
        misOriX = np.max(misOriX, axis=0)
        misOriY = np.max(misOriY, axis=0)

        # convert to misorientation in degrees
        misOriX = 2 * np.arccos(misOriX) * 180 / np.pi
        misOriY = 2 * np.arccos(misOriY) * 180 / np.pi

        # GRAIN boundary POINTS where misOriX or misOriY are greater
        # than set value
        self.boundariesX = misOriX > boundDef
        self.boundariesY = misOriY > boundDef

        # PHASE boundary POINTS
        self.phaseBoundariesX = np.not_equal(
            self.phaseArray, np.roll(self.phaseArray, -1, axis=1))
        self.phaseBoundariesX[:, -1] = False

        self.phaseBoundariesY = np.not_equal(
            self.phaseArray, np.roll(self.phaseArray, -1, axis=0))
        self.phaseBoundariesY[-1, :] = False

        self.phaseBoundaries = np.logical_or(
            self.phaseBoundariesX, self.phaseBoundariesY)
        self.phaseBoundaries = -self.phaseBoundaries.astype(int)

        # add PHASE boundary POINTS to GRAIN boundary POINTS
        self.boundariesX = np.logical_or(self.boundariesX, self.phaseBoundariesX)
        self.boundariesY = np.logical_or(self.boundariesY, self.phaseBoundariesY)
        self.boundaries = np.logical_or(self.boundariesX, self.boundariesY)
        self.boundaries = -self.boundaries.astype(int)

        _, _, self.boundaryLines = Map.create_boundary_lines(
            boundaries_x=self.boundariesX,
            boundaries_y=self.boundariesY
        )
        _, _, self.phaseBoundaryLines = Map.create_boundary_lines(
            boundaries_x=self.phaseBoundariesX,
            boundaries_y=self.phaseBoundariesY
        )

        yield 1.

    @staticmethod
    def create_boundary_lines(*, boundaries_x=None, boundaries_y=None):
        boundary_data = {}
        if boundaries_x is not None:
            boundary_data['x'] = boundaries_x
        if boundaries_y is not None:
            boundary_data['y'] = boundaries_y
        if not boundary_data:
            raise ValueError("No boundaries provided.")

        deltas = {
            'x': (0.5, -0.5, 0.5, 0.5),
            'y': (-0.5, 0.5, 0.5, 0.5)
        }
        all_lines = []
        for mode, boundaries in boundary_data.items():
            points = np.where(boundaries)
            lines = []
            for i, j in zip(*points):
                lines.append((
                    (j + deltas[mode][0], i + deltas[mode][1]),
                    (j + deltas[mode][2], i + deltas[mode][3])
                ))
            all_lines.append(lines)

        if len(all_lines) == 2:
            all_lines.append(all_lines[0] + all_lines[1])
            return tuple(all_lines)
        else:
            return all_lines[0]

    @reportProgress("constructing neighbour network")
    def buildNeighbourNetwork(self):
        # create network
        nn = nx.Graph()
        nn.add_nodes_from(self.grainList)

        for i, boundaries in enumerate((self.boundariesX, self.boundariesY)):
            yLocs, xLocs = np.nonzero(boundaries)
            totalPoints = len(xLocs)

            for iPoint, (x, y) in enumerate(zip(xLocs, yLocs)):
                # report progress, assumes roughly equal number of x and
                # y boundary points
                yield 0.5 * (i + iPoint / totalPoints)

                if (x == 0 or y == 0 or x == self.grains.shape[1] - 1 or
                        y == self.grains.shape[0] - 1):
                    # exclude boundary pixels of map
                    continue

                grainID = self.grains[y, x] - 1
                neiGrainID = self.grains[y + i, x - i + 1] - 1

                if neiGrainID == grainID:
                    # ignore if neighbour is same as grain
                    continue
                if neiGrainID < 0 or grainID < 0:
                    # ignore if not a grain (boundary points -1 and
                    # points in small grains -2)
                    continue

                grain = self[grainID]
                neiGrain = self[neiGrainID]

                try:
                    # look up boundary segment if it exists
                    bSeg = nn[grain][neiGrain]['boundary']
                except KeyError:
                    # neighbour relation doesn't exist so add it
                    bSeg = BoundarySegment(self, grain, neiGrain)
                    nn.add_edge(grain, neiGrain, boundary=bSeg)

                # add the boundary point
                bSeg.addBoundaryPoint((x, y), i, grain)

        self.neighbourNetwork = nn

    @reportProgress("finding phase boundaries")

    def plotPhaseBoundaryMap(self, dilate=False, **kwargs):
        """Plot phase boundary map.

        Parameters
        ----------
        dilate : bool
            If true, dilate boundary.
        kwargs
            All other arguments are passed to :func:`defdap.plotting.MapPlot.create`.

        Returns
        -------
        defdap.plotting.MapPlot

        """
        # Set default plot parameters then update with any input
        plotParams = {
            'vmax': 1,
            'plotColourBar': True,
            'cmap': 'gray'
        }
        plotParams.update(kwargs)

        boundariesImage = -self.phaseBoundaries
        if dilate:
            boundariesImage = mph.binary_dilation(boundariesImage)

        plot = MapPlot.create(self, boundariesImage, **plotParams)

        return plot

    def plotBoundaryMap(self, **kwargs):
        """Plot grain boundary map.

        Parameters
        ----------
        kwargs
            All arguments are passed to :func:`defdap.plotting.MapPlot.create`.

        Returns
        -------
        defdap.plotting.MapPlot

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
        """Find grains and assign IDs.

        Parameters
        ----------
        minGrainSize : int
            Minimum grain area in pixels.

        """
        # TODO: grains need to be assigned a phase
        # Initialise the grain map
        # TODO: Look at grain map compared to boundary map
        # self.grains = np.copy(self.boundaries)
        self.grains = np.zeros_like(self.boundaries)

        self.grainList = []

        # List of points where no grain has be set yet
        points_left = self.phaseArray != 0
        total_points = points_left.sum()
        found_point = 0
        next_point = points_left.tobytes().find(b'\x01')

        # Start counter for grains
        grainIndex = 1

        # Loop until all points (except boundaries) have been assigned
        # to a grain or ignored
        i = 0
        while found_point >= 0:
            # Flood fill first unknown point and return grain object
            idx = np.unravel_index(next_point, self.grains.shape)
            currentGrain = self.floodFill(idx[1], idx[0], grainIndex,
                                          points_left)

            if len(currentGrain) < minGrainSize:
                # if grain size less than minimum, ignore grain and set
                # values in grain map to -2
                for coord in currentGrain.coordList:
                    self.grains[coord[1], coord[0]] = -2
            else:
                # add grain to list and increment grain index
                self.grainList.append(currentGrain)
                grainIndex += 1

            # find next search point
            points_left_sub = points_left.reshape(-1)[next_point + 1:]
            found_point = points_left_sub.tobytes().find(b'\x01')
            next_point += found_point + 1

            # report progress
            i += 1
            if i == defaults['find_grain_report_freq']:
                yield 1. - points_left_sub.sum() / total_points
                i = 0

        # Assign phase to each grain
        for grain in self:
            phaseVals = grain.grainData(self.phaseArray)
            if np.max(phaseVals) != np.min(phaseVals):
                warn(f"Grain {grain.grainID} could not be assigned a "
                     f"phase, phase vals not constant.")
                continue
            phaseID = phaseVals[0] - 1
            if not (0 <= phaseID < self.numPhases):
                warn(f"Grain {grain.grainID} could not be assigned a "
                     f"phase, invalid phase {phaseID}.")
                continue
            grain.phaseID = phaseID
            grain.phase = self.phases[phaseID]

    def plotGrainMap(self, **kwargs):
        """Plot a map with grains coloured.

        Parameters
        ----------
        kwargs
            All arguments are passed to :func:`defdap.plotting.MapPlot.create`.

        Returns
        -------
        defdap.plotting.MapPlot

        """
        # Set default plot parameters then update with any input
        plotParams = {
            'clabel': "Grain number"
        }
        plotParams.update(kwargs)

        plot = MapPlot.create(self, self.grains, **plotParams)

        return plot

    def floodFill(self, x, y, grainIndex, points_left):
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
        points_left : numpy.ndarray
            Boolean map of the points that have not been assigned a grain yet

        Returns
        -------
        currentGrain : defdap.ebsd.Grain
            New grain object with points added
        """
        # create new grain
        currentGrain = Grain(grainIndex - 1, self)

        # add first point to the grain
        currentGrain.addPoint((x, y), self.quatArray[y, x])
        self.grains[y, x] = grainIndex
        points_left[y, x] = False
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
                    points_left[t, s] = False
                    edge.append((s, t))

        return currentGrain

    @reportProgress("calculating grain mean orientations")
    def calcGrainAvOris(self):
        """Calculate the average orientation of grains.

        """
        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        numGrains = len(self)
        for iGrain, grain in enumerate(self):
            grain.calcAverageOri()

            # report progress
            yield (iGrain + 1) / numGrains

    @reportProgress("calculating grain misorientations")
    def calcGrainMisOri(self, calcAxis=False):
        """Calculate the misorientation within grains.

        Parameters
        ----------
        calcAxis : bool
            Calculate the misorientation axis if True.

        """
        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        numGrains = len(self)
        for iGrain, grain in enumerate(self):
            grain.buildMisOriList(calcAxis=calcAxis)

            # report progress
            yield (iGrain + 1) / numGrains

    def plotMisOriMap(self, component=0, **kwargs):
        """Plot misorientation map.

        Parameters
        ----------
        component : int, {0, 1, 2, 3}
            0 gives misorientation, 1, 2, 3 gives rotation about x, y, z
        kwargs
            All other arguments are passed to :func:`defdap.plotting.MapPlot.create`.

        Returns
        -------
        defdap.plotting.MapPlot

        """
        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        self.misOri = np.ones([self.yDim, self.xDim])

        if component in [1, 2, 3]:
            # Calculate misorientation axis if not calculated
            if np.any([grain.misOriAxisList is None for grain in self.grainList]):
                self.calcGrainMisOri(calcAxis = True)
            for grain in self.grainList:
                for coord, misOriAxis in zip(grain.coordList, np.array(grain.misOriAxisList)):
                    self.misOri[coord[1], coord[0]] = misOriAxis[component - 1]

            misOri = self.misOri * 180 / np.pi
            clabel = "Rotation around {:} axis ($^\circ$)".format(
                ['X', 'Y', 'Z'][component-1]
            )
        else:
            # Calculate misorientation if not calculated
            if np.any([grain.misOriList is None for grain in self.grainList]):
                self.calcGrainMisOri(calcAxis = False)
            for grain in self.grainList:
                for coord, misOri in zip(grain.coordList, grain.misOriList):
                    self.misOri[coord[1], coord[0]] = misOri

            misOri = np.arccos(self.misOri) * 360 / np.pi
            clabel = "Grain reference orienation deviation (GROD) ($^\circ$)"

        # Set default plot parameters then update with any input
        plotParams = {
            'plotColourBar': True,
            'clabel': clabel
        }
        plotParams.update(kwargs)

        plot = MapPlot.create(self, misOri, **plotParams)

        return plot

    def loadSlipSystems(self, name):
        """Load slip system definitions from file.

        Parameters
        ----------
        name : str
            name of the slip system file (without file extension)
            stored in the defdap install dir or path to a file.

        """
        # TODO: should be loaded into the phases of the map
        self.slipSystems, self.slipTraceColours = SlipSystem.loadSlipSystems(
            name, self.crystalSym, cOverA=self.cOverA
        )

        if self.checkGrainsDetected(raiseExc=False):
            for grain in self:
                grain.slipSystems = self.slipSystems

    def printSlipSystems(self):
        """Print a list of slip planes (with colours) and slip directions.

        """
        # TODO: this should be moved to static method of the SlipSystem class
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
        based on average grain orientation.

        Parameters
        ----------
        loadVector :
            Loading vector, e.g. [1, 0, 0].
        slipSystems : list, optional
            Slip planes to calculate Schmid factor for,
            maximum of all planes calculated if not given.

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
        orientation (for all slip systems unless specified).

        Parameters
        ----------
        planes : list, optional
            Plane ID(s) to consider. All planes considered if not given.
        directions : list, optional
            Direction ID(s) to consider. All directions considered if not given.
        kwargs
            All other arguments are passed to :func:`defdap.plotting.MapPlot.create`.

        Returns
        -------
        defdap.plotting.MapPlot

        """
        # Set default plot parameters then update with any input
        plot_params = {
            'vmin': 0,
            'vmax': 0.5,
            'cmap': 'gray',
            'plotColourBar': True,
            'clabel': "Schmid factor"
        }
        plot_params.update(kwargs)

        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        if self[0].averageSchmidFactors is None:
            raise Exception("Run 'calcAverageGrainSchmidFactors' first")

        grains_sf = []
        for grain in self.grainList:
            current_sf = []

            if planes is not None:
                for plane in planes:
                    if directions is not None:
                        for direction in directions:
                            current_sf.append(
                                grain.averageSchmidFactors[plane][direction]
                            )
                    else:
                        current_sf += grain.averageSchmidFactors[plane]
            else:
                for sf_group in grain.averageSchmidFactors:
                    current_sf += sf_group

            grains_sf.append(current_sf)

        grains_sf = np.array(grains_sf)
        grains_sf_max = np.max(grains_sf, axis=1)

        plot = self.plotGrainDataMap(grainData=grains_sf_max, bg=0.5,
                                     **plot_params)

        return plot


class Grain(base.Grain):
    """
    Class to encapsulate a grain in an EBSD map and useful analysis and
    plotting methods.

    Attributes
    ----------
    crystalSym : str
        Symmetry of material e.g. "cubic", "hexagonal"
    slipSystems : list(list(defdap.crystal.SlipSystem))
        Slip systems
    ebsdMap : defdap.ebsd.Map
        EBSD map this grain is a member of.
    ownerMap : defdap.ebsd.Map
        EBSD map this grain is a member of.
    quatList : list
        List of quats.
    misOriList : list
        MisOri at each point in grain.
    misOriAxisList : list
        MisOri axes at each point in grain.
    refOri : defdap.quat.Quat
        Average ori of grain
    averageMisOri
        Average misOri of grain.
    averageSchmidFactors : list
        List of list Schmid factors (grouped by slip plane).
    slipTraceAngles : list
        Slip trace angles in screen plane.
    slipTraceInclinations : list
         Angle between slip plane and screen plane.

    """

    # TODO: each grain should be assigned a phase and slip systems
    # slip systems accessed from the phase
    def __init__(self, grainID, ebsdMap):
        # Call base class constructor
        super(Grain, self).__init__(grainID, ebsdMap)

        self.crystalSym = ebsdMap.crystalSym    # symmetry of material e.g. "cubic", "hexagonal"
        self.slipSystems = ebsdMap.slipSystems
        self.ebsdMap = self.ownerMap            # ebsd map this grain is a member of
        self.quatList = []                      # list of quats
        self.misOriList = None                  # list of misOri at each point in grain
        self.misOriAxisList = None              # list of misOri axes at each point in grain
        self.refOri = None                      # (quat) average ori of grain
        self.averageMisOri = None               # average misOri of grain

        self.averageSchmidFactors = None        # list of list Schmid factors (grouped by slip plane)
        self.slipTraceAngles = None             # list of slip trace angles
        self.slipTraceInclinations = None

    @property
    def plotDefault(self):
        return lambda *args, **kwargs: self.plotUnitCell(
            *args, **kwargs
        )

    def addPoint(self, coord, quat):
        """Append a coordinate and a quat to a grain.

        Parameters
        ----------
        coord : tuple
            (x,y) coordinate to append
        quat : defdap.quat.Quat
            Quaternion to append.

        """
        self.coordList.append(coord)
        self.quatList.append(quat)

    def calcAverageOri(self):
        """Calculate the average orientation of a grain.

        """
        quatCompsSym = Quat.calcSymEqvs(self.quatList, self.crystalSym)

        self.refOri = Quat.calcAverageOri(quatCompsSym)

    def buildMisOriList(self, calcAxis=False):
        """Calculate the misorientation within given grain.

        Parameters
        ----------
        calcAxis : bool
            Calculate the misorientation axis if True.

        """
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
        """Plot the average grain orientation on an IPF.

        Parameters
        ----------
        direction : numpy.ndarray
            Sample direction for IPF.
        kwargs
            All other arguments are passed to :func:`defdap.quat.Quat.plotIPF`.

        Returns
        -------
        defdap.plotting.PolePlot

        """
        plotParams = {'marker': '+'}
        plotParams.update(kwargs)
        return Quat.plotIPF([self.refOri], direction, self.crystalSym,
                            **plotParams)

    def plotOriSpread(self, direction=np.array([0, 0, 1]), **kwargs):
        """Plot all orientations within a given grain, on an IPF.

        Parameters
        ----------
        direction : numpy.ndarray
            Sample direction for IPF.
        kwargs
            All other arguments are passed to :func:`defdap.quat.Quat.plotIPF`.

        Returns
        -------
        defdap.plotting.PolePlot

        """
        plotParams = {'marker': '.'}
        plotParams.update(kwargs)
        return Quat.plotIPF(self.quatList, direction, self.crystalSym,
                            **plotParams)

    def plotUnitCell(self, fig=None, ax=None, plot=None, **kwargs):
        """Plot an unit cell of the average grain orientation.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Matplotlib figure to plot on
        ax : matplotlib.figure.Figure
            Matplotlib figure to plot on
        plot : defdap.plotting.PolePlot
            defdap plot to plot the figure to.
        kwargs
            All other arguments are passed to :func:`defdap.quat.Quat.plotUnitCell`.

        """
        crystalStructure = self.ebsdMap.phases[self.phaseID].crystalStructure
        plot = Quat.plotUnitCell(self.refOri, fig=fig, ax=ax, plot=plot,
                          crystalStructure=crystalStructure, **kwargs)

        return plot

    def plotMisOri(self, component=0, **kwargs):
        """Plot misorientation map for a given grain.

        Parameters
        ----------
        component : int, {0, 1, 2, 3}
            0 gives misorientation, 1, 2, 3 gives rotation about x, y, z.
        kwargs
            All other arguments are passed to :func:`defdap.ebsd.plotGrainData`.

        Returns
        -------
        defdap.plotting.GrainPlot

        """
        component = int(component)

        # Set default plot parameters then update with any input
        plotParams = {
            'plotColourBar': True
        }
        if component == 0:
            if self.misOriList is None: self.buildMisOriList()
            plotParams['clabel'] = "Grain reference orientation " \
                                   "deviation (GROD) ($^\circ$)"
            plotData = np.rad2deg(2 * np.arccos(self.misOriList))

        elif 0 < component < 4:
            if self.misOriAxisList is None: self.buildMisOriList(calcAxis=True)
            plotParams['clabel'] = "Rotation around {:} ($^\circ$)".format(
                ['X', 'Y', 'Z'][component-1]
            )
            plotData = np.rad2deg(np.array(self.misOriAxisList)[:, component-1])

        else:
            raise ValueError("Component must between 0 and 3")
        plotParams.update(kwargs)

        plot = self.plotGrainData(grainData=plotData, **plotParams)

        return plot

    # define load axis as unit vector
    def calcAverageSchmidFactors(self, loadVector, slipSystems=None):
        """Calculate Schmid factors for grain, using average orientation.

        Parameters
        ----------
        loadVector : numpy.ndarray
            Loading vector, i.e. [1, 0, 0]
        slipSystems : list, optional
            Slip planes to calculate Schmid factor for. Maximum for all planes
            used if not set.

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
        """Returns list of slip trace angles.

        Returns
        -------
        list
            Slip trace angles based on grain orientation in calcSlipTraces.

        """
        if self.slipTraceAngles is None:
            self.calcSlipTraces()

        return self.slipTraceAngles

    def printSlipTraces(self):
        """Print a list of slip planes (with colours) and slip directions

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
        """Calculates list of slip trace angles based on grain orientation.

        Parameters
        -------
        slipSystems : defdap.crystal.SlipSystem, optional

        """
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


class BoundarySegment(object):
    def __init__(self, ebsdMap, grain1, grain2):
        self.ebsdMap = ebsdMap

        self.grain1 = grain1
        self.grain2 = grain2

        # list of boundary points (x, y) for horizontal (X) and
        # vertical (Y) boundaries
        self.boundaryPointsX = []
        self.boundaryPointsY = []
        # Boolean value for each point above, True if boundary point is
        # in grain1 and False if in grain2
        self.boundaryPointOwnersX = []
        self.boundaryPointOwnersY = []

    def __eq__(self, right):
        if type(self) is not type(right):
            raise NotImplementedError()

        return ((self.grain1 is right.grain1 and
                self.grain2 is right.grain2) or
                (self.grain1 is right.grain2 and
                 self.grain2 is right.grain1))

    def __len__(self):
        return len(self.boundaryPointsX) + len(self.boundaryPointsY)

    def addBoundaryPoint(self, point, kind, ownerGrain):
        if kind == 0:
            self.boundaryPointsX.append(point)
            self.boundaryPointOwnersX.append(ownerGrain is self.grain1)
        elif kind == 1:
            self.boundaryPointsY.append(point)
            self.boundaryPointOwnersY.append(ownerGrain is self.grain1)
        else:
            raise ValueError("Boundary point kind is 0 for x and 1 for y")

    def boundaryPointPairs(self, kind):
        """Return pairs of points either side of the boundary. The first
        point is always in grain1
        """
        if kind == 0:
            boundaryPoints = self.boundaryPointsX
            boundaryPointOwners = self.boundaryPointOwnersX
            delta = (1, 0)
        else:
            boundaryPoints = self.boundaryPointsY
            boundaryPointOwners = self.boundaryPointOwnersY
            delta = (0, 1)

        boundaryPointPairs = []
        for point, owner in zip(boundaryPoints, boundaryPointOwners):
            otherPoint = (point[0] + delta[0], point[1] + delta[1])
            if owner:
                boundaryPointPairs.append((point, otherPoint))
            else:
                boundaryPointPairs.append((otherPoint, point))

        return boundaryPointPairs

    @property
    def boundaryPointPairsX(self):
        """Return pairs of points either side of the boundary. The first
        point is always in grain1
        """
        return self.boundaryPointPairs(0)

    @property
    def boundaryPointPairsY(self):
        """Return pairs of points either side of the boundary. The first
        point is always in grain1
        """
        return self.boundaryPointPairs(1)

    def misorientation(self):
        misOri, minSymm = self.grain1.refOri.misOri(
            self.grain2.refOri, self.ebsdMap.crystalSym, returnQuat=2
        )
        misOri = 2 * np.arccos(misOri)
        misOriAxis = self.grain1.refOri.misOriAxis(minSymm)

        # should this be a unit vector already?
        misOriAxis /= np.sqrt(np.dot(misOriAxis, misOriAxis))

        return misOri, misOriAxis

        # compVector = np.array([1., 1., 1.])
        # deviation = np.arccos(
        #     np.dot(misOriAxis, np.array([1., 1., 1.])) /
        #     (np.sqrt(np.dot(misOriAxis, misOriAxis) * np.dot(compVector,
        #                                                      compVector))))
        # print(deviation * 180 / np.pi)


class Linker(object):
    """Class for linking multiple EBSD maps of the same region for analysis of deformation.

    Parameters
    ----------
    ebsdMaps : list(ebsd.Map)
        List of ebsd.Map objects that are linked.
    links : list
        List of grain link. Each link is stored as a tuple of
        grain IDs (one from each map stored in same order of maps).
    numMaps : int
        Number of linked maps.

    """

    def __init__(self, maps):
        self.ebsdMaps = maps
        self.numMaps = len(maps)
        self.links = []
        return

    def setOrigin(self):
        """Interacive tool to set origin of each EBSD map.

        """
        for ebsdMap in self.ebsdMaps:
            ebsdMap.locateGrainID(clickEvent=self.clickSetOrigin)

    def clickSetOrigin(self, event, currentEbsdMap):
        """Event handler for clicking to set origin of map.

        Parameters
        ----------
        event
            Click event.
        currentEbsdMap : defdap.ebsd.Map
            EBSD map to set origin for.

        """
        currentEbsdMap.origin = (int(event.ydata), int(event.xdata))
        print("Origin set to ({:}, {:})".format(currentEbsdMap.origin[0], currentEbsdMap.origin[1]))

    def startLinking(self):
        """Start interactive grain linking process of each EBSD map.

        """
        for ebsdMap in self.ebsdMaps:
            ebsdMap.locateGrainID(clickEvent=self.clickGrainGuess)

            # Add make link button to axes
            btnAx = ebsdMap.fig.add_axes([0.8, 0.0, 0.1, 0.07])
            Button(btnAx, 'Make link', color='0.85', hovercolor='0.95')

    def clickGrainGuess(self, event, currentEbsdMap):
        """Guesses grain position in other maps, given click on one.

        Parameters
        ----------
        event
            Click handler.
        currentEbsdMap : defdap.ebsd.Map
            EBSD map that is clicked on.

        """
        # self is current linker instance even if run as click event handler from map class
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
        """Make a link between the EBSD maps after clicking.

        """
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
        """Reset links.

        """
        self.links = []

#   Analysis routines

    def setAvOriFromInitial(self):
        """Loop over each map (not first/reference) and each link.
        Sets refOri of linked grains to refOri of grain in first map.

        """
        masterMap = self.ebsdMaps[0]

        for i, ebsdMap in enumerate(self.ebsdMaps[1:], start=1):
            for link in self.links:
                ebsdMap.grainList[link[i]].refOri = copy.deepcopy(masterMap.grainList[link[0]].refOri)

        return

    def updateMisOri(self, calcAxis=False):
        """Recalculate misorientation for linked grain (not for first map)

        Parameters
        ----------
        calcAxis : bool
            Calculate the misorientation axis if True.

        """
        for i, ebsdMap in enumerate(self.ebsdMaps[1:], start=1):
            for link in self.links:
                ebsdMap.grainList[link[i]].buildMisOriList(calcAxis=calcAxis)

        return
