import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import skimage.morphology as mph


class Map(object):

    def __init__(self):
        self.homogPoints = []
        self.selPoint = None

        self.proxigramArr = None

    def __len__(self):
        return len(self.grainList)

    # allow array like getting of grains
    def __getitem__(self, key):
        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        return self.grainList[key]

    def checkGrainsDetected(self):
        """Check if grains have been detected

        Returns:
            bool: Returns True if grains detected

        Raises:
            Exception: if grains not detected
        """
        if self.grainList is None or type(self.grainList) is not list or len(self.grainList) < 1:
            raise Exception("No grains detected.")
        return True

    def plotGBs(self, ax=None, colour='white', dilate=False):
        # create colourmap for boundaries and plot. colourmap goes transparent white to opaque white/colour
        cmap1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap', ['white', colour], 256)
        cmap1._init()
        cmap1._lut[:, -1] = np.linspace(0, 1, cmap1.N + 3)

        boundariesImage = -self.boundaries

        if dilate:
            boundariesImage = mph.binary_dilation(boundariesImage)

        if ax is not None:
            ax.imshow(boundariesImage, cmap=cmap1, interpolation='None', vmin=0, vmax=1)
        else:
            plt.imshow(boundariesImage, cmap=cmap1, interpolation='None', vmin=0, vmax=1)

    def setHomogPoint(self, binSize=1):
        self.selPoint = None

        self.plotHomog()
        # Plot stored homogo points if there are any
        if len(self.homogPoints) > 0:
            homogPoints = np.array(self.homogPoints) * binSize
            self.ax.scatter(x=homogPoints[:, 0], y=homogPoints[:, 1], c='y', s=60)

        btnAx = self.fig.add_axes([0.8, 0.0, 0.1, 0.07])
        Button(btnAx, 'Save point', color='0.85', hovercolor='0.95')

        # connect click handler
        self.fig.canvas.mpl_connect('button_press_event', lambda x: self.clickHomog(x, binSize=binSize))

    def clickHomog(self, event, binSize=1):
        if event.inaxes is not None:
            # save current zoom state of axis
            currXLim = self.ax.get_xlim()
            currYLim = self.ax.get_ylim()

            # clear current axis and redraw map
            self.ax.clear()
            self.plotHomog(updateCurrent=True)

            if event.inaxes is self.fig.axes[0]:
                # axis 0 then is a click on the map. Update selected point and plot
                self.selPoint = (int(event.xdata), int(event.ydata))
                self.ax.scatter(x=self.selPoint[0], y=self.selPoint[1], c='w', s=60, marker='x')

            elif (event.inaxes is self.fig.axes[1]) and (self.selPoint is not None):
                # axis 1 then is a click on the button. Check if a point is selected and add to list
                self.selPoint = tuple(int(round(x / binSize)) for x in self.selPoint)
                self.homogPoints.append(self.selPoint)
                self.selPoint = None

            # Plot stored homogo points if there are any
            if len(self.homogPoints) > 0:
                homogPoints = np.array(self.homogPoints) * binSize
                self.ax.scatter(x=homogPoints[:, 0], y=homogPoints[:, 1], c='y', s=60)

            # Set zoom state back and redraw axis
            self.ax.set_xlim(currXLim)
            self.ax.set_ylim(currYLim)

            self.fig.canvas.draw()

    def updateHomogPoint(self, homogID, newPoint=None, delta=None):
        """Update a homog by either over wrting it with a new point or
        incrementing the current values.

        Args:
            homogID (int): ID (place in list) of point to update or -1 for all
            newPoint (tuple, optional): New point
            delta (tuple, optional): Increments to current point (dx, dy)
        """
        if type(homogID) is not int:
            raise Exception("homogID must be an integer.")
        if homogID >= len(self.homogPoints):
            raise Exception("homogID is out of range.")

        # Update all points
        if homogID < 0:
            for i in range(len(self.homogPoints)):
                self.updateHomogPoint(homogID=i, delta=delta)
        # Update a single point
        else:
            # overwrite point
            if newPoint is not None:
                if type(newPoint) is not tuple and len(newPoint) != 2:
                    raise Exception("newPoint must be a 2 component tuple")

            # increment current point
            elif delta is not None:
                if type(delta) is not tuple and len(delta) != 2:
                    raise Exception("delta must be a 2 component tuple")
                newPoint = list(self.homogPoints[homogID])
                newPoint[0] += delta[0]
                newPoint[1] += delta[1]
                newPoint = tuple(newPoint)

            self.homogPoints[homogID] = newPoint

    def highlightGrains(self, grainIds, grainColours):
        if grainColours is None:
            grainColours = ['white']

        outline = np.zeros((self.yDim, self.xDim), dtype=int)
        for i, grainId in enumerate(grainIds, start=1):
            if i > len(grainColours):
                i = len(grainColours)
            # outline of highlighted grain
            grainOutline = self.grainList[grainId].grainOutline(bg=0, fg=i)
            x0, y0, xmax, ymax = self.grainList[grainId].extremeCoords

            # use logical of same are in entire area to ensure neigbouring grains display correctly
            # grainOutline = np.logical_or(outline[y0:ymax + 1, x0:xmax + 1], grainOutline).astype(int)
            # outline[y0:ymax + 1, x0:xmax + 1] = grainOutline

            outline[y0:ymax + 1, x0:xmax + 1] = outline[y0:ymax + 1, x0:xmax + 1] + grainOutline

        grainColours.insert(0, 'white')

        # Custom colour map where 0 is tranparent white for bg and 255 is opaque white for fg
        cmap1 = mpl.colors.ListedColormap(grainColours)
        cmap1._init()
        alpha = np.full(cmap1.N + 3, self.highlightAlpha)
        alpha[0] = 0
        cmap1._lut[:, -1] = alpha

        self.ax.imshow(outline, interpolation='none', cmap=cmap1)
        return

    def buildNeighbourNetwork(self):
        # Construct a list of neighbours

        yLocs, xLocs = np.nonzero(self.boundaries)
        neighboursList = []

        for y, x in zip(yLocs, xLocs):
            if x == 0 or y == 0 or x == self.grains.shape[1] - 1 or y == self.grains.shape[0] - 1:
                # exclude boundary pixel of map
                continue
            else:
                # use sets as they do not allow duplicate elements
                # minus 1 on all as the grain image starts labeling at 1
                neighbours = {self.grains[y + 1, x] - 1, self.grains[y - 1, x] - 1,
                              self.grains[y, x + 1] - 1, self.grains[y, x - 1] - 1}
                # neighbours = set(neighbours)
                # remove boundary points (-2) and points in small grains (-3) (Normally -1 and -2)
                neighbours.discard(-2)
                neighbours.discard(-3)

                nunNeig = len(neighbours)

                if nunNeig == 1:
                    continue
                elif nunNeig == 2:
                    neighboursSplit = [neighbours]
                elif nunNeig > 2:
                    neighbours = list(neighbours)
                    neighboursSplit = []
                    for i in range(nunNeig):
                        for j in range(i + 1, nunNeig):
                            neighboursSplit.append({neighbours[i], neighbours[j]})

                for trialNeig in neighboursSplit:
                    if trialNeig not in neighboursList:
                        neighboursList.append(trialNeig)

        # create network
        import networkx as nx
        self.neighbourNetwork = nx.Graph()
        self.neighbourNetwork.add_nodes_from(range(len(self)))
        self.neighbourNetwork.add_edges_from(neighboursList)

    def displayNeighbours(self):
        self.locateGrainID(clickEvent=self.clickGrainNeighbours)

    def clickGrainNeighbours(self, event, map):
        if event.inaxes is not None:
            # grain id of selected grain
            grainId = int(self.grains[int(event.ydata), int(event.xdata)] - 1)
            if grainId < 0:
                return
            self.currGrainId = grainId

            # clear current axis and redraw euler map with highlighted grain overlay
            self.ax.clear()
            highlightGrains = [self.currGrainId] + list(self.neighbourNetwork.neighbors(self.currGrainId))

            secondNeighbours = []

            for firstNeighbour in list(self.neighbourNetwork.neighbors(self.currGrainId)):
                trialSecondNeighbours = list(self.neighbourNetwork.neighbors(firstNeighbour))
                for secondNeighbour in trialSecondNeighbours:
                    if secondNeighbour not in highlightGrains and secondNeighbour not in secondNeighbours:
                        secondNeighbours.append(secondNeighbour)

            highlightColours = ['white']
            highlightColours.extend(['yellow'] * (len(highlightGrains) - 1))
            highlightColours.append('green')

            highlightGrains.extend(secondNeighbours)

            self.plotDefault(updateCurrent=True, highlightGrains=highlightGrains,
                             highlightColours=highlightColours)
            self.fig.canvas.draw()

    @property
    def proxigram(self):
        self.calcProxigram(forceCalc=False)

        return self.proxigramArr

    def calcProxigram(self, numTrials=500, forceCalc=True):
        if self.proxigramArr is not None and not forceCalc:
            return

        proxBoundaries = np.copy(self.boundaries)
        proxShape = proxBoundaries.shape

        # ebsd boundary arrays have extra boundary along right and bottom edge. These need to be removed
        # rigth edge
        if np.all(proxBoundaries[:, -1] == -1):
            proxBoundaries[:, -1] = proxBoundaries[:, -2]
        # bottom edge
        if np.all(proxBoundaries[-1, :] == -1):
            proxBoundaries[-1, :] = proxBoundaries[-2, :]

        # create list of positions of each boundary point
        indexBoundaries = []
        for index, value in np.ndenumerate(proxBoundaries):
            if value == -1:
                indexBoundaries.append(index)
        # add 0.5 to boundary coordiantes as they are placed on the bottom right edge pixels of grains
        indexBoundaries = np.array(indexBoundaries) + 0.5

        # array of x and y coordinate of each pixel in the map
        coords = np.zeros((2, proxShape[0], proxShape[1]), dtype=float)
        coords[0], coords[1] = np.meshgrid(range(proxShape[0]), range(proxShape[1]), indexing='ij')

        # array to store trial distance from each boundary point
        trialDistances = np.full((numTrials + 1, proxShape[0], proxShape[1]), 1000, dtype=float)

        # loop over each boundary point (p) and calcuale distance from p to all points in the map
        # store minimum once numTrails have been made and start a new batch of trials
        print("Calculating proxigram ", end='')
        numBoundaryPoints = len(indexBoundaries)
        j = 1
        for i, indexBoundary in enumerate(indexBoundaries):
            trialDistances[j] = np.sqrt((coords[0] - indexBoundary[0])**2 + (coords[1] - indexBoundary[1])**2)

            if j == numTrials:
                # find current minimum distances and store
                trialDistances[0] = trialDistances.min(axis=0)
                j = 0
                print("{:.1f}% ".format(i / numBoundaryPoints * 100), end='')
            j += 1

        # find final minimum distances to a boundary
        self.proxigramArr = trialDistances.min(axis=0)

        trialDistances = None


class Grain(object):

    def __init__(self):
        self.coordList = []         # list of coords stored as tuples (x, y). These are corrds in a cropped image if crop exists

    def __len__(self):
        return len(self.coordList)

    @property
    def extremeCoords(self):
        unzippedCoordlist = list(zip(*self.coordList))
        x0 = min(unzippedCoordlist[0])
        y0 = min(unzippedCoordlist[1])
        xmax = max(unzippedCoordlist[0])
        ymax = max(unzippedCoordlist[1])

        return x0, y0, xmax, ymax

    @property
    def centreCoords(self):
        x0, y0, xmax, ymax = self.extremeCoords
        xCentre = int((xmax - x0) / 2)
        yCentre = int((ymax - y0) / 2)

        return xCentre, yCentre

    def grainOutline(self, bg=np.nan, fg=0):
        x0, y0, xmax, ymax = self.extremeCoords

        # initialise array with nans so area not in grain displays white
        outline = np.full((ymax - y0 + 1, xmax - x0 + 1), bg, dtype=int)

        for coord in self.coordList:
            outline[coord[1] - y0, coord[0] - x0] = fg

        return outline

    def plotOutline(self):
        plt.figure()
        plt.imshow(self.grainOutline(), interpolation='none')
        plt.colorbar()

    def plotSlipTraces(self, colours=None, ax=None, pos=None):
        if colours is None:
            colours = self.ebsdMap.slipTraceColours

        if pos is None:
            pos = self.centreCoords

        for i, slipTraceAngle in enumerate(self.slipTraces):
            slipTrace = np.array((-np.sin(slipTraceAngle), np.cos(slipTraceAngle)))
            colour = colours[len(colours) - 1] if i >= len(colours) else colours[i]
            if ax is None:
                plt.quiver(
                    pos[0], pos[1],
                    slipTrace[0], slipTrace[1],
                    scale=1, pivot="middle",
                    color=colour, headwidth=1,
                    headlength=0
                )
            else:
                ax.quiver(
                    pos[0], pos[1],
                    slipTrace[0], slipTrace[1],
                    scale=1, pivot="middle",
                    color=colour, headwidth=1,
                    headlength=0
                )

    def grainData(self, mapData):
        """Takes this grains data from the given map data

        Args:
            mapData (np.array): Array of map data. This must be cropped!

        Returns:
            np.array: Array containing this grains values from the given map data
        """
        grainData = np.zeros(len(self), dtype=mapData.dtype)

        for i, coord in enumerate(self.coordList):
            grainData[i] = mapData[coord[1], coord[0]]

        return grainData

    def grainMapData(self, mapData, bg=np.nan):
        """Creates a map of this grain only from the given map data

        Args:
            mapData (np.array): Array of map data. This must be cropped!
            bg (float, optional): Value to fill the backgraound with. Must be same dtype as input.

        Returns:
            np.array: Map of this grains data
        """
        grainData = self.grainData(mapData)
        x0, y0, xmax, ymax = self.extremeCoords

        grainMapData = np.full((ymax - y0 + 1, xmax - x0 + 1), bg, dtype=mapData.dtype)

        for coord, data in zip(self.coordList, grainData):
            grainMapData[coord[1] - y0, coord[0] - x0] = data

        return grainMapData

    def grainMapDataCoarse(self, mapData, kernelSize=2, bg=np.nan):
        """Creates a coarsed map of this grain only from the given map data.
           Data is coarsened using a kenel at each pixel in the grain using only this grains data.

        Args:
            mapData (np.array): Array of map data. This must be cropped!
            kernelSize (int, optional): Size of kernel as the number of pixels to dilate by i.e 1 gives a 3x3 kernel.
            bg (float, optional): Value to fill the backgraound with. Must be same dtype as input.

        Returns:
            np.array: Map of this grains coarsened data
        """
        grainMapData = self.grainMapData(mapData)
        grainMapDataCoarse = np.full_like(grainMapData, np.nan)

        for i, j in np.ndindex(grainMapData.shape):
            if np.isnan(grainMapData[i, j]):
                grainMapDataCoarse[i, j] = bg
            else:
                coarseValue = 0

                yLow = i - kernelSize if i - kernelSize >= 0 else 0
                yHigh = i + kernelSize + 1 if i + kernelSize + 1 <= grainMapData.shape[0] else grainMapData.shape[0]

                xLow = j - kernelSize if j - kernelSize >= 0 else 0
                xHigh = j + kernelSize + 1 if j + kernelSize + 1 <= grainMapData.shape[1] else grainMapData.shape[1]

                numPoints = 0
                for k in range(yLow, yHigh):
                    for l in range(xLow, xHigh):
                        if not np.isnan(grainMapData[k, l]):
                            coarseValue += grainMapData[k, l]
                            numPoints += 1

                grainMapDataCoarse[i, j] = coarseValue / numPoints if numPoints > 0 else np.nan

        return grainMapDataCoarse

    def plotGrainData(self, mapData, vmin=None, vmax=None, clabel='', cmap='viridis'):
        """Plot a map of this grain only from the given map data.

        Args:
            mapData (np.array): Array of map data. This must be cropped!
            vmin (float, optional): Minimum value of colour scale
            vmax (float, optional): Maximum value for colour scale
            clabel (str, optional): Colour bar label text
            cmap (str, optional): Colour map to use, default is viridis.
        """
        grainMapData = self.grainMapData(mapData)

        plt.figure()
        plt.imshow(grainMapData, interpolation='none', vmin=vmin, vmax=vmax, cmap=cmap)

        plt.colorbar(label=clabel)
        plt.xticks([])
        plt.yticks([])

        return grainMapData


class SlipSystem(object):
    def __init__(self, slipPlane, slipDir, crystalSym, cOverA=None):
        # Currently only for cubic
        self.crystalSym = crystalSym    # symmetry of material e.g. "cubic", "hexagonal"

        # Stored as Miller indicies (Miller-Bravais for hexagonal)
        self.slipPlaneMiller = slipPlane
        self.slipDirMiller = slipDir

        # Stored as vectors in a cartesian basis
        if crystalSym == "cubic":
            self.slipPlaneOrtho = slipPlane / np.sqrt(np.dot(slipPlane, slipPlane))
            self.slipDirOrtho = slipDir / np.sqrt(np.dot(slipDir, slipDir))
        elif crystalSym == "hexagonal":
            if cOverA is None:
                raise Exception("No c over a ratio given")
            self.cOverA = cOverA

            # Convert plane and dir from Miller-Bravais to Miller
            slipPlaneM = slipPlane[[0, 1, 3]]
            slipDirM = slipDir[[0, 1, 3]]
            slipDirM[[0, 1]] -= slipDir[2]

            # Create L matrix. Transformation from crystal to orthonormal coords
            lMatrix = SlipSystem.lMatrix(1, 1, cOverA, np.pi / 2, np.pi / 2, np.pi * 2 / 3)

            # Create Q matrix fro transforming planes
            qMatrix = SlipSystem.qMatrix(lMatrix)

            # Transform into orthonormal basis and then normalise
            self.slipPlaneOrtho = np.matmul(qMatrix, slipPlaneM)
            self.slipDirOrtho = np.matmul(lMatrix, slipDirM)
            self.slipPlaneOrtho /= np.sqrt(np.dot(self.slipPlaneOrtho, self.slipPlaneOrtho))
            self.slipDirOrtho /= np.sqrt(np.dot(self.slipDirOrtho, self.slipDirOrtho))
        else:
            raise Exception("Only cubic and hexagonal currently supported.")

    # overload ==. Two slip systems are equal if they have the same slip plane in miller
    def __eq__(self, right):
        return np.all(self.slipPlaneMiller == right.slipPlaneMiller)

    @property
    def slipPlane(self):
        return self.slipPlaneOrtho

    @property
    def slipDir(self):
        return self.slipDirOrtho

    @property
    def slipPlaneLabel(self):
        slipPlane = self.slipPlaneMiller
        if self.crystalSym == "hexagonal":
            return "({:d}{:d}{:d}{:d})".format(slipPlane[0], slipPlane[1], slipPlane[2], slipPlane[3])
        else:
            return "({:d}{:d}{:d})".format(slipPlane[0], slipPlane[1], slipPlane[2])

    @property
    def slipDirLabel(self):
        slipDir = self.slipDirMiller
        if self.crystalSym == "hexagonal":
            return "[{:d}{:d}{:d}{:d}]".format(slipDir[0], slipDir[1], slipDir[2], slipDir[3])
        else:
            return "[{:d}{:d}{:d}]".format(slipDir[0], slipDir[1], slipDir[2])

    @staticmethod
    def loadSlipSystems(filepath, crystalSym, cOverA=None):
        """Load in slip systems from file. 3 integers for slip plane normal and
           3 for slip direction. Returns a list of list of slip systems
           grouped by slip plane.

        Args:
            filepath (string): Path to file containing slip systems
            crystalSym (string): The crystal symmetry ("cubic" or "hexagonal")

        Returns:
            list(list(SlipSystem)): A list of list of slip systems grouped slip plane.

        Raises:
            IOError: Raised if not 6/8 integers per line
        """

        f = open(filepath)
        f.readline()
        colours = f.readline().strip()
        slipTraceColours = colours.split(',')
        f.close()

        if crystalSym == "hexagonal":
            vectSize = 4
        else:
            vectSize = 3

        ssData = np.loadtxt(filepath, delimiter='\t', skiprows=2, dtype=int)
        if ssData.shape[1] != 2 * vectSize:
            raise IOError("Slip system file not valid")

        # Create list of slip system objects
        slipSystems = []
        for row in ssData:
            slipSystems.append(SlipSystem(row[0:vectSize], row[vectSize:2 * vectSize], crystalSym, cOverA=cOverA))

        # Group slip sytems by slip plane
        groupedSlipSystems = SlipSystem.groupSlipSystems(slipSystems)

        return groupedSlipSystems, slipTraceColours

    @staticmethod
    def groupSlipSystems(slipSystems):
        """Groups slip systems by there slip plane.

        Args:
            slipSytems (list(SlipSystem)): A list of slip systems

        Returns:
            list(list(SlipSystem)): A list of list of slip systems grouped slip plane.
        """
        distSlipSystems = [slipSystems[0]]
        groupedSlipSystems = [[slipSystems[0]]]

        for slipSystem in slipSystems[1:]:

            for i, distSlipSystem in enumerate(distSlipSystems):
                if slipSystem == distSlipSystem:
                    groupedSlipSystems[i].append(slipSystem)
                    break
            else:
                distSlipSystems.append(slipSystem)
                groupedSlipSystems.append([slipSystem])

        return groupedSlipSystems

    @staticmethod
    def lMatrix(a, b, c, alpha, beta, gamma):
        lMatrix = np.zeros((3, 3))

        cosAlpha = np.cos(alpha)
        cosBeta = np.cos(beta)
        cosGamma = np.cos(gamma)

        sinGamma = np.sin(gamma)

        # From Randle and Engle - Intro to texture analysis
        lMatrix[0, 0] = a
        lMatrix[0, 1] = b * cosGamma
        lMatrix[0, 2] = c * cosBeta

        lMatrix[1, 1] = b * sinGamma
        lMatrix[1, 2] = c * (cosAlpha - cosBeta * cosGamma) / sinGamma

        lMatrix[2, 2] = c * np.sqrt(1 + 2 * cosAlpha * cosBeta * cosGamma -
                                    cosAlpha**2 - cosBeta**2 - cosGamma**2) / sinGamma

        # Swap 00 with 11 and 01 with 10 due to how OI orthonormalises
        # From Brad Wynne
        t1 = lMatrix[0, 0]
        t2 = lMatrix[1, 0]

        lMatrix[0, 0] = lMatrix[1, 1]
        lMatrix[1, 0] = lMatrix[0, 1]

        lMatrix[1, 1] = t1
        lMatrix[0, 1] = t2

        # Set small components to 0
        lMatrix[np.abs(lMatrix) < 1e-10] = 0

        return lMatrix

    @staticmethod
    def qMatrix(lMatrix):
        # Construct matix of reciprocal lattice zectors to transform plane normals
        # See C. T. Young and J. L. Lytton, J. Appl. Phys., vol. 43, no. 4, pp. 1408–1417, 1972.
        a = lMatrix[:, 0]
        b = lMatrix[:, 1]
        c = lMatrix[:, 2]

        volume = abs(np.dot(a, np.cross(b, c)))
        aStar = np.cross(b, c) / volume
        bStar = np.cross(c, a) / volume
        cStar = np.cross(a, b) / volume

        qMatrix = np.stack((aStar, bStar, cStar), axis=1)

        return qMatrix
