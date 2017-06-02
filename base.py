import numpy as np

import matplotlib as mpl
from matplotlib.widgets import Button


class Map(object):

    def __init__(self):
        self.homogPoints = []
        self.selPoint = None

    def __len__(self):
        return len(self.grainList)

    # allow array like getting of grains
    def __getitem__(self, key):
        return self.grainList[key]

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
            highlightGrains = [self.currGrainId] + self.neighbourNetwork.neighbors(self.currGrainId)

            secondNeighbours = []

            for firstNeighbour in self.neighbourNetwork.neighbors(self.currGrainId):
                trialSecondNeighbours = self.neighbourNetwork.neighbors(firstNeighbour)
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
            IOError: Raised if not 6 integers per line
        """
        if crystalSym == "hexagonal":
            vectSize = 4
        else:
            vectSize = 3

        ssData = np.loadtxt(filepath, delimiter='\t', skiprows=1, dtype=int)
        if ssData.shape[1] != 2 * vectSize:
            raise IOError("Slip system file not valid")

        # Create list of slip system objects
        slipSystems = []
        for row in ssData:
            slipSystems.append(SlipSystem(row[0:vectSize], row[vectSize:2 * vectSize], crystalSym, cOverA=cOverA))

        # Group slip sytems by slip plane
        groupedSlipSystems = SlipSystem.groupSlipSystems(slipSystems)

        return groupedSlipSystems

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
