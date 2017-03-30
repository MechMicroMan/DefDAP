import numpy as np

from matplotlib.widgets import Button


class Map(object):
    def __len__(self):
        return len(self.grainList)

    # allow array like getting of grains
    def __getitem__(self, key):
        return self.grainList[key]

    def setHomogPoint(self):
        self.selPoint = None

        self.plotDefault()
        homogPoints = np.array(self.homogPoints)
        self.ax.scatter(x=homogPoints[:, 0], y=homogPoints[:, 1], c='y', s=60)

        btnAx = self.fig.add_axes([0.8, 0.0, 0.1, 0.07])
        Button(btnAx, 'Save point', color='0.85', hovercolor='0.95')

        # coonect click handler
        self.fig.canvas.mpl_connect('button_press_event', self.clickHomog)

    def clickHomog(self, event):
        if event.inaxes is not None:
            # clear current axis and redraw map
            self.ax.clear()
            self.plotDefault(updateCurrent=True)

            if event.inaxes is self.fig.axes[0]:
                # axis 0 then is a click on the map. Update selected point and plot
                self.selPoint = (int(event.xdata), int(event.ydata))
                self.ax.scatter(x=self.selPoint[0], y=self.selPoint[1], c='w', s=60, marker='x')

            elif (event.inaxes is self.fig.axes[1]) and (self.selPoint is not None):
                # axis 1 then is a click on the button. Add selected point to list
                self.homogPoints.append(self.selPoint)
                self.selPoint = None

            homogPoints = np.array(self.homogPoints)
            self.ax.scatter(x=homogPoints[:, 0], y=homogPoints[:, 1], c='y', s=60)

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

            # Create L matrix
            lMatrix = SlipSystem.lMatrix(1, 1, cOverA, np.pi / 2, np.pi / 2, np.pi * 2 / 3)

            # Transform into ortho-normal basis and then normalise
            self.slipPlaneOrtho = lMatrix.dot(slipPlaneM)
            self.slipDirOrtho = lMatrix.dot(slipDirM)
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

        lMatrix[0, 0] = a
        lMatrix[0, 1] = b * cosGamma
        lMatrix[0, 2] = c * cosBeta

        lMatrix[1, 1] = b * sinGamma
        lMatrix[1, 2] = c * (cosAlpha - cosBeta * cosGamma) / sinGamma

        lMatrix[2, 2] = c * np.sqrt(1 + 2 * cosAlpha * cosBeta * cosGamma -
                                    cosAlpha**2 - cosBeta**2 - cosGamma**2) / sinGamma

        # t1 = lMatrix[0, 0]
        # t2 = lMatrix[1, 0]

        # lMatrix[0, 0] = lMatrix[1, 1]
        # lMatrix[1, 0] = lMatrix[0, 1]

        # lMatrix[1, 1] = t1
        # lMatrix[0, 1] = t2

        return lMatrix
