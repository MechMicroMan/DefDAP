import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from skimage import transform as tf
from skimage import morphology as mph

from scipy.stats import mode

import peakutils

from . import base
from .quat import Quat


class Map(base.Map):

    def __init__(self, path, fname):
        # Call base class constructor
        super(Map, self).__init__()

        self.ebsdMap = None
        self.ebsdTransform = None       # Transform from EBSD to DIC coordinates
        self.ebsdTransformInv = None    # Transform from DIC to EBSD coordinates
        self.grainList = None
        self.currGrainId = None     # Id of last selected grain
        # ...
        self.ebsdGrainIds = None
        self.patternImPath = None   # Path to BSE image of map
        self.windowSize = None      # Window size for map

        self.plotHomog = self.plotMaxShear
        self.highlightAlpha = 0.6

        self.path = path
        self.fname = fname
        # Load in data
        self.data = np.loadtxt(self.path + self.fname, skiprows=1)
        self.xc = self.data[:, 0]  # x coordinates
        self.yc = self.data[:, 1]  # y coordinates
        self.xd = self.data[:, 2]  # x displacement
        self.yd = self.data[:, 3]  # y displacement

        # Calculate size of map
        self.xdim = int((self.xc.max() - self.xc.min()) /
                        min(abs((np.diff(self.xc)))) + 1)  # size of map along x
        self.ydim = int((self.yc.max() - self.yc.min()) /
                        max(abs((np.diff(self.yc)))) + 1)  # size of map along y

        # *dim are full size of data. *Dim are size after cropping
        self.xDim = self.xdim
        self.yDim = self.ydim

        self.x_map = self._map(self.xd)     # u (displacement component along x)
        self.y_map = self._map(self.yd)     # v (displacement component along x)
        xDispGrad = self._grad(self.x_map)
        yDispGrad = self._grad(self.y_map)
        self.f11 = xDispGrad[1] + 1     # f11
        self.f22 = yDispGrad[0] + 1     # f22
        self.f12 = xDispGrad[0]         # f12
        self.f21 = yDispGrad[1]         # f21

        self.max_shear = np.sqrt((((self.f11 - self.f22) / 2.)**2) +
                                 ((self.f12 + self.f21) / 2.)**2)  # max shear component
        self.mapshape = np.shape(self.max_shear)

        self.cropDists = np.array(((0, 0), (0, 0)), dtype=int)

    def plotDefault(self, *args, **kwargs):
        self.plotMaxShear(plotGBs=True, *args, **kwargs)

    def _map(self, data_col):
        data_map = np.reshape(np.array(data_col), (self.ydim, self.xdim))
        return data_map

    def _grad(self, data_map):
        grad_step = min(abs((np.diff(self.xc))))
        data_grad = np.gradient(data_map, grad_step, grad_step)
        return data_grad

    def setCrop(self, xMin=None, xMax=None, yMin=None, yMax=None, updateHomogPoints=False):
        # changes in homog points
        dx = 0
        dy = 0

        # update crop distances
        if xMin is not None:
            if updateHomogPoints:
                dx = self.cropDists[0, 0] - int(xMin)
            self.cropDists[0, 0] = int(xMin)
        if xMax is not None:
            self.cropDists[0, 1] = int(xMax)
        if yMin is not None:
            if updateHomogPoints:
                dy = self.cropDists[1, 0] - int(yMin)
            self.cropDists[1, 0] = int(yMin)
        if yMax is not None:
            self.cropDists[1, 1] = int(yMax)

        # update homogo points if required
        if updateHomogPoints and (dx != 0 or dy != 0):
            self.updateHomogPoint(homogID=-1, delta=(dx, dy))

        # set new cropped dimensions
        self.xDim = self.xdim - self.cropDists[0, 0] - self.cropDists[0, 1]
        self.yDim = self.ydim - self.cropDists[1, 0] - self.cropDists[1, 1]

    def crop(self, mapData, binned=True):
        if binned:
            multiplier = 1
        else:
            multiplier = self.windowSize

        return mapData[int(self.cropDists[1, 0] * multiplier):int((self.ydim - self.cropDists[1, 1]) * multiplier),
                       int(self.cropDists[0, 0] * multiplier):int((self.xdim - self.cropDists[0, 1]) * multiplier)]

    def setHomogPoint(self, display=None):
        if display is None:
            display = "maxshear"

        # Set plot dafault to display selected image
        display = display.lower().replace(" ", "")
        if display == "bse" or display == "pattern":
            self.plotHomog = self.plotPattern
            binSize = self.windowSize
        else:
            self.plotHomog = self.plotMaxShear
            binSize = 1

        # Call set homog points from base class setting the bin size
        super(type(self), self).setHomogPoint(binSize=binSize)

    def linkEbsdMap(self, ebsdMap, transformType="affine", order=2):
        self.ebsdMap = ebsdMap
        if transformType == "piecewiseAffine":
            self.ebsdTransform = tf.PiecewiseAffineTransform()
            self.ebsdTransformInv = self.ebsdTransform.inverse
        elif transformType == "polynomial":
            self.ebsdTransform = tf.PolynomialTransform()
            # You can't calculate the inverse of a polynomial transform so have to estimate
            # by swapping source and destination homog points
            self.ebsdTransformInv = tf.PolynomialTransform()
            self.ebsdTransformInv.estimate(np.array(self.ebsdMap.homogPoints), np.array(self.homogPoints), order=order)
            # calculate transform from EBSD to DIC frame
            self.ebsdTransform.estimate(np.array(self.homogPoints), np.array(self.ebsdMap.homogPoints), order=order)
            return
        else:
            self.ebsdTransform = tf.AffineTransform()
            self.ebsdTransformInv = self.ebsdTransform.inverse

        # calculate transform from EBSD to DIC frame
        self.ebsdTransform.estimate(np.array(self.homogPoints), np.array(self.ebsdMap.homogPoints))

    def warpToDicFrame(self, mapData, cropImage=True):
        if (cropImage or type(self.ebsdTransform) is not tf.AffineTransform):
            # crop to size of DIC map
            outputShape = (self.yDim, self.xDim)
            # warp the map
            warpedMap = tf.warp(mapData, self.ebsdTransform, output_shape=outputShape)
        else:
            # copy ebsd transform and change translation to give an extra 5% border
            # to show the entire image after rotation/shearing
            tempEbsdTransform = tf.AffineTransform(matrix=np.copy(self.ebsdTransform.params))
            tempEbsdTransform.params[0:2, 2] = -0.05 * np.array(mapData.shape)

            # output the entire warped image with 5% border (add some extra to fix a bug)
            outputShape = np.array(mapData.shape) * 1.4 / tempEbsdTransform.scale

            # warp the map
            warpedMap = tf.warp(mapData, tempEbsdTransform, output_shape=outputShape.astype(int))

        # return map
        return warpedMap

    @property
    def boundaries(self):
        # image is returned cropped if a piecewise transform is being used
        boundaries = self.warpToDicFrame(-self.ebsdMap.boundaries.astype(float), cropImage=False) > 0.1

        boundaries = mph.skeletonize(boundaries)
        mph.remove_small_objects(boundaries, min_size=10, in_place=True, connectivity=2)

        # crop image if it is a simple affine transform
        if type(self.ebsdTransform) is tf.AffineTransform:
            # need to apply the translation of ebsd transform and remove 5% border
            crop = np.copy(self.ebsdTransform.params[0:2, 2])
            crop += 0.05 * np.array(self.ebsdMap.boundaries.shape)
            # the crop is defined in EBSD coords so need to transform it
            transformMatrix = np.copy(self.ebsdTransform.params[0:2, 0:2])
            crop = np.matmul(np.linalg.inv(transformMatrix), crop)
            crop = crop.round().astype(int)

            boundaries = boundaries[crop[1]:crop[1] + self.yDim,
                                    crop[0]:crop[0] + self.xDim]

        boundaries = -boundaries.astype(int)

        return boundaries

    def setPatternPath(self, filePath, windowSize):
        """Set path of BSE image of pattern. filePath is relative to the path set when constructing."""

        self.patternImPath = self.path + filePath
        self.windowSize = windowSize

    def plotPattern(self, updateCurrent=False, vmin=None, vmax=None):
        """Plot BSE image of Map. For use with setting homog points"""

        # Check image path is set
        if self.patternImPath is not None:
            if not updateCurrent:
                self.fig, self.ax = plt.subplots()

            bseImage = plt.imread(self.patternImPath)
            self.ax.imshow(self.crop(bseImage, binned=False), cmap='gray', vmin=vmin, vmax=vmax)

        else:
            raise Exception("First set path to pattern image.")

    def plotMaxShear(self, plotGBs=False, dilateBoundaries=False, boundaryColour='white', plotSlipTraces=False, plotPercent=True,
                     updateCurrent=False, highlightGrains=None, highlightColours=None,
                     plotColourBar=False, vmin=None, vmax=None,
                     slipTraceColours=["white", "green", "red", "black"]):
        if not updateCurrent:
            # self.fig, self.ax = plt.subplots(figsize=(5.75, 4))
            self.fig, self.ax = plt.subplots()

        multiplier = 100 if plotPercent else 1
        img = self.ax.imshow(self.crop(self.max_shear) * multiplier,
                             cmap='viridis', interpolation='None', vmin=vmin, vmax=vmax)
        if plotColourBar:
            plt.colorbar(img, ax=self.ax, label="Effective shear strain (%)")

        if plotGBs:
            self.plotGBs(ax=self.ax, colour=boundaryColour, dilate=dilateBoundaries)

        if highlightGrains is not None:
            self.highlightGrains(highlightGrains, highlightColours)

        # plot slip traces
        if plotSlipTraces:
            numGrains = len(self.grainList)
            numSS = len(self.ebsdMap.slipSystems)
            grainSizeData = np.zeros((numGrains, 4))
            slipTraceData = np.zeros((numGrains, numSS, 2))

            i = 0   # keep track of number of slip traces
            for grain in self.grainList:
                if len(grain) < 1000:
                    continue

                # x0, y0, xmax, ymax
                grainSizeData[i, 0], grainSizeData[i, 1], grainSizeData[i, 2], grainSizeData[i, 3] = grain.extremeCoords

                for j, slipTrace in enumerate(grain.slipTraces()):
                    slipTraceData[i, j, 0:2] = slipTrace[0:2]

                i += 1

            grainSizeData = grainSizeData[0:i, :]
            slipTraceData = slipTraceData[0:i, :, :]

            scale = 4 / ((grainSizeData[:, 2] - grainSizeData[:, 0]) / self.xDim +
                         (grainSizeData[:, 3] - grainSizeData[:, 1]) / self.xDim)

            xPos = grainSizeData[:, 0] + (grainSizeData[:, 2] - grainSizeData[:, 0]) / 2
            yPos = grainSizeData[:, 1] + (grainSizeData[:, 3] - grainSizeData[:, 1]) / 2

            colours = slipTraceColours

            for i, colour in enumerate(colours[0:numSS]):
                self.ax.quiver(xPos, yPos, slipTraceData[:, i, 0], slipTraceData[:, i, 1],
                               scale=scale, pivot="middle", color=colour, headwidth=1,
                               headlength=0, width=0.002)

        return

    def plotGrainAvMaxShear(self, plotGBs=False, plotColourBar=True, vmin=None, vmax=None, clabel=''):
        """Plot grain map with grains filled with average value of max shear.
        This uses the max shear values stored in grain objects, to plot other data
        use plotGrainAv().

        Args:
            plotGBs (bool, optional): Set to True to draw grain boundaries
            plotColourBar (bool, optional): Set to Flase to exclude the colour bar
            vmin (float, optional): Minimum value of colour scale
            vmax (float, optional): Maximum value for colour scale
            clabel (str, optional): Colour bar label text
        """
        plt.figure()

        grainAvMaxShear = np.zeros([self.yDim, self.xDim])

        for grain in self.grainList:
            avMaxShear = np.array(grain.maxShearList).mean()

            for coord in grain.coordList:
                grainAvMaxShear[coord[1], coord[0]] = avMaxShear

        plt.imshow(grainAvMaxShear * 100, vmin=0, vmax=6)

        if plotColourBar:
                plt.colorbar(label="Effective shear strain (%)")

        if plotGBs:
            self.plotGBs()

    def calcGrainAv(self, mapData):
        """Calculate grain average of any DIC map data.

        Args:
            mapData (np.array): Array of map data to grain average. This must be cropped!

        Returns:
            np.array: Array containing the grain average values
        """
        grainAvData = np.zeros(len(self))

        for grainId, grain in enumerate(self.grainList):
            grainData = grain.grainData(mapData)
            grainAvData[grainId] = grainData.mean()

        return grainAvData

    def plotGrainAv(self, mapData, plotGBs=False, plotColourBar=True, vmin=None, vmax=None, clabel=''):
        """Plot grain map with grains filled with average value of from any DIC map data

        Args:
            mapData (np.array): Array of map data to grain average. This must be cropped!
            plotGBs (bool, optional): Set to True to draw grain boundaries
            plotColourBar (bool, optional): Set to Flase to exclude the colour bar
            vmin (float, optional): Minimum value of colour scale
            vmax (float, optional): Maximum value for colour scale
            clabel (str, optional): Colour bar label text
        """
        plt.figure()

        grainAvData = self.calcGrainAv(mapData)

        grainAvMap = np.zeros([self.yDim, self.xDim])

        for grainId, grain in enumerate(self.grainList):
            grainAv = grainAvData[grainId]

            for coord in grain.coordList:
                grainAvMap[coord[1], coord[0]] = grainAv

        plt.imshow(grainAvMap, vmin=vmin, vmax=vmax)

        if plotColourBar:
            plt.colorbar(label=clabel)

        if plotGBs:
            self.plotGBs()

    def plotGrainAvIPF(self, mapData, direction, plotColourBar=True, vmin=None, vmax=None, clabel=''):
        """Plot IPF of grain reference (average) orientations with points coloured
        by grain average values from map data.

        Args:
            mapData (np.array): Array of map data to grain average. This must be cropped!
            direction (np.array): Vector of reference direction for the IPF
            plotColourBar (bool, optional): Set to Flase to exclude the colour bar
            vmin (float, optional): Minimum value of colour scale
            vmax (float, optional): Maximum value for colour scale
            clabel (str, optional): Colour bar label text
        """
        plt.figure()
        grainAvData = self.calcGrainAv(mapData)

        grainOri = np.empty(len(self), dtype=Quat)

        for grainId, grain in enumerate(self.grainList):
            grainOri[grainId] = grain.ebsdGrain.refOri

        Quat.plotIPF(grainOri, direction, self.ebsdMap.symGroup, c=grainAvData,
                     marker='o', vmin=vmin, vmax=vmax)

        if plotColourBar:
            plt.colorbar(label=clabel)

    def locateGrainID(self, clickEvent=None, displaySelected=False, vmin=None, vmax=None, dilateBoundaries=False):
        if (self.grainList is not None) and (self.grainList != []):
            # reset current selected grain and plot max shear map with click handler
            self.currGrainId = None
            self.plotMaxShear(plotGBs=True, vmin=vmin, vmax=vmax, dilateBoundaries=dilateBoundaries)
            if clickEvent is None:
                # default click handler which highlights grain and prints id
                self.fig.canvas.mpl_connect('button_press_event', lambda x: self.clickGrainId(x, displaySelected, vmin=None, vmax=None))
            else:
                # click handler loaded in as parameter. Pass current map object to it.
                self.fig.canvas.mpl_connect('button_press_event', lambda x: clickEvent(x, self))

            # unset figure for plotting grains
            self.grainFig = None
            self.grainAx = None

        else:
            raise Exception("Grain list empty")

    def clickGrainId(self, event, displaySelected, vmin=0, vmax=10):
        if event.inaxes is not None:
            # grain id of selected grain
            self.currGrainId = int(self.grains[int(event.ydata), int(event.xdata)] - 1)
            print("Grain ID: {}".format(self.currGrainId))

            # clear current axis and redraw map with highlighted grain overlay
            self.ax.clear()
            self.plotMaxShear(plotGBs=True, updateCurrent=True, highlightGrains=[self.currGrainId],
                              vmin=vmin, vmax=vmax, highlightColours=['green'])
            self.fig.canvas.draw()

            if displaySelected:
                if self.grainFig is None:
                    self.grainFig, self.grainAx = plt.subplots()
                self.grainList[self.currGrainId].calcSlipTraces()
                self.grainAx.clear()
                self.grainList[self.currGrainId].plotMaxShear(plotSlipTraces=True,
                                                              plotShearBands=True,
                                                              ax=self.grainAx,
                                                              vmin=vmin,
                                                              vmax=vmax)
                self.grainFig.canvas.draw()

    def findGrains(self, minGrainSize=10):
        # Initialise the grain map
        self.grains = np.copy(self.boundaries)

        self.grainList = []

        # List of points where no grain has been set yet
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

        # Now link grains to those in ebsd Map
        # Warp DIC grain map to EBSD frame
        dicGrains = self.grains
        warpedDicGrains = tf.warp(dicGrains.astype(float), self.ebsdTransformInv,
                                  output_shape=(self.ebsdMap.yDim, self.ebsdMap.xDim), order=0).astype(int)

        # Initalise list to store ID of corresponding grain in EBSD map. Also stored in grain objects
        self.ebsdGrainIds = []

        for i in range(len(self.grainList)):
            # Find grain by masking the native ebsd grain image with selected grain from
            # the warped dic grain image. The modal value is the EBSD grain label.
            modeId, _ = mode(self.ebsdMap.grains[warpedDicGrains == i + 1])

            self.ebsdGrainIds.append(modeId[0] - 1)
            self.grainList[i].ebsdGrainId = modeId[0] - 1
            self.grainList[i].ebsdGrain = self.ebsdMap.grainList[modeId[0] - 1]

        return

    def floodFill(self, x, y, grainIndex):
        currentGrain = Grain(self)

        currentGrain.addPoint((x, y), self.max_shear[y + self.cropDists[1, 0], x + self.cropDists[0, 0]])

        edge = [(x, y)]
        grain = [(x, y)]

        self.grains[y, x] = grainIndex
        while edge:
            newedge = []

            for (x, y) in edge:
                moves = np.array([(x + 1, y),
                                  (x - 1, y),
                                  (x, y + 1),
                                  (x, y - 1)])

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
                        currentGrain.addPoint((s, t), self.max_shear[y + self.cropDists[1, 0],
                                                                     x + self.cropDists[0, 0]])
                        newedge.append((s, t))
                        grain.append((s, t))
                        self.grains[t, s] = grainIndex
                    elif self.grains[t, s] == -1 and (s > x or t > y):
                        currentGrain.addPoint((s, t), self.max_shear[y + self.cropDists[1, 0],
                                                                     x + self.cropDists[0, 0]])
                        grain.append((s, t))
                        self.grains[t, s] = grainIndex

            if newedge == []:
                return currentGrain
            else:
                edge = newedge


class Grain(object):
    def __init__(self, dicMap):
        self.dicMap = dicMap       # dic map this grain is a member of
        self.coordList = []         # list of coords stored as tuples (x, y). These are corrds in a cropped image
        self.maxShearList = []
        self.ebsdGrain = None
        return

    def __len__(self):
        return len(self.coordList)

    # coord is a tuple (x, y)
    def addPoint(self, coord, maxShear):
        self.coordList.append(coord)
        self.maxShearList.append(maxShear)
        return

    @property
    def extremeCoords(self):
        unzippedCoordlist = list(zip(*self.coordList))
        x0 = min(unzippedCoordlist[0])
        y0 = min(unzippedCoordlist[1])
        xmax = max(unzippedCoordlist[0])
        ymax = max(unzippedCoordlist[1])
        return x0, y0, xmax, ymax

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
        return

    def plotMaxShear(self, plotPercent=True, plotSlipTraces=False, plotShearBands=False,
                     vmin=None, vmax=None, cmap="viridis", ax=None,
                     slipTraceColours=["white", "green", "red", "black"]):

        multiplier = 100 if plotPercent else 1
        x0, y0, xmax, ymax = self.extremeCoords

        # initialise array with nans so area not in grain displays white
        grainMaxShear = np.full((ymax - y0 + 1, xmax - x0 + 1), np.nan, dtype=float)

        for coord, maxShear in zip(self.coordList, self.maxShearList):
            grainMaxShear[coord[1] - y0, coord[0] - x0] = maxShear

        if ax is None:
            plt.figure()
            plt.imshow(grainMaxShear * multiplier, interpolation='none', vmin=vmin, vmax=vmax, cmap=cmap)
            plt.colorbar(label="Effective shear strain (%)")
            plt.xticks([])
            plt.yticks([])
        else:
            ax.imshow(grainMaxShear * multiplier, interpolation='none', vmin=vmin, vmax=vmax, cmap=cmap)
            # ax.colorbar()

        if plotSlipTraces:
            if self.slipTraces() is None:
                raise Exception("First calculate slip traces")

            colours = slipTraceColours
            xPos = int((xmax - x0) / 2)
            yPos = int((ymax - y0) / 2)
            for i, slipTrace in enumerate(self.slipTraces()):
                colour = colours[len(colours) - 1] if i >= len(colours) else colours[i]
                if ax is None:
                    plt.quiver(xPos, yPos, slipTrace[0], slipTrace[1], scale=1, pivot="middle",
                               color=colour, headwidth=1, headlength=0)
                else:
                    ax.quiver(xPos, yPos, slipTrace[0], slipTrace[1], scale=1, pivot="middle",
                              color=colour, headwidth=1, headlength=0)

        if plotShearBands:
            grainMaxShear = np.nan_to_num(grainMaxShear)

            sin_map = tf.radon(grainMaxShear, circle=False)
            profile = np.max(sin_map, axis=0)

            x = np.arange(180)
            y = profile

            indexes = peakutils.indexes(y, thres=0.5, min_dist=30)
            peaks = peakutils.interpolate(x, y, ind=indexes)
            print("Number of bands detected: {:}".format(len(peaks)))

            shearBandAngles = peaks
            print("Angles: {:}".format(180-(90-shearBandAngles)))
            shearBandAngles = -shearBandAngles * np.pi / 180
            shearBandVectors = np.array((np.sin(shearBandAngles), np.cos(shearBandAngles)))

            xPos = int((xmax - x0) / 2)
            yPos = int((ymax - y0) / 2)

            if ax is None:
                plt.quiver(xPos, yPos, shearBandVectors[0], shearBandVectors[1], scale=1, pivot="middle",
                           color='yellow', headwidth=1, headlength=0)
            else:
                ax.quiver(xPos, yPos, shearBandVectors[0], shearBandVectors[1], scale=1, pivot="middle",
                          color='yellow', headwidth=1, headlength=0)

        return

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

    def grainMapDataCoarse(self, mapData, kernelSize=2):
        grainMapData = self.grainMapData(mapData)
        grainMapDataCoarse = np.full_like(grainMapData, np.nan)

        for i, j in np.ndindex(grainMapData.shape):
            if np.isnan(grainMapData[i, j]):
                grainMapDataCoarse[i, j] = np.nan
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

    def slipTraces(self, correctAvOri=False):
        if correctAvOri:
            # need to correct slip traces due to warping of map
            return self.ebsdGrain.slipTraces
        else:
            return self.ebsdGrain.slipTraces

    def calcSlipTraces(self, slipSystems=None):
        self.ebsdGrain.calcSlipTraces(slipSystems=slipSystems)

    # def calcSlipTraces(self, correctAvOri=False):

    #     if correctAvOri:
    #         # transformRotation = Quat(-DicMap.ebsdTransform.rotation, 0, 0)
    #         transformRotation = Quat(0.1329602509925417, 0, 0)
    #         grainAvOri = grainAvOri * transformRotation
