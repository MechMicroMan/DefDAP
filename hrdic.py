import numpy as np
import matplotlib.pyplot as plt
import inspect
import pandas as pd
from matplotlib_scalebar.scalebar import ScaleBar

from skimage import transform as tf
from skimage import morphology as mph

from scipy.stats import mode

import peakutils

import base
from quat import Quat


class Map(base.Map):

    def __init__(self, path, fname):
        """Initialise class and import DIC data from file

        Args:
            path(str): Path to file
            fname(str): Name of file including extension
        """

        # Call base class constructor
        super(Map, self).__init__()

        print("Loading DIC data...", end="")

        # Initialise variables
        self.ebsdMap = None                 # EBSD map linked to DIC map
        self.ebsdTransform = None           # Transform from EBSD to DIC coordinates
        self.ebsdTransformInv = None        # Transform from DIC to EBSD coordinates
        self.grainList = None
        self.currGrainId = None             # ID of last selected grain
        self.ebsdGrainIds = None
        self.patternImPath = None           # Path to BSE image of map
        self.windowSize = None              # Window size for map
        self.plotHomog = self.plotMaxShear  # Use max shear map for defining homologous points
        self.highlightAlpha = 0.6
        self.bseScale = None
        self.path = path                    # file path
        self.fname = fname                  # file name

        # Load metadata
        with open(self.path + self.fname, 'r') as f:
            header = f.readline()
        metadata = header.split()

        self.format = metadata[0].strip('#')    # Software name
        self.version = metadata[1]              # Software version
        self.binning = int(metadata[3])         # Sub-window width in pixels
        self.xdimfile = int(metadata[5])        # size of map along x (from header)
        self.ydimfile = int(metadata[4])        # size of map along y (from header)

        # Load data

        self.data = pd.read_table(self.path + self.fname, delimiter='\t', skiprows=1, header=None)
        self.xc = self.data.values[:, 0]           # x coordinates
        self.yc = self.data.values[:, 1]           # y coordinates
        self.xd = self.data.values[:, 2]           # x displacement
        self.yd = self.data.values[:, 3]           # y displacement

        # Calculate size of map
        self.xdim = int((self.xc.max() - self.xc.min()) /
                        min(abs((np.diff(self.xc)))) + 1)  # size of map along x
        self.ydim = int((self.yc.max() - self.yc.min()) /
                        max(abs((np.diff(self.yc)))) + 1)  # size of map along y

        if (self.xdim != self.xdimfile) or (self.ydim != self.ydimfile):
            raise Exception("Dimensions of data and header do not match")

        # *dim are full size of data. *Dim are size after cropping
        self.xDim = self.xdim
        self.yDim = self.ydim

        self.x_map = self._map(self.xd)     # u (displacement component along x)
        self.y_map = self._map(self.yd)     # v (displacement component along x)
        xDispGrad = self._grad(self.x_map)
        yDispGrad = self._grad(self.y_map)
        self.f11 = xDispGrad[1] + 1         # f11
        self.f22 = yDispGrad[0] + 1         # f22
        self.f12 = xDispGrad[0]             # f12
        self.f21 = yDispGrad[1]             # f21

        self.max_shear = np.sqrt((((self.f11 - self.f22) / 2.)**2) +
                                 ((self.f12 + self.f21) / 2.)**2)   # max shear component
        self.mapshape = np.shape(self.max_shear)                    # map shape

        self.cropDists = np.array(((0, 0), (0, 0)), dtype=int)      # crop distances (default all zeros)

        print("\rLoaded {0} {1} data (dimensions: {2} x {3} pixels, sub-window size: {4} x {4} pixels)".
              format(self.format, self.version, self.xdimfile, self.ydimfile, self.binning))

    def plotDefault(self, *args, **kwargs):
        self.plotMaxShear(plotGBs=True, *args, **kwargs)

    def _map(self, data_col):
        data_map = np.reshape(np.array(data_col), (self.ydim, self.xdim))
        return data_map

    def _grad(self, data_map):
        grad_step = min(abs((np.diff(self.xc))))
        data_grad = np.gradient(data_map, grad_step, grad_step)
        return data_grad

    def retrieveName(self):
        """
        Gets the first name assigned to the a map, as a string

        var(obj) variable to get name of
        """
        for fi in reversed(inspect.stack()):
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is self]
            if len(names) > 0:
                return names[0]

    def setScale(self, micrometrePerPixel):
        """
        Sets the scale of the map

        micrometrePerPixel(float) length of pixel in original BSE image in micrometres
        """
        self.bseScale = micrometrePerPixel

    @property
    def strainScale(self):
        return None if self.bseScale is None else self.bseScale * self.binning

    def printStatsTable(self, percentiles, components):
        """
        Print out statistics table for a DIC map

        percentiles(list) list of percentiles to print (number, Min, Mean or Max)
        components(list of str) lis of map components to print i.e. f11, mss
        """

        # Print map info
        print('\033[1m', end='')    # START BOLD
        print("{0} (dimensions: {1} x {2} pixels, sub-window size: {3} x {3} pixels, number of points: {4})\n".format(
            self.retrieveName(),
            self.xdimfile,
            self.ydimfile,
            self.binning,
            self.xdimfile * self.ydimfile
        ))

        # Print table header
        print("Component\t".format(), end="")
        for x in percentiles:
            if x == 'Min' or x == 'Mean' or x == 'Max':
                print("{0}\t".format(x), end='')
            else:
                print("P{0}\t".format(x), end='')
        print('\033[0m', end='')    # END BOLD
        print()

        # Print table
        for c in components:
            selmap = []
            if c == 'mss':
                selmap = self.crop(self.max_shear) * 100
            if c == 'f11':
                selmap = (self.crop(self.f11) - 1) * 100
            if c == 'f12':
                selmap = self.crop(self.f12) * 100
            if c == 'f21':
                selmap = self.crop(self.f21) * 100
            if c == 'f22':
                selmap = (self.crop(self.f22) - 1) * 100
            plist = []
            for p in percentiles:
                if p == 'Min':
                    plist.append(np.min(selmap))
                elif p == 'Mean':
                    plist.append(np.mean(selmap))
                elif p == 'Max':
                    plist.append(np.max(selmap))
                else:
                    plist.append(np.percentile(selmap, p))
            print("{0}\t\t".format(c), end="")
            for l in plist:
                print("{0:.2f}\t".format(l), end='')
            print()
        print()

    def setCrop(self, xMin=None, xMax=None, yMin=None, yMax=None, updateHomogPoints=False):
        """Set a crop for the DIC map

        Args:
            xMin(int): Distance to crop from left in pixels
            xMax(int): Distance to crop from right in pixels
            yMin(int): Distance to crop from top in pixels
            yMax(int): Distance to crop from bottom in pixels
            updateHomogPoints (bool, optional): Change homologous points to reflect crop
        """
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
        """Calculates the transformation required to align EBSD dataset to DIC

        Args:
            ebsdMap(ebsd.Map): EBSD map object to link
            transformType(string, optional): affine, piecewiseAffine or polynomial
            order(int, optional): Order of polynomial transform to apply
        """
        print("Linking EBSD <-> DIC...", end="")

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

        print("\r", end="")

    def checkEbsdLinked(self):
        """Check if an EBSD map has been linked.

        Returns:
            bool: Returns True if EBSD map linked

        Raises:
            Exception: if EBSD map not linked
        """
        if self.ebsdMap is None:
            raise Exception("No EBSD map linked.")
        return True

    def warpToDicFrame(self, mapData, cropImage=True, order=1, preserve_range=False):
        """Warps a map to the DIC frame

        Returns:
            warpedMap: Map (i.e. EBSD map) warped to the DIC frame

        Args:
            mapData(map): data to warp
            cropImage(bool, optional):
            order(int, optional): order of interpolation (0: Nearest-neighbor, 1: Bi-linear...)
            preserve_range(bool, optional): keep the original range of values
        """
        # Check a EBSD map is linked
        self.checkEbsdLinked()

        if (cropImage or type(self.ebsdTransform) is not tf.AffineTransform):
            # crop to size of DIC map
            outputShape = (self.yDim, self.xDim)
            # warp the map
            warpedMap = tf.warp(mapData, self.ebsdTransform, output_shape=outputShape, order=order, preserve_range=preserve_range)
        else:
            # copy ebsd transform and change translation to give an extra 5% border
            # to show the entire image after rotation/shearing
            tempEbsdTransform = tf.AffineTransform(matrix=np.copy(self.ebsdTransform.params))
            tempEbsdTransform.params[0:2, 2] = -0.05 * np.array(mapData.shape)

            # output the entire warped image with 5% border (add some extra to fix a bug)
            outputShape = np.array(mapData.shape) * 1.4 / tempEbsdTransform.scale

            # warp the map
            warpedMap = tf.warp(mapData, tempEbsdTransform, output_shape=outputShape.astype(int), order=order, preserve_range=preserve_range)

        # return map
        return warpedMap

    @property
    def boundaries(self):
        # Check a EBSD map is linked
        self.checkEbsdLinked()

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

    def plotGrainNumbers(self, dilate=False):
        """Plot a map with grains numbered

        Args:
            dilate(bool, optional): Set to true to dilate boundaries by one pixel
        """

        self.fig, self.ax = plt.subplots()

        for grainID in range(0, len(self.grainList)):
            xmiddle = (self.grainList[grainID].extremeCoords[2] + self.grainList[grainID].extremeCoords[0]) / 2
            ymiddle = (self.grainList[grainID].extremeCoords[3] + self.grainList[grainID].extremeCoords[1]) / 2
            self.ax.text(xmiddle, ymiddle, grainID, fontsize=10)

        self.plotGBs(ax=self.ax, colour='black', dilate=dilate)

    def plotMaxShear(self, plotGBs=False, dilateBoundaries=False, boundaryColour='white', plotSlipTraces=False, plotPercent=False,
                     scaleBar=False, updateCurrent=False, highlightGrains=None, highlightColours=None,
                     plotColourBar=False, vmin=None, vmax=None):
        """Plot a map of maximum shear strain

        Args:
            plotGBs(bool, optional): Set to true to overlay grain boundaries
            dilateBoundaries(bool, optional): Set to true to dilate boundaries by one pixel
            boundaryColour(string, optional): Colour of boundaries
            plotSlipTraces(bool, optional): Set to true to plot slip traces for each grain
            plotPercent(bool, optional): Set to true to plot maps using percentage
            plotColourBar(bool, optional): Set to true to plot colour bar
            vmin(bool, optional): Minimum value to plot
            vmax(bool, optional): Maximum value to plot
        """
        if not updateCurrent:
            # self.fig, self.ax = plt.subplots(figsize=(5.75, 4))
            self.fig, self.ax = plt.subplots()

        multiplier = 100 if plotPercent else 1
        img = self.ax.imshow(self.crop(self.max_shear) * multiplier,
                             cmap='viridis', interpolation='None', vmin=vmin, vmax=vmax)
        if plotColourBar:
            if plotPercent:
                plt.colorbar(img, ax=self.ax, label="Effective shear strain (%)")
            plt.colorbar(img, ax=self.ax, label="Effective shear strain")

        if plotGBs:
            self.plotGBs(ax=self.ax, colour=boundaryColour, dilate=dilateBoundaries)

        if highlightGrains is not None:
            self.highlightGrains(highlightGrains, highlightColours)

        if scaleBar:
            if self.strainScale is None:
                raise Exception("First set image scale using setScale")
            else:
                scalebar = ScaleBar(self.strainScale * (1e-6))  # 1 pixel = 0.2 meter
                plt.gca().add_artist(scalebar)

        # # plot slip traces
        # if plotSlipTraces:
        #     # Check that grains have been detected in the map
        #     self.checkGrainsDetected()

        #     numGrains = len(self.grainList)
        #     numSS = len(self.ebsdMap.slipSystems)
        #     grainSizeData = np.zeros((numGrains, 4))
        #     slipTraceData = np.zeros((numGrains, numSS, 2))

        #     i = 0   # keep track of number of slip traces
        #     for grain in self.grainList:
        #         if len(grain) < 1000:
        #             continue

        #         # x0, y0, xmax, ymax
        #         grainSizeData[i, 0], grainSizeData[i, 1], grainSizeData[i, 2], grainSizeData[i, 3] = grain.extremeCoords

        #         for j, slipTrace in enumerate(grain.slipTraces()):
        #             slipTraceData[i, j, 0:2] = slipTrace[0:2]

        #         i += 1

        #     grainSizeData = grainSizeData[0:i, :]
        #     slipTraceData = slipTraceData[0:i, :, :]

        #     scale = 4 / ((grainSizeData[:, 2] - grainSizeData[:, 0]) / self.xDim +
        #                  (grainSizeData[:, 3] - grainSizeData[:, 1]) / self.xDim)

        #     xPos = grainSizeData[:, 0] + (grainSizeData[:, 2] - grainSizeData[:, 0]) / 2
        #     yPos = grainSizeData[:, 1] + (grainSizeData[:, 3] - grainSizeData[:, 1]) / 2

        #     colours = self.ebsdMap.slipTraceColours

        #     for i, colour in enumerate(colours[0:numSS]):
        #         self.ax.quiver(xPos, yPos, slipTraceData[:, i, 0], slipTraceData[:, i, 1],
        #                        scale=scale, pivot="middle", color=colour, headwidth=1,
        #                        headlength=0, width=0.002)

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
        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        plt.figure()

        grainAvMaxShear = np.zeros([self.yDim, self.xDim])

        for grain in self.grainList:
            avMaxShear = np.array(grain.maxShearList).mean()

            for coord in grain.coordList:
                grainAvMaxShear[coord[1], coord[0]] = avMaxShear

        plt.imshow(grainAvMaxShear * 100, vmin=vmin, vmax=vmin)

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
        # Check that grains have been detected in the map
        self.checkGrainsDetected()

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

        grainAvData = self.calcGrainAv(mapData)

        self.plotGrainData(
            grainAvData,
            grainIds=-1,
            plotGBs=plotGBs,
            plotColourBar=plotColourBar,
            vmin=vmin,
            vmax=vmax,
            clabel=clabel
        )

    def plotGrainData(self, grainData, grainIds=-1, bg=0, ax=None, plotGBs=False, plotColourBar=True, cmap=None, vmin=None, vmax=None, clabel=''):
        """Plot grain map with grains filled with average value of from any DIC map data

        Args:
            grainData (np.array): Data value for each grain
            grainIds (iterable): Grain IDs of each grain to plot, -1 for all grains
            bg (float, optional): Background value for areas with no grain data.
            plotGBs (bool, optional): Set to True to draw grain boundaries
            plotColourBar (bool, optional): Set to Flase to exclude the colour bar
            cmap (str, optional): Colour map to use
            vmin (float, optional): Minimum value of colour scale
            vmax (float, optional): Maximum value for colour scale
            clabel (str, optional): Colour bar label text
        """
        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        if type(grainIds) is int and grainIds == -1:
            grainIds = range(len(self))

        if len(grainData) != len(grainIds):
            raise Exception("Must be 1 value for each grain in grainData.")

        grainMap = np.full([self.yDim, self.xDim], bg, dtype=grainData.dtype)
        for grainId, grainValue in zip(grainIds, grainData):
            grain = self.grainList[grainId]
            for coord in grain.coordList:
                grainMap[coord[1], coord[0]] = grainValue

        if ax is None:
            plt.figure()
            ax = plt.gca()

        im = ax.imshow(grainMap, vmin=vmin, vmax=vmax, cmap=cmap)

        if plotColourBar:
            plt.colorbar(im, ax=ax, label=clabel)

        if plotGBs:
            self.plotGBs()

    def plotGrainAvIPF(self, mapData, direction, ax=None, plotColourBar=True, vmin=None, vmax=None, clabel=''):
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
        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        plt.figure()
        grainAvData = self.calcGrainAv(mapData)

        grainOri = np.empty(len(self), dtype=Quat)

        for grainId, grain in enumerate(self.grainList):
            grainOri[grainId] = grain.ebsdGrain.refOri

        if ax is None:
            plt.figure()
            ax = plt.gca()

        Quat.plotIPF(grainOri, direction, self.ebsdMap.crystalSym, ax=ax,
                     c=grainAvData, marker='o', vmin=vmin, vmax=vmax)

        if plotColourBar:
            plt.colorbar(label=clabel)

    def locateGrainID(self, clickEvent=None, displaySelected=False, **kwargs):
        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        # reset current selected grain and plot max shear map with click handler
        self.currGrainId = None
        self.plotMaxShear(plotGBs=True, **kwargs)
        if clickEvent is None:
            # default click handler which highlights grain and prints id
            self.fig.canvas.mpl_connect(
                'button_press_event',
                lambda x: self.clickGrainId(x, displaySelected, **kwargs)
            )
        else:
            # click handler loaded in as parameter. Pass current map object to it.
            self.fig.canvas.mpl_connect('button_press_event', lambda x: clickEvent(x, self))

        # unset figure for plotting grains
        self.grainFig = None
        self.grainAx = None

    def clickGrainId(self, event, displaySelected, **kwargs):
        if event.inaxes is not None:
            # grain id of selected grain
            self.currGrainId = int(self.grains[int(event.ydata), int(event.xdata)] - 1)
            print("Grain ID: {}".format(self.currGrainId))

            # clear current axis and redraw map with highlighted grain overlay
            self.ax.clear()
            self.plotMaxShear(plotGBs=True, updateCurrent=True, highlightGrains=[self.currGrainId],
                              highlightColours=['green'], **kwargs)
            self.fig.canvas.draw()

            if displaySelected:
                if self.grainFig is None:
                    self.grainFig, self.grainAx = plt.subplots()
                self.grainList[self.currGrainId].calcSlipTraces()
                self.grainAx.clear()
                self.grainList[self.currGrainId].plotMaxShear(plotSlipTraces=True,
                                                              plotShearBands=True,
                                                              ax=self.grainAx,
                                                              **kwargs)
                self.grainFig.canvas.draw()

    def findGrains(self, minGrainSize=10):
        print("Finding grains in DIC map...", end="")

        # Check a EBSD map is linked
        self.checkEbsdLinked()

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
            self.grainList[i].ebsdMap = self.ebsdMap

        print("\r", end="")
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


class Grain(base.Grain):

    def __init__(self, dicMap):
        # Call base class constructor
        super(Grain, self).__init__()

        self.dicMap = dicMap       # dic map this grain is a member of
        self.maxShearList = []
        self.ebsdGrain = None
        self.ebsdMap = None

    # coord is a tuple (x, y)
    def addPoint(self, coord, maxShear):
        self.coordList.append(coord)
        self.maxShearList.append(maxShear)

    def plotMaxShear(self, plotPercent=True, plotSlipTraces=False, plotShearBands=False,
                     vmin=None, vmax=None, cmap="viridis", ax=None):

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
            self.plotSlipTraces(ax=ax)

        if plotShearBands:
            self.plotShearBands(grainMaxShear, ax=ax)

    @property
    def slipTraces(self):
        return self.ebsdGrain.slipTraces

    def calcSlipTraces(self, slipSystems=None):
        self.ebsdGrain.calcSlipTraces(slipSystems=slipSystems)

    def calcSlipBands(self, grainMapData, thres=0.3, min_dist=30):
        """Use Radon transform to detect slip band angles

        Args:
            grainMapData (numpy.array): Data to find bands in
            thres (float, optional): Normalised threshold for peaks
            min_dist (int, optional): Minimum angle between bands

        Returns:
            list(float): Detected slip band angles
        """
        grainMapData = np.nan_to_num(grainMapData)

        if grainMapData.min() < 0:
            print("Negeative values in data, taking absolute value.")
            # grainMapData = grainMapData**2
            grainMapData = np.abs(grainMapData)

        sin_map = tf.radon(grainMapData, circle=False)
        profile = np.max(sin_map, axis=0)

        x = np.arange(180)

        indexes = peakutils.indexes(profile, thres=thres, min_dist=min_dist)
        peaks = x[indexes]
        # peaks = peakutils.interpolate(x, profile, ind=indexes)
        print("Number of bands detected: {:}".format(len(peaks)))

        slipBandAngles = peaks
        slipBandAngles = slipBandAngles * np.pi / 180

        return slipBandAngles

    def plotSlipBands(self, grainMapData, ax=None, thres=0.3, min_dist=30, slipBandAngles=None):
        if slipBandAngles is None:
            slipBandAngles = self.calcSlipBands(grainMapData, thres=thres, min_dist=min_dist)
        slipBandVectors = np.array((-np.sin(slipBandAngles), np.cos(slipBandAngles)))

        xPos, yPos = self.centreCoords

        if ax is None:
            plt.quiver(
                xPos, yPos,
                slipBandVectors[0], slipBandVectors[1],
                scale=1, pivot="middle",
                color='yellow', headwidth=1,
                headlength=0
            )
        else:
            ax.quiver(
                xPos, yPos,
                slipBandVectors[0], slipBandVectors[1],
                scale=1, pivot="middle",
                color='yellow', headwidth=1,
                headlength=0
            )

        return slipBandAngles
