import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from skimage import transform as tf
from skimage import morphology as mph

from scipy.stats import mode

import peakutils

from quat import Quat
import base


class Map(base.Map):

    def __init__(self, path, fname):
        # Call base class constructor
        super(Map, self).__init__()

        self.ebsdMap = None
        self.ebsdTransform = None
        self.ebsdShift = (0, 0)
        self.grainList = None
        self.currGrainId = None     # Id of last selected grain
        # ...
        self.ebsdGrainIds = None
        self.patternImPath = None   # Path to BSE image of map
        self.windowSize = None      # Window size for map

        self.plotDefault = self.plotMaxShear

        self.path = path
        self.fname = fname
        # Load in data
        self.data = np.loadtxt(self.path + self.fname, skiprows=1)
        self.xc = self.data[:, 0]  # x coordinates
        self.yc = self.data[:, 1]  # y coordinates
        self.xd = self.data[:, 2]  # x displacement
        self.yd = self.data[:, 3]  # y displacement

        # Calculate size of map
        self.xdim = ((self.xc.max() - self.xc.min()) /
                     min(abs((np.diff(self.xc)))) + 1)  # size of map along x
        self.ydim = ((self.yc.max() - self.yc.min()) /
                     max(abs((np.diff(self.yc)))) + 1)  # size of map along y

        # *dim are full size of data. *Dim are size after cropping
        self.xDim = self.xdim
        self.yDim = self.ydim

        self.x_map = self._map(self.xd)  # u (displacement component along x)
        self.y_map = self._map(self.yd)  # v (displacement component along x)
        self.f11 = self._grad(self.x_map)[1]  # f11
        self.f22 = self._grad(self.y_map)[0]  # f22
        self.f12 = self._grad(self.x_map)[0]  # f12
        self.f21 = self._grad(self.y_map)[1]  # f21

        self.max_shear = np.sqrt((((self.f11 - self.f22) / 2.)**2) +
                                 ((self.f12 + self.f21) / 2.)**2)  # max shear component
        self.mapshape = np.shape(self.max_shear)

        self.cropDists = np.array(((0, 0), (0, 0)), dtype=int)

    def _map(self, data_col):
        data_map = np.reshape(np.array(data_col), (int(self.ydim), int(self.xdim)))
        return data_map

    def _grad(self, data_map):
        grad_step = min(abs((np.diff(self.xc))))
        data_grad = np.gradient(data_map, grad_step, grad_step)
        return data_grad

    def setCrop(self, xMin=None, xMax=None, yMin=None, yMax=None):
        if xMin is not None:
            self.cropDists[0, 0] = int(xMin)
        if xMax is not None:
            self.cropDists[0, 1] = int(xMax)
        if yMin is not None:
            self.cropDists[1, 0] = int(yMin)
        if yMax is not None:
            self.cropDists[1, 1] = int(yMax)

        self.xDim = int(self.xdim - xMin - xMax)
        self.yDim = int(self.ydim - yMin - yMax)

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
            self.plotDefault = self.plotPattern
            binSize = self.windowSize
        else:
            self.plotDefault = self.plotMaxShear
            binSize = 1

        # Call set homog points from base class setting the bin size
        super(Map, self).setHomogPoint(binSize=binSize)

    def linkEbsdMap(self, ebsdMap):
        self.ebsdMap = ebsdMap
        self.ebsdTransform = tf.AffineTransform()
        # calculate transform from EBSD to DIC frame
        self.ebsdTransform.estimate(np.array(self.homogPoints), np.array(self.ebsdMap.homogPoints))

    def setEbsdShift(self, xShift=None, yShift=None):
        if xShift is None:
            xShift = self.ebsdShift[0]
        if yShift is None:
            yShift = self.ebsdShift[1]

        self.ebsdShift = (xShift, yShift)

    def warpToDicFrame(self, mapData, cropImage=True):
        # calculate transform from EBSD to DIC frame
        # self.ebsdTransform.estimate(np.array(self.homogPoints), np.array(self.ebsdMap.homogPoints))

        if cropImage:
            # crop to size of DIC map
            outputShape = (self.yDim + self.ebsdShift[1], self.xDim + self.ebsdShift[0])
            # warp the map
            warpedMap = tf.warp(mapData, self.ebsdTransform, output_shape=outputShape)
            # return map with shift
            return warpedMap[self.ebsdShift[1]:, self.ebsdShift[0]:]
        else:
            # copy ebsd transform and change translation to give an extra 5% border
            # to show the entire image after rotation/shearing
            tempEbsdTransform = tf.AffineTransform(matrix=np.copy(self.ebsdTransform.params))
            tempEbsdTransform.params[0:2, 2] = -0.05 * np.array(mapData.shape)

            # output the entire warped image with 5% border
            outputShape = np.array(mapData.shape) * 1.1 / tempEbsdTransform.scale

            # warp the map
            warpedMap = tf.warp(mapData, tempEbsdTransform, output_shape=outputShape.astype(int))

            return warpedMap

    @property
    def boundaries(self):
        boundaries = self.warpToDicFrame(-self.ebsdMap.boundaries.astype(float), cropImage=False) > 0.1

        boundaries = mph.skeletonize(boundaries)
        mph.remove_small_objects(boundaries, min_size=10, in_place=True, connectivity=2)

        # need to apply the translation of ebsd transform and remove 5% border
        crop = np.copy(self.ebsdTransform.params[0:2, 2])
        crop += 0.05 * np.array(self.ebsdMap.boundaries.shape)
        # the crop is defined in EBSD coords so need to transform it
        transformMatrix = np.copy(self.ebsdTransform.params[0:2, 0:2])
        crop = np.matmul(np.linalg.inv(transformMatrix), crop)

        crop = crop.round().astype(int) + np.array(self.ebsdShift)

        boundaries = boundaries[crop[1]:crop[1] + self.yDim + self.ebsdShift[1],
                                crop[0]:crop[0] + self.xDim + self.ebsdShift[0]]

        boundaries = -boundaries.astype(int)

        return boundaries

    def setPatternPath(self, filePath, windowSize):
        """Set path of BSE of pattern. filePath is relative to the path set when constructing."""

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

    def plotMaxShear(self, plotGBs=False, plotSlipTraces=False, plotPercent=True, vMax=0.5,
                     updateCurrent=False, highlightGrains=None, plotColourBar=False,
                     saveFig=False,saveFile=None,dpi=None,title=None,plotAxis="on"):
        
        if not updateCurrent:
            # self.fig, self.ax = plt.subplots(figsize=(5.75, 4))
            self.fig, self.ax = plt.subplots()

        multiplier = 100 if plotPercent else 1
        if plotPercent:
            img = self.ax.imshow(self.crop(self.max_shear) * multiplier,
                             cmap='viridis', interpolation='None', vmin=0, vmax=10)
        else:
            img = self.ax.imshow(self.crop(self.max_shear) * multiplier,
                             cmap='viridis', interpolation='None', vmin=0, vmax=vMax)
        if plotColourBar:
            if plotPercent:
                plt.colorbar(img, ax=self.ax, label="Effective shear strain (%)", shrink = 0.75)
            else:
               plt.colorbar(img, ax=self.ax, label="HRDIC $\epsilon_{eff}$ [ ]", shrink = 0.75)

        if plotGBs:
            cmap1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap', ['white', 'white'], 256)
            cmap1._init()
            cmap1._lut[:, -1] = np.linspace(0, 1, cmap1.N + 3)

            self.ax.imshow(-self.boundaries, cmap=cmap1, interpolation='None', vmin=0, vmax=1)

        if highlightGrains is not None:
            self.highlightGrains(highlightGrains)

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
                grainSizeData[i, 0], grainSizeData[i, 1], grainSizeData[i, 2], grainSizeData[i, 3] = grain.extremeCoords()

                for j, slipTrace in enumerate(grain.slipTraces()):
                    slipTraceData[i, j, 0:2] = slipTrace[0:2]

                i += 1

            grainSizeData = grainSizeData[0:i, :]
            slipTraceData = slipTraceData[0:i, :, :]

            scale = 4 / ((grainSizeData[:, 2] - grainSizeData[:, 0]) / self.xDim +
                         (grainSizeData[:, 3] - grainSizeData[:, 1]) / self.xDim)

            xPos = grainSizeData[:, 0] + (grainSizeData[:, 2] - grainSizeData[:, 0]) / 2
            yPos = grainSizeData[:, 1] + (grainSizeData[:, 3] - grainSizeData[:, 1]) / 2

            colours = ["white", "green", "red", "black"]

            for i, colour in enumerate(colours[0:numSS]):
                self.ax.quiver(xPos, yPos, slipTraceData[:, i, 0], slipTraceData[:, i, 1],
                               scale=scale, pivot="middle", color=colour, headwidth=1,
                               headlength=0, width=0.002)
        if saveFig:
            plt.title(title)
            plt.axis(plotAxis)
            plt.savefig(saveFile,dpi=dpi,bbox_inches='tight')
            
        return

    def highlightGrains(self, grainIds):
        outline = np.zeros((self.yDim, self.xDim), dtype=int)
        for grainId in grainIds:
            # outline of highlighted grain
            grainOutline = self.grainList[grainId].grainOutline(bg=0, fg=1)
            x0, y0, xmax, ymax = self.grainList[grainId].extremeCoords()

            # use logical of same are in entire area to ensure neigbouring grains display correctly
            grainOutline = np.logical_or(outline[y0:ymax + 1, x0:xmax + 1], grainOutline).astype(int)
            outline[y0:ymax + 1, x0:xmax + 1] = grainOutline

        # Custom colour map where 0 is tranparent white for bg and 255 is opaque white for fg
        cmap1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap', ['green', 'green'], 256)
        cmap1._init()
        cmap1._lut[:, -1] = np.linspace(0, 0.6, cmap1.N + 3)

        # plot highlighted grain overlay
        self.ax.imshow(outline, interpolation='none', vmin=0, vmax=1, cmap=cmap1)

        return

    def locateGrainID(self, clickEvent=None, displaySelected=False):
        if (self.grainList is not None) and (self.grainList != []):
            # reset current selected grain and plot max shear map with click handler
            self.currGrainId = None
            self.plotMaxShear(plotGBs=True)
            if clickEvent is None:
                # default click handler which highlights grain and prints id
                self.fig.canvas.mpl_connect('button_press_event', lambda x: self.clickGrainId(x, displaySelected))
            else:
                # click handler loaded in as parameter. Pass current map object to it.
                self.fig.canvas.mpl_connect('button_press_event', lambda x: clickEvent(x, self))

            # unset figure for plotting grains
            self.grainFig = None
            self.grainAx = None

        else:
            raise Exception("Grain list empty")

    def clickGrainId(self, event, displaySelected):
        if event.inaxes is not None:
            # grain id of selected grain
            self.currGrainId = int(self.grains[int(event.ydata), int(event.xdata)] - 1)
            print(self.currGrainId)

            # clear current axis and redraw map with highlighted grain overlay
            self.ax.clear()
            self.plotMaxShear(plotGBs=True, updateCurrent=True, highlightGrains=[self.currGrainId])
            self.fig.canvas.draw()

            if displaySelected:
                if self.grainFig is None:
                    self.grainFig, self.grainAx = plt.subplots()
                self.grainList[self.currGrainId].calcSlipTraces()
                self.grainAx.clear()
                self.grainList[self.currGrainId].plotMaxShear(plotSlipTraces=True,
                                                              plotShearBands=True,
                                                              ax=self.grainAx)
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
        # Warp DIC grain map to EBSD frame, accounting for shift (only positive)
        dicGrains = self.grains
        if self.ebsdShift[1] != 0:
            dicGrains = np.vstack([np.zeros((self.ebsdShift[1], dicGrains.shape[1])), dicGrains])
        if self.ebsdShift[0] != 0:
            dicGrains = np.hstack([np.zeros((dicGrains.shape[0], self.ebsdShift[0])), dicGrains])

        warpedDicGrains = tf.warp(dicGrains.astype(float), self.ebsdTransform.inverse,
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
        self.coordList = []         # list of coords stored as tuples (x, y)
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

    def extremeCoords(self):
        unzippedCoordlist = list(zip(*self.coordList))
        x0 = min(unzippedCoordlist[0])
        y0 = min(unzippedCoordlist[1])
        xmax = max(unzippedCoordlist[0])
        ymax = max(unzippedCoordlist[1])
        return x0, y0, xmax, ymax

    def grainOutline(self, bg=np.nan, fg=0):
        x0, y0, xmax, ymax = self.extremeCoords()

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
                     saveFig=False,saveFile=None,dpi=None,title=None,plotAxis="on"):
        
        multiplier = 100 if plotPercent else 1
        x0, y0, xmax, ymax = self.extremeCoords()

        # initialise array with nans so area not in grain displays white
        grainMaxShear = np.full((ymax - y0 + 1, xmax - x0 + 1), np.nan, dtype=float)

        for coord, maxShear in zip(self.coordList, self.maxShearList):
            grainMaxShear[coord[1] - y0, coord[0] - x0] = maxShear

        if ax is None:
            plt.figure()
            plt.imshow(grainMaxShear * multiplier, interpolation='none', vmin=vmin, vmax=vmax, cmap=cmap)
            if plotPercent:
                plt.colorbar(label="Effective shear strain (%)", shrink = 0.75)
            else:
               plt.colorbar(label="HRDIC $\epsilon_{eff}$ [ ]", shrink = 0.75)
            plt.xticks([])
            plt.yticks([])
        else:
            ax.imshow(grainMaxShear * multiplier, interpolation='none', vmin=vmin, vmax=vmax, cmap=cmap)
            # ax.colorbar()

        if plotSlipTraces:
            if self.slipTraces() is None:
                raise Exception("First calculate slip traces")

            colours = ["white", "green", "red", "black"]
            xPos = int((xmax - x0) / 2)
            yPos = int((ymax - y0) / 2)
            for slipTrace, colour in zip(self.slipTraces(), colours):
                if ax is None:
                    plt.quiver(xPos, yPos, slipTrace[0], slipTrace[1], scale=1, pivot="middle",
                               color=colour, headwidth=1, headlength=0)
                else:
                    ax.quiver(xPos, yPos, slipTrace[0], slipTrace[1], scale=1, pivot="middle",
                              color=colour, headwidth=1, headlength=0)

        if plotShearBands:
            grainMaxShear = np.nan_to_num(grainMaxShear)

            sin_map = tf.radon(grainMaxShear)
            profile = np.max(sin_map, axis=0)

            x = np.arange(180)
            y = profile

            indexes = peakutils.indexes(y, thres=0.5, min_dist=30)
            peaks = peakutils.interpolate(x, y, ind=indexes)
            print("Number of bands detected: {:}".format(len(peaks)))

            shearBandAngles = peaks
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
        if saveFig:
            plt.title(title)
            plt.axis(plotAxis)
            plt.savefig(saveFile,dpi=dpi,bbox_inches='tight')                     

        return

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

    
    def findSlipBands(self,c_range=[], cropXMin=0,cropXMax=-1,cropYMin=0,cropYMax=-1,strainMin=0):

        x0, y0, xmax, ymax = self.extremeCoords()

        # initialise array with nans so area not in grain displays white
        grainMaxShear = np.full((ymax - y0 + 1, xmax - x0 + 1), np.nan, dtype=float)

        for coord, maxShear in zip(self.coordList, self.maxShearList):
            grainMaxShear[coord[1] - y0, coord[0] - x0] = maxShear  

        image = np.nan_to_num(grainMaxShear[cropXMin:cropXMax,cropYMin:cropYMax])

        imageTemp = []
        for i in image:
            for j in i:
                if j < strainMin:
                    j = 0
                imageTemp.append(j)
        image = np.reshape(imageTemp,(np.shape(image)))

        plt.imshow(image,vmax=0.05)


    #     # calculate the fft
    #     acorr=(np.fft.fft2(image)*np.conjugate(np.fft.fft2(image)))
    #     ashift=np.fft.fftshift(acorr)
    #     corr_map=np.log(np.fft.fftshift((np.abs(np.fft.ifft2(ashift)))))
    #     if c_range==[]:
    #         plt.imshow(corr_map, interpolation='nearest', cmap='viridis');
    #     else:
    #         plt.imshow(corr_map, interpolation='nearest', cmap='viridis',
    #                    vmin=c_range[0], vmax=c_range[1]);


    #      # run an fft filter
    #     strain_map_fft=np.fft.fft2(image)
    #     cut=0
    #     lcut=int(np.shape(strain_map_fft)[0]/2-cut)
    #     rcut=int(np.shape(strain_map_fft)[0]/2+cut)
    #     tcut=int(np.shape(strain_map_fft)[1]/2-cut)
    #     bcut=int(np.shape(strain_map_fft)[1]/2+cut)
    #     strain_map_fft[lcut:rcut,tcut:bcut]=0

    #     plt.imshow((np.abs(strain_map_fft)),cmap='viridis',vmax=10)

    #     filt_image=np.abs(np.fft.ifft2(strain_map_fft))
    #     plt.imshow(filt_image,cmap='viridis', vmax=0.03)
    #     plt.colorbar()

    
        # run the Hough transform
        from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
        from matplotlib import cm
        h, theta, d = hough_line(image)

        # Generating figure 1
        fig, axes = plt.subplots(1, 3, figsize=(15, 6),
                                 subplot_kw={'adjustable': 'box-forced'})
        ax = axes.ravel()

        ax[0].imshow(image, cmap=cm.gray)
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        ax[1].imshow(np.log(1 + h),
                     extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                     cmap=cm.gray, aspect=1/1.5)
        ax[1].set_title('Hough transform')
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].axis('image')

        ax[2].imshow(image, cmap=cm.gray)
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
            ax[2].plot((0, image.shape[1]), (y0, y1), '-r')
        ax[2].set_xlim((0, image.shape[1]))
        ax[2].set_ylim((image.shape[0], 0))
        ax[2].set_axis_off()
        ax[2].set_title('Detected lines')

        plt.tight_layout()
        plt.show()
