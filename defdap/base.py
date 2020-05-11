# Copyright 2020 Mechanics of Microstructures Group
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
import networkx as nx

from defdap.quat import Quat
from defdap import plotting
from defdap.plotting import MapPlot, GrainPlot

from defdap.utils import reportProgress


class Map(object):
    """
    Base class for a map.
    """
    def __init__(self):
        self.grainList = None
        self.homogPoints = []

        self.proxigramArr = None
        self.neighbourNetwork = None

        self.grainPlot = None

    def __len__(self):
        return len(self.grainList)

    # allow array like getting of grains
    def __getitem__(self, key):
        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        return self.grainList[key]

    @property
    def shape(self):
        return self.yDim, self.xDim

    def checkGrainsDetected(self):
        """Check if grains have been detected.

        Returns
        -------
        bool:
            True if grains detected.

        Raises
        -------
        Exception
            If grains not detected.

        """

        if (self.grainList is None or
                type(self.grainList) is not list or
                len(self.grainList) < 1):
            raise Exception("No grains detected.")
        return True

    def plotGrainNumbers(self, dilateBoundaries=False, ax=None, **kwargs):
        """Plot a map with grains numbered.

        Parameters
        ----------
        dilateBoundaries : bool, optional
            Set to true to dilate boundaries.
        ax : matplotlib.axes.Axes, optional
            axis to plot on, if not provided the current active axis is used.
        kwargs : dict
            Keyword arguments to pass to matplotlib.

        Returns
        -------
        defdap.plotting.MapPlot

        """

        plot = plotting.MapPlot(self, ax=ax)
        plot.addGrainBoundaries(colour='black', dilate=dilateBoundaries)
        plot.addGrainNumbers(**kwargs)

        return plot

    def locateGrainID(self, clickEvent=None, displaySelected=False, **kwargs):
        """Interactive plot for identifying grains.

        Parameters
        ----------
        clickEvent : optional
            Click handler to use.
        displaySelected : bool, optional
            If true, plot slip traces for grain selected by click.

        """
        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        # reset current selected grain and plot euler map with click handler
        self.currGrainId = None
        plot = self.plotDefault(makeInteractive=True, **kwargs)
        if clickEvent is None:
            # default click handler which highlights grain and prints id
            plot.addEventHandler(
                'button_press_event',
                lambda e, p: self.clickGrainID(e, p, displaySelected)
            )
        else:
            # click handler loaded in as parameter. Pass current map
            # object to it.
            plot.addEventHandler('button_press_event', clickEvent)

        return plot

    def clickGrainID(self, event, plot, displaySelected):
        """Event handler to capture clicking on a map.

        Parameters
        ----------
        event :
            Click event.
        plot : defdap.plotting.Plot
            Plot to capture clicks from.
        displaySelected : bool
            If true, plot the selected grain alone in pop-out window.

        """
        if event.inaxes is plot.ax:
            # grain id of selected grain
            self.currGrainId = int(self.grains[int(event.ydata), int(event.xdata)] - 1)
            print("Grain ID: {}".format(self.currGrainId))

            # update the grain highlights layer in the plot
            plot.addGrainHighlights([self.currGrainId], alpha=self.highlightAlpha)

            # TODO: Check display selected works for ebsd map
            if displaySelected:
                currGrain = self[self.currGrainId]
                if self.grainPlot is None or not self.grainPlot.exists:
                    self.grainPlot = currGrain.plotDefault(makeInteractive=True)
                else:
                    self.grainPlot.clear()
                    self.grainPlot.callingGrain = currGrain
                    currGrain.plotDefault(plot=self.grainPlot)

    def setHomogPoint(self, binSize=1, points=None, **kwargs):
        """
        Interactive tool to set homologous points. Right-click on a point
        then click 'save point' to append to the homologous points list.

        Parameters
        ----------
        binSize : int, optional
            Binning applied to image, if applicable.
        points : numpy.ndarray, optional
            Array of (x,y) homologous points to set explicitly.
        kwargs : dict, optional
            Keyword arguments for matplotlib.

        """
        if points is None:
            plot = self.plotHomog(makeInteractive=True, **kwargs)
            # Plot stored homogo points if there are any
            if len(self.homogPoints) > 0:
                homogPoints = np.array(self.homogPoints) * binSize
                plot.addPoints(homogPoints[:, 0], homogPoints[:, 1],
                               c='y', s=60)
            else:
                # add empty points layer to update later
                plot.addPoints([None], [None], c='y', s=60)

            # add empty points layer for current selected point
            plot.addPoints([None], [None], c='w', s=60, marker='x')

            plot.addEventHandler('button_press_event', self.clickHomog)
            plot.addEventHandler('key_press_event', self.keyHomog)
            plot.addButton("Save point",
                           lambda e, p: self.clickSaveHomog(e, p, binSize),
                           color="0.85", hovercolor="blue")
        else:
            self.homogPoints = points

    def clickHomog(self, event, plot):
        """Event handler for capturing position when clicking on a map.

        Parameters
        ----------
        event :
            Click event.
        plot : defdap.plotting.Plot
            Plot to monitor.

        """
        if event.inaxes is plot.ax:
            # right mouse click or shift + left mouse click
            # shift click doesn't work in osx backend
            if (event.button == 3 or
                    (event.button == 1 and event.key == 'shift')):
                plot.addPoints([int(event.xdata)], [int(event.ydata)],
                               updateLayer=1)

    def keyHomog(self, event, plot):
        """Event handler for moving position using keyboard after clicking on a map.

        Parameters
        ----------
        event :
            Keypress event.
        plot : defdap.plotting.Plot
            Plot to monitor.

        """
        keys = ['left', 'right', 'up', 'down']
        key = event.key.split('+')
        if key[-1] in keys:
            # get the selected point
            points = plot.imgLayers[plot.pointsLayerIDs[1]]
            selPoint = points.get_offsets()[0]

            # check if a point is selected
            if selPoint[0] is not None and selPoint[1] is not None:
                # print(event.key)
                move = 1
                if len(key) == 2 and key[0] == 'shift':
                    move = 10

                if key[-1] == keys[0]:
                    selPoint[0] -= move
                elif key[-1] == keys[1]:
                    selPoint[0] += move
                elif key[-1] == keys[2]:
                    selPoint[1] -= move
                elif key[-1] == keys[3]:
                    selPoint[1] += move

                plot.addPoints([selPoint[0]], [selPoint[1]], updateLayer=1)

    def clickSaveHomog(self, event, plot, binSize):
        """Append the selected point on the map to homogPoints.

        Parameters
        ----------
        event :
            Button click event.
        plot : defdap.plotting.Plot
            Plot to monitor.
        binSize : int, optional
            Binning applied to image, if applicable.

        """
        # get the selected point
        points = plot.imgLayers[plot.pointsLayerIDs[1]]
        selPoint = points.get_offsets()[0]

        # Check if a point is selected
        if selPoint[0] is not None and selPoint[1] is not None:
            # remove selected point from plot
            plot.addPoints([None], [None], updateLayer=1)

            # then scale and add to homog points list
            selPoint = tuple((selPoint / binSize).round().astype(int))
            self.homogPoints.append(selPoint)

            # update the plotted homog points
            homogPoints = np.array(self.homogPoints) * binSize
            plot.addPoints(homogPoints[:, 0], homogPoints[:, 1], updateLayer=0)

    def updateHomogPoint(self, homogID, newPoint=None, delta=None):
        """
        Update a homog point by either over writing it with a new point or
        incrementing the current values.

        Parameters
        ----------
        homogID : int
            ID (place in list) of point to update or -1 for all.
        newPoint : tuple, optional
            (x, y) coordinates of new point.
        delta : tuple, optional
            Increments to current point (dx, dy).

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

    def buildNeighbourNetwork(self):
        """Construct a list of neighbours

        """
        yLocs, xLocs = np.nonzero(self.boundaries)
        neighboursList = []

        for y, x in zip(yLocs, xLocs):
            if (x == 0 or y == 0 or x == self.grains.shape[1] - 1 or
                    y == self.grains.shape[0] - 1):
                # exclude boundary pixel of map
                continue
            else:
                # use 4 nearest neighbour points as potential neighbour grains
                # (this maybe needs changing considering the position of
                # boundary pixels relative to the actual edges)
                # use sets as they do not allow duplicate elements
                # minus 1 on all as the grain image starts labeling at 1
                neighbours = {
                    self.grains[y + 1, x] - 1,
                    self.grains[y - 1, x] - 1,
                    self.grains[y, x + 1] - 1,
                    self.grains[y, x - 1] - 1
                }
                # neighbours = set(neighbours)
                # remove boundary points (-2) and points in small
                # grains (-3) (Normally -1 and -2)
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
        self.neighbourNetwork = nx.Graph()
        self.neighbourNetwork.add_nodes_from(range(len(self)))
        self.neighbourNetwork.add_edges_from(neighboursList)

    def displayNeighbours(self):
        self.locateGrainID(clickEvent=self.clickGrainNeighbours)

    def clickGrainNeighbours(self, event, plot):
        """Event handler to capture clicking and show neighbours of selected grain.

        Parameters
        ----------
        event :
            Click event.
        plot : defdap.plotting.Plot
            Plot to monitor.

        """
        if event.inaxes is plot.ax:
            # grain id of selected grain
            grainId = int(self.grains[int(event.ydata), int(event.xdata)] - 1)
            if grainId < 0:
                return
            self.currGrainId = grainId

            # find first and second nearest neighbours
            firstNeighbours = list(self.neighbourNetwork.neighbors(self.currGrainId))
            highlightGrains = [self.currGrainId] + firstNeighbours

            secondNeighbours = []
            for firstNeighbour in firstNeighbours:
                trialSecondNeighbours = list(self.neighbourNetwork.neighbors(firstNeighbour))
                for secondNeighbour in trialSecondNeighbours:
                    if (secondNeighbour not in highlightGrains and
                            secondNeighbour not in secondNeighbours):
                        secondNeighbours.append(secondNeighbour)
            highlightGrains.extend(secondNeighbours)

            highlightColours = ['white']
            highlightColours.extend(['yellow'] * len(firstNeighbours))
            highlightColours.append('green')

            # update the grain highlights layer in the plot
            plot.addGrainHighlights(highlightGrains, grainColours=highlightColours)

    @property
    def proxigram(self):
        """Proxigram for a map.

        Returns
        -------
        numpy.ndarray
            Distance from a grain boundary at each point in map.

        """
        self.calcProxigram(forceCalc=False)

        return self.proxigramArr

    @reportProgress("calculating proxigram")
    def calcProxigram(self, numTrials=500, forceCalc=True):
        """Calculate distance from a grain boundary at each point in map.

        Parameters
        ----------
        numTrials : int, optional
            number of trials.
        forceCalc : bool, optional
            Force calculation even is proxigramArr is populated.

        """
        if self.proxigramArr is not None and not forceCalc:
            return

        proxBoundaries = np.copy(self.boundaries)
        proxShape = proxBoundaries.shape

        # ebsd boundary arrays have extra boundary along right and
        # bottom edge. These need to be removed right edge
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
        # add 0.5 to boundary coordiantes as they are placed on the
        # bottom right edge pixels of grains
        indexBoundaries = np.array(indexBoundaries) + 0.5

        # array of x and y coordinate of each pixel in the map
        coords = np.zeros((2, proxShape[0], proxShape[1]), dtype=float)
        coords[0], coords[1] = np.meshgrid(
            range(proxShape[0]), range(proxShape[1]), indexing='ij'
        )

        # array to store trial distance from each boundary point
        trialDistances = np.full((numTrials + 1, proxShape[0], proxShape[1]),
                                 1000, dtype=float)

        # loop over each boundary point (p) and calculate distance from
        # p to all points in the map store minimum once numTrails have
        # been made and start a new batch of trials
        numBoundaryPoints = len(indexBoundaries)
        j = 1
        for i, indexBoundary in enumerate(indexBoundaries):
            trialDistances[j] = np.sqrt((coords[0] - indexBoundary[0])**2
                                        + (coords[1] - indexBoundary[1])**2)

            if j == numTrials:
                # find current minimum distances and store
                trialDistances[0] = trialDistances.min(axis=0)
                j = 0
                # report progress
                yield i / numBoundaryPoints
            j += 1

        # find final minimum distances to a boundary
        self.proxigramArr = trialDistances.min(axis=0)

        trialDistances = None

    def calcGrainAv(self, mapData, grainIds=-1):
        """Calculate grain average of any DIC map data.

        Parameters
        ----------
        mapData : numpy.ndarray
            Array of map data to grain average. This must be cropped!
        grainIds : list, optional
            grainIDs to perform operation on, set to -1 for all grains.

        Returns
        -------
        numpy.ndarray
            Array containing the grain average values.

        """

        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        if type(grainIds) is int and grainIds == -1:
            grainIds = range(len(self))

        grainAvData = np.zeros(len(grainIds))

        for i, grainId in enumerate(grainIds):
            grain = self[grainId]
            grainData = grain.grainData(mapData)
            grainAvData[i] = grainData.mean()

        return grainAvData

    def plotGrainDataMap(self, mapData=None, grainData=None,
                         grainIds=-1, bg=0, **kwargs):
        """
        Plot a grain map with grains coloured by given data. The data
        can be provided as a list of values per grain or as a map which
        a grain average will be applied.

        Parameters
        ----------
        mapData : numpy.ndarray, optional
            Array of map data. This must be cropped! You must supply either
            mapData or grainData.
        grainData : list or numpy.array, optional
            Grain values. This an be a single value per grain or RGB
            values. You must supply either mapData or grainData.
        grainIds : list(int) or int, optional
            IDs of grains to plot for. Use -1 for all grains in the map.
        bg : int or real, optional
            Value to fill the background with.
        kwargs :
            Other parameters are passed to defdap.plotting.MapPlot.create.

        Returns
        -------
        plot : defdap.plotting.MapPlot
            Plot object created.

        """
        # Set default plot parameters then update with any input
        plotParams = {}
        plotParams.update(kwargs)

        if grainData is None:
            if mapData is None:
                raise ValueError("Either 'mapData' or 'grainData' must "
                                 "be supplied.")
            else:
                grainData = self.calcGrainAv(mapData, grainIds=grainIds)

        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        if type(grainIds) is int:
            if grainIds == -1:
                grainIds = range(len(self))
            else:
                grainIds = [grainIds]

        grainData = np.array(grainData)
        if grainData.shape[0] != len(grainIds):
            raise Exception("The length of supplied grain data does not"
                            "match the number of grains.")
        if len(grainData.shape) == 1:
            mapShape = [self.yDim, self.xDim]
        elif len(grainData.shape) == 2 and grainData.shape[1] == 3:
            mapShape = [self.yDim, self.xDim, 3]
        else:
            raise Exception("The grain data supplied must be either a"
                            "single value or RGB values per grain.")

        grainMap = np.full(mapShape, bg, dtype=grainData.dtype)
        for grainId, grainValue in zip(grainIds, grainData):
            grain = self.grainList[grainId]
            for coord in grain.coordList:
                grainMap[coord[1], coord[0]] = grainValue

        plot = MapPlot.create(self, grainMap, **plotParams)

        return plot

    def plotGrainDataIPF(
            self, direction, mapData=None, grainData=None,
            grainIds=-1, **kwargs
    ):
        """
        Plot IPF of grain reference (average) orientations with
        points coloured by grain average values from map data.

        Parameters
        ----------
        mapData : numpy.ndarray
            Array of map data to grain average. This must be cropped!
        direction : numpy.ndarray
            Vector of reference direction for the IPF.
        plotColourBar : bool, optional
            Set to False to exclude the colour bar from the plot.
        vmin : float, optional
            Minimum value of colour scale.
        vmax : float, optional
            Maximum value for colour scale.
        cLabel : str, optional
            Colour bar label text.

        """
        # Set default plot parameters then update with any input
        plotParams = {}
        plotParams.update(kwargs)

        if grainData is None:
            if mapData is None:
                raise ValueError("Either 'mapData' or 'grainData' must "
                                 "be supplied.")
            else:
                grainData = self.calcGrainAv(mapData, grainIds=grainIds)

        # Check that grains have been detected in the map
        self.checkGrainsDetected()

        if type(grainIds) is int and grainIds == -1:
            grainIds = range(len(self))

        if len(grainData) != len(grainIds):
            raise Exception("Must be 1 value for each grain in grainData.")

        grainOri = np.empty(len(grainIds), dtype=Quat)

        for i, grainId in enumerate(grainIds):
            grain = self[grainId]
            grainOri[i] = grain.refOri

        plot = Quat.plotIPF(grainOri, direction, self.crystalSym,
                            c=grainData, **plotParams)

        return plot


class Grain(object):
    """
    Base class for a grain.
    """
    def __init__(self):
        # list of coords stored as tuples (x, y). These are coords in a
        # cropped image if crop exists.
        self.coordList = []

    def __len__(self):
        return len(self.coordList)

    @property
    def extremeCoords(self):
        """Coordinates of the bounding box for a grain.

        Returns
        -------
        int, int, int, int
            minimum x, minimum y, maximum x, maximum y.

        """
        coords = np.array(self.coordList, dtype=int)

        x0, y0 = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)

        return x0, y0, xmax, ymax

    def centreCoords(self, centreType="box", grainCoords=True):
        """
        Calculates the centre of the grain, either as the centre of the
        bounding box or the grains centre of mass.

        Parameters
        ----------
        centreType : str, optional, {'box', 'com'}
            Set how to calculate the centre. Either 'box' for centre of
            bounding box or 'com' for centre of mass. Default is 'box'.
        grainCoords : bool, optional
            If set True the centre is returned in the grain coordinates
            otherwise in the map coordinates. Defaults is grain.

        Returns
        -------
        int, int
            Coordinates of centre of grain.

        """
        x0, y0, xmax, ymax = self.extremeCoords
        if centreType == "box":
            xCentre = round((xmax + x0) / 2)
            yCentre = round((ymax + y0) / 2)
        elif centreType == "com":
            xCentre, yCentre = np.array(self.coordList).mean(axis=0).round()
        else:
            raise ValueError("centreType must be box or com")

        if grainCoords:
            xCentre -= x0
            yCentre -= y0

        return int(xCentre), int(yCentre)

    def grainOutline(self, bg=np.nan, fg=0):
        """Generate an array of the grain outline.

        Parameters
        ----------
        bg : int
            Value for points not within grain.
        fg : int
            Value for points within grain.

        Returns
        -------
        numpy.ndarray
            Bounding box for grain with np.nan outside the grain and given number within.

        """
        x0, y0, xmax, ymax = self.extremeCoords

        # initialise array with nans so area not in grain displays white
        outline = np.full((ymax - y0 + 1, xmax - x0 + 1), bg, dtype=int)

        for coord in self.coordList:
            outline[coord[1] - y0, coord[0] - x0] = fg

        return outline

    def plotOutline(self, ax=None, plotScaleBar=False, **kwargs):
        """Plot the outline of the grain.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            axis to plot on, if not provided the current active axis is used.
        plotScaleBar : bool
            plots the scale bar on the grain if true.
        kwargs : dict
            keyword arguments to pass to plotting.GrainPlot.addMap.

        Returns
        -------
        defdap.plotting.GrainPlot

        """
        plot = plotting.GrainPlot(self, ax=ax)
        plot.addMap(self.grainOutline(), **kwargs)

        if plotScaleBar:
            plot.addScaleBar()

        return plot

    def grainData(self, mapData):
        """Extract this grains data from the given map data.

        Parameters
        ----------
        mapData : numpy.ndarray
            Array of map data. This must be cropped!

        Returns
        -------
        numpy.ndarray
            Array containing this grains values from the given map data.

        """
        grainData = np.zeros(len(self), dtype=mapData.dtype)

        for i, coord in enumerate(self.coordList):
            grainData[i] = mapData[coord[1], coord[0]]

        return grainData

    def grainMapData(self, mapData=None, grainData=None, bg=np.nan):
        """Extract a single grain map from the given map data.

        Parameters
        ----------
        mapData : numpy.ndarray
            Array of map data. This must be cropped! Either this or
            'grainData' must be supplied and 'grainData' takes precedence.
        grainData : numpy.ndarray
            Array of data at each point in the grain. Either this or
            'mapData' must be supplied and 'grainData' takes precedence.
        bg : various, optional
            Value to fill the background with. Must be same dtype as
            input array.

        Returns
        -------
        numpy.ndarray
            Grain map extracted from given data.

        """
        if grainData is None:
            if mapData is None:
                raise ValueError("Either 'mapData' or 'grainData' must "
                                 "be supplied.")
            else:
                grainData = self.grainData(mapData)
        x0, y0, xmax, ymax = self.extremeCoords

        grainMapData = np.full((ymax - y0 + 1, xmax - x0 + 1), bg,
                               dtype=type(grainData[0]))

        for coord, data in zip(self.coordList, grainData):
            grainMapData[coord[1] - y0, coord[0] - x0] = data

        return grainMapData

    def grainMapDataCoarse(self, mapData=None, grainData=None,
                           kernelSize=2, bg=np.nan):
        """
        Create a coarsed data map of this grain only from the given map
        data. Data is coarsened using a kernel at each pixel in the
        grain using only data in this grain.

        Parameters
        ----------
        mapData : numpy.ndarray
            Array of map data. This must be cropped! Either this or
            'grainData' must be supplied and 'grainData' takes precedence.
        grainData : numpy.ndarray
            List of data at each point in the grain. Either this or
            'mapData' must be supplied and 'grainData' takes precedence.
        kernelSize : int, optional
            Size of kernel as the number of pixels to dilate by i.e 1
            gives a 3x3 kernel.
        bg : various, optional
            Value to fill the background with. Must be same dtype as
            input array.

        Returns
        -------
        numpy.ndarray
            Map of this grains coarsened data.

        """
        grainMapData = self.grainMapData(mapData=mapData, grainData=grainData)
        grainMapDataCoarse = np.full_like(grainMapData, np.nan)

        for i, j in np.ndindex(grainMapData.shape):
            if np.isnan(grainMapData[i, j]):
                grainMapDataCoarse[i, j] = bg
            else:
                coarseValue = 0

                if i - kernelSize >= 0:
                    yLow = i - kernelSize
                else:
                    yLow = 0
                if i + kernelSize + 1 <= grainMapData.shape[0]:
                    yHigh = i + kernelSize + 1
                else:
                    yHigh = grainMapData.shape[0]
                if j - kernelSize >= 0:
                    xLow = j - kernelSize
                else:
                    xLow = 0
                if j + kernelSize + 1 <= grainMapData.shape[1]:
                    xHigh = j + kernelSize + 1
                else:
                    xHigh = grainMapData.shape[1]

                numPoints = 0
                for k in range(yLow, yHigh):
                    for l in range(xLow, xHigh):
                        if not np.isnan(grainMapData[k, l]):
                            coarseValue += grainMapData[k, l]
                            numPoints += 1

                if numPoints > 0:
                    grainMapDataCoarse[i, j] = coarseValue / numPoints
                else:
                    grainMapDataCoarse[i, j] = np.nan

        return grainMapDataCoarse

    def plotGrainData(self, mapData=None, grainData=None, **kwargs):
        """
        Plot a map of this grain from the given map data.

        Parameters
        ----------
        mapData : numpy.ndarray
            Array of map data. This must be cropped! Either this or
            'grainData' must be supplied and 'grainData' takes precedence.
        grainData : numpy.ndarray
            List of data at each point in the grain. Either this or
            'mapData' must be supplied and 'grainData' takes precedence.
        vmin : float, optional
            Minimum value of colour scale.
        vmax : float, optional
            Minimum value of colour scale.
        cLabel : str, optional
            Colour bar label text.
        cmap : str, optional
            Colour map to use, default is viridis.

        """
        # Set default plot parameters then update with any input
        plotParams = {}
        plotParams.update(kwargs)

        grainMapData = self.grainMapData(mapData=mapData, grainData=grainData)

        plot = GrainPlot.create(self, grainMapData, **plotParams)

        return plot
