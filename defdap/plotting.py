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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

from skimage import morphology as mph

from defdap import quat
# TODO: add plot parameter to add to current figure


class Plot(object):
    """ Class used for creating and manipulating plots.

    """
    def __init__(self, ax, axParams={}, fig=None, makeInteractive=False,
                 title=None, **kwargs):
        self.interactive = makeInteractive
        if makeInteractive:
            if fig is not None and ax is not None:
                self.fig = fig
                self.ax = ax
            else:
                # self.fig, self.ax = plt.subplots(**kwargs)
                self.fig = plt.figure(**kwargs)
                self.ax = self.fig.add_subplot(111, **axParams)
            self.btnStore = []
            self.txtStore = []
            self.txtBoxStore = []
        else:
            self.fig = fig
            # TODO: flag for new figure
            if ax is None:
                self.fig = plt.figure(**kwargs)
                self.ax = self.fig.add_subplot(111, **axParams)
            else:
                self.ax = ax
        self.colourBar = None

        if title is not None:
            self.setTitle(title)

    def checkInteractive(self):
        """Checks if current plot is interactive.

        Raises
        -------
        Exception
            If plot is not interactive

        """
        if not self.interactive:
            raise Exception("Plot must be interactive")

    def addEventHandler(self, eventName, eventHandler):
        self.checkInteractive()

        self.fig.canvas.mpl_connect(eventName, lambda e: eventHandler(e, self))

    def addAxes(self, loc, proj='2d'):
        """Add axis to current plot

        Parameters
        ----------
        loc
            Location of axis.
        proj : str, {2d, 3d}
            2D or 3D projection.

        Returns
        -------
        matplotlib.Axes.axes

        """
        if proj == '2d':
            return self.fig.add_axes(loc)
        if proj == '3d':
            return Axes3D(self.fig, rect=loc, proj_type='ortho', azim=270, elev=90)

    def addButton(self, label, clickHandler, loc=(0.8, 0.0, 0.1, 0.07), **kwargs):
        """Add a button to the plot.

        Parameters
        ----------
        label : str
            Label for the button.
        clickHandler
            Click handler to assign.
        loc : list(float), len 4
            Left, bottom, width, height.
        kwargs
            All other arguments passed to :class:`matplotlib.widgets.Button`.

        """
        self.checkInteractive()
        btnAx = self.fig.add_axes(loc)
        btn = Button(btnAx, label, **kwargs)
        btn.on_clicked(lambda e: clickHandler(e, self))

        self.btnStore.append(btn)

    def addTextBox(self, label, submitHandler, loc=(0.8, 0.0, 0.1, 0.07), **kwargs):
        """Add a text box to the plot.

        Parameters
        ----------
        label : str
            Label for the button.
        submitHandler
            Submit handler to assign.
        loc : list(float), len 4
            Left, bottom, width, height.
        kwargs
            All other arguments passed to :class:`matplotlib.widgets.TextBox`.

        Returns
        -------
        matplotlotlib.widgets.TextBox

        """
        self.checkInteractive()
        txtBoxAx = self.fig.add_axes(loc)
        txtBox = TextBox(txtBoxAx, label, **kwargs)
        txtBox.on_submit(lambda e: submitHandler(e, self))

        self.txtBoxStore.append(txtBox)

        return txtBox

    def addText(self, ax, x, y, txt, **kwargs):
        """Add text to the plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matplotlib axis to plot on.
        x : float
            x position.
        y : float
            y position.
        txt : str
            Text to write onto the plot.

        kwargs :
            All other arguments passed to :func:`matplotlib.pyplot.text`.

        """
        txt = ax.text(x, y, txt, **kwargs)
        self.txtStore.append(txt)

    def setSize(self, size):
        """Set size of plot.

        Parameters
        ----------
        size : float, float
            Width and height in inches.

        """
        self.fig.set_size_inches(size[0], size[1], forward=True)

    def setTitle(self, txt):
        """Set title of plot.

        Parameters
        ----------
        txt : str
            Title to set.

        """
        self.fig.canvas.set_window_title(txt)

    @property
    def exists(self):
        self.checkInteractive()

        return plt.fignum_exists(self.fig.number)

    def clear(self):
        """Clear plot.

        """
        self.checkInteractive()

        self.ax.clear()
        if self.colourBar is not None:
            self.colourBar.remove()
        self.draw()

    def draw(self):
        """Draw plot

        """
        self.fig.canvas.draw()


class MapPlot(Plot):
    """ Class for creating a map plot.

    """
    def __init__(self, callingMap, fig=None, ax=None, axParams={},
                 makeInteractive=False, **kwargs):
        """Initialise a map plot.

        Parameters
        ----------
        callingMap : Map
            DIC or EBSD map which called this plot.
        fig : matplotlib.figure.Figure
            Matplotlib figure to plot on
        ax : matplotlib.axes.Axes
            Matplotlib axis to plot on
        axParams :
            Passed to defdap.plotting.Plot as axParams.
        makeInteractive : bool, optional
            If true, make interactive
        kwargs
            Other arguments passed to :class:`defdap.plotting.Plot`.
        """
        super(MapPlot, self).__init__(
            ax, axParams=axParams, fig=fig, makeInteractive=makeInteractive,
            **kwargs
        )

        self.callingMap = callingMap
        self.imgLayers = []
        self.highlightsLayerID = None
        self.pointsLayerIDs = []

        self.ax.set_xticks([])
        self.ax.set_yticks([])

    def addMap(self, mapData, vmin=None, vmax=None, cmap='viridis', **kwargs):
        """Add a map to a plot.

        Parameters
        ----------
        mapData : numpy.ndarray
            Map data to plot.
        vmin : float
            Minimum value for the colour scale.
        vmax : float
            Maximum value for the colour scale.
        cmap
            Colour map.
        kwargs
            Other arguments are passed to :func:`matplotlib.pyplot.imshow`.

        Returns
        -------
        matplotlib.image.AxesImage

        """
        img = self.ax.imshow(mapData, vmin=vmin, vmax=vmax,
                             interpolation='None', cmap=cmap, **kwargs)
        self.draw()

        self.imgLayers.append(img)

        return img

    def addColourBar(self, label, layer=0, **kwargs):
        """Add a colour bar to plot.

        Parameters
        ----------
        label : str
            Label for the colour bar.
        layer : int
            Layer ID.
        kwargs
            Other arguments are passed to :func:`matplotlib.pyplot.colorbar`.

        """
        img = self.imgLayers[layer]
        self.colourBar = plt.colorbar(img, ax=self.ax, label=label, **kwargs)

    def addScaleBar(self, scale=None):
        """Add scale bar to plot.

        Parameters
        ----------
        scale : float
            Size of a pixel in microns.

        """
        if scale is None:
            scale = self.callingMap.scale * 1e-6
        self.ax.add_artist(ScaleBar(scale))

    def addGrainBoundaries(self, colour=None, dilate=False):
        """Add grain boundaries to the plot.

        Parameters
        ----------
        colour : str
            Colour of grain boundaries.
        dilate : bool
            If true, dilate the grain boundaries

        Returns
        -------
        matplotlib.image.AxesImage

        """
        if colour is None:
            colour = "white"

        boundariesImage = -self.callingMap.boundaries

        if dilate:
            boundariesImage = mph.binary_dilation(boundariesImage)

        # create colourmap for boundaries going from transparent to
        # opaque of the given colour
        boundariesCmap = mpl.colors.LinearSegmentedColormap.from_list(
            'my_cmap', ['white', colour], 256
        )
        boundariesCmap._init()
        boundariesCmap._lut[:, -1] = np.linspace(0, 1, boundariesCmap.N + 3)

        img = self.ax.imshow(boundariesImage, cmap=boundariesCmap,
                             interpolation='None', vmin=0, vmax=1)
        self.draw()

        self.imgLayers.append(img)

        return img

    def addGrainHighlights(self, grainIds, grainColours=None, alpha=None,
                           newLayer=False):
        """Highlight grains in the plot.

        Parameters
        ----------
        grainIds : list
            List of grain IDs to highlight.
        grainColours :
            Colour to use for grain highlight.
        alpha : float
            Alpha (transparency) to use for grain highlight.
        newLayer : bool
            If true, make a new layer in imgLayers.

        Returns
        -------
        matplotlib.image.AxesImage

        """
        if grainColours is None:
            grainColours = ['white']
        if alpha is None:
            alpha = self.callingMap.highlightAlpha

        xDim = self.callingMap.xDim
        yDim = self.callingMap.yDim

        outline = np.zeros((yDim, xDim), dtype=int)
        for i, grainId in enumerate(grainIds, start=1):
            if i > len(grainColours):
                i = len(grainColours)

            # outline of highlighted grain
            grain = self.callingMap.grainList[grainId]
            grainOutline = grain.grainOutline(bg=0, fg=i)
            x0, y0, xmax, ymax = grain.extremeCoords

            # add to highlight image
            outline[y0:ymax + 1, x0:xmax + 1] += grainOutline

        # Custom colour map where 0 is transparent white for bg and
        # then a patch for each grain colour
        grainColours.insert(0, 'white')
        hightlightsCmap = mpl.colors.ListedColormap(grainColours)
        hightlightsCmap._init()
        alphaMap = np.full(hightlightsCmap.N + 3, alpha)
        alphaMap[0] = 0
        hightlightsCmap._lut[:, -1] = alphaMap

        if self.highlightsLayerID is None or newLayer:
            img = self.ax.imshow(outline, interpolation='none',
                                 cmap=hightlightsCmap)
            if self.highlightsLayerID is None:
                self.highlightsLayerID = len(self.imgLayers)
            self.imgLayers.append(img)
        else:
            img = self.imgLayers[self.highlightsLayerID]
            img.set_data(outline)
            img.set_cmap(hightlightsCmap)
            img.autoscale()

        self.draw()

        return img

    def addGrainNumbers(self, fontsize=10, **kwargs):
        """Add grain numbers to a map.

        Parameters
        ----------
        fontsize : float
            Font size.
        kwargs
            Pass other arguments to :func:`matplotlib.pyplot.text`.

        """
        for grainID, grain in enumerate(self.callingMap):
            xCentre, yCentre = grain.centreCoords(centreType="com",
                                                  grainCoords=False)

            self.ax.text(xCentre, yCentre, grainID,
                         fontsize=fontsize, **kwargs)
        self.draw()

    def addLegend(self, values, labels, layer=0, **kwargs):
        """Add a legend to a map.

        Parameters
        ----------
        values : list
            Values to find colour patched for.
        labels : list
            Labels to assign to values.
        layer : int
            Image layer to generate legend for.
        kwargs
            Pass other arguments to :func:`matplotlib.pyplot.legend`.

        """
        # Find colour values for given values
        img = self.imgLayers[layer]
        colors = [img.cmap(img.norm(value)) for value in values]

        # Get colour patches for each phase and make legend
        patches = [mpl.patches.Patch(
            color=colors[i], label=labels[i]
        ) for i in range(len(values))]

        self.ax.legend(handles=patches, **kwargs)

    def addPoints(self, x, y, updateLayer=None, **kwargs):
        """Add points to plot.

        Parameters
        ----------
        x : float
            x coordinate.
        y : float
            y coordinate.
        updateLayer : int, optional
            Layer to place points on
        kwargs
            Other arguments passed to :func:`matplotlib.pyplot.scatter`.

        """
        x, y = np.array(x), np.array(y)
        if len(self.pointsLayerIDs) == 0 or updateLayer is None:
            points = self.ax.scatter(x, y, **kwargs)
            self.pointsLayerIDs.append(len(self.imgLayers))
            self.imgLayers.append(points)
        else:
            points = self.imgLayers[self.pointsLayerIDs[updateLayer]]
            points.set_offsets(np.hstack((x[:, np.newaxis], y[:, np.newaxis])))

        self.draw()

        return points

    @classmethod
    def create(
        cls, callingMap, mapData,
        fig=None, figParams={}, ax=None, axParams={},
        plot=None, makeInteractive=False,
        plotColourBar=False, vmin=None, vmax=None, cmap=None, clabel="",
        plotGBs=False, dilateBoundaries=False, boundaryColour=None,
        plotScaleBar=False, scale=None,
        highlightGrains=None, highlightColours=None, highlightAlpha=None,
        **kwargs
    ):
        """Create a plot for a map.

        Parameters
        ----------
        callingMap : Map
            DIC or EBSD map which called this plot.
        mapData : numpy.ndarray
            Data to be plotted.
        fig : matplotlib.figure.Figure
            Matplotlib figure to plot on.
        figParams :
            Passed to defdap.plotting.Plot.
        ax : matplotlib.axes.Axes
            Matplotlib axis to plot on.
        axParams :
            Passed to defdap.plotting.Plot as axParams.
        plot : defdap.plotting.Plot
            If none, use current plot.
        makeInteractive :
            If true, make plot interactive
        plotColourBar : bool
            If true, plot a colour bar next to the map.
        vmin : float, optional
            Minimum value for the colour scale.
        vmax : float, optional
            Maximum value for the colour scale.
        cmap : str
            Colour map.
        clabel : str
            Label for the colour bar.
        plotGBs : bool
            If true, plot the grain boundaries on the map.
        dilateBoundaries : bool
            If true, dilate the grain boundaries.
        boundaryColour : str
            Colour to use for the grain boundaries.
        plotScaleBar : bool
            If true, plot a scale bar in the map.
        scale : float
            Size of pizel in microns.
        highlightGrains : list(int)
            List of grain IDs to highlight.
        highlightColours : str
            Colour to hightlight grains.
        highlightAlpha : float
            Alpha (transparency) by which to highlight grains.
        kwargs :
            All other arguments passed to :func:`defdap.plotting.MapPlot.addMap`

        Returns
        -------
        defdap.plotting.MapPlot

        """
        if plot is None:
            plot = cls(callingMap, fig=fig, ax=ax, axParams=axParams,
                       makeInteractive=makeInteractive, **figParams)

        if mapData is not None:
            plot.addMap(mapData, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

        if plotColourBar:
            plot.addColourBar(clabel)

        if plotGBs:
            plot.addGrainBoundaries(colour=boundaryColour, dilate=dilateBoundaries)

        if highlightGrains is not None:
            plot.addGrainHighlights(
                highlightGrains,
                grainColours=highlightColours, alpha=highlightAlpha
            )

        if plotScaleBar:
            plot.addScaleBar(scale=scale)

        return plot


class LineSlice:
    """ Class to catch click and drag and return start and end positions.

    """
    def __init__(self, fig, ax, action):
        self.p1=[0,0]
        self.p2=[0,0]
        self.ax = ax
        self.cidclick = plt.connect('button_press_event', self)
        self.cidrelease = plt.connect('button_release_event', self)
        self.action = action
        self.fig=fig

    def __call__(self, event):
        if event.name == 'button_press_event':
            self.p1 = (event.xdata, event.ydata)    # save 1st point
        elif event.name == 'button_release_event':
            self.p2 = (event.xdata, event.ydata)    # save 2nd point

            self.action(startEnd=(self.p1[0], self.p1[1], self.p2[0], self.p2[1]))
            self.fig.canvas.draw()

            self.points = (self.p1[0], self.p1[1], self.p2[0], self.p2[1])

            return self.p1[0], self.p1[1], self.p2[0], self.p2[1]


class GrainPlot(Plot):
    """ Class for creating a map for a grain.

    """
    def __init__(self, callingGrain, fig=None, ax=None, axParams={},
                 makeInteractive=False, **kwargs):
        super(GrainPlot, self).__init__(
            ax, axParams=axParams, fig=fig, makeInteractive=makeInteractive,
            **kwargs
        )

        self.callingGrain = callingGrain
        self.imgLayers = []

        self.ax.set_xticks([])
        self.ax.set_yticks([])

    def addMap(self, mapData, vmin=None, vmax=None, cmap='viridis', **kwargs):
        """Add a map to a grain plot.

        Parameters
        ----------
        mapData : numpy.ndarray
            Grain data to plot
        vmin : float
            Minimum value for the colour scale.
        vmax : float
            Maximum value for the colour scale.
        cmap
            Colour map to use.
        kwargs
            Other arguments are passed to :func:`matplotlib.pyplot.imshow`.

        Returns
        -------
        matplotlib.image.AxesImage

        """
        img = self.ax.imshow(mapData, vmin=vmin, vmax=vmax,
                             interpolation='None', cmap=cmap, **kwargs)
        self.draw()
        self.arrow = None

        self.imgLayers.append(img)

        return img

    def addArrow(self, startEnd, persistent=False, clearPrev=True, label=None):
        """Add arrow to grain plot.

        Parameters
        ----------
        startEnd: 4-tuple
            Starting (x, y), Ending (x, y).
        persistent :
            If persistent, do not clear arrow with clearPrev.
        clearPrev :
            Clear all non-persistent arrows.
        label
            Label to place near arrow.

        """

        x0 = startEnd[0]
        y0 = startEnd[1]
        x1 = startEnd[2]
        y1 = startEnd[3]

        if persistent:
            self.ax.annotate(
                "", xy=(x0, y0), xycoords='data', xytext=(x1, y1),
                textcoords='data', arrowprops=dict(
                    arrowstyle="<-", connectionstyle="arc3",
                    color='red', alpha=0.7, linewidth=2
                )
            )

        if not persistent:
            if clearPrev:
                if self.arrow is not None:
                    self.arrow.remove()

            if None in (x0, y0, x1, y1):
                pass
            else:
                self.arrow = self.ax.annotate(
                    "", xy=(x0, y0), xycoords='data', xytext=(x1, y1),
                    textcoords='data', arrowprops=dict(
                        arrowstyle="<-", connectionstyle="arc3",
                        color='red',alpha=0.7,linewidth=2
                    )
                )

        if label is not None:
            self.ax.annotate(label, xy=(x1, y1), xycoords='data',
                             xytext=(15, 15), textcoords='offset pixels',
                             c='red', fontsize=12)

    def addColourBar(self, label, layer=0, **kwargs):
        """Add colour bar to grain plot.

        Parameters
        ----------
        label : str
            Label to add to colour bar.
        layer : int
            Layer on which to add colourbar.
        kwargs
            Other arguments passed to :func:`matplotlib.pyplot.colorbar`.

        """
        img = self.imgLayers[layer]
        self.colourBar = plt.colorbar(img, ax=self.ax, label=label, **kwargs)

    def addScaleBar(self, scale=None):
        """Add scale bar to grain plot.

        Parameters
        ----------
        scale : float
            Size of pixel in micron.

        """
        if scale is None:
            scale = self.callingGrain.ownerMap.scale * 1e-6
        self.ax.add_artist(ScaleBar(scale))

    def addTraces(self, angles, colours, topOnly=False, pos=None, **kwargs):
        """Add slip trace angles to grain plot. Illustrated by lines
        crossing through central pivot point to create a circle.

        Parameters
        ----------
        angles : list
            Angles of slip traces.
        colours : list
            Colours to plot.
        topOnly : bool, optional
            If true, plot only a semi-circle instead of a circle.
        pos : tuple
            Position of slip traces.
        kwargs
            Other arguments are passed to :func:`matplotlib.pyplot.quiver`

        """
        if pos is None:
            pos = self.callingGrain.centreCoords()
        traces = np.array((-np.sin(angles), np.cos(angles)))

        # When plotting top half only, move all 'traces' to +ve y
        # and set the pivot to be in the tail instead of centre
        if topOnly:
            pivot='tail'
            for idx, (x,y) in enumerate(zip(traces[0], traces[1])):
                if x < 0 and y < 0:
                    traces[0][idx] *= -1
                    traces[1][idx] *= -1
            self.ax.set_ylim(pos[1]-0.001, pos[1]+0.1)
            self.ax.set_xlim(pos[0]-0.1, pos[0]+0.1)
        else:
            pivot = 'middle'

        for i, trace in enumerate(traces.T):
            colour = colours[len(colours) - 1] if i >= len(colours) else colours[i]
            self.ax.quiver(
                pos[0], pos[1],
                trace[0], trace[1],
                scale=1, pivot=pivot,
                color=colour, headwidth=1,
                headlength=0, **kwargs
            )
            self.draw()

    def addSlipTraces(self, topOnly=False, colours=None, pos=None, **kwargs):
        """Add slip traces to plot, based on the calling grain's slip systems.

        Parameters
        ----------
        colours : list
            Colours to plot.
        topOnly : bool, optional
            If true, plot only a semi-circle instead of a circle.
        pos : tuple
            Position of slip traces.
        kwargs
            Other arguments are passed to :func:`matplotlib.pyplot.quiver`

        """

        if colours is None:
            colours = self.callingGrain.ebsdMap.slipTraceColours
        slipTraceAngles = self.callingGrain.slipTraces

        self.addTraces(slipTraceAngles, colours, topOnly, pos=pos, **kwargs)

    def addSlipBands(self, topOnly=False, grainMapData=None, angles=None, pos=None,
                     thres=None, min_dist=None, **kwargs):
        """Add lines representing slip bands detected by Radon transform
        in :func:`~defdap.hrdic.grain.calcSlipBands`.

        Parameters
        ----------
        topOnly : bool, optional
            If true, plot only a semi-circle instead of a circle.
        grainMapData :
            Map data to pass to :func:`~defdap.hrdic.Grain.calcSlipBands`.
        angles : list(float), optional
            List of angles to plot, otherwise, use angles
            detected in :func:`~defdap.hrdic.Grain.calcSlipBands`.
        pos : tuple
            Position in which to plot slip traces.
        thres : float
            Threshold to use in :func:`~defdap.hrdic.Grain.calcSlipBands`.
        min_dist :
            Minimum angle between bands in :func:`~defdap.hrdic.Grain.calcSlipBands`.
        kwargs
            Other arguments are passed to :func:`matplotlib.pyplot.quiver`.

        """

        if angles is None:
            slipBandAngles = self.callingGrain.calcSlipBands(grainMapData,
                                                             thres=thres,
                                                             min_dist=min_dist)
        else:
            slipBandAngles = angles

        self.addTraces(slipBandAngles, ["black"], topOnly,  pos=pos, **kwargs)

    @classmethod
    def create(
        cls, callingGrain, mapData,
        fig=None, figParams={}, ax=None, axParams={},
        plot=None, makeInteractive=False,
        plotColourBar=False, vmin=None, vmax=None, cmap=None, clabel="",
        plotScaleBar=False, scale=None,
        plotSlipTraces=False, plotSlipBands=False, **kwargs
    ):
        """Create grain plot.

        Parameters
        ----------
        callingGrain : Grain
            DIC or EBSD grain which called this plot.
        mapData :
            Data to be plotted.
        fig : matplotlib.figure.Figure
            Matplotlib figure to plot on.
        figParams :
            Passed to defdap.plotting.Plot.
        ax : matplotlib.axes.Axes
            Matplotlib axis to plot on.
        axParams :
            Passed to defdap.plotting.Plot as axParams.
        plot : defdap.plotting.Plot
            If none, use current plot.
        makeInteractive :
            If true, make plot interactive
        plotColourBar : bool
            If true, plot a colour bar next to the map.
        vmin : float
            Minimum value for the colour scale.
        vmax : float
            Maximum value for the colour scale.
        cmap :
            Colour map.
        clabel : str
            Label for the colour bar.
        plotScaleBar : bool
            If true, plot a scale bar in the map.
        scale : float
            Size of pizel in microns.
        plotSlipTraces : bool
            If true, plot slip traces with :func:`~defdap.plotting.GrainPlot.addSlipTraces`
        plotSlipBands : bool
            If true, plot slip traces with :func:`~defdap.plotting.GrainPlot.addSlipBands`
        kwargs :
            All other arguments passed to :func:`defdap.plotting.GrainPlot.addMap`

        Returns
        -------
        defdap.plotting.GrainPlot

        """
        if plot is None:
            plot = cls(callingGrain, fig=fig, ax=ax, axParams=axParams,
                       makeInteractive=makeInteractive, **figParams)
        plot.addMap(mapData, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

        if plotColourBar:
            plot.addColourBar(clabel)

        if plotScaleBar:
            plot.addScaleBar(scale=scale)

        if plotSlipTraces:
            plot.addSlipTraces()

        if plotSlipBands:
            plot.addSlipBands(grainMapData=mapData)

        return plot


class PolePlot(Plot):
    """ Class for creating an inverse pole figure plot.

    """
    defaultProjection = "stereographic"

    def __init__(self, plotType, crystalSym, projection=None,
                 fig=None, ax=None, axParams={}, makeInteractive=False,
                 **kwargs):
        super(PolePlot, self).__init__(
            ax, axParams=axParams, fig=fig, makeInteractive=makeInteractive,
            **kwargs)

        self.plotType = plotType
        self.crystalSym = crystalSym
        self.projection = self._validateProjection(projection)

        self.imgLayers = []

        self.addAxis()

    def addAxis(self):
        """Draw axes on the IPF based on crystal symmetry.

        Raises
        -------
        NotImplementedError
            If a crystal type other than 'cubic' or 'hexagonal' are selected.

        """
        if self.plotType == "IPF" and self.crystalSym == "cubic":
            # line between [001] and [111]
            self.addLine([0, 0, 1], [1, 1, 1], c='k', lw=2)

            # line between [001] and [101]
            self.addLine([0, 0, 1], [1, 0, 1], c='k', lw=2)

            # line between [101] and [111]
            self.addLine([1, 0, 1], [1, 1, 1], c='k', lw=2)

            # label poles
            self.labelPoint([0, 0, 1], '001',
                            padY=-0.005, va='top', ha='center', fontsize=12)
            self.labelPoint([1, 0, 1], '101',
                            padY=-0.005, va='top', ha='center', fontsize=12)
            self.labelPoint([1, 1, 1], '111',
                            padY=0.005, va='bottom', ha='center', fontsize=12)

        elif self.plotType == "IPF" and self.crystalSym == "hexagonal":
            # line between [0001] and [10-10] ([001] and [210])
            # converted to cubic axes
            self.addLine([0, 0, 1], [np.sqrt(3), 1, 0], c='k', lw=2)

            # line between [0001] and [2-1-10] ([001] and [100])
            self.addLine([0, 0, 1], [1, 0, 0], c='k', lw=2)

            # line between [2-1-10] and [10-10] ([100] and [210])
            self.addLine([1, 0, 0], [np.sqrt(3), 1, 0], c='k', lw=2)

            # label poles
            self.labelPoint([0, 0, 1], '0001',
                            padY=-0.012, va='top', ha='center', fontsize=12)
            self.labelPoint([1, 0, 0], r'$2\bar{1}\bar{1}0$',
                            padY=-0.012, va='top', ha='center', fontsize=12)
            self.labelPoint([np.sqrt(3), 1, 0], r'$10\bar{1}0$',
                            padY=0.009, va='bottom', ha='center', fontsize=12)

        else:
            raise NotImplementedError("Only works for cubic and hexagonal IPFs")

        self.ax.axis('equal')
        self.ax.axis('off')

    def addLine(self, startPoint, endPoint, plotSyms=False, res=100, **kwargs):
        """Draw lines on the IPF plot.

        Parameters
        ----------
        startPoint : numpy.ndarray
            Start point in crystal coordinates (i.e. [0,0,1]).
        endPoint : numpy.ndarray
            End point in crystal coordinates, (i.e. [1,0,0]).
        plotSyms : bool, optional
            If true, plot all symmetrically equivelant points.
        res : int
            Number of points within each line to plot.
        kwargs
            All other arguments are passed to :func:`matplotlib.pyplot.plot`.

        """
        lines = [(startPoint, endPoint)]
        if plotSyms:
            for symm in quat.Quat.symEqv(self.crystalSym)[1:]:
                startPointSymm = symm.transformVector(startPoint).astype(int)
                endPointSymm = symm.transformVector(endPoint).astype(int)

                if startPointSymm[2] < 0:
                    startPointSymm *= -1
                if endPointSymm[2] < 0:
                    endPointSymm *= -1

                lines.append((startPointSymm, endPointSymm))

        linePoints = np.zeros((3, res), dtype=float)
        for line in lines:
            for i in range(3):
                if line[0][i] == line[1][i]:
                    linePoints[i] = np.full(res, line[0][i])
                else:
                    linePoints[i] = np.linspace(line[0][i], line[1][i], res)

            xp, yp = self.projection(linePoints[0], linePoints[1], linePoints[2])
            self.ax.plot(xp, yp, **kwargs)

    def labelPoint(self, point, label, padX=0, padY=0, **kwargs):
        """Place a label near a coordinate in the pole plot.

        Parameters
        ----------
        point : tuple
            (x, y) coordinate to place text.
        label : str
            Text to use in label.
        padX : int, optional
            Pad added to x coordinate.
        padY : int, optional
            Pad added to y coordinate.
        kwargs
            Other arguments will be passed to :func:`matplotlib.axes.Axes.text`.

        """
        xp, yp = self.projection(*point)
        self.ax.text(xp + padX, yp + padY, label, **kwargs)

    def addPoints(self, alphaAng, betaAng, markerColour=None, markerSize=None, **kwargs):
        """Add a point to the pole plot.

        Parameters
        ----------
        alphaAng
            Inclination angle to plot.
        betaAng
            Azimuthal angle (around z axis from x in anticlockwise as per ISO) to plot.
        markerColour : str or list(str), optional
            Colour of marker. If two specified, then the point will have two
            semicircles of different colour.
        markerSize : float
            Size of marker.
        kwargs
            Other arguments will be passed to :func:`matplotlib.axes.Axes.scatter`.

        Raises
        -------
        Exception
            If more than two colours are specified

        """
        # project onto equatorial plane
        xp, yp = self.projection(alphaAng, betaAng)

        # plot poles
        # plot markers with 'half and half' colour
        if type(markerColour) is str:
            markerColour = [markerColour]

        if markerColour is None:
            points = self.ax.scatter(xp, yp, **kwargs)
            self.imgLayers.append(points)
        elif len(markerColour) == 2:
            pos = (xp, yp)
            r1 = 0.5
            r2 = r1 + 0.5
            markerSize = np.sqrt(markerSize)

            x = [0] + np.cos(np.linspace(0, 2 * np.pi * r1, 10)).tolist()
            y = [0] + np.sin(np.linspace(0, 2 * np.pi * r1, 10)).tolist()
            xy1 = list(zip(x, y))

            x = [0] + np.cos(np.linspace(2 * np.pi * r1, 2 * np.pi * r2, 10)).tolist()
            y = [0] + np.sin(np.linspace(2 * np.pi * r1, 2 * np.pi * r2, 10)).tolist()
            xy2 = list(zip(x, y))

            points = self.ax.scatter(
                pos[0], pos[1], marker=(xy1, 0),
                s=markerSize, c=markerColour[0], **kwargs
            )
            self.imgLayers.append(points)
            points = self.ax.scatter(
                pos[0], pos[1], marker=(xy2, 0),
                s=markerSize, c=markerColour[1], **kwargs
            )
            self.imgLayers.append(points)
        else:
            raise Exception("specify one colour for solid markers or list two for 'half and half'")

    def addColourBar(self, label, layer=0, **kwargs):
        """Add colourbar to pole plot.

        Parameters
        ----------
        label : str
            Label to place next to colour bar.
        layer : int
            Layer number to add the colour bar to.
        kwargs
            Other argument will be passed to :func:`matplotlib.pyplot.colorbar`.

        """
        img = self.imgLayers[layer]
        self.colourBar = plt.colorbar(img, ax=self.ax, label=label, **kwargs)

    @staticmethod
    def _validateProjection(projectionIn, validateDefault=False):
        if validateDefault:
            defaultProjection = None
        else:
            defaultProjection = PolePlot._validateProjection(
                PolePlot.defaultProjection, validateDefault=True
            )

        if projectionIn is None:
            projection = defaultProjection

        elif type(projectionIn) is str:
            projectionName = projectionIn.replace(" ", "").lower()
            if projectionName in ["lambert", "equalarea"]:
                projection = PolePlot.lambertProject
            elif projectionName in ["stereographic", "stereo", "equalangle"]:
                projection = PolePlot.stereoProject
            else:
                print("Unknown projection name, using default")
                projection = defaultProjection

        elif callable(projectionIn):
            projection = projectionIn

        else:
            print("Unknown projection, using default")
            projection = defaultProjection

        if projection is None:
            raise Exception("Problem with default projection.")

        return projection

    @staticmethod
    def stereoProject(*args):
        """Stereographic projection of pole direction or pair of polar angles.

        Parameters
        ----------
        args : numpy.ndarray, len 2 or 3
            2 arguments for polar angles or 3 arguments for pole directions.

        Returns
        -------
        float, float
            x coordinate, y coordinate

        Raises
        -------
        Exception
            If input array has incorrect length

        """
        if len(args) == 3:
            alpha, beta = quat.Quat.polarAngles(args[0], args[1], args[2])
        elif len(args) == 2:
            alpha, beta = args
        else:
            raise Exception("3 arguments for pole directions and 2 for polar angles.")

        alphaComp = np.tan(alpha / 2)
        xp = alphaComp * np.cos(beta)
        yp = alphaComp * np.sin(beta)

        return xp, yp

    @staticmethod
    def lambertProject(*args):
        """Lambert Projection of pole direction or pair of polar angles.

        Parameters
        ----------
        args : numpy.ndarray, len 2 or 3
            2 arguments for polar angles or 3 arguments for pole directions.

        Returns
        -------
        float, float
            x coordinate, y coordinate

        Raises
        -------
        Exception
            If input array has incorrect length

        """
        if len(args) == 3:
            alpha, beta = quat.Quat.polarAngles(args[0], args[1], args[2])
        elif len(args) == 2:
            alpha, beta = args
        else:
            raise Exception("3 arguments for pole directions and 2 for polar angles.")

        alphaComp = np.sqrt(2 * (1 - np.cos(alpha)))
        xp = alphaComp * np.cos(beta)
        yp = alphaComp * np.sin(beta)

        return xp, yp


class HistPlot(Plot):
    """ Class for creating a histogram.

    """
    def __init__(self, plotType="linear", density=True, fig=None,
                 ax=None, axParams={}, makeInteractive=False, **kwargs):
        """Initialise a histogram plot

        Parameters
        ----------
        plotType : str, {'log', 'None'}, optional
            If 'log' is specified, logarithmic y scale will be used.
        density :
            If true, histogram will be normalised such that the integral sums to 1.
        fig : matplotlib.figure.Figure
            Matplotlib figure to plot on.
        ax : matplotlib.axes.Axes
            Matplotlib axis to plot on.
        axParams :
            Passed to defdap.plotting.Plot as axParams.
        makeInteractive : bool
            If true, make the plot interactive.
        kwargs
            Other arguments will be passed to :class:`defdap.plotting.Plot`

        """
        super(HistPlot, self).__init__(
            ax, axParams=axParams, fig=fig, makeInteractive=makeInteractive,
            **kwargs
        )

        plotType = plotType.lower()
        if plotType in ["linear", "log"]:
            self.plotType = plotType
        else:
            raise ValueError("plotType must be linear or log.")

        self.density = bool(density)

        # set y-axis label
        yLabel = "Normalised frequency" if self.density else "Frequency"
        if self.plotType == "log":
            yLabel = "ln({})".format(yLabel)
        self.ax.set_ylabel(yLabel)

    def addHist(self, histData, bins=10, range=None, line='o',
                label=None, **kwargs):
        """Add a histogram to the current plot

        Parameters
        ----------
        histData : numpy.ndarray
            Data to be used in the histogram.
        bins : int
            Number of bins to use for histogram.
        range : tuple or None, optional
            The lower and upper range of the bins
        line : str, optional
            Marker or line type to be used.
        label : str, optional
            Label to use for data (will be used for legend).
        kwargs
            Other arguments will be passed to :func:`numpy.histogram`

        """

        hist = np.histogram(histData.flatten(), bins=bins, range=range,
                            density=self.density)

        yVals = hist[0]
        if self.plotType == "log":
            yVals = np.log(yVals)
        xVals = 0.5 * (hist[1][1:] + hist[1][:-1])

        self.ax.plot(xVals, yVals, line, label=label, **kwargs)

    def addLegend(self, **kwargs):
        """Add legend to histogram.

        Parameters
        ----------
        kwargs
            All arguments passed to :func:`matplotlib.axes.Axes.legend`.

        """
        self.ax.legend(**kwargs)

    @classmethod
    def create(
        cls, histData, fig=None, figParams={}, ax=None, axParams={},
        plot=None, makeInteractive=False,
        plotType="linear", density=True, bins=10, range=None,
        line='o', label=None, **kwargs
    ):
        """Create a histogram plot.

        Parameters
        ----------
        histData : numpy.ndarray
            Data to be used in the histogram.
        fig : matplotlib.figure.Figure
            Matplotlib figure to plot on.
        figParams :
            Passed to defdap.plotting.Plot.
        ax : matplotlib.axes.Axes
            Matplotlib axis to plot on.
        axParams :
            Passed to defdap.plotting.Plot as axParams.
        plot : defdap.plotting.HistPlot
            Plot where histgram will be places. If none, placed on current active plot.
        makeInteractive : bool, optional
            If true, make plot interactive.
        plotType : str, {'log', 'None'}, optional
            If 'log' is specified, logarithmic y scale will be used.
        density :
            If true, histogram will be normalised such that the integral sums to 1.
        bins : int
            Number of bins to use for histogram.
        range : tuple or None, optional
            The lower and upper range of the bins
        line : str, optional
            Marker or line type to be used.
        label : str, optional
            Label to use for data (will be used for legend).
        kwargs
            Other arguments will be passed to :func:`defdap.plotting.HistPlot.addHist`

        Returns
        -------
        defdap.plotting.HistPlot

        """
        if plot is None:
            plot = cls(plotType=plotType, density=density, fig=fig, ax=ax,
                       axParams=axParams, makeInteractive=makeInteractive,
                       **figParams)
        plot.addHist(histData, bins=bins, range=range, line=line,
                     label=label, **kwargs)

        return plot


class CrystalPlot(Plot):
    """ Class for creating a 3D plot for plotting unit cells.

    """
    def __init__(self, fig=None, ax=None, axParams={},
                 makeInteractive=False, **kwargs):
        """Initialise a 3D plot.

        Parameters
        ----------
        fig : matplotlib.pyplot.Figure
            Figure to plot to.
        ax : matplotlib.pyplot.Axis
            Axis to plot to.
        axParams
            Passed to defdap.plotting.Plot as axParams.
        makeInteractive : bool, optional
            If true, make plot interactive.
        kwargs
            All other arguments are passed to :class:`defdap.plotting.Plot`.

        """
        # Set default plot parameters then update with input
        figParams = {
            'figsize': (6, 6)
        }
        figParams.update(kwargs)
        axParamsDefault = {
            'projection': '3d',
            'proj_type': 'ortho'
        }
        axParamsDefault.update(axParams)
        axParams = axParamsDefault

        super(CrystalPlot, self).__init__(
            ax, axParams=axParams, fig=fig, makeInteractive=makeInteractive,
            **figParams
        )

        # Set plotting parameters
        self.ax.set_xlim3d(-0.15, 0.15)
        self.ax.set_ylim3d(-0.15, 0.15)
        self.ax.set_zlim3d(-0.15, 0.15)
        self.ax.view_init(azim=270, elev=90)
        self.ax._axis3don = False

    def addVerts(self, verts, **kwargs):
        """Plots planes, defined by the vertices provided.

        Parameters
        ----------
        verts : list
            List of vertices.
        kwargs
            All other arguments are passed to :class:`matplotlib.collections.PolyCollection`.

        """
        # Set default plot parameters then update with any input
        plotParams = {
            'alpha' : 0.6,
            'facecolor' : '0.8',
            'linewidths' : 3,
            'edgecolor' : 'k'
        }
        plotParams.update(kwargs)

        # Add list of planes defined by given vertices to the 3D plot
        pc = Poly3DCollection(verts, **plotParams)
        self.ax.add_collection3d(pc)
