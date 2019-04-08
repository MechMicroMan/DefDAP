import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

from skimage import morphology as mph

from defdap.quat import Quat


class Plot(object):
    def __init__(self, fig, ax, makeInteractive):

        self.interactive = makeInteractive
        if makeInteractive:
            if fig is not None and ax is not None:
                self.fig = fig
                self.ax = ax
            else:
                self.fig, self.ax = plt.subplots()
        else:
            self.fig = fig
            # TODO: flag for new figure
            self.ax = plt.gca() if ax is None else ax

class MapPlot(Plot):
    def __init__(self, callingMap, fig=None, ax=None, makeInteractive=False):
        super(MapPlot, self).__init__(fig, ax, makeInteractive)

        self.callingMap = callingMap
        self.imgLayers = []

        ax.set_xticks([])
        ax.set_yticks([])

    def addMap(self, mapData, vmin=None, vmax=None, cmap='viridis', **kwargs):

        img = self.ax.imshow(mapData, vmin=vmin, vmax=vmax,
                             interpolation='None', cmap=cmap, **kwargs)

        self.imgLayers.append(img)

        return img

    def addColourBar(self, label, layer=0, **kwargs):
        img = self.imgLayers[layer]
        plt.colorbar(img, ax=self.ax, label=label, **kwargs)

    def addScaleBar(self, scale):
        scalebar = ScaleBar(scale)
        self.ax.add_artist(scalebar)

    def addGBs(self, colour='white', dilate=False):
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

        self.imgLayers.append(img)

        return img

    def addGrainHighlights(self, grainIds, grainColours=None, alpha=None):
        if grainColours is None:
            grainColours = ['white']
        if alpha is None:
            alpha = self.highlightAlpha

        cMap = self.callingMap

        outline = np.zeros((cMap.yDim, cMap.xDim), dtype=int)
        for i, grainId in enumerate(grainIds, start=1):
            if i > len(grainColours):
                i = len(grainColours)

            # outline of highlighted grain
            grainOutline = cMap.grainList[grainId].grainOutline(bg=0, fg=i)
            x0, y0, xmax, ymax = cMap.grainList[grainId].extremeCoords

            # add to highlight image
            outline[y0:ymax + 1, x0:xmax + 1] += grainOutline

        # Custom colour map where 0 is transparent white for bg and
        # then a patch for each grain colour
        grainColours.insert(0, 'white')
        hightlightsCmap = mpl.colors.ListedColormap(grainColours)
        hightlightsCmap._init()
        alphaMap = np.full(cmap1.N + 3, alpha)
        alphaMap[0] = 0
        hightlightsCmap._lut[:, -1] = alphaMap

        img = self.ax.imshow(outline, interpolation='none',
                             cmap=hightlightsCmap)

        return img

    def addLegend(self, values, lables, layer=0, **kwargs):
        # Find colour values for given values
        img = self.imgLayers[layer]
        colors = [img.cmap(img.norm(value)) for value in values]

        # Get colour patches for each phase and make legend
        patches = [mpl.patches.Patch(
            color=colors[i], label=lables[i]
        ) for i in range(len(values))]

        self.ax.legend(handles=patches, **kwargs)


class PolePlot(Plot):
    defaultProjection = "stereographic"

    def __init__(self, plotType, crystalSym, projection=None, ax=None):
        super(PolePlot, self).__init__(ax)

        self.plotType = plotType
        self.crystalSym = crystalSym
        self.projection = self._validateProjection(projection)

        self.addAxis()

    def addAxis(self):
        if self.plotType == "IPF" and self.crystalSym == "cubic":
            # line between [001] and [111]
            self.addLine([0, 0, 1], [1, 1, 1], c='k', lw=2)

            # line between [001] and [101]
            self.addLine([0, 0, 1], [1, 0, 1], c='k', lw=2)

            # line between [101] and [111]
            self.addLine([1, 0, 1], [1, 1, 1], c='k', lw=2)

            # label poles
            self.labelPoint([0, 0, 1], '001',
                            padY=-0.005, va='top', ha='center')
            self.labelPoint([1, 0, 1], '101',
                            padY=-0.005, va='top', ha='center')
            self.labelPoint([1, 1, 1], '111',
                            padY=0.005, va='bottom', ha='center')

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
                            padY=-0.008, va='top', ha='center')
            self.labelPoint([1, 0, 0], '2-1-10',
                            padY=-0.008, va='top', ha='center')
            self.labelPoint([np.sqrt(3), 1, 0], '10-10',
                            padY=0.008, va='bottom', ha='center')

        else:
            raise NotImplementedError("Only works for cubic and hexagonal IPFs")

        self.ax.axis('equal')
        self.ax.axis('off')

    def addLine(self, startPoint, endPoint, plotSyms=False, res=100, **kwargs):
        lines = [(startPoint, endPoint)]
        if plotSyms:
            for symm in Quat.symEqv(self.crystalSym)[1:]:
                startPointSymm = symm.transformVector(startPoint).astype(int)
                endPointSymm = symm.transformVector(endPoint).astype(int)

                if startPointSymm[2] < 0: startPointSymm *= -1
                if endPointSymm[2] < 0: endPointSymm *= -1

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
        xp, yp = self.projection(*point)
        self.ax.text(xp + padX, yp + padY, label, **kwargs)

    def addPoints(self, alpha, beta, markerColour=None, **kwargs):
        # project onto equatorial plane
        xp, yp = self.projection(alpha, beta)

        # plot poles
        # plot markers with 'half and half' colour
        if type(markerColour) is str:
            markerColour = [markerColour]

        if markerColour is None:
            self.ax.scatter(xp, yp, **kwargs)
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

            self.ax.scatter(pos[0], pos[1], marker=(xy1, 0),
                            s=markerSize, c=markerColour[0], **kwargs)
            self.ax.scatter(pos[0], pos[1], marker=(xy2, 0),
                            s=markerSize, c=markerColour[1], **kwargs)
        else:
            raise Exception("specify one colour for solid markers or list two for 'half and half'")

    @staticmethod
    def _validateProjection(projectionIn, validateDefault=False):
        if validateDefault:
            defaultProjection = None
        else:
            defaultProjection = PolePlot._validateProjection(
                Quat.defaultProjection, validateDefault=True
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
        if len(args) == 3:
            alpha, beta = Quat.polarAngles(args[0], args[1], args[2])
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
        if len(args) == 3:
            alpha, beta = Quat.polarAngles(args[0], args[1], args[2])
        elif len(args) == 2:
            alpha, beta = args
        else:
            raise Exception("3 arguments for pole directions and 2 for polar angles.")

        alphaComp = np.sqrt(2 * (1 - np.cos(alpha)))
        xp = alphaComp * np.cos(beta)
        yp = alphaComp * np.sin(beta)

        return xp, yp


class HistPlot(Plot)
    def __init__(self, plotType,  ax=None):
        super(HistPlot, self).__init__(ax)

        self.plotType = plotType