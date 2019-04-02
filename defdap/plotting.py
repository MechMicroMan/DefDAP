import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

from skimage import morphology as mph

from defdap.quat import Quat


class Plot(object):
    def __init__(self, ax):
        self.ax = plt.gca() if ax is None else ax


class MapPlot(Plot):
    def __init__(self, callingMap, ax=None):
        super(MapPlot, self).__init__(ax)

        self.callingMap = callingMap

        self.imgLayers = []

    def addMap(self, mapData, vmin=None, vmax=None, cmap='viridis'):

        img = self.ax.imshow(mapData, vmin=vmin, vmax=vmax,
                             interpolation='None', cmap=cmap)

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


class PolePlot(Plot):
    def __init__(self, crystalSym, ax=None):
        super(PolePlot, self).__init__(ax)

        self.crystalSym = crystalSym

        Quat.plotPoleAxis('IPF', self.crystalSym, ax=self.ax)


class HistPlot(Plot)
    def __init__(self, crystalSym, ax=None):
        super(HistPlot, self).__init__(ax)