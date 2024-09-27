import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

import inspect

from skimage import transform as tf
from skimage import morphology as mph

from scipy.stats import mode#
from scipy.ndimage import binary_dilation

import peakutils

from defdap.file_readers import DICDataLoader
from defdap import base
from defdap.quat import Quat

from defdap import defaults
from defdap.plotting import MapPlot, GrainPlot
from defdap.inspector import GrainInspector
from defdap.utils import reportProgress

from defdap import quat
from defdap import ebsd
from defdap import hrdic
from defdap import plotting

class Map(base.Map):
    def __init__(self, path, fname, file_extension):
        """Initialise class and import DIC data from file.

        Parameters
        ----------
        path : str
            Path to file.
        fname : str
            Name of file including extension.
        dataType : str
            Type of data file.

        """
        # Call base class constructor
        super(Map, self).__init__()

        # Initialise variables
        self.format = None      # Software name
        self.version = None     # Software version
        self.extension =  file_extension  #file extension
        self.xc = None          # x coordinates
        self.yc = None          # y coordinates

        self.ebsdMap = None                 # EBSD map linked to DIC map
        self.ebsdTransform = None           # Transform from EBSD to Optical coordinates
        self.ebsdTransformInv = None        # Transform from Optical to EBSD coordinates
        self.ebsdGrainIds = None
        self.plotHomog = self.plotMap       # Use max shear map for defining homologous points
        self.highlightAlpha = 0.6
        #self.bseScale = None                # size of a pixel in the correlated images
        #self.patScale = None                # size of pixel in loaded
        self.opticalScale = None
        # pattern relative to pixel size of dic data i.e 1 means they
        # are the same size and 2 means the pixels in the pattern are
        # half the size of the dic data.
        self.path = path                    # file path
        self.fname = fname                  # file name
        self.patternImPath = self.path + f"{self.fname}.{self.extension}"
        #self.loadData(path, fname, dataType=dataType
        
        # crop distances (default all zeros)
        self.cropDists = np.array(((0, 0), (0, 0)), dtype=int)
        self.optical = plt.imread(self.patternImPath)
        self.xdim = np.shape(self.optical)[0]       # size of map along x (from header)
        self.ydim = np.shape(self.optical)[1]       # size of map along y (from header)
        #self.plotDefault = lambda *args, **kwargs: self.plotMaxShear(plotGBs=True, *args, **kwargs)# come back to --------------------------------------
        
        # *dim are full size of data. *Dim are size after cropping
        self.xDim = self.xdim
        self.yDim = self.ydim
#########################################################################################################################################################
    def setScale(self, micrometrePerPixel):
        """Sets the scale of the map.

        Parameters
        ----------
        micrometrePerPixel : float
            Length of pixel in original BSE image in micrometres.

        """
        self.opticalScale = micrometrePerPixel

    @property
    def scale(self):
        """Returns the number of micrometers per pixel in the DIC map.

        """
        if self.opticalScale is None:
            raise ValueError("Map scale not set. Set with setScale()")

        return self.opticalScale 

    def setCrop(self, xMin=None, xMax=None, yMin=None, yMax=None, updateHomogPoints=False):
        """Set a crop for the DIC map.

        Parameters
        ----------
        xMin : int
            Distance to crop from left in pixels.
        xMax : int
            Distance to crop from right in pixels.
        yMin : int
            Distance to crop from top in pixels.
        yMax : int
            Distance to crop from bottom in pixels.
        updateHomogPoints : bool, optional
            If true, change homologous points to reflect crop.

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
        """ Crop given data using crop parameters stored in map
        i.e. cropped_data = DicMap.crop(DicMap.data_to_crop).

        Parameters
        ----------
        mapData : numpy.ndarray
            Bap data to crop.
        binned : bool
            True if mapData is binned i.e. binned BSE pattern.
        """
        if binned:
            multiplier = 1
        else:
            multiplier = self.opticalScale

        minY = int(self.cropDists[1, 0] * multiplier)
        maxY = int((self.ydim - self.cropDists[1, 1]) * multiplier)

        minX = int(self.cropDists[0, 0] * multiplier)
        maxX = int((self.xdim - self.cropDists[0, 1]) * multiplier)

        return mapData[minY:maxY, minX:maxX]
    
    def retrieveName(self):
        """Gets the first name assigned to the a map, as a string

        """
        for fi in reversed(inspect.stack()):
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is self]
            if len(names) > 0:
                return names[0]

    def plotMap(self, component=None, **kwargs):
        """Plot a map from the DIC/EBSD data or an optical image.
    
        Parameters
        ----------
        component : str, optional
            Map component to plot, e.g., 'e11', 'f11', 'eMaxShear'. If None, the optical image is plotted.
        kwargs : dict
            Additional arguments passed to :func:`defdap.plotting.MapPlot.create`.
    
        Returns
        -------
        defdap.plotting.MapPlot or None
            Returns the plot if a map component is plotted, otherwise None for optical images.
        """
    
        # If a map component is provided, plot the DIC/EBSD map component
        if component is not None:
            plotParams = {
                'plotColourBar': True,
                'clabel': component
            }
            plotParams.update(kwargs)
    
            # Assuming self.component[component] holds the map data (e.g., strain)
            mapData = self.crop(self.component[component])
    
            # Use the create method for map plotting
            plot = MapPlot.create(self, mapData, **plotParams)
    
            return plot
        
        else:
            # Default figure/axis setup for optical image
            mapData = self.crop(self.optical)
            plotParams = {'plotColourBar': False, 'clabel': None}
            plotParams.update(kwargs)
            plot = MapPlot.create(self, mapData, **plotParams)
            return plot
            
    def plotOptical(self,  **kwargs):
        """
        Plot optical image of Map. For use with setting with homogPoints. 
                Parameters
        ----------
        kwargs
            All arguments are passed to :func:`defdap.plotting.MapPlot.create`.

        Returns
        -------
        defdap.plotting.MapPlot
        """
        # Set default plot parameters then update with any input
        plotParams = {
            'cmap': 'gray'
        }
        #scale = 1
        #try:
        #    plotParams['scale'] = self.scale / self.opticalScale * 1e-6
        #except(ValueError):
        #    pass
        #plotParams.update(kwargs)

        # Check image path is set
        if self.patternImPath is None:
            raise Exception("First set path to pattern image.")

        polarised = imread(self.patternImPath)
        #polarised = self.crop(polarised, binned=False)

        plot = MapPlot.create(self, polarised,makeInteractive=True, **plotParams)

        return plot        
    
    def setHomogPoint(self, points=None, display=None, **kwargs):#1
        """Set homologous points. Uses interactive GUI if points is None.
    
        Parameters
        ----------
        points : list, optional
            Homologous points to set.
        display : string, optional
            Use max shear map if set to 'maxshear', pattern if set to 'pattern', 
            or optical image if set to 'optical'.
        """
        
        if points is not None:
            self.homogPoints = points
            
        if points is None:
            # Default display setting
            if display is None:
                display = "optical"
    
            # Normalize the display string for comparison
            display = display.lower().replace(" ", "")
    
            # Handle different display modes
            if display == "optical":
                # Set to optical image display (new case for optical data)
                self.plotHomog = self.plotOptical  # This should be a method for handling optical data
                self.opticalScale = 1
                binSize = self.opticalScale  # Adjust to whatever scaling factor is appropriate for optical data
 
            # Call setHomogPoint from the base class, passing binSize and points
            super(type(self), self).setHomogPoint(binSize=binSize, points=points, **kwargs)


    def linkEbsdMap(self, ebsdMap, transformType="affine", **kwargs):
        """Calculates the transformation required to align EBSD dataset to DIC.

        Parameters
        ----------
        ebsdMap : defdap.ebsd.Map
            EBSD map object to link.
        transformType : str, optional
            affine, piecewiseAffine or polynomial.
        kwargs
            All arguments are passed to `estimate` method of the transform.

        """
        self.ebsdMap = ebsdMap
        calc_inv = False
        if transformType.lower() == "piecewiseaffine":
            self.ebsdTransform = tf.PiecewiseAffineTransform()
        elif transformType.lower() == "projective":
            self.ebsdTransform = tf.ProjectiveTransform()
        elif transformType.lower() == "polynomial":
            calc_inv = True
            self.ebsdTransform = tf.PolynomialTransform()
            self.ebsdTransformInv = tf.PolynomialTransform()
        else:
            # default to using affine
            self.ebsdTransform = tf.AffineTransform()

        # calculate transform from EBSD to DIC frame
        self.ebsdTransform.estimate(
            np.array(self.homogPoints),
            np.array(self.ebsdMap.homogPoints),
            **kwargs
        )
        # Calculate inverse if required
        if calc_inv:
            self.ebsdTransformInv.estimate(
                np.array(self.ebsdMap.homogPoints),
                np.array(self.homogPoints),
                **kwargs
            )
        else:
            self.ebsdTransformInv = self.ebsdTransform.inverse

    def warp_lines_to_optical_frame(self, lines):
        """Warp a set of lines to the DIC reference frame.

        Parameters
        ----------
        lines : list of tuples
            Lines to warp. Each line is represented as a tuple of start
            and end coordinates (x, y).

        Returns
        -------
        list of tuples
            List of warped lines with same representation as input.

        """
        # Flatten to coord list
        lines = np.array(lines).reshape(-1, 2)
        # Transform & reshape back
        lines = self.ebsdTransformInv(lines).reshape(-1, 2, 2)
        # Round to nearest
        lines = np.round(lines - 0.5) + 0.5
        lines = [(tuple(l[0]), tuple(l[1])) for l in lines]

        return lines
    
    @property
    def boundaries(self):
        """Returns EBSD map grain boundaries warped to DIC frame.

        """
        # Check a EBSD map is linked
        self.checkEbsdLinked()

        # image is returned cropped if a piecewise transform is being used
        boundaries = self.ebsdMap.boundaries
        boundaries = self.warpToOpticalFrame(  
            -boundaries.astype(float), cropImage=False
        )
        boundaries = boundaries > 0.1
        boundaries = mph.skeletonize(boundaries)
        boundaries = mph.remove_small_objects(
            boundaries, min_size=10, connectivity=2
        )

        # crop image if it is a simple affine transform
        if type(self.ebsdTransform) is tf.AffineTransform:
            # need to apply the translation of ebsd transform and
            # remove 5% border
            crop = np.copy(self.ebsdTransform.params[0:2, 2])
            crop += 0.05 * np.array(self.ebsdMap.boundaries.shape)
            # the crop is defined in EBSD coords so need to transform it
            transformMatrix = np.copy(self.ebsdTransform.params[0:2, 0:2])
            crop = np.matmul(np.linalg.inv(transformMatrix), crop)
            crop = crop.round().astype(int)

            boundaries = boundaries[crop[1]:crop[1] + self.yDim,
                                    crop[0]:crop[0] + self.xDim]

        return -boundaries.astype(int)

    
    @property
    def boundaryLines(self):
        return self.warp_lines_to_optical_frame(self.ebsdMap.boundaryLines)

    @property
    def phaseBoundaryLines(self):
        return self.warp_lines_to_optical_frame(self.ebsdMap.phaseBoundaryLines)

    def checkEbsdLinked(self):
        """Check if an EBSD map has been linked.

        Returns
        ----------
        bool
            Returns True if EBSD map linked.

        Raises
        ----------
        Exception
            If EBSD map not linked.

        """
        if self.ebsdMap is None:
            raise Exception("No EBSD map linked.")
        return True

    def warpToOpticalFrame(self, mapData, cropImage=True, order=1, preserve_range=False):
        """Warps a map to the DIC frame.

        Parameters
        ----------
        mapData : numpy.ndarray
            Data to warp.
        cropImage : bool, optional
            Crop to size of DIC map if true.
        order : int, optional
            Order of interpolation (0: Nearest-neighbor, 1: Bi-linear...).
        preserve_range: bool, optional
            Keep the original range of values.

        Returns
        ----------
        numpy.ndarray
            Map (i.e. EBSD map data) warped to the DIC frame.

        """
        # Check a EBSD map is linked
        self.checkEbsdLinked()

        if (cropImage or type(self.ebsdTransform) is not tf.AffineTransform):
            # crop to size of DIC map
            outputShape = (self.yDim, self.xDim)
            # warp the map
            warpedMap = tf.warp(
                mapData, self.ebsdTransform,
                output_shape=outputShape,
                order=order, preserve_range=preserve_range
            )
        else:
            # copy ebsd transform and change translation to give an extra
            # 5% border to show the entire image after rotation/shearing
            tempEbsdTransform = tf.AffineTransform(matrix=np.copy(self.ebsdTransform.params))
            tempEbsdTransform.params[0:2, 2] = -0.05 * np.array(mapData.shape)

            # output the entire warped image with 5% border (add some
            # extra to fix a bug)
            outputShape = np.array(mapData.shape) * 1.4 / tempEbsdTransform.scale

            # warp the map
            warpedMap = tf.warp(
                mapData, tempEbsdTransform,
                output_shape=outputShape.astype(int),
                order=order, preserve_range=preserve_range
            )

        return warpedMap


    @reportProgress("finding grains")
    def findGrains(self, algorithm=None, minGrainSize=10):
        """Finds grains in the DIC map.

        Parameters
        ----------
        algorithm : str {'warp', 'floodfill'}
            Use floodfill or warp algorithm.
        minGrainSize : int
            Minimum grain area in pixels for floodfill algorithm.
        """
        # Check a EBSD map is linked
        self.checkEbsdLinked()

        if algorithm is None:
            algorithm = defaults['hrdic_grain_finding_method']

        if algorithm == 'warp':
            # Warp EBSD grain map to DIC frame
            self.grains = self.warpToOpticalFrame(self.ebsdMap.grains, cropImage=True,
                                              order=0, preserve_range=True)

            # Find all unique values (these are the EBSD grain IDs in the DIC area, sorted)
            self.ebsdGrainIds = np.array([int(i) for i in np.unique(self.grains) if i>0])

            # Make a new list of sequential IDs of same length as number of grains
            opticalGrainsIds = np.arange(1, len(self.ebsdGrainIds)+1)

            # Map the EBSD IDs to the DIC IDs (keep the same mapping for values <= 0)
            negVals = np.array([i for i in np.unique(self.grains) if i<=0])
            old = np.concatenate((negVals, self.ebsdGrainIds))
            new = np.concatenate((negVals, opticalGrainsIds))
            index = np.digitize(self.grains.ravel(), old, right=True)
            self.grains = new[index].reshape(self.grains.shape)

            self.grainList = []
            for i, (opticalGrainId, ebsdGrainId) in enumerate(zip(opticalGrainsIds, self.ebsdGrainIds)):
                yield i / len(opticalGrainsIds)          # Report progress

                # Make grain object
                currentGrain = Grain(grainID=opticalGrainId, opticalMap=self)

                # Find (x,y) coordinates and corresponding max shears of grain
                coords = np.argwhere(self.grains == opticalGrainId)       # (y,x)
                currentGrain.coordList = np.flip(coords, axis=1)      # (x,y)
                currentGrain.optical = self.optical[coords[:,0]+ self.cropDists[1, 0], 
                                                           coords[:,1]+ self.cropDists[0, 0]]

                # Assign EBSD grain ID to DIC grain and increment grain list
                currentGrain.ebsdGrainId = ebsdGrainId - 1
                currentGrain.ebsdGrain = self.ebsdMap.grainList[ebsdGrainId - 1]
                currentGrain.ebsdMap = self.ebsdMap
                self.grainList.append(currentGrain)

        elif algorithm == 'floodfill':
            # Initialise the grain map
            self.grains = np.copy(self.boundaries)

            self.grainList = []

            # List of points where no grain has been set yet
            points_left = self.grains == 0
            total_points = points_left.sum()
            found_point = 0
            next_point = points_left.tobytes().find(b'\x01')

            # Start counter for grains
            grainIndex = 1

            # Loop until all points (except boundaries) have been assigned
            # to a grain or ignored
            i = 0
            while found_point >= 0:
                # Flood fill first unknown point and return grain object
                idx = np.unravel_index(next_point, self.grains.shape)
                currentGrain = self.floodFill(idx[1], idx[0], grainIndex,
                                              points_left)

                if len(currentGrain) < minGrainSize:
                    # if grain size less than minimum, ignore grain and set
                    # values in grain map to -2
                    for coord in currentGrain.coordList:
                        self.grains[coord[1], coord[0]] = -2
                else:
                    # add grain to list and increment grain index
                    self.grainList.append(currentGrain)
                    grainIndex += 1

                # find next search point
                points_left_sub = points_left.reshape(-1)[next_point + 1:]
                found_point = points_left_sub.tobytes().find(b'\x01')
                next_point += found_point + 1

                # report progress
                i += 1
                if i == defaults['find_grain_report_freq']:
                    yield 1. - points_left_sub.sum() / total_points
                    i = 0

            # Now link grains to those in ebsd Map
            # Warp DIC grain map to EBSD frame
            dicGrains = self.grains
            warpedDicGrains = tf.warp(
                np.ascontiguousarray(dicGrains.astype(float)),
                self.ebsdTransformInv,
                output_shape=(self.ebsdMap.yDim, self.ebsdMap.xDim),
                order=0
            ).astype(int)

            # Initialise list to store ID of corresponding grain in EBSD map.
            # Also stored in grain objects
            self.ebsdGrainIds = []

            for i in range(len(self)):
                # Find grain by masking the native ebsd grain image with
                # selected grain from the warped dic grain image. The modal
                # value is the EBSD grain label.
                modeId, _ = mode(self.ebsdMap.grains[warpedDicGrains == i + 1])
                ebsd_grain_idx = modeId[0] - 1
                self.ebsdGrainIds.append(ebsd_grain_idx)
                self[i].ebsdGrainId = ebsd_grain_idx
                self[i].ebsdGrain = self.ebsdMap[ebsd_grain_idx]
                self[i].ebsdMap = self.ebsdMap

        else:
            raise ValueError(f"Unknown grain finding algorithm '{algorithm}'.")


    def floodFill(self, x, y, grainIndex, points_left):
        """Flood fill algorithm that uses the combined x and y boundary array 
        to fill a connected area around the seed point. The points are inserted
        into a grain object and the grain map array is updated.

        Parameters
        ----------
        x : int
            Seed point x for flood fill
        y : int
            Seed point y for flood fill
        grainIndex : int
            Value to fill in grain map
        points_left : numpy.ndarray
            Boolean map of the points that have not been assigned a grain yet

        Returns
        -------
        currentGrain : defdap.hrdic.Grain
            New grain object with points added

        """
        # create new grain
        currentGrain = Grain(grainIndex - 1, self)

        # add first point to the grain
        currentGrain.addPoint((x, y), self.optical[y + self.cropDists[1, 0],
                                                     x + self.cropDists[0, 0]])
        self.grains[y, x] = grainIndex
        points_left[y, x] = False
        edge = [(x, y)]

        while edge:
            x, y = edge.pop(0)

            moves = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            # get rid of any that go out of the map area
            if x <= 0:
                moves.pop(1)
            elif x >= self.xDim - 1:
                moves.pop(0)
            if y <= 0:
                moves.pop(-1)
            elif y >= self.yDim - 1:
                moves.pop(-2)

            for (s, t) in moves:
                addPoint = False

                if self.grains[t, s] == 0:
                    addPoint = True
                    edge.append((s, t))

                elif self.grains[t, s] == -1 and (s > x or t > y):
                    addPoint = True

                if addPoint:
                    currentGrain.addPoint(
                        (s, t),
                        self.optical[t + self.cropDists[1, 0],
                                       s + self.cropDists[0, 0]]
                    )
                    self.grains[t, s] = grainIndex
                    points_left[t, s] = False

        return currentGrain

    def runGrainInspector(self, vmax=0.1, corrAngle=0):
        """Run the grain inspector interactive tool.

        Parameters
        ----------
        vmax : float
            Maximum value of the colour map.
        corrAngle: float
            Correction angle in degrees to subtract from measured angles to account
            for small rotation between DIC and EBSD frames. Approximately the rotation
            component of affine transform.

        """
        GrainInspector(currMap=self, vmax=vmax, corrAngle=corrAngle)




#---------------------------------------------------------------------------------------------------------------------------------------------------------


class Grain(base.Grain):
    """
    Class to encapsulate DIC grain data and useful analysis and plotting
    methods.

    Attributes
    ----------
    opticalMap : defdap.hrdic.Map
        DIC map this grain is a member of
    ownerMap : defdap.hrdic.Map
        DIC map this grain is a member of
    optical_data : list
        List of maximum shear values for grain.
    ebsdGrain : defdap.ebsd.Grain
        EBSD grain ID that this DIC grain corresponds to.
    ebsdMap : defdap.ebsd.Map
        EBSD map that this DIC grain belongs to.
    pointsList : numpy.ndarray
        Start and end points for lines drawn using defdap.inspector.GrainInspector.
    groupsList :
        Groups, angles and slip systems detected for
        lines drawn using defdap.inspector.GrainInspector.

    """
    def __init__(self, grainID, opticalMap):
        # Call base class constructor
        super(Grain, self).__init__(grainID, opticalMap)

        self.optical_map = self.ownerMap     # DIC map this grain is a member of
        self.optical_data = []
        self.ebsdGrain = None
        self.ebsdMap = None

        self.pointsList = []            # Lines drawn for STA
        self.groupsList = []            # Unique angles drawn for STA
        
        
    def printData(self):
        """Print the data stored in the Grain object."""
        print(f"Grain ID: {self.grainID}")
        print(f"Coordinate List: {self.coordList}")
        print(f"Optical Data: {self.optical_data}")
        print(f"Points List: {self.pointsList}")
        print(f"Groups List: {self.groupsList}")

        if self.ebsdGrain is not None:
            print(f"EBSD Grain: {self.ebsdGrain}")
        else:
            print("EBSD Grain: None")

        if self.ebsdMap is not None:
            print(f"EBSD Map: {self.ebsdMap}")
        else:
            print("EBSD Map: None")

    @property
    def plotDefault(self):
        return lambda *args, **kwargs: self.plotMaxShear(
            plotColourBar=True, plotScaleBar=True, plotSlipTraces=True,
            plotSlipBands=True, *args, **kwargs
        )

    # coord is a tuple (x, y)
    def addPoint(self, coord, optical_data):
        if len(self.coordList) != len(self.optical_data):
            raise ValueError("coordList and optical_data lengths are inconsistent.")
            
        if optical_data is None or len(optical_data) == 0:
            raise ValueError("optical_data is empty or None.")
            
        if not isinstance(coord, (tuple, list)) or len(coord) != 2:
            raise ValueError("coord should be a tuple or list with two elements (x, y).")
        
        print(f"Adding coord: {coord}, optical_data: {optical_data}")        
         
        self.coordList.append(coord)
        self.optical_data.append(optical_data)
        
        print(f"coordList length: {len(self.coordList)}, optical_data length: {len(self.optical_data)}")
        
    def plotDarkfield(self, **kwargs):
        """Plot a maximum shear map for a grain.

        Parameters
        ----------
        kwargs
            All arguments are passed to :func:`defdap.base.plotGrainData`.

        Returns
        -------
        defdap.plotting.GrainPlot

        """
        # Set default plot parameters then update with any input
        plotParams = {
            'plotColourBar': False,
            'clabel': "Effective shear strain"
        }
        plotParams.update(kwargs)

        plot = self.plotGrainData(grainData=self.optical_map, **plotParams)

        return plot
'''
    @property
    def refOri(self):
        """Returns average grain orientation.

        Returns
        -------
        defdap.quat.Quat

        """
        return self.ebsdGrain.refOri

    @property
    def slipTraces(self):
        """Returns list of slip trace angles based on EBSD grain orientation.

        Returns
        -------
        list

        """
        return self.ebsdGrain.slipTraces

    def calcSlipTraces(self, slipSystems=None):
        """Calculates list of slip trace angles based on EBSD grain orientation.

        Parameters
        -------
        slipSystems : defdap.crystal.SlipSystem, optional

        """
        self.ebsdGrain.calcSlipTraces(slipSystems=slipSystems)

    def calcSlipBands(self, grainMapData, thres=None, min_dist=None):
        """Use Radon transform to detect slip band angles.

        Parameters
        ----------
        grainMapData : numpy.ndarray
            Data to find bands in.
        thres : float, optional
            Normalised threshold for peaks.
        min_dist : int, optional
            Minimum angle between bands.

        Returns
        ----------
        list(float)
            Detected slip band angles

        """
        if thres is None:
            thres = 0.3
        if min_dist is None:
            min_dist = 30
        grainMapData = np.nan_to_num(grainMapData)

        if grainMapData.min() < 0:
            print("Negative values in data, taking absolute value.")
            # grainMapData = grainMapData**2
            grainMapData = np.abs(grainMapData)
        suppGMD = np.zeros(grainMapData.shape) #array to hold shape / support of grain
        suppGMD[grainMapData!=0]=1
        sin_map = tf.radon(grainMapData, circle=False)
        #profile = np.max(sin_map, axis=0) # old method
        supp_map = tf.radon(suppGMD, circle=False)
        supp_1 = np.zeros(supp_map.shape)
        supp_1[supp_map>0]=1
        mindiam = np.min(np.sum(supp_1, axis=0), axis=0) # minimum diameter of grain
        crop_map = np.zeros(sin_map.shape)
        # only consider radon rays that cut grain with mindiam*2/3 or more, and scale by length of the cut
        crop_map[supp_map>mindiam*2/3] = sin_map[supp_map>mindiam*2/3]/supp_map[supp_map>mindiam*2/3] 
        supp_crop = np.zeros(crop_map.shape)
        supp_crop[crop_map>0] = 1
        profile = np.sum(crop_map**4, axis=0) / np.sum(supp_crop, axis=0) # raise to power to accentuate local peaks

        x = np.arange(180)

        # indexes = peakutils.indexes(profile, thres=thres, min_dist=min_dist, thres_abs=False)
        indexes = peakutils.indexes(profile, thres=thres, min_dist=min_dist)
        peaks = x[indexes]
        # peaks = peakutils.interpolate(x, profile, ind=indexes)
        print("Number of bands detected: {:}".format(len(peaks)))

        slipBandAngles = peaks
        slipBandAngles = slipBandAngles * np.pi / 180
        return slipBandAngles
'''