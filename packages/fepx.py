import glob, os, fnmatch
import io
import copy

from IPython.display import clear_output

import pyevtk.vtk

import numpy as np

from .quat import Quat


class Mesh(object):
    ### object variables ###
    # meshDir
    # dataDir

    ### MESH DATA ###
    # numElmts(int)
    # numNodes(int)
    # elType(int)
    # elmtCon[numElmnts, 10](int) - Nodes for each element - node numbers are 0 based
    # nodePos[numNodes, 3](float) - Initial position of each node
    # numGrains(int) - number of grains in mesh
    # elmtGrain[numElmts](int) - Grain ID of each element - grain ID is 1 based
    # elmtPhase[numElmts](int) - Phase ID of each element - phase ID is 1 based
    # grainOris[numGrains, 3](float) - Initial Euler angles of each grain (Kocks, degrees)

    ### SIMULATION DATA ###
    # numFrames(int) - Number of load steps output
    # numProcs(int) - Number of cores the simulation was run on
    # numSlipSys(int) - Number of slip systems

    # nodePosData[numElmts, 3, numFrames](float) - Node positions
    # angles[numElmts, 3, numFrames](float) - Euler angles of each element (Bunge, radians)
    # oris[numElmts, numFrames](Quat) - Quat of orientation of each element

    # misOri

    def __init__(self, name, meshDir="./", dataDir="./"):
        MESH_FILE_EX = "mesh"
        GRAIN_FILE_EX = "grain"
        ORI_FILE_EX = "ori"

        self.meshDir = meshDir if meshDir[-1] == "/" else "{:s}{:s}".format(meshDir, "/")
        self.dataDir = dataDir if dataDir[-1] == "/" else "{:s}{:s}".format(dataDir, "/")

        # open mesh file
        fileName = "{:s}{:s}.{:s}".format(self.meshDir, name, MESH_FILE_EX)
        meshFile = open(fileName, "r")

        # read header
        header = meshFile.readline()
        self.numElmts, self.numNodes, self.elType = (int(x) for x in header.split())

        # read element connectivity data
        self.elmtCon = np.genfromtxt(meshFile, dtype=int, max_rows=self.numElmts, usecols=list(range(1, 11)))

        # read node positions
        self.nodePos = np.genfromtxt(meshFile, dtype=float, max_rows=self.numNodes, usecols=[1, 2, 3])

        # close mesh file
        meshFile.close()

        # open grain file
        fileName = "{:s}{:s}.{:s}".format(self.meshDir, name, GRAIN_FILE_EX)
        grainFile = open(fileName, "r")

        # read header
        header = grainFile.readline()
        _, self.numGrains = (int(x) for x in header.split())

        # read grain and phase info for each element
        self.elmtGrain, self.elmtPhase = np.genfromtxt(grainFile, dtype=int, unpack=True, max_rows=self.numElmts)

        # close grain file
        grainFile.close()

        # open ori file
        fileName = "{:s}{:s}.{:s}".format(self.meshDir, name, ORI_FILE_EX)
        oriFile = open(fileName, "r")

        # read orientation of each grain (Kocks Euler angles)
        self.grainOris = np.genfromtxt(oriFile, dtype=float, skip_header=2, max_rows=self.numGrains, usecols=[0, 1, 2])

        # close ori file
        oriFile.close()

    def setSimParams(self, numProcs=-1, numFrames=-1, numSlipSys=-1):
        if numProcs > 0:
            self.numProcs = numProcs
        if numFrames > 0:
            self.numFrames = numFrames
        if numSlipSys > 0:
            self.numSlipSys = numSlipSys

    def loadFrameData(self, dataName, initialIncd, usecols=None, numProcs=-1, numFrames=-1, numSlipSys=-1):
        self.setSimParams(numProcs=numProcs, numFrames=numFrames, numSlipSys=numSlipSys)

        loadedData = []
        for i in range(self.numProcs):
            # load data per processor
            fileName = "{:s}post.{:s}.{:d}".format(self.dataDir, dataName, i)
            loadedData.append(np.loadtxt(fileName, dtype=float, comments="%", usecols=usecols, unpack=True))

            # reshape into 3d array with 3rd dim for each frame
            rows, cols = loadedData[i].shape
            # +1 if initial values are also stored
            numFrames = (self.numFrames + 1) if initialIncd else self.numFrames
            perFrame = cols / numFrames
            loadedData[i] = np.reshape(loadedData[i], (rows, perFrame, numFrames), order='F')

        # concatenate data from all processors into one array and transpose first 2 axes
        return np.transpose(np.concatenate(loadedData, axis=1), axes=(1, 0, 2))

    # load node positions for each frame of the simulation
    def loadNodePosData(self, numProcs=-1, numFrames=-1):
        self.nodePosData = self.loadFrameData("adx",
                                              True,
                                              numProcs=numProcs,
                                              numFrames=numFrames)

    def loadAngleData(self, numProcs=-1, numFrames=-1):
        self.angles = self.loadFrameData("ang",
                                         True,
                                         usecols=[1, 2, 3],
                                         numProcs=numProcs,
                                         numFrames=numFrames)

        # Covert to Bunge rep. in radians (were Kocks in degrees)
        self.angles *= (np.pi / 180)
        self.angles[:, 0, :] += np.pi / 2
        self.angles[:, 2, :] *= -1
        self.angles[:, 2, :] += np.pi / 2

        # construct quat array
        self.oris = np.empty([self.angles.shape[0], self.angles.shape[2]], dtype=Quat)

        for i in range(self.numFrames + 1):
            for j, row in enumerate(self.angles[..., i]):
                self.oris[j, i] = Quat(row[0], row[1], row[2])

        # create arrays to store misori data
        self.misOri = np.zeros(self.oris.shape, dtype=float)
        self.avMisOri = np.zeros([self.numGrains, self.oris.shape[1]], dtype=float)

    def loadShearRateData(self, numProcs=-1, numFrames=-1, numSlipSys=-1):
        self.shearRates = self.loadFrameData("gammadot",
                                             False,
                                             numProcs=numProcs,
                                             numFrames=numFrames,
                                             numSlipSys=numSlipSys)

    def loadBackstressData(self, numProcs=-1, numFrames=-1):
        self.backstress = self.loadFrameData("backstress",
                                             False,
                                             numProcs=numProcs,
                                             numFrames=numFrames)

    def loadMeshElStatsData(self, name):
        # open mesh stats file
        ELSTAT_FILE_EX = "stelt3d"
        fileName = "{:s}{:s}.{:s}".format(self.meshDir, name, ELSTAT_FILE_EX)
        elStatFile = open(fileName, "r")

        # read grain and phase info for each element
        elStats = np.loadtxt(elStatFile)

        if elStats.shape[0] != self.numElmts:
            print("Problem with element stats file. Missing element data.")
        else:
            self.elStats = elStats

        self.meshElStatNames = ["ID",
                                "coord-x", "coord-y", "coord-z",
                                "elset (grains)", "partition", "volume",
                                "2dmeshp-x", "2dmeshp-y", "2dmeshp-z",
                                "2dmeshd",
                                "2dmeshv-x", "2dmeshv-y", "2dmeshv-z",
                                "2dmeshn-x", "2dmeshn-y", "2dmeshn-z"]

    def __validateFrameNums(self, frameNums):
        # frameNums: frame or list of frames to run calculation for. -1 for all
        if type(frameNums) != list:
            if type(frameNums) == int:
                if frameNums < 0:
                    frameNums = list(range(self.numFrames + 1))
                else:
                    frameNums = [frameNums]
        return frameNums

    def calcMisori(self, frameNums):
        frameNums = self.__validateFrameNums(frameNums)

        symmetry = 'cubic'
        # Loop over frames
        for frameNum in frameNums:
            # Loop over grains
            for grainID in range(1, self.numGrains + 1):
                # select grains based on grainID
                selected = np.where(self.elmtGrain == grainID)[0]
                quats = self.oris[selected, frameNum]

                # Calculate average orientation
                averageOri = copy.deepcopy(quats[0])  # start average
                for quat in quats[1:]:
                    # loop over symmetries and find min misorientation for average
                    # add the symetric equivelent of quat with the minimum misorientation (relative to the average)
                    # to the average. Then normalise.
                    averageOri += averageOri.misOri(quat, symmetry, returnQuat=True)
                averageOri.normalise()

                # Calculate misorientation of each element
                meanMisOri = 0
                for i, quat in enumerate(quats):
                    currentMisOri = quat.misOri(averageOri, symmetry)
                    if currentMisOri > 1:
                        currentMisOri = 1

                    self.misOri[selected[i], frameNum] = currentMisOri
                    meanMisOri += currentMisOri

                self.avMisOri[grainID - 1, frameNum] = meanMisOri / (i + 1)

                clear_output()
                print("Grain {:d} of {:d} done. On frame {:d}.".format(grainID, self.numGrains + 1, frameNum))

        self.misOri[:, frameNums] = np.arccos(self.misOri[:, frameNums]) * 360 / np.pi
        self.avMisOri[:, frameNums] = np.arccos(self.avMisOri[:, frameNums]) * 360 / np.pi

    def writeVTU(self, fileName, frameNums, outputs, times=None, useInitialNodePos=True):
        """Write data out to VTK compatible files. Files are output to the data directory.

        Args:
            fileName (string): Base name of output files.
            frameNums (lst(int)): The simlation frames to output. Either an integer for a single
                                  frame, a list of ints for many or -1 for all.
            outputs (list(string)): The properties to be output. Current options are misOri, gammadot, backstress and elStats.
            times (list(float), optional): The time at each frame (only used for labeling)
            useInitialNodePos (bool, optional): If false uses updateded node positions from each frame. Default: True.
        """
        # Constants
        CON_ORDER = [0, 2, 4, 9, 1, 3, 5, 6, 7, 8]  # corners, then midpoints
        FILE_POSTFIX = "_misorientation"

        # create arrays for element type, conneectivity offset and node positions
        cell_types = np.empty(self.numElmts, dtype='uint8')
        # cell_types[:] = pyevtk.vtk.VtkQuadraticTetra.tid
        cell_types[:] = 24

        offsets = np.arange(start=10, stop=10 * (self.numElmts + 1), step=10, dtype=int)

        if useInitialNodePos:
            x = np.ascontiguousarray(self.nodePos[:, 0])
            y = np.ascontiguousarray(self.nodePos[:, 1])
            z = np.ascontiguousarray(self.nodePos[:, 2])

        # check frameNums and times are valid
        frameNums = self.__validateFrameNums(frameNums)
        if times is None:
            times = frameNums

        # open vtk group file
        fileNameFull = "{:s}{:s}{:s}".format(self.dataDir, fileName, FILE_POSTFIX)
        print(fileNameFull)
        vtgFile = pyevtk.vtk.VtkGroup(fileNameFull)

        for frameNum in frameNums:
            # write frame vtu file path to vtk group file (this needed modification to
            # evtk module to write paths not relative to the current working directory)
            fileNameFull = "{:s}{:s}.{:d}.vtu".format(fileName, FILE_POSTFIX, frameNum)
            vtgFile.addFile(fileNameFull, times[frameNum], relToCWD=False)

            # open vtu file and write root elements
            fileNameFull = "{:s}{:s}{:s}.{:d}".format(self.dataDir, fileName, FILE_POSTFIX, frameNum)
            vtuFile = pyevtk.vtk.VtkFile(fileNameFull, pyevtk.vtk.VtkUnstructuredGrid)
            vtuFile.openGrid()
            vtuFile.openPiece(ncells=self.numElmts, npoints=self.numNodes)

            if not useInitialNodePos:
                x = np.ascontiguousarray(self.nodePosData[:, 0, frameNum])
                y = np.ascontiguousarray(self.nodePosData[:, 1, frameNum])
                z = np.ascontiguousarray(self.nodePosData[:, 2, frameNum])

            vtuFile.openElement("Points")
            vtuFile.addData("points", (x, y, z))
            vtuFile.closeElement("Points")

            vtuFile.openElement("Cells")
            # vtuFile.addData("connectivity", self.elmtCon.flatten())
            vtuFile.addHeader("connectivity", self.elmtCon.dtype.name, self.elmtCon.size, 1)
            vtuFile.addData("offsets", offsets)
            vtuFile.addData("types", cell_types)
            vtuFile.closeElement("Cells")

            # Add headers of cell data
            vtuFile.openElement("CellData")

            if "misOri" in outputs:
                # vtuFile.addData("misorientation", self.misOri[:, 4])
                vtuFile.addHeader("misorientation", self.misOri[:, frameNum].dtype.name, self.misOri[:, 4].size, 1)

            if frameNum > 0:

                if "gammadot" in outputs:
                    for i in range(self.numSlipSys):
                        vtuFile.addHeader("gammadot {:d}".format(i + 1),
                                          self.shearRates[:, i, frameNum - 1].dtype.name,
                                          self.shearRates[:, i, frameNum - 1].size, 1)

                if "backstress" in outputs:
                    maxBackstress = np.abs(self.backstress).max(axis=1)
                    vtuFile.addHeader("backstress max",
                                      maxBackstress[:, frameNum - 1].dtype.name,
                                      maxBackstress[:, frameNum - 1].size, 1)
                    for i in range(self.numSlipSys):
                        vtuFile.addHeader("backstress {:d}".format(i + 1),
                                          self.backstress[:, i, frameNum - 1].dtype.name,
                                          self.backstress[:, i, frameNum - 1].size, 1)

            else:

                if "elStats" in outputs:
                    for i in range(self.elStats.shape[1]):
                        vtuFile.addHeader("Element stat - {:}".format(self.meshElStatNames[i]),
                                          self.elStats[:, i].dtype.name,
                                          self.elStats[:, i].size, 1)

            vtuFile.closeElement("CellData")

            vtuFile.closePiece()
            vtuFile.closeGrid()

            # add actual data to file
            vtuFile.appendData((x, y, z))
            vtuFile.appendData(self.elmtCon[:, CON_ORDER].flatten()).appendData(offsets).appendData(cell_types)
            if "misOri" in outputs:
                vtuFile.appendData(np.ascontiguousarray(self.misOri[:, frameNum]))

            if frameNum > 0:

                if "gammadot" in outputs:
                    for i in range(self.numSlipSys):
                        vtuFile.appendData(np.ascontiguousarray(self.shearRates[:, i, frameNum - 1]))

                if "backstress" in outputs:
                    vtuFile.appendData(np.ascontiguousarray(maxBackstress[:, frameNum - 1]))
                    for i in range(self.numSlipSys):
                        vtuFile.appendData(np.ascontiguousarray(self.backstress[:, i, frameNum - 1]))

            else:

                if "elStats" in outputs:
                    for i in range(self.elStats.shape[1]):
                        vtuFile.appendData(np.ascontiguousarray(self.elStats[:, i]))

            vtuFile.save()

        vtgFile.save()

    def writeVTK(self, fileName, frameNums):
        # constants - taken from ExportVTKFepx.m
        DESCRIPTION = 'Fepx Misorientation Data'
        SUF = 'vtk'
        VERSION = '# vtk DataFile Version 3.0'
        FORMAT = 'ASCII'
        DSET_TYPE = 'DATASET UNSTRUCTURED_GRID'
        DTYPE_R = 'double'
        CDATA = 'CELL_DATA'
        POINTS = 'POINTS'
        CELLS = 'CELLS'
        CTYPES = 'CELL_TYPES'
        CTYPE_TET10 = 24  # VTK_QUADRATIC_TETRA
        CON_ORDER = [0, 2, 4, 9, 1, 3, 5, 6, 7, 8]  # corners, then midpoints
        DAT_SCA = 'SCALARS'
        LOOKUP_DFLT = 'LOOKUP_TABLE default'

        # open file and write header
        fileName = "{:s}{:s}_misorientation.{:s}".format(self.dataDir, fileName, SUF)
        outFile = open(fileName, "w")

        outFile.write("{:s}\n".format(VERSION))
        outFile.write("{:s}\n".format(DESCRIPTION))
        outFile.write("{:s}\n".format(FORMAT))

        # write mesh
        outFile.write("{:s}\n".format(DSET_TYPE))
        outFile.write("{:s} {:d} {:s}\n".format(POINTS, self.numNodes, DTYPE_R))
        np.savetxt(outFile, self.nodePos, fmt='%e')

        outFile.write("{:s} {:d} {:d}\n".format(CELLS, self.numElmts, self.numElmts * 11))
        np.savetxt(outFile, np.concatenate((np.ones((self.numElmts, 1), dtype=int) * 10,
                   self.elmtCon[:, CON_ORDER]), axis=1), fmt='%d')

        outFile.write("{:s} {:d}\n".format(CTYPES, self.numElmts))
        np.savetxt(outFile, np.ones((self.numElmts, 1), dtype=int) * CTYPE_TET10, fmt='%d')

        # write element/cell data
        outFile.write("{:s} {:d}\n".format(CDATA, self.numElmts))

        # phase
        label = "phase"
        outFile.write("{:s} {:s} {:s}\n".format(DAT_SCA, label, DTYPE_R))
        outFile.write("{:s}\n".format(LOOKUP_DFLT))
        np.savetxt(outFile, self.elmtPhase, fmt='%d')

        # grain
        label = "grain"
        outFile.write("{:s} {:s} {:s}\n".format(DAT_SCA, label, DTYPE_R))
        outFile.write("{:s}\n".format(LOOKUP_DFLT))
        np.savetxt(outFile, self.elmtGrain, fmt='%d')

        # misorientation for given frames
        frameNums = self.__validateFrameNums(frameNums)
        for frameNum in frameNums:
            label = "{:s}-{:d}".format("misorientation", frameNum)
            outFile.write("{:s} {:s} {:s}\n".format(DAT_SCA, label, DTYPE_R))
            outFile.write("{:s}\n".format(LOOKUP_DFLT))
            np.savetxt(outFile, self.misOri[:, frameNum], fmt='%e')

        # close file
        outFile.close()

    @staticmethod
    def combineFiles(baseDir, inDirs, outDir):
        """Combine output files from multiple simulations. If a simulation was run in smaller parts

        Args:
            baseDir (string): Base directory of whole simulation. No trailing slash
            inDirs (List(string)): List of simulation directory names to
                                   combine (baseDir/inDir). No trailing slash
            outDir (string): Directory to output combined files to (baseDir/outDir)
        """
        # no trailing slashes on directory names and inDirs is a list
        fileNames = []
        for fileName in os.listdir("{:s}/{:s}".format(baseDir, inDirs[0])):
            if (fnmatch.fnmatch(fileName, "post.*") and
                fileName != "post.conv" and fileName != "post.stats" and not
                    (fnmatch.fnmatch(fileName, "post.debug.*") or
                     fnmatch.fnmatch(fileName, "post.log.*") or
                     fnmatch.fnmatch(fileName, "post.restart.*")
                     )):

                fileNames.append(fileName)

        if not os.path.isdir("{:s}/{:s}".format(baseDir, outDir)):
            os.mkdir("{:s}/{:s}".format(baseDir, outDir))

        for fileName in fileNames:
            outFile = open("{:s}/{:s}/{:s}".format(baseDir, outDir, fileName), 'w')

            for inDir in inDirs:
                inFile = open("{:s}/{:s}/{:s}".format(baseDir, inDir, fileName), 'r')
                for line in inFile:
                    outFile.write(line)
                inFile.close()

            outFile.close()
