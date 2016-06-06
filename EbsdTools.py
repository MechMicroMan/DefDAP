import numpy as np
import matplotlib.pyplot as plt
import copy

from Quat import Quat

#import ipyparallel as ipp

class Map(object):
    #defined instance variables
    #xDim, yDim - (int) dimensions of maps
    #binData - imported binary data
    
    def __init__(self):
        self.crystalSym = None #symmetry of material e.g. "cubic", "hexagonal"
        self.xDim = None #(int) dimensions of maps
        self.yDim = None
        self.binData = None #imported binary data
        self.quatArray = None #(array) array of quaterions for each point of map
        self.misOx = None #(array) map of misorientation with single neighbour pixel in positive x dir
        self.misOy = None
        self.boundaries = None #(array) map of boundariers
        self.grains = None #(array) map of grains
        self.grainList = None #(list) list of grains
        self.misOri = None #(array) map of misorientation
        self.averageSchmidFactor = None #(array) map of average Schmid factor
        return
    
    def loadData(self, fileName, crystalSym):
        #open meta data file and read in x and y dimensions
        f = open(fileName + ".cpr", 'r')
        for line in f:
            if line[:6] == 'xCells':
                self.xDim = int(line[7:])
            if line[:6] == 'yCells':
                self.yDim = int(line[7:])
        f.close()
        #now read the binary .crc file
        fmt_np=np.dtype([('Phase','b'), ('Eulers', [('ph1','f'), ('phi','f'), ('ph2','f')]),
                         ('mad','f'), ('IB2','uint8'), ('IB3','uint8'), ('IB4','uint8'),
                         ('IB5','uint8'), ('IB6','f')])
        self.binData = np.fromfile(fileName + ".crc", fmt_np, count=-1)
        self.crystalSym = crystalSym
        return
                         
    def plotBandContrastMap(self):
        self.checkDataLoaded()
                                 
        bcmap = np.reshape(self.binData[('IB2')], (self.yDim, self.xDim))
        plt.imshow(bcmap, cmap='gray')
        plt.colorbar()
        return

    def plotEulerMap(self):
        self.checkDataLoaded()
        
        emap = np.transpose(np.array([self.binData['Eulers']['ph1'], self.binData['Eulers']['phi'],
                                      self.binData['Eulers']['ph2']]))
        #this is the normalization for the
        norm = np.tile(np.array([2*np.pi, np.pi/2, np.pi/2]), (self.yDim,self.xDim))
        norm = np.reshape(norm, (self.yDim,self.xDim,3))
        eumap = np.reshape(emap, (self.yDim,self.xDim,3))
        #make non-indexed points green
        eumap = np.where(eumap!=[0.,0.,0.], eumap, [0.,1.,0.])
        
        fig, ax = plt.subplots()
        ax.imshow(eumap/norm, aspect='equal')
        return fig
    
    def checkDataLoaded(self):
        if self.binData is None:
            raise Exception("Data not loaded")
        return
    
    def buildQuatArray(self):
        self.checkDataLoaded()
        
        if self.quatArray is None:
            self.quatArray = np.empty([self.yDim, self.xDim], dtype=Quat)
            for j in range(self.yDim):
                for i in range(self.xDim):
                    eulers = self.binData[j*self.xDim + i][('Eulers')]
                    self.quatArray[j, i] = Quat(eulers[0], eulers[1], eulers[2])
        return
    
    
    def findBoundaries(self, boundDef = 10):
        self.buildQuatArray()
        
        self.misOx = np.zeros([self.yDim, self.xDim])
        self.misOy = np.zeros([self.yDim, self.xDim])
        self.boundaries = np.zeros([self.yDim, self.xDim])
        
        
        self.smap = np.zeros([self.yDim, self.xDim])
        for i in range(self.xDim):
            for j in range(self.yDim):
                self.smap[j,i] = np.arccos(self.quatArray[j,i][0])
        
        
        #sweep in positive x and y dirs calculating misorientation with neighbour
        #if > boundDef then mark as a grain boundary
        for i in range(self.xDim):
            for j in range(self.yDim - 1):
                aux = abs(self.quatArray[j,i] % self.quatArray[j+1,i])
                if aux > 1:
                    aux = 1
                
                self.misOx[j,i] = 360 * np.arccos(aux) / np.pi
                
                if self.misOx[j,i] > boundDef:
                    self.misOx[j,i] = 0.0
                    self.boundaries[j,i] = 255
                    
                    self.smap[j,i] = 0
        
        
        for i in range(self.xDim - 1):
            for j in range(self.yDim):
                
                aux = abs(self.quatArray[j,i] % self.quatArray[j,i+1])
                if aux > 1:
                    aux = 1
                
                self.misOy[j,i] = 360 * np.arccos(aux) / np.pi
                
                if self.misOy[j,i] > boundDef:
                    self.misOy[j,i] = 0.0
                    self.boundaries[j,i] = 255
                    
                    self.smap[j,i] = 0
        
        return

    def plotBoundaryMap(self):
        plt.figure(), plt.imshow(self.smap, interpolation='none')
        plt.figure(), plt.imshow(self.boundaries, vmax=15),plt.colorbar()
        return
  
  
    def findGrains(self):
        self.grains = np.copy(self.boundaries)
        
        self.grainList = []
        
        unknownPoints = np.where(self.grains == 0)
        
        grainIndex = 1
        
        while unknownPoints[0].shape[0] > 0:
            self.floodFill(unknownPoints[1][0], unknownPoints[0][0], grainIndex)
            
            grainIndex += 1
            unknownPoints = np.where(self.grains == 0)
        return
            
    def plotGrainMap(self):
        plt.figure()
        plt.imshow(self.grains)
        return
    
    def locateGrainID(self):
        if (self.grainList is not None) and (self.grainList != []):
            fig = self.plotEulerMap()
            fig.canvas.callbacks.connect('button_press_event', self.fig_on_click)
        else:
            raise Exception("Grain list empty")
            

    def fig_on_click(self, event):
        if event.inaxes is not None:
            print self.grains[int(event.ydata), int(event.xdata)] - 1
    

    def floodFill(self, x, y, grainIndex):
        currentGrain = Grain(self.crystalSym)

        currentGrain.addPoint(self.quatArray[y, x], (x, y))
        
        edge = [(x, y)]
        grain = [(x, y)]
        
        self.grains[y, x] = grainIndex
        while edge:
            newedge = []
            
            for (x, y) in edge:
                moves = np.array([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
                
                movesIndexShift = 0
                if x <= 0:
                    moves = np.delete(moves, 1, 0)
                    movesIndexShift = 1
                elif x >= self.xDim-1:
                    moves = np.delete(moves, 0, 0)
                    movesIndexShift = 1
                
                if y <= 0:
                    moves = np.delete(moves, 3-movesIndexShift, 0)
                elif y >= self.yDim-1:
                    moves = np.delete(moves, 2-movesIndexShift, 0)
                
                
                for (s, t) in moves:
                    if self.grains[t, s] == 0:
                        currentGrain.addPoint(self.quatArray[t, s], (s, t))
                        newedge.append((s, t))
                        grain.append((s, t))
                        self.grains[t, s] = grainIndex
                    elif self.grains[t, s] == 255 and (s > x or t > y):
                        currentGrain.addPoint(self.quatArray[t, s], (s, t))
                        grain.append((s, t))
                        self.grains[t, s] = grainIndex
            
            if newedge == []:
                self.grainList.append(currentGrain)
                return grain
            else:
                edge = newedge

    def calcGrainAvOris(self):
        for grain in self.grainList:
            grain.calcAverageOri()
        return

    def calcGrainMisOri(self):
        #localGrainList = self.grainList[0:10]

        #paraClient = ipp.Client()
        
        
        
        #dview = paraClient[:]
        #dview.map_sync(lambda x: x**10, range(32000000))
        #print dview.map_sync(lambda grain: grain.buildMisOriList(), localGrainList)
        
        #paraClient.close()
        
        for grain in self.grainList:
            grain.buildMisOriList()
        return

    def buildMisOriMap(self):
        self.misOri = np.zeros([self.yDim, self.xDim])
        
        for grain in self.grainList:
            for coord, misOri in zip(grain.coordList, grain.misOriList):
                self.misOri[coord[1], coord[0]] = misOri
    
        return

    def plotMisOriMap(self, vMin=None, vMax=None):
        plt.figure()
        plt.imshow(np.arccos(self.misOri) * 360 / np.pi, interpolation='none', vmin=vMin, vmax=vMax)
        plt.colorbar()
        return



    def calcAverageGrainSchmidFactors(self, loadVector = np.array([0, 0, 1])):
        for grain in self.grainList:
            grain.calcAverageSchmidFactors(loadVector = loadVector)
        return

    def buildAverageGrainSchmidFactorsMap(self):
        self.averageSchmidFactor = np.zeros([self.yDim, self.xDim])
        
        for grain in self.grainList:
            currentSchmidFactor = max(np.array(grain.averageSchmidFactors))
            for coord in grain.coordList:
                self.averageSchmidFactor[coord[1], coord[0]] = currentSchmidFactor
        return

    def plotAverageGrainSchmidFactorsMap(self):
        plt.figure()
        plt.imshow(self.averageSchmidFactor, interpolation='none')
        plt.colorbar()
        return





class Grain(object):
    
    def __init__(self, crystalSym):
        self.crystalSym = crystalSym    #symmetry of material e.g. "cubic", "hexagonal"
        self.coordList = []             #list of coords stored as tuples (x, y)
        self.quatList = []              #list of quats
        self.misOriList = None          #list of misOri at each point in grain
        self.averageOri = None          #(quat) average ori of grain
        self.averageMisOri = None       #average misOri of grain
        self.loadVectorCrystal = None   #load vector in crystal coordinates
        self.averageSchmidFactors = None    #list of Schmid factors for each system
        return
    
    def __len__(self):
        return len(self.quatList)
    
    #quat is a quaterion and coord is a tuple (x, y)
    def addPoint(self, quat, coord):
        self.coordList.append(coord)
        self.quatList.append(quat)
        return

    def calcAverageOri(self):
        firstQuat = True
        for quat in self.quatList:
            if firstQuat: #if 1st orientation, start the average
                self.averageOri = copy.deepcopy(quat)
                firstQuat = False
            else: #otherwise need to loop over symmetries and find min misorientation for average
                #add the symetric equivelent of quat with the minimum misorientation (relative to the average)
                #to the average. Then normalise.
                self.averageOri += self.averageOri.misOri(quat, self.crystalSym, returnQuat = True)
                self.averageOri.normalise()
        return

    def buildMisOriList(self):
        if self.averageOri is None:
            self.calcAverageOri()

        self.misOriList = []
        for quat in self.quatList:
            currentMisOri = quat.misOri(self.averageOri, self.crystalSym)
            if currentMisOri > 1:
                currentMisOri = 1
            self.misOriList.append(currentMisOri)

        self.averageMisOri = np.array(self.misOriList).mean()

        return
        #return self#to make it work with parallel
    
    
    
    def extremeCoords(self):
        unzippedCoordlist = list(zip(*self.coordList))
        x0 = min(unzippedCoordlist[0])
        y0 = min(unzippedCoordlist[1])
        xmax = max(unzippedCoordlist[0])
        ymax = max(unzippedCoordlist[1])
        return x0, y0, xmax, ymax

    def plotOutline(self):
        x0, y0, xmax, ymax = self.extremeCoords()
    
        #initialise array with nans so area not in grain displays white
        grainOuline = np.full([ymax - y0 + 1, xmax - x0 + 1], np.nan, dtype=float)

        for coord in self.coordList:
            grainOuline[coord[1] - y0, coord[0] - x0] = 0

        plt.figure()
        plt.imshow(grainOuline, interpolation='none')
        plt.colorbar()
        
        return

    def plotMisOri(self, vMin=None, vMax=None):
        x0, y0, xmax, ymax = self.extremeCoords()
        
        #initialise array with nans so area not in grain displays white
        grainMisOri = np.full([ymax - y0 + 1, xmax - x0 + 1], np.nan, dtype=float)

        for coord, misOri in zip(self.coordList, self.misOriList):
            grainMisOri[coord[1] - y0, coord[0] - x0] = misOri

        plt.figure()
        plt.imshow(np.arccos(grainMisOri) * 360 / np.pi, interpolation='none', vmin=vMin, vmax=vMax)
        plt.colorbar()

        return
    
    
    
    
    
    
    #define load axis as unit vector to save calculations
    def calcAverageSchmidFactors(self, loadVector = np.array([0, 0, 1])):
        if self.averageOri is None:
            self.calcAverageOri()
        
        #Transform the load vector into crystal coordinates
        loadQuat = Quat(0, loadVector[0], loadVector[1], loadVector[2])

        loadQuatCrystal = (self.averageOri * loadQuat) * self.averageOri.conjugate()
    
        loadVectorCrystal = loadQuatCrystal[1:4]    #will still be a unit vector as aveageOri is a unit quat
        self.loadVectorCrystal = loadVectorCrystal
        
        self.averageSchmidFactors = []
    
        #calculated Schmid factor of average ori with all slip systems
        for slipSystem in self.slipSystems():
            self.averageSchmidFactors.append(abs(np.dot(loadVectorCrystal, slipSystem[1])
                                                 * np.dot(loadVectorCrystal, slipSystem[0])))


    #slip systems defined as list with 1st value the slip direction, 2nd the slip plane and 3rd a label
    #define as unit vectors to save calculations
    def slipSystems(self):
        systems = []
        if self.crystalSym == "cubic":
            systems.append([np.array([0, 0.707107, -0.707107]), np.array([0.577350, 0.577350, 0.577350]), "(111)[01-1]"])
            systems.append([np.array([-0.707107, 0, 0.707107]), np.array([0.577350, 0.577350, 0.577350]), "(111)[-101]"])
            systems.append([np.array([0.707107, -0.707107, 0]), np.array([0.577350, 0.577350, 0.577350]), "(111)[1-10]"])

            systems.append([np.array([0, 0.707107, 0.707107]), np.array([0.577350, 0.577350, -0.577350]), "(11-1)[011]"])
            systems.append([np.array([-0.707107, 0, -0.707107]), np.array([0.577350, 0.577350, -0.577350]), "(11-1)[-10-1]"])
            systems.append([np.array([0.707107, -0.707107, 0]), np.array([0.577350, 0.577350, -0.577350]), "(11-1)[1-10]"])

            systems.append([np.array([0, 0.707107, -0.707107]), np.array([-0.577350, 0.577350, 0.577350]), "(-111)[01-1]"])
            systems.append([np.array([0.707107, 0, 0.707107]), np.array([-0.577350, 0.577350, 0.577350]), "(-111)[101]"])
            systems.append([np.array([-0.707107, -0.707107, 0]), np.array([-0.577350, 0.577350, 0.577350]), "(-111)[-1-10]"])

            systems.append([np.array([0, -0.707107, -0.707107]), np.array([0.577350, -0.577350, 0.577350]), "(1-11)[0-1-1]"])
            systems.append([np.array([-0.707107, 0, 0.707107]), np.array([0.577350, -0.577350, 0.577350]), "(1-11)[-101]"])
            systems.append([np.array([0.707107, 0.707107, 0]), np.array([0.577350, -0.577350, 0.577350]), "(1-11)[110]"])

        return systems

