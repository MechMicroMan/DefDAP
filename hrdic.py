import numpy as np
import matplotlib.pyplot as plt


class Map(object):
    
    def __init__(self, path, fname) :
        
        self.path = path
        self.fname = fname
        #Load in data
        self.data = np.loadtxt(self.path + self.fname, skiprows=1)
        self.xc = self.data[:,0] #x coordinates
        self.yc = self.data[:,1] #y coordinates
        self.xd = self.data[:,2] #x displacement
        self.yd = self.data[:,3] #y displacement
        
        #Calculate size of map
        self.xdim = (self.xc.max()-self.xc.min()
                  )/min(abs((np.diff(self.xc))))+1 #size of map along x
        self.ydim = (self.yc.max()-self.yc.min()
                  )/max(abs((np.diff(self.yc))))+1 #size of map along y


        self.x_map = self._map(self.xd) #u (displacement component along x) 
        self.y_map = self._map(self.yd) #v (displacement component along x) 
        self.f11 = self._grad(self.x_map)[1]#f11
        self.f22 = self._grad(self.y_map)[0]#f22
        self.f12 = self._grad(self.x_map)[0]#f12
        self.f21 = self._grad(self.y_map)[1]#f21
        
        self.max_shear = np.sqrt((((self.f11-self.f22)/2.)**2)
                               + ((self.f12+self.f21)/2.)**2)# max shear component
        self.mapshape = np.shape(self.max_shear)
        
    def _map(self, data_col):
        data_map = np.reshape(np.array(data_col), (int(self.ydim), int(self.xdim)))
        return data_map
    
    def _grad(self, data_map) :
        grad_step = min(abs((np.diff(self.xc))))
        data_grad = np.gradient(data_map, grad_step, grad_step)
        return data_grad