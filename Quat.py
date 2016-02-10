import numpy as np

#be careful with deep and shallow copies
class Quat(object):
    def __init__(self, *args, **kwargs):
        self.quatCoef = np.zeros(4, dtype=float)
        #construt with euler angles
        if len(args) == 3:
            ph1 = args[0]
            phi = args[1]
            ph2 = args[2]
            
            self.quatCoef[0] = np.cos(phi/2.) * np.cos((ph1+ph2)/2.)
            self.quatCoef[1] = -np.sin(phi/2.) * np.cos((ph2-ph1)/2.)
            self.quatCoef[2] = np.sin(phi/2.) * np.sin((ph2-ph1)/2.)
            self.quatCoef[3] = -np.cos(phi/2.) * np.sin((ph1+ph2)/2.)
        #construt with array of quat coefficients
        elif len(args) == 1:
            self.quatCoef = args[0]
        
        if (self.quatCoef[0] < 0):
            self.quatCoef = self.quatCoef * -1
    
    #overload * operator for quaterion product
    def __mul__(self, right):
        if isinstance(right, type(self)):
            newQuatCoef = np.zeros(4, dtype=float)
            newQuatCoef[0] = self.quatCoef[0]*right.quatCoef[0] - np.dot(self.quatCoef[1:4], right.quatCoef[1:4])
            newQuatCoef[1:4] = self.quatCoef[0]*right.quatCoef[1:4] + right.quatCoef[0]*self.quatCoef[1:4] + np.cross(self.quatCoef[1:4],right.quatCoef[1:4])
            return Quat(newQuatCoef)
        raise TypeError()
    
    #overload % operator for dot product
    def __mod__(self, right):
        if isinstance(right, type(self)):
            return np.dot(self.quatCoef, right.quatCoef)
        raise TypeError()
    
    #overload + operator for dot product
    def __add__(self, right):
        if isinstance(right, type(self)):
            return Quat(self.quatCoef + right.quatCoef)
        raise TypeError()
    
    #overload += operator for dot product
    def __iadd__(self, right):
        if isinstance(right, type(self)):
            self.quatCoef += right.quatCoef
            return self
        raise TypeError()
    
    #allow array like setting/getting of components
    def __getitem__(self, key):
        return self.quatCoef[key]
    
    def __setitem__(self, key, value):
        self.quatCoef[key] = value
        return
    
    def norm(self):
        return np.sqrt(np.dot(self.quatCoef[0:4], self.quatCoef[0:4]))
    
    def normalise(self):
        self.quatCoef /= self.norm()
        return
    
    def misOri(self, right, symGroup, returnQuat = False):
        if isinstance(right, type(self)):
            minMisOri = 0   #actually looking for max of this as it is cos of misoriention angle
            for sym in Quat.symEqv(symGroup):   #loop over symmetrically equivelent orienations
                quatSym = right * sym
                currentMisOri = abs(self % quatSym)
                if currentMisOri > minMisOri:   #keep if misorientation lower
                    minMisOri = currentMisOri
                    minQuatSym = quatSym
        
            if returnQuat:
                return minQuatSym
            else:
                return minMisOri
        raise TypeError()
    
    @staticmethod
    def symEqv(group):
        qsym=[]
        qsym.append(Quat(np.array([1.0 , 0.0 , 0.0 , 0.0])))
        
        #from Pete Bate's fspl_orir.f90 code
        #cubic tetrads(100)
        qsym.append(Quat(np.array([0.7071068 , 0.7071068, 0.0, 0.0])))
        qsym.append(Quat(np.array([0.0 , 1.0 , 0.0, 0.0 ])))
        qsym.append(Quat(np.array([0.7071068 , -0.7071068, 0.0, 0.0])))
        
        qsym.append(Quat(np.array([0.7071068, 0.0, 0.7071068, 0.0 ])))
        qsym.append(Quat(np.array([0.0 , 0.0 , 1.0 , 0.0])))
        qsym.append(Quat(np.array([0.7071068, 0.0 ,-0.7071068, 0.0])))
        
        qsym.append(Quat(np.array([0.7071068, 0.0 , 0.0 , 0.7071068])))
        qsym.append(Quat(np.array([0.0 , 0.0 , 0.0 , 1.0])))
        qsym.append(Quat(np.array([0.7071068, 0.0 , 0.0 , -0.7071068])))
        
        #cubic dyads (110)
        qsym.append(Quat(np.array([0.0 , 0.7071068 , 0.7071068 , 0.0])))
        qsym.append(Quat(np.array([0.0 , -0.7071068 , 0.7071068 , 0.0])))
        
        qsym.append(Quat(np.array([0.0 , 0.7071068 , 0.0 , 0.7071068])))
        qsym.append(Quat(np.array([0.0 , -0.7071068 , 0.0 , 0.7071068])))
        
        qsym.append(Quat(np.array([0.0 , 0.0 , 0.7071068 , 0.7071068])))
        qsym.append(Quat(np.array([0.0 , 0.0 , -0.7071068 , 0.7071068])))
        
        #cubic triads (111)
        qsym.append(Quat(np.array([0.5, 0.5 , 0.5 , 0.5])))
        qsym.append(Quat(np.array([0.5, -0.5 , -0.5 , -0.5])))
        
        qsym.append(Quat(np.array([0.5, -0.5 , 0.5 , 0.5])))
        qsym.append(Quat(np.array([0.5, 0.5 , -0.5 , -0.5])))
        
        qsym.append(Quat(np.array([0.5, 0.5 , -0.5 , 0.5])))
        qsym.append(Quat(np.array([0.5, -0.5 , 0.5 , -0.5])))
        
        qsym.append(Quat(np.array([0.5, 0.5 , 0.5 , -0.5])))
        qsym.append(Quat(np.array([0.5, -0.5 , -0.5 , 0.5])))
        
        #hexagonal hexads
        qsym.append(Quat(np.array([0.866254 , 0.0 , 0.0 , 0.5])))
        qsym.append(Quat(np.array([0.5 , 0.0 , 0.0 , 0.866254])))
        qsym.append(Quat(np.array([0.5 , 0.0 , 0.0 , -0.866254])))
        qsym.append(Quat(np.array([0.866254 , 0.0 , 0.0 , -0.5])))
        
        #hexagonal diads
        qsym.append(Quat(np.array([0.0, -0.5 , 0.866254 , 0.0])))
        qsym.append(Quat(np.array([0.0, -0.5 , -0.866254 , 0.0])))
        
        if (group == 'cubic'):
            return qsym[0:24]
        elif (group == 'hexagonal'):
            return [qsym[0], qsym[2], qsym[8]] + qsym[-6:30]
        else:
            return qsym
