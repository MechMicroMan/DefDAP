import numpy as np

#be careful with deep and shallow copies
class Quat(object):
    def __init__(self, *args, **kwargs):
        self.quatCoef = np.zeros(4, dtype=float)
        #construt with Bunge euler angles (radians)
        if len(args) == 3:
            ph1 = args[0]
            phi = args[1]
            ph2 = args[2]
            
            self.quatCoef[0] = np.cos(phi/2.0) * np.cos((ph1+ph2)/2.0)
            self.quatCoef[1] = np.sin(phi/2.0) * np.cos((ph1-ph2)/2.0)
            self.quatCoef[2] = np.sin(phi/2.0) * np.sin((ph1-ph2)/2.0)
            self.quatCoef[3] = np.cos(phi/2.0) * np.sin((ph1+ph2)/2.0)
        #construt with array of quat coefficients
        elif len(args) == 1:
            self.quatCoef = args[0]
        #construt with quat coefficients
        elif len(args) == 4:
            self.quatCoef[0] = args[0]
            self.quatCoef[1] = args[1]
            self.quatCoef[2] = args[2]
            self.quatCoef[3] = args[3]
        
        if (self.quatCoef[0] < 0):
            self.quatCoef = self.quatCoef * -1

    def eulerAngles(self):
        #See Melcher, a. Unser, A. Reichhardt, M. Nestler, B. Conversion of EBSD data by a quaternion based algorithm to be used for grain structure simulations
    
        eulers = np.empty(3, dtype=float)
        eulers1 = np.empty(3, dtype=float)
        
        chi = np.sqrt((self.quatCoef[0]**2 + self.quatCoef[3]**2) * (self.quatCoef[1]**2 + self.quatCoef[2]**2))
        
        cosPh1 = (self.quatCoef[0]*self.quatCoef[1] - self.quatCoef[2]*self.quatCoef[3]) / (2*chi)
        sinPh1 = (self.quatCoef[0]*self.quatCoef[2] + self.quatCoef[1]*self.quatCoef[3]) / (2*chi)
        
        cosPhi = self.quatCoef[0]**2 + self.quatCoef[3]**2 - self.quatCoef[1]**2 - self.quatCoef[2]**2
        sinPhi = 2*chi
        
        cosPh2 = (self.quatCoef[0]*self.quatCoef[1] + self.quatCoef[2]*self.quatCoef[3]) / (2*chi)
        sinPh2 = (self.quatCoef[1]*self.quatCoef[3] - self.quatCoef[0]*self.quatCoef[2]) / (2*chi)
        
        #eulers[0] = np.arctan(sinPh1/cosPh1)
        #eulers[1] = np.arctan(sinPhi/cosPhi)
        #eulers[2] = np.arctan(sinPh2/cosPh2)
        
        eulers[0] = np.arctan2(sinPh1, cosPh1)
        eulers[1] = np.arctan2(sinPhi, cosPhi)
        eulers[2] = np.arctan2(sinPh2, cosPh2)
        
        if eulers[0] < 0: eulers[0] += 2*np.pi
        if eulers[2] < 0: eulers[2] += 2*np.pi
        
        
        
        
        #at1 = atan2(qd,qa);
        #at2 = atan2(qb,qc);

#alpha = at1 - at2;
#beta = 2*atan2(sqrt(qb.^2+qc.^2), sqrt(qa.^2+qd.^2));
#gamma = at1 + at2;
        
        
        #phi1
        #eulers[0] = np.arccos((self.quatCoef[0]*self.quatCoef[1] - self.quatCoef[2]*self.quatCoef[3]) / (2*chi))
        #eulers1[0] = np.arcsin((self.quatCoef[0]*self.quatCoef[2] + self.quatCoef[1]*self.quatCoef[3]) / (2*chi))
        
        #Phi
        #eulers[1] = np.arccos((self.quatCoef[0]**2 + self.quatCoef[3]**2) - (self.quatCoef[1]**2 + self.quatCoef[2]**2))
        #eulers1[1] = np.arcsin(2*chi)
        
        #phi2
        #eulers[2] = np.arccos((self.quatCoef[0]*self.quatCoef[1] + self.quatCoef[2]*self.quatCoef[3]) / (2*chi))
        #eulers1[2] = np.arcsin((self.quatCoef[1]*self.quatCoef[3] - self.quatCoef[0]*self.quatCoef[2]) / (2*chi))
    
    
    
        return eulers

    #show something meaningful when the quat is printed
    def __repr__(self):
        return "[%.4f, %.4f, %.4f, %.4f]" % (self.quatCoef[0], self.quatCoef[1], self.quatCoef[2], self.quatCoef[3])
    def __str__(self):
        return "[%.4f, %.4f, %.4f, %.4f]" % (self.quatCoef[0], self.quatCoef[1], self.quatCoef[2], self.quatCoef[3])
    
    #overload * operator for quaterion product and vector product
    def __mul__(self, right):
        if isinstance(right, type(self)):   #another quat
            newQuatCoef = np.zeros(4, dtype=float)
            newQuatCoef[0] = self.quatCoef[0]*right.quatCoef[0] - np.dot(self.quatCoef[1:4], right.quatCoef[1:4])
            newQuatCoef[1:4] = self.quatCoef[0]*right.quatCoef[1:4] + right.quatCoef[0]*self.quatCoef[1:4] + np.cross(self.quatCoef[1:4],right.quatCoef[1:4])
            return Quat(newQuatCoef)
        #elif isinstance(right, np.ndarray) and len(right) == 3: #a 3 vector
        #    returnVector = np.empty(3, dtype=float)
        #    returnVector = self.quatCoef[1:4] * right
        #    return returnVector
        raise TypeError()
    
    #overload % operator for dot product
    def __mod__(self, right):
        if isinstance(right, type(self)):
            return np.dot(self.quatCoef, right.quatCoef)
        raise TypeError()
    
    #overload + operator
    def __add__(self, right):
        if isinstance(right, type(self)):
            return Quat(self.quatCoef + right.quatCoef)
        raise TypeError()
    
    #overload += operator
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
    
    #also the inverse if this is a unit quaterion
    def conjugate(self):
        return Quat(self.quatCoef[0], -self.quatCoef[1], -self.quatCoef[2], -self.quatCoef[3])
    
    def misOri(self, right, symGroup, returnQuat = 0):
        if isinstance(right, type(self)):
            minMisOri = 0   #actually looking for max of this as it is cos of misoriention angle
            for sym in Quat.symEqv(symGroup):   #loop over symmetrically equivelent orienations
                quatSym = right * sym
                currentMisOri = abs(self % quatSym)
                if currentMisOri > minMisOri:   #keep if misorientation lower
                    minMisOri = currentMisOri
                    minQuatSym = quatSym
        
            if returnQuat == 1:
                return minQuatSym
            elif returnQuat == 2:
                return minMisOri, minQuatSym
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
