class Quat(object):
    def __init__(self, *args, **kwargs):
        self.quat = np.zeros(4, dtype=float)
        #construt with euler angles
        if len(args) == 3:
            ph1 = args[0]
            phi = args[1]
            ph2 = args[2]
            
            self.quat[0] = np.cos(phi/2.) * np.cos((ph1+ph2)/2.)
            self.quat[1] = -np.sin(phi/2.) * np.cos((ph2-ph1)/2.)
            self.quat[2] = np.sin(phi/2.) * np.sin((ph2-ph1)/2.)
            self.quat[3] = -np.cos(phi/2.) * np.sin((ph1+ph2)/2.)
        #construt with array of quat coefficients
        elif len(args) == 1:
            self.quat = args[0]
        
        if (self.quat[0] < 0):
            self.quat = self.quat * -1
    
    #overload * operator for quaterion product
    def __mul__(self, right):
        if type(right) is Quat:
            quatCoef = np.zeros(4, dtype=float)
            quatCoef[0] = self.quat[0]*right.quat[0] - np.dot(self.quat[1:4], right.quat[1:4])
            quatCoef[1:4] = self.quat[0]*right.quat[1:4] + right.quat[0]*self.quat[1:4] + np.cross(self.quat[1:4],right.quat[1:4])
            return Quat(quatCoef)
        return
    
    #overload % operator for dot product
    def __mod__(self, right):
        if type(right) is Quat:
            return np.dot(self.quat[0:4], right.quat[0:4])
        return
    
    #allow array like setting/getting of components
    def __getitem__(self, key):
        return self.quat[key]
    
    def __setitem__(self, key, value):
        self.quat[key] = value
        return
    
    def norm(self):
        return np.sqrt(np.dot(self.quat[0:4], self.quat[0:4]))
    
    def disOri(self, right, symGroup):
        misori = []
        for sym in Quat.symEqv(symGroup):
            c = np.dot(self.quat, (right * sym).quat)
            misori.append(c)
        return max(np.array(abs(np.array(misori))))
    
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
