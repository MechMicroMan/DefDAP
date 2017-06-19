import numpy as N
import pylab as P
import sys
import re
import scipy.ndimage.interpolation as S

class ReadEBSD:
    
	
    def __init__(self):
        self.data=[] #creates an empty list for each EBSD name
        self.alist=[]
        self.ori_list=[]
        
    def c5read(self,fname):
    
        
    #f=open('34 compression twin.ctf','r')
        f=open(fname,'r')
        print ('Reading file...')
        for line in f:
            if line[0:2]=='XC':
                xdim=int(line.split()[1])
                x_number=float(xdim)
                print ('XCells = ' +str(xdim))
                self.xcells=xdim
            if line[0:2]=='YC':
                    ydim=int(line.split()[1])
                    print ('YCells = ' +str(ydim))
                    self.ycells=ydim
        	
            if re.match('\d\s',line) != None: 
                if int(line.split()[0])==1:
                    eulers=line.split()[5:8] 
                    for x in range(len(eulers)):
                        eulers[x]=float(eulers[x])
                    self.data.append(N.array(eulers))
                else:
                    eulers=line.split()[5:8]
                    for x in range(len(eulers)):
                        eulers[x]=float('0.0')
                    self.data.append(N.array(eulers)) 
        f.close()
        print ('Done!')
        
    def create_map(self,y1,y2,x1,x2,step,rot_val,euler):
        self.step=step
        xdim,ydim=self.xcells,self.ycells
        print (xdim,ydim)
        data_list=self.data
        l_map=N.reshape(data_list,(ydim,xdim,3))
        r_map=S.rotate(l_map[::,::,:],rot_val,order=0)#front
        s_map=r_map[y1:y2:step,x1:x2:step]
        s_map=s_map[:,:]
        
        print (N.shape( s_map))
        cs_map_user=s_map[:,:,euler]
        
        cs_map_p1=s_map[:,:,0]
        cs_map_P=s_map[:,:,1]
        cs_map_p2=s_map[:,:,2]

        shape=N.shape(s_map)[0]*N.shape(s_map)[1]
        
        self.total=shape
        self.xels=N.shape(s_map)[1]
        self.zels=N.shape(s_map)[0]

        print (shape)
        X = N.ma.masked_equal(cs_map_user,0)
        X_p1,X_P,X_p2=N.ma.masked_equal(cs_map_p1,0),N.ma.masked_equal(cs_map_P,0),N.ma.masked_equal(cs_map_p2,0)
        self.map=[X_p1,X_P,X_p2]	
        P.colorbar((P.imshow(X,interpolation='none',cmap='nipy_spectral')))
        P.axis('on')
        a_list=N.reshape(s_map,(N.shape(s_map)[0]*N.shape(s_map)[1],3))
        print (N.shape(a_list))
        self.alist.append(a_list)
        
    
    def orifile(self,yels,ori_name):
        self.yels=yels
        total=self.total
        xels=self.xels
        zels=self.zels
        xpos,ypos,zpos=N.arange(0,total,xels),N.arange(1,yels+1),N.arange(1,zels+1)
        print (xpos)
        eul_last=N.array([0.0,0.0,0.0])
        fdata_list=self.alist[0]
        print (ori_name)
        lines=0
        xinit=0
        f=open(ori_name,'w')
        for xi,z in zip(xpos,zpos):
            for yi in ypos:
                print ([xi,z*xels])
                for eul in fdata_list[xi:(z*xels)]:
    
                    lines=lines+1
                    #eul[1]=eul[1]+30.0  #Bate correction
                    #if eul.sum()==0:
                        #eul=eul*N.pi/180.0	
                        #f.write('1.0 %2.5f %2.5f %2.5f\n' % (eul_last[0],eul_last[1],eul_last[2]))
                        
                    #else:
                    eul=eul*N.pi/180.0	
                    f.write('1.0 %2.5f %2.5f %2.5f\n' % (eul[0],eul[1],eul[2]))
                    eul_last=eul.copy()
                        
        print ('Ori File created Successfully')
        print ('Number of lines = '+str(lines))

    def ori_test(self,ori_name):
        xels,yels,zels=self.xels,self.yels,self.zels
        ori_list=N.loadtxt(ori_name,delimiter=' ')
        Phi=ori_list[:,2]
        phi1=ori_list[:,1]
        phi2=ori_list[:,3]
        
        indices=N.arange(0,len(ori_list)+xels,xels)
        phi_indices_f=N.arange(0,yels*len(ori_list)+yels*xels,yels*xels)
        f_region=N.zeros(shape=[self.total/yels,3])
        print (indices)
       
        for i,j,p in zip(indices[:],indices[1:],phi_indices_f):
            print (i,j,p)
            f_region[i:j,0]=phi1[p:p+xels]
            f_region[i:j,1]=Phi[p:p+xels]
            f_region[i:j,2]=phi2[p:p+xels]
        fface=N.reshape(f_region[:,0],[zels,xels])
        self.face=N.reshape(f_region[:],[zels,xels,3])
        P.imshow(fface,interpolation='none',cmap='nipy_spectral')
