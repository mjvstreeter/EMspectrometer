import numpy as np
import scipy.constants as C
np.seterr(divide='ignore')

class ParticlePushObj:
    
    def __init__(self,p_6d,EM_function,dtMin=0,dtMax=1,ppc=16,q=-1,m=1):
        self.p_6d =  p_6d
        self.EM_function = EM_function
        self.dtMin=dtMin
        self.dtMax=dtMax
        self.ppc=ppc
        self.q=q
        self.m=m
        '''
        p_6d is a 6 by N array giving the 3 spatial and 3 momentum values for N particles
        p_6d[:,0] = x,y,z,p_x,p_y,p_z
        EM_function is a user defined function which takes a 3 by N array of particle positions
        and returns (E,B) fields at those positions
        '''
    
    def _dt_calc(self,B_vec,gamma,rqm):
        ''' Calculates the time step given current particle properties and magnetic field
        gives a single dt for all particles to use for simplicity
        '''
        B_mod = np.sqrt(np.sum(B_vec**2,axis=0))

        if np.max(B_mod)>0:
            r = np.min(np.abs(np.divide(gamma*C.c*rqm,B_mod)))
            dt = np.clip((2*np.pi*r)/C.c/(self.ppc),self.dtMin,self.dtMax)
        else:
            dt = self.dtMax
        if np.isnan(dt) or np.isinf(dt):
            dt = self.dtMax
        return dt

    def _gamma_calc(self,p):
        return np.sqrt(1+np.sum(p**2,axis=0))

    def _check_tracking_conditions(self,p_x,p_px,xLims,pLims,n,N_max):
        p_inside = n<N_max
        if p_inside:
            for n in range(3):
                if xLims[n][0] is not None:
                    p_inside = np.logical_and(p_x[n,:]>=xLims[n][0],p_inside)
                    #print(xLims[n][0])
                if xLims[n][1] is not None:
                    p_inside = np.logical_and(p_x[n,:]<=xLims[n][1],p_inside)

            for n in range(3):
                if pLims[n][0] is not None:
                    p_inside = np.logical_and(p_px[n,:]>=pLims[n][0],p_inside)
                if pLims[n][1] is not None:
                    p_inside = np.logical_and(p_px[n,:]<=pLims[n][1],p_inside)
        return np.any(p_inside)

    def trackParticles(self,xLims=[[None]*2]*3,pLims=[[None]*2]*3,N_max=10000):
        ''' Tracks particles using the Boris push method
        '''
        p_m = self.m*C.m_e
        p_q = self.q*C.e
        rqm = p_m/p_q
        p_x = self.p_6d[0:3,:]
        p_px = self.p_6d[3:6,:]
        t = 0
        p_6d_t = []
        t_list = []

        keepTracking = True
        n=0
        while keepTracking:
            
            gamma = self._gamma_calc(p_px)
            E_vec, B_vec = self.EM_function(p_x)
            
            dt = self._dt_calc(B_vec,gamma,rqm)
            # offset velocity by -half step in first iteration
            if n==0:
                dt = -dt/2
            else:
                t = t + dt
            t_list.append(t)
            if n==0:
                p_x_dump, p_px = self._boris_step(p_x,p_px,rqm,dt,E_vec,B_vec)
            else:
                p_x, p_px = self._boris_step(p_x,p_px,rqm,dt,E_vec,B_vec)
            p_6d_t.append(np.concatenate((p_x,p_px),axis=0))
            keepTracking = self._check_tracking_conditions(p_x,p_px,xLims,pLims,n,N_max)
            n=n+1
            
        p_6d_t = np.array(p_6d_t)
        t = np.array(t_list)
        self.p_6d_t = p_6d_t
        self.t = t
        return t,p_6d_t


    def _boris_step(self,p_x,p_px,rqm,dt,E_vec,B_vec):
        # Boris method
        tem = 0.5 * dt / rqm
        # half acceleration with E
        p = p_px +E_vec*tem/C.c
        # compute inverse gamma and multiply by tem
        
        g_tem = np.tile(tem/self._gamma_calc(p),(3,1))
        B_vec = B_vec*g_tem
        
        # half rotation with B
        pp = p + np.cross(p,B_vec,axis=0)
    
        # Second half rotation
        B_vec = B_vec*2/(1+B_vec**2)
        p =  p + np.cross(pp,B_vec,axis=0)
    
        # second half acceleration
        p = p + E_vec*tem/C.c
        
        # move particles
        g = np.tile(self._gamma_calc(p),(3,1))
        v = C.c*p/g
        p_x =  p_x + v*dt
        p_px = p
        return p_x, p_px

        
        
