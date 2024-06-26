from . import particle_pusher as pp3d
import numpy as np
from scipy.constants import c, m_e, e
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from .general_tools import circular_pattern

import matplotlib, os

def g2b(g):
    return np.sqrt(1-1./g**2)
def b2g(b):
    return 1/np.sqrt(1-b**2)

class Espec:
    ''' Class for modelling an electro-magnetic spectrometer. 
            Calculates particle trajectories and can return intersections of particles with detector planes and calculate dispersion function (with angular dependance)
    '''
    def __init__(self,q=-1,m=1):
        self.N_MeV = 1000 # number of points on electron energy axis
        self.p_6d_0 = None
        self.q = q # -1 for electrons
        self.m = m # 1 for electrons 1836.15 for protons

    def plotTracks(self,ax=None,plot_Bfield=True,plot_cbar=True):
        xLims = self.xLims
        N_p = self.N_p
        Nx_img= 1000
        Ny_img = 500
        x_img =np.linspace(xLims[0][0],xLims[0][1],num=Nx_img)
        y_img =np.linspace(xLims[1][1],xLims[1][0],num=Ny_img)
        [X,Y] = np.meshgrid(x_img,y_img)
        Z = X*0
        XYZ = np.array((X.flatten(),Y.flatten(),Z.flatten()))

        E_vec, B_vec =  self.EM_function(0,XYZ)
        B_img = np.reshape(B_vec[2,:],(Ny_img,Nx_img))
        if ax is None:
            fig = plt.figure(figsize = (16/2.54,5/2.54),dpi=150)
            ax = plt.axes()
        if plot_Bfield:
            if np.abs(np.max(B_img))>np.abs(np.min(B_img)): 
                cmap = 'gray_r'
            else:
                cmap = 'gray'
            ih = ax.pcolormesh(x_img,y_img,B_img,cmap=cmap,vmin=np.min(B_img),vmax=np.max(B_img),shading='auto')
        if plot_cbar:
            if plot_Bfield:
                cbh = plt.colorbar(ih,ax=ax)
        else:
            cbh = None
        pSel = range(0,N_p,int(N_p/self.N_g))
        for p_ind in pSel:
            ax.plot(self.p_6d_t[:,0,p_ind],self.p_6d_t[:,1,p_ind],'r-',alpha=0.3)
        if self.screens is not None:
            for dS in self.screens:
                ax.plot((dS['origin'][0],dS['end'][0]),(dS['origin'][1],dS['end'][1]),'b-')
        ax.set_xlim(*xLims[0])
        ax.set_ylim(*xLims[1])
        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$y$ [m]')
        return ax, cbh

    def findScreenParticles(self):
        p_6d_t = self.p_6d_t
        N_g = self.N_g
        g = self.g
        G = self.G
        t = self.t
        N_p = self.N_p
        for dS in self.screens:
            d_o = dS['origin']
            d_e = dS['end']
            d_s = dS['side']
            d_x_vec = (d_e-d_o)/np.linalg.norm(d_e-d_o)
            d_y_vec = (d_s-d_o)/np.linalg.norm(d_s-d_o)
            l = np.linalg.norm(d_e-d_o)

            screenNormal = np.cross(d_x_vec,d_y_vec)
            screenNormal = np.expand_dims(screenNormal,1)
            screenNormal = screenNormal/np.linalg.norm(screenNormal)
            screen_x = []
            screen_y = []
            screen_eAng = []
            screen_g = []
            for n in range(N_p):
                x = p_6d_t[:,0,n]
                y = p_6d_t[:,1,n]
                z = p_6d_t[:,2,n]
                p_x = p_6d_t[:,3,n]
                p_y = p_6d_t[:,4,n]
                p_z = p_6d_t[:,5,n]
                v = np.dot(np.reshape(
                    p_6d_t[:,0:3,n],(-1,3))-np.expand_dims(dS['origin'],0),screenNormal)
                t_v = interp1d(v.flatten(),t,bounds_error=False)
                x_t = interp1d(t,x,bounds_error=False)
                y_t = interp1d(t,y,bounds_error=False)
                z_t = interp1d(t,z,bounds_error=False)
                px_t = interp1d(t,p_x,bounds_error=False)
                py_t = interp1d(t,p_y,bounds_error=False)
                pz_t = interp1d(t,p_z,bounds_error=False)
                
                t_0 = t_v(0)
                if np.isfinite(t_0):
                    
                    p_0 = np.array((x_t(t_0),y_t(t_0),z_t(t_0)))
                    v_0 = np.array((px_t(t_0),py_t(t_0),pz_t(t_0)))
                    v_0 = v_0 /np.linalg.norm(v_0)
                    screen_x.append(np.dot(p_0-d_o,d_x_vec))
                    screen_y.append(np.dot(p_0-d_o,d_y_vec))
                    screen_g.append(G.flatten()[n])
                    screen_eAng.append(np.arccos(np.dot(v_0,screenNormal.flatten())))
                else:
                    screen_x.append(np.nan)
                    screen_y.append(np.nan)
                    screen_g.append(np.nan)
                    screen_eAng.append(np.nan)
            screen_x = np.array(screen_x)
            screen_y = np.array(screen_y)   
            screen_g = np.array(screen_g) 
            screen_eAng = np.array(screen_eAng)

            dS['screen_x'] = screen_x
            dS['screen_y'] = screen_y
            dS['screen_g'] = screen_g
            dS['screen_E'] = (screen_g-1)*self.m*m_e*c**2/e/1e6
            dS['screen_eAng'] = screen_eAng


    def calc_dispersion(self):
            
        for dS in self.screens:
            d_o = dS['origin']
            d_e = dS['end']
            d_s = dS['side']
            d_x_vec = (d_e-d_o)/np.linalg.norm(d_e-d_o)
            d_y_vec = (d_s-d_o)/np.linalg.norm(d_s-d_o)
            l = np.linalg.norm(d_e-d_o)
            spec_x = np.linspace(0,l,num=self.N_MeV,endpoint=True)

            N_g = self.N_g 
            x_mean = np.zeros(N_g)
            e_ang_mean = np.zeros(N_g)
            x_rms = np.zeros(N_g)
            x_min = np.zeros(N_g)
            x_max = np.zeros(N_g)
            
            for n in range(N_g):
                g_sel = dS['screen_g'] ==self.g[n]
                if np.any(g_sel):
                    x_sel = dS['screen_x'][g_sel]
                    x_mean[n] = np.mean(x_sel)
                    x_rms[n] = np.sqrt(np.mean((x_sel - x_mean[n])**2))
                    x_min[n] = np.min(x_sel)
                    x_max[n] = np.max(x_sel)
                    e_ang_mean[n] = np.mean(dS['screen_eAng'][g_sel])
                else:
                    x_mean[n]= np.nan
            

            iSel= np.isfinite(x_mean)
            b = np.sqrt(1-1./self.g[iSel]**2)
            E = (self.g[iSel]-1)*self.m*m_e*c**2/e/1e6
            E_x = interp1d(x_mean[iSel],E,kind='cubic',bounds_error=False)
            E_x_min = interp1d(x_min[iSel],E,kind='cubic',bounds_error=False)
            E_x_max = interp1d(x_max[iSel],E,kind='cubic',bounds_error=False)
            
            spec_MeV = E_x(spec_x)
            spec_MeV_min = E_x_min(spec_x)
            spec_MeV_max = E_x_max(spec_x)
            spec_eAng = interp1d(x_mean[iSel],e_ang_mean[iSel],
                                 kind='cubic',bounds_error=False)(spec_x)
            screen_b = np.sqrt(1-1./dS['screen_g']**2)
            
            dS['spec_x_mm'] = spec_x*1e3
            dS['spec_eAng'] = spec_eAng
            dS['spec_MeV'] = spec_MeV
            dS['spec_MeV_min'] = spec_MeV_min
            dS['spec_MeV_max'] = spec_MeV_max
            dS['spec_MeV_per_mm'] = np.gradient(spec_MeV)/np.gradient(spec_x*1e3)
            dS['spec_percentage_err'] = (spec_MeV_max-spec_MeV_min)/(2*spec_MeV)*100
            if self.plot_tracks:
                fig,axs = plt.subplots(1,3,figsize=(16/2.54,3/2.54),dpi=150)
                plt.subplots_adjust(wspace=0.3)

                axs[0].plot(dS['spec_x_mm'],dS['spec_MeV'],'k-')
                axs[0].fill_between(dS['spec_x_mm'],dS['spec_MeV_min'],dS['spec_MeV_max'],alpha=0.3,color='k')
                axs[0].set_xlabel('Screen position [mm]')
                axs[0].set_ylabel('Energy [MeV]')
                axs[0].set_title(dS['label'])
                axs[1].plot(dS['spec_MeV'],dS['spec_MeV_per_mm'])
                axs[1].set_xlabel('Energy [MeV]')
                axs[1].set_ylabel('Dispersion [MeV/mm]')

                axs[2].plot(dS['spec_MeV'],dS['spec_percentage_err'])
                axs[2].set_xlabel('Energy [MeV]')
                axs[2].set_ylabel(r'Energy error $\pm$ [$\%$]')
                plt.show()

    def initialise_particles(self,N_g=51,N_t=11 ,N_phi=1, dist_type='fan',
                             g_lims = [50,4000], div_mrad = 10, phi =  0):
        ''' Initialises a particle distribution for tracking through the spectrometer
                N_g (=51) is the number of distinct particle energies to use
                N_t (=11) is the number of angles polar angles to use
                N_phi (=1) is the number of azimuthal angles to use. If type='fan' then this is ignored as only one azimuthal angles is used.
                type='fan' is beam type either 'fan' or 'beam'
                g_lims =[50,4000] maximum and minimum gamma factor for the beam
                div_mrad =10 is the maximum divergence of the beam in mrad
                phi= 0 orientation of beam for fan mode. 0 means beam will be in x-y plane

            returns
                initial beam distribution with dimensions [6,N_p] where N_p is number of particles
                    elements are [x,y,z,px,py,pz] with x,y,z are positions in meters and px,py,pz are momenta normalised to m*c

        '''
        assert dist_type.lower() in ['fan','beam']
        # initialise particle energies (log-spaced)
        g = np.logspace(np.log10(g_lims[0]),np.log10(g_lims[1]),num=N_g,endpoint=True)
        if dist_type == 'fan':
            N_phi=1 # if beam type is fan, then there is only one azimuthal angle
            t = np.linspace(-div_mrad*1e-3,div_mrad*1e-3,N_t,endpoint=True)
            P,G,T = np.meshgrid(phi,g,t)
            b = g2b(G.flatten())
            px = b*G.flatten()*np.sqrt(1-(np.sin(T.flatten()))**2)
            py = b*G.flatten()*np.sin(T.flatten())*np.sin(P.flatten())
            pz = b*G.flatten()*np.sin(T.flatten())*np.cos(P.flatten())
            N_a = N_t # number of angles (number of particles per energy slice)

        elif dist_type =='beam':
            # code for evenly distributing points over the area of a circle
            y_dash,z_dash =  circular_pattern(N_t*N_phi,R=div_mrad)
            N_a = len(y_dash) # number of angles
            YD,G = np.meshgrid(y_dash,g)
            ZD,G = np.meshgrid(z_dash,g)
            r_dash = np.sqrt(Y**2+ZD**2).flatten()
            px = b*G.flatten()*np.sqrt(1-r_dash**2)
            py = b*G.flatten()*YD.flatten()
            pz = b*G.flatten()*ZD.flatten()


        N_p = len(G.flatten())# total number of particles
        # build particle distribution
        p_6d_0 = np.zeros((6,N_p))
        p_6d_0[3,:] = px
        p_6d_0[4,:] = py
        p_6d_0[5,:] = pz

        self.p_6d_0 = p_6d_0
        self.g=g
        self.G = G
        self.N_p = N_p
        self.N_g = N_g
        self.N_a = N_a

        return p_6d_0

    def modelSpectrometer(self,EM_function,screens=None,dx=1e-3,xLims=None,N_max = 10000,pLims=None,
                          calc_dispersion = True, plot_tracks=True):
        self.plot_tracks=plot_tracks
        self.screens = screens
        self.EM_function = EM_function
        self.xLims=xLims
        
        if self.p_6d_0 is None:
            self.initialise_particles()

        b  = np.sqrt(1-1/self.G**2)

        v_max = np.max(b)*c
        P_tracker = pp3d.ParticlePusher(self.p_6d_0,EM_function,dtMin=0,dtMax=dx/v_max,ppc=32,q=self.q,m=self.m)
        self.t,self.p_6d_t = P_tracker.trackParticles(xLims=self.xLims,pLims=pLims,N_max=N_max)
        if plot_tracks:
            self.plotTracks()
        if screens is not None:
            self.findScreenParticles()
        if calc_dispersion:
            self.calc_dispersion()

def load_radia_field(file_path):
    ''' inputs: file_path  path to csv file save by radia
            csv file has 6 headings and then a list of positions and fields
            
        returns: x,y,z positions in mm
                 Bx,By,Bz field in T
                 field data is shaped into 3D arrays
                 x,y,z axes are 1D arrays along dimensions 0,1,2 respectively
            '''
    df = pd.read_csv(file_path)
    Nx = len(np.unique(df['x'].values))
    Ny = len(np.unique(df['y'].values))
    Nz = len(np.unique(df['z'].values))
        
    X = np.reshape(df['x'].values,[Nx,Ny,Nz])
    Y = np.reshape(df['y'].values,[Nx,Ny,Nz])
    Z = np.reshape(df['z'].values,[Nx,Ny,Nz])
    Bx = np.reshape(df['Bx'].values,[Nx,Ny,Nz])
    By = np.reshape(df['By'].values,[Nx,Ny,Nz])
    Bz = np.reshape(df['Bz'].values,[Nx,Ny,Nz])

    x = X[:,0,0]
    y = Y[0,:,0]
    z = Z[0,0,:]
    return x,y,z,Bx,By,Bz