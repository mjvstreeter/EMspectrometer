import ParticlePusher3D as pp3d
import numpy as np
import matplotlib
import scipy.constants as C
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def EM_function(p_x):
    E_vec = np.zeros_like(p_x)
    B_vec = np.zeros_like(p_x)
    B_vec[2,:] = 1.0
    return E_vec, B_vec

g_0 = 10000
p_6d_0 = np.zeros((6,2))
p_6d_0[3,:] = g_0
P_tracker = pp3d.ParticlePushObj(p_6d_0,EM_function,dtMin=0.01e-3/3e8,dtMax=1,ppc=32,q=-1,m=1)

t,p_6d_t = P_tracker.trackParticles(N_max=32*10)


plt.figure()
plt.plot(p_6d_t[:,0,0],p_6d_t[:,1,0],'k',label='Boris pusher',alpha=0.2)
theta = np.linspace(0,2*np.pi,num=1000)
r = g_0*C.m_e/C.e*C.c
plt.plot(r*np.sin(theta),r*np.cos(theta)+r,'r--',label='Analytical')
plt.xlabel(r'$x$ [m]')
plt.ylabel(r'$y$ [m]')
plt.legend()
plt.show()
