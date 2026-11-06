import numpy as np
from scipy.constants import c, m_e, e

class ParticlePusher:
    """Relativistic particle pusher using normalized momentum and SI fields."""

    def __init__(self,p_6d,EM_function,dtMin=0,dtMax=1,ppc=16,q=-1,m=1,pushMethod='boris'):
        """Set up a pusher for positions in metres and momentum in units of m*c.

        ``EM_function(t, positions)`` must return electric and magnetic fields
        in SI units as two arrays with shape ``(3, number_of_particles)``.
        """
        self.p_6d =  p_6d
        self.EM_function = EM_function
        self.dtMin=dtMin
        self.dtMax=dtMax
        self.ppc=ppc
        self.q=q
        self.m=m
        method = str(pushMethod).lower().replace('-', '_')
        if method not in {'boris', 'hc', 'boris_hc', 'higuera_cary'}:
            raise ValueError("pushMethod must be 'boris' or 'boris_hc'")
        self.pushMethod = 'boris_hc' if method in {'hc', 'boris_hc', 'higuera_cary'} else 'boris'
    
    def _dt_calc(self,B_vec,gamma,rqm):
        """Choose a shared step from the shortest relativistic gyroperiod.

        ``ppc`` is the target number of steps per gyroperiod. ``dtMin`` and
        ``dtMax`` clamp the result; zero-field steps use ``dtMax``.
        """
        B_mod = np.sqrt(np.sum(B_vec**2,axis=0))

        if np.max(B_mod)>0:
            gyro_period_over_2pi = np.divide(
                gamma*np.abs(rqm), B_mod,
                out=np.full_like(B_mod, np.inf, dtype=float), where=B_mod > 0)
            gyro_period = 2*np.pi*np.min(gyro_period_over_2pi)
            dt = np.clip(gyro_period/self.ppc,self.dtMin,self.dtMax)
        else:
            dt = self.dtMax
        if np.isnan(dt) or np.isinf(dt):
            dt = self.dtMax
        return dt

    def _gamma_calc(self,p):
        """Return Lorentz factors for normalized momenta shaped ``(3, N)``."""
        return np.sqrt(1+np.sum(p**2,axis=0))

    def _check_tracking_conditions(self,p_x,p_px,xLims,pLims,n,N_max):
        """Continue while below the step limit and at least one particle is in bounds."""
        p_inside = n<N_max
        if p_inside:
            for axis in range(3):
                if xLims[axis][0] is not None:
                    p_inside = np.logical_and(p_x[axis,:]>=xLims[axis][0],p_inside)
                if xLims[axis][1] is not None:
                    p_inside = np.logical_and(p_x[axis,:]<=xLims[axis][1],p_inside)

            for axis in range(3):
                if pLims[axis][0] is not None:
                    p_inside = np.logical_and(p_px[axis,:]>=pLims[axis][0],p_inside)
                if pLims[axis][1] is not None:
                    p_inside = np.logical_and(p_px[axis,:]<=pLims[axis][1],p_inside)
        return np.any(p_inside)

    def track_particles(self,xLims=[[None]*2]*3,pLims=[[None]*2]*3,N_max=10000):
        """Track the initial particle ensemble and save every sampled state.

        Parameters
        ----------
        xLims, pLims : sequence of three ``(min, max)`` pairs
            Optional position and normalized-momentum bounds. ``None`` means
            unbounded. Tracking continues while any particle remains inside.
        N_max : int
            Maximum number of pushes (the returned history also includes the
            initial state at ``t=0``).

        Returns
        -------
        t : ndarray, shape (n_steps + 1,)
            Sample times in seconds.
        p_6d_t : ndarray, shape (n_steps + 1, 6, n_particles)
            Position in metres followed by momentum in units of ``m*c``.
        """
        if self.EM_function is None:
            raise ValueError('EM_function must be provided')
        p_6d = np.asarray(self.p_6d, dtype=float)
        if p_6d.ndim != 2 or p_6d.shape[0] != 6 or p_6d.shape[1] == 0:
            raise ValueError('p_6d must have shape (6, number_of_particles) with at least one particle')
        if self.q == 0 or self.m <= 0:
            raise ValueError('Particle charge must be non-zero and mass must be positive')
        if self.ppc <= 0 or self.dtMin < 0 or self.dtMax <= 0 or self.dtMin > self.dtMax:
            raise ValueError('Require ppc > 0 and 0 <= dtMin <= dtMax with dtMax > 0')
        if int(N_max) != N_max or N_max < 0:
            raise ValueError('N_max must be a non-negative integer')
        xLims = [[None, None] for _ in range(3)] if xLims is None else xLims
        pLims = [[None, None] for _ in range(3)] if pLims is None else pLims

        p_m = self.m*m_e
        p_q = self.q*e
        rqm = p_m/p_q
        p_x = p_6d[0:3,:].copy()
        p_px = p_6d[3:6,:].copy()
        t = 0
        # Keep the initial state so first-step screen crossings can be found
        # and the time array always includes t=0.
        p_6d_t = [np.concatenate((p_x, p_px), axis=0)]
        t_list = [t]

        n = 0
        keepTracking = self._check_tracking_conditions(
            p_x, p_px, xLims, pLims, n, N_max)
        while keepTracking:
            
            gamma = self._gamma_calc(p_px)
            _, B_vec = self.EM_function(t,p_x)
            B_vec = np.asarray(B_vec, dtype=float)
            if B_vec.shape != p_x.shape or not np.all(np.isfinite(B_vec)):
                raise ValueError('EM_function must return a finite B field with shape (3, number_of_particles)')
            
            dt = self._dt_calc(B_vec,gamma,rqm)
            t_mid = t + dt/2
            p_x, p_px = self._boris_step(p_x,p_px,rqm,dt,t_mid)
            t += dt
            t_list.append(t)
            p_6d_t.append(np.concatenate((p_x,p_px),axis=0))
            n += 1
            keepTracking = self._check_tracking_conditions(
                p_x, p_px, xLims, pLims, n, N_max)
            
        p_6d_t = np.asarray(p_6d_t, dtype=float).reshape((-1, 6, p_6d.shape[1]))
        t = np.array(t_list)
        self.p_6d_t = p_6d_t
        self.t = t
        return t,p_6d_t


    def _boris_step(self,p_x,p_px,rqm,dt,t):
        """Advance one symmetric drift/kick/rotation/kick/drift step.

        Fields are sampled at the time and position midpoint. ``boris_hc``
        uses the Higuera–Cary implicit gamma correction for the magnetic
        rotation; the electric half-kicks and position drifts are shared.
        """
        # Relativistic Boris split: drift to the spatial midpoint, apply
        # fields at the temporal and spatial midpoint, then drift to the end.
        tem = 0.5 * dt / rqm

        # half step move
        g = np.tile(self._gamma_calc(p_px),(3,1))
        p_x = p_x + p_px*c/g*dt/2

        # get fields
        E_vec, B_vec = self.EM_function(t,p_x)
        E_vec = np.asarray(E_vec, dtype=float)
        B_vec = np.asarray(B_vec, dtype=float)
        if E_vec.shape != p_x.shape or B_vec.shape != p_x.shape:
            raise ValueError('EM_function must return E and B arrays with shape (3, number_of_particles)')
        if not np.all(np.isfinite(E_vec)) or not np.all(np.isfinite(B_vec)):
            raise ValueError('EM_function returned non-finite field values')

        # half acceleration with E
        p = p_px +E_vec*tem/c

        # compute inverse gamma and multiply by tem
        g = self._gamma_calc(p)
    
        if self.pushMethod == 'boris_hc':
            g2 = 1+np.sum(p**2,axis=0)
            
            tau = tem * B_vec
            tau2 = np.sum(tau**2,axis=0)
            sigma =g2 - tau2
           
            tau_dot_u = np.sum(tau*p,axis=0)

            g = np.sqrt(0.5*(sigma + np.sqrt(sigma**2 + 4.0 * (tau2 + tau_dot_u**2))))

        g_tem = np.tile(tem/g,(3,1))
        B_vec = B_vec*g_tem
        
        # half rotation with B
        pp = p + np.cross(p,B_vec,axis=0)
    
        # Second half rotation
        B_vec = B_vec*2/(1+np.sum(B_vec**2, axis=0, keepdims=True))
        p =  p + np.cross(pp,B_vec,axis=0)
    
        # second half acceleration
        p = p + E_vec*tem/c
        
        # final half move particles
        g = np.tile(self._gamma_calc(p),(3,1))
        v = c*p/g
        p_x =  p_x + v*dt/2
        p_px = p
        return p_x, p_px

        
        
