import numpy as np
from scipy.constants import c, m_e, e

class ParticlePusher:
    """Relativistic Higuera–Cary pusher using normalized momentum and SI fields."""

    def __init__(self,p_6d,EM_function,dtMin=0,dtMax=1,ppc=16,q=-1,m=1):
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
        if not np.all(np.isfinite([self.q, self.m, self.ppc, self.dtMin, self.dtMax])):
            raise ValueError('q, m, ppc, dtMin, and dtMax must be finite')
        if self.q == 0 or self.m <= 0:
            raise ValueError('Particle charge must be non-zero and mass must be positive')
        if self.ppc <= 0 or self.dtMin < 0 or self.dtMax <= 0 or self.dtMin > self.dtMax:
            raise ValueError('Require ppc > 0 and 0 <= dtMin <= dtMax with dtMax > 0')
        if int(N_max) != N_max or N_max < 0:
            raise ValueError('N_max must be a non-negative integer')
        if not np.all(np.isfinite(p_6d)):
            raise ValueError('p_6d must contain only finite values')
        xLims = [[None, None] for _ in range(3)] if xLims is None else xLims
        pLims = [[None, None] for _ in range(3)] if pLims is None else pLims
        for name, limits in (('xLims', xLims), ('pLims', pLims)):
            if len(limits) != 3 or any(len(axis_limits) != 2 for axis_limits in limits):
                raise ValueError(f'{name} must contain three (min, max) pairs')
            for lower, upper in limits:
                if (lower is not None and not np.isfinite(lower)) or (upper is not None and not np.isfinite(upper)):
                    raise ValueError(f'{name} bounds must be finite or None')
                if lower is not None and upper is not None and lower > upper:
                    raise ValueError(f'{name} lower bounds must not exceed upper bounds')

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
            p_x, p_px = self._higuera_cary_step(p_x,p_px,rqm,dt,t_mid)
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


    def _higuera_cary_step(self,p_x,p_px,rqm,dt,t):
        """Advance one symmetric drift/kick/rotation/kick/drift step.

        Fields are sampled at the time and position midpoint. The magnetic
        rotation uses the Higuera–Cary gamma correction.
        """
        half_qmdt = 0.5 * dt / rqm

        # Drift to the spatial midpoint using the incoming momentum.
        gamma_n = np.tile(self._gamma_calc(p_px), (3, 1))
        x_mid = p_x + p_px*c/gamma_n*dt/2

        # Sample fields once at the time and position midpoint.
        E_vec, B_vec = self.EM_function(t, x_mid)
        E_vec = np.asarray(E_vec, dtype=float)
        B_vec = np.asarray(B_vec, dtype=float)
        if E_vec.shape != x_mid.shape or B_vec.shape != x_mid.shape:
            raise ValueError('EM_function must return E and B arrays with shape (3, number_of_particles)')
        if not np.all(np.isfinite(E_vec)) or not np.all(np.isfinite(B_vec)):
            raise ValueError('EM_function returned non-finite field values')

        # Apply the first electric half-kick, then solve the Higuera–Cary
        # quadratic for the gamma used in the magnetic rotation.
        u_minus = p_px + E_vec*half_qmdt/c
        tau = half_qmdt*B_vec
        tau_sq = np.sum(tau**2, axis=0)
        sigma = 1 + np.sum(u_minus**2, axis=0) - tau_sq
        tau_dot_u = np.sum(tau*u_minus, axis=0)
        gamma_rotation = np.sqrt(
            0.5*(sigma + np.sqrt(sigma**2 + 4*(tau_sq + tau_dot_u**2))))

        t_vec = tau/np.tile(gamma_rotation, (3, 1))
        u_prime = u_minus + np.cross(u_minus, t_vec, axis=0)
        s_vec = 2*t_vec/(1 + np.sum(t_vec**2, axis=0, keepdims=True))
        u_plus = u_minus + np.cross(u_prime, s_vec, axis=0)

        # Finish the electric kick and drift to the end of the step.
        p_next = u_plus + E_vec*half_qmdt/c
        gamma_next = np.tile(self._gamma_calc(p_next), (3, 1))
        x_next = x_mid + c*p_next/gamma_next*dt/2
        return x_next, p_next

        
        
