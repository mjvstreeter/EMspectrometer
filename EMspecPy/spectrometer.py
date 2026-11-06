from . import particle_pusher as pp3d
import numpy as np
from scipy.constants import c, m_e, e
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from .general_tools import circular_pattern
import warnings

def g2b(g):
    """Convert Lorentz factor(s) to speed fraction ``beta = v/c``."""
    return np.sqrt(1-1./g**2)
def b2g(b):
    """Convert speed fraction(s) ``beta = v/c`` to Lorentz factor(s)."""
    return 1/np.sqrt(1-b**2)

class EMspec:
    """Track a particle ensemble through user-supplied electromagnetic fields.

    Particle coordinates use SI metres and momentum is normalized to ``m*c``.
    The field callback must have the signature ``EM_function(t, positions)``
    and return electric and magnetic field arrays, each shaped ``(3, N)`` in
    SI units. Screens are rectangular planes defined by ``origin``, ``end``
    and ``side`` 3-vectors.
    """
    def __init__(self,EM_function=None,q=-1,m=1):
        """Create a spectrometer model.

        ``q`` and ``m`` are charge and mass in units of the electron charge
        magnitude and electron mass, respectively (for example, ``q=-1,
        m=1`` for electrons).
        """
        self.N_spec_points = 1000 # number of samples along each screen calibration
        self.p_6d_0 = None
        self.q = q # -1 for electrons
        self.m = m # 1 for electrons 1836.15 for protons
        self.EM_function = EM_function

    def plot_tracks(self,ax=None,plot_Bfield=True,plot_cbar=True):
        """Plot tracked x-y trajectories and optionally the z magnetic field.

        Spatial plot limits come from ``self.xLims`` when supplied, otherwise
        they are inferred from the trajectory history. Returns ``(axes,
        colorbar)``; the colorbar is ``None`` when no field image is drawn.
        """
        xLims = self.xLims
        N_p = self.N_p
        if ax is None:
            fig = plt.figure(figsize = (16/2.54,5/2.54),dpi=150)
            ax = plt.axes()
        cbh = None
        x_bounds = list(xLims[0]) if xLims is not None else [None, None]
        y_bounds = list(xLims[1]) if xLims is not None else [None, None]
        tracks = self.p_6d_t
        for bounds, values in ((x_bounds, tracks[:, 0, :]), (y_bounds, tracks[:, 1, :])):
            finite = values[np.isfinite(values)]
            if finite.size:
                pad = max((finite.max() - finite.min()) * 0.05, 1e-6)
                if bounds[0] is None:
                    bounds[0] = finite.min() - pad
                if bounds[1] is None:
                    bounds[1] = finite.max() + pad
        field_image = None
        if plot_Bfield and self.EM_function is not None and all(v is not None for v in x_bounds + y_bounds):
            Nx_img, Ny_img = 500, 250
            x_img = np.linspace(x_bounds[0], x_bounds[1], num=Nx_img)
            y_img = np.linspace(y_bounds[1], y_bounds[0], num=Ny_img)
            X, Y = np.meshgrid(x_img, y_img)
            XYZ = np.array((X.ravel(), Y.ravel(), np.zeros(X.size)))
            _, B_vec = self.EM_function(0, XYZ)
            B_img = np.reshape(B_vec[2, :], (Ny_img, Nx_img))
            if np.abs(np.max(B_img))>np.abs(np.min(B_img)): 
                cmap = 'gray_r'
            else:
                cmap = 'gray'
            field_image = ax.pcolormesh(x_img,y_img,B_img,cmap=cmap,vmin=np.min(B_img),vmax=np.max(B_img),shading='auto')
        if plot_cbar and field_image is not None:
            cbh = plt.colorbar(field_image,ax=ax)
        stride = max(1, int(N_p / max(self.N_g, 1)))
        pSel = range(0, N_p, stride)
        for p_ind in pSel:
            ax.plot(self.p_6d_t[:,0,p_ind],self.p_6d_t[:,1,p_ind],'r-',alpha=0.3)
        if self.screens is not None:
            for dS in self.screens:
                ax.plot((dS['origin'][0],dS['end'][0]),(dS['origin'][1],dS['end'][1]),'b-')
        if all(v is not None for v in x_bounds):
            ax.set_xlim(*x_bounds)
        if all(v is not None for v in y_bounds):
            ax.set_ylim(*y_bounds)
        ax.set_xlabel(r'$x$ [m]')
        ax.set_ylabel(r'$y$ [m]')
        return ax, cbh

    def find_screen_particles(self):
        """Find each particle's first in-bounds crossing of every screen.

        For each trajectory, signed distances at adjacent time samples are
        checked for a plane crossing and the state is linearly interpolated
        between those samples. Results are written into each screen dict as
        ``screen_x``, ``screen_y``, ``screen_g``, ``screen_E``,
        ``screen_eAng`` and ``p_6d_screen``. Misses are represented by NaNs.
        """
        p_6d_t = self.p_6d_t
        G = self.G
        N_p = self.N_p
        for dS in self.screens:
            d_o = dS['origin']
            d_e = dS['end']
            d_s = dS['side']
            x_vec = d_e-d_o
            y_vec = d_s-d_o
            l = np.linalg.norm(x_vec)
            y_len = np.linalg.norm(y_vec)
            if l == 0 or y_len == 0:
                raise ValueError('Screen origin, end, and side must define non-zero edges')
            d_x_vec = x_vec/l
            y_orth = y_vec - np.dot(y_vec, d_x_vec)*d_x_vec
            y_orth_len = np.linalg.norm(y_orth)
            if y_orth_len == 0:
                raise ValueError('Screen end and side must not lie on the same line')
            d_y_vec = y_orth/y_orth_len
            screenNormal = np.cross(d_x_vec, d_y_vec)
            width = np.dot(y_vec, d_y_vec)
            screen_x = []
            screen_y = []
            screen_eAng = []
            screen_g = []
            p_6d_screen = []
            for n in range(N_p):
                positions = p_6d_t[:, 0:3, n]
                momenta = p_6d_t[:, 3:6, n]
                signed_distance = (positions-d_o) @ screenNormal
                crosses_forward = (signed_distance[:-1] <= 0) & (signed_distance[1:] >= 0)
                crosses_backward = (signed_distance[:-1] >= 0) & (signed_distance[1:] <= 0)
                crossing = np.flatnonzero(crosses_forward | crosses_backward)
                hit = None
                for i in crossing:
                    delta = signed_distance[i+1] - signed_distance[i]
                    fraction = 0.0 if delta == 0 else -signed_distance[i]/delta
                    p_x_0 = positions[i] + fraction*(positions[i+1]-positions[i])
                    screen_x_0 = np.dot(p_x_0-d_o, d_x_vec)
                    screen_y_0 = np.dot(p_x_0-d_o, d_y_vec)
                    if 0 <= screen_x_0 <= l and 0 <= screen_y_0 <= width:
                        p_p_0 = momenta[i] + fraction*(momenta[i+1]-momenta[i])
                        hit = (p_x_0, p_p_0, screen_x_0, screen_y_0)
                        break
                if hit is None:
                    screen_x.append(np.nan)
                    screen_y.append(np.nan)
                    screen_g.append(np.nan)
                    screen_eAng.append(np.nan)
                    p_6d_screen.append(np.array([np.nan]*6))
                else:
                    p_x_0, p_p_0, screen_x_0, screen_y_0 = hit
                    p_mag = np.linalg.norm(p_p_0)
                    p_norm = p_p_0/p_mag if p_mag > 0 else np.full(3, np.nan)
                    screen_x.append(screen_x_0)
                    screen_y.append(screen_y_0)
                    screen_g.append(G.flatten()[n])
                    screen_eAng.append(np.arccos(np.clip(np.dot(p_norm, screenNormal), -1, 1)))
                    p_6d_screen.append(np.concatenate([p_x_0, p_p_0]))
            screen_x = np.array(screen_x)
            screen_y = np.array(screen_y)   
            screen_g = np.array(screen_g) 
            screen_eAng = np.array(screen_eAng)
            p_6d_screen = np.array(p_6d_screen)

            dS['p_6d_screen'] = p_6d_screen
            dS['screen_x'] = screen_x
            dS['screen_y'] = screen_y
            dS['screen_g'] = screen_g
            dS['screen_E'] = (screen_g-1)*self.m*m_e*c**2/e/1e6
            dS['screen_eAng'] = screen_eAng


    def calc_dispersion(self):
        """Build each screen's energy calibration and dispersion arrays.

        Per-energy mean hit position is inverted with a shape-preserving
        interpolator. The result is stored on each screen as ``spec_x_mm``,
        ``spec_MeV``, uncertainty bounds, angular mean, and ``spec_MeV_per_mm``.
        A screen with fewer than two hits at distinct energies emits a warning
        and receives NaN result arrays instead of raising during interpolation.
        """

        if self.screens is None:
            raise ValueError('No screens defined. Cannot calculate dispersion')
        for dS in self.screens:
            d_e = dS['end']
            l = np.linalg.norm(d_e-dS['origin'])
            if l == 0:
                raise ValueError('Screen origin and end must be different')
            spec_x = np.linspace(0,l,num=self.N_spec_points,endpoint=True)

            def set_nan_dispersion():
                nan_values = np.full(spec_x.shape, np.nan, dtype=float)
                dS['spec_x_mm'] = spec_x*1e3
                dS['spec_eAng'] = nan_values.copy()
                dS['spec_MeV'] = nan_values.copy()
                dS['spec_MeV_min'] = nan_values.copy()
                dS['spec_MeV_max'] = nan_values.copy()
                dS['spec_MeV_per_mm'] = nan_values.copy()
                dS['spec_percentage_err'] = nan_values.copy()
                if self.show_tracks:
                    fig, axs = plt.subplots(1, 3, figsize=(16/2.54, 3/2.54), dpi=150)
                    plt.subplots_adjust(wspace=0.3)
                    axs[0].plot(dS['spec_x_mm'], dS['spec_MeV'], 'k-')
                    axs[0].set_xlabel('Screen position [mm]')
                    axs[0].set_ylabel('Energy [MeV]')
                    axs[0].set_title(dS.get('label', 'Screen'))
                    axs[1].plot(dS['spec_MeV'], dS['spec_MeV_per_mm'])
                    axs[1].set_xlabel('Energy [MeV]')
                    axs[1].set_ylabel('Dispersion [MeV/mm]')
                    axs[2].plot(dS['spec_MeV'], dS['spec_percentage_err'])
                    axs[2].set_xlabel('Energy [MeV]')
                    axs[2].set_ylabel(r'Energy error $\pm$ [$\%$]')
                    plt.show()

            hit_mask = np.isfinite(dS['screen_g']) & np.isfinite(dS['screen_x'])
            hit_count = np.count_nonzero(hit_mask)
            distinct_energies = np.unique(dS['screen_g'][hit_mask]).size
            if hit_count < 2 or distinct_energies < 2:
                label = dS.get('label', 'Screen')
                warnings.warn(
                    f"{label}: only {hit_count} particle hit(s) across "
                    f"{distinct_energies} distinct energy point(s); at least two "
                    "hits at distinct energies are needed for dispersion. "
                    "Returning NaN dispersion arrays.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                set_nan_dispersion()
                continue

            N_g = self.N_g
            x_mean = np.zeros(N_g)
            e_ang_mean = np.zeros(N_g)
            x_min = np.zeros(N_g)
            x_max = np.zeros(N_g)
            
            for n in range(N_g):
                g_sel = dS['screen_g'] ==self.g[n]
                if np.any(g_sel):
                    x_sel = dS['screen_x'][g_sel]
                    x_mean[n] = np.mean(x_sel)
                    x_min[n] = np.min(x_sel)
                    x_max[n] = np.max(x_sel)
                    e_ang_mean[n] = np.mean(dS['screen_eAng'][g_sel])
                else:
                    x_mean[n]= np.nan
            

            iSel= np.isfinite(x_mean)
            if np.count_nonzero(iSel) < 2:
                label = dS.get('label', 'Screen')
                warnings.warn(
                    f'{label}: fewer than two energy slices have screen hits; '
                    'returning NaN dispersion arrays.',
                    RuntimeWarning,
                    stacklevel=2,
                )
                set_nan_dispersion()
                continue
            E = (self.g[iSel]-1)*self.m*m_e*c**2/e/1e6
            def interpolate_calibration(x, y, label):
                valid = np.isfinite(x) & np.isfinite(y)
                x_valid, y_valid = x[valid], y[valid]
                if x_valid.size < 2:
                    raise ValueError(f'{dS.get("label", "Screen")}: at least two valid {label} points are required')
                dx = np.diff(x_valid)
                if not (np.all(dx > 0) or np.all(dx < 0)):
                    raise ValueError(f'{dS.get("label", "Screen")}: {label} position must vary monotonically with energy')
                if np.all(dx < 0):
                    x_valid, y_valid = x_valid[::-1], y_valid[::-1]
                return PchipInterpolator(x_valid, y_valid, extrapolate=False)(spec_x)

            spec_MeV = interpolate_calibration(x_mean[iSel], E, 'mean calibration')
            if np.allclose(x_min[iSel], x_mean[iSel]) and np.allclose(x_max[iSel], x_mean[iSel]):
                spec_MeV_min = spec_MeV.copy()
                spec_MeV_max = spec_MeV.copy()
            else:
                spec_MeV_min = interpolate_calibration(x_min[iSel], E, 'lower envelope')
                spec_MeV_max = interpolate_calibration(x_max[iSel], E, 'upper envelope')
            spec_eAng = interpolate_calibration(x_mean[iSel], e_ang_mean[iSel], 'angle calibration')
            
            dS['spec_x_mm'] = spec_x*1e3
            dS['spec_eAng'] = spec_eAng
            dS['spec_MeV'] = spec_MeV
            dS['spec_MeV_min'] = spec_MeV_min
            dS['spec_MeV_max'] = spec_MeV_max
            dS['spec_MeV_per_mm'] = np.gradient(spec_MeV)/np.gradient(spec_x*1e3)
            dS['spec_percentage_err'] = np.divide(
                (spec_MeV_max-spec_MeV_min)*50, spec_MeV,
                out=np.full_like(spec_MeV, np.nan), where=spec_MeV != 0)
            if self.show_tracks:
                fig,axs = plt.subplots(1,3,figsize=(16/2.54,3/2.54),dpi=150)
                plt.subplots_adjust(wspace=0.3)

                axs[0].plot(dS['spec_x_mm'],dS['spec_MeV'],'k-')
                axs[0].fill_between(dS['spec_x_mm'],dS['spec_MeV_min'],dS['spec_MeV_max'],alpha=0.3,color='k')
                axs[0].set_xlabel('Screen position [mm]')
                axs[0].set_ylabel('Energy [MeV]')
                axs[0].set_title(dS.get('label', 'Screen'))
                axs[1].plot(dS['spec_MeV'],dS['spec_MeV_per_mm'])
                axs[1].set_xlabel('Energy [MeV]')
                axs[1].set_ylabel('Dispersion [MeV/mm]')

                axs[2].plot(dS['spec_MeV'],dS['spec_percentage_err'])
                axs[2].set_xlabel('Energy [MeV]')
                axs[2].set_ylabel(r'Energy error $\pm$ [$\%$]')
                plt.show()

    def initialise_particles(self,N_g=51,N_t=11 ,N_phi=1, dist_type='fan',
                             g_lims=(50,4000), div_rad=0.01, phi=0):
        """Create a mono-directional-energy grid with angular divergence.

        ``g_lims`` are minimum and maximum Lorentz factors, sampled
        logarithmically. ``div_rad`` is the maximum angular divergence in
        radians. ``fan`` varies one polar angle in the plane set by ``phi``;
        ``beam`` distributes ``N_t*N_phi`` directions across a circular cone.
        Positions start at the origin and momenta are normalized to ``m*c``.

        Returns the initial state array shaped ``(6, n_particles)`` and stores
        the energy/direction indexing arrays on this instance.
        """
        dist_type = str(dist_type).lower()
        if dist_type not in {'fan', 'beam'}:
            raise ValueError("dist_type must be 'fan' or 'beam'")
        if any(int(n) != n or n < 1 for n in (N_g, N_t, N_phi)):
            raise ValueError('N_g, N_t, and N_phi must all be positive integers')
        if len(g_lims) != 2 or min(g_lims) < 1 or g_lims[0] > g_lims[1]:
            raise ValueError('g_lims must contain two ordered gamma values >= 1')
        if not 0 <= div_rad < np.pi/2:
            raise ValueError('div_rad must be in the range [0, pi/2) radians')
        # initialise particle energies (log-spaced)
        g = np.logspace(np.log10(g_lims[0]),np.log10(g_lims[1]),num=N_g,endpoint=True)
        if dist_type == 'fan':
            N_phi=1 # if beam type is fan, then there is only one azimuthal angle
            t = np.linspace(-div_rad, div_rad, N_t, endpoint=True)
            P,G,T = np.meshgrid(phi,g,t)
            b = g2b(G.flatten())
            px = b*G.flatten()*np.sqrt(1-(np.sin(T.flatten()))**2)
            py = b*G.flatten()*np.sin(T.flatten())*np.sin(P.flatten())
            pz = b*G.flatten()*np.sin(T.flatten())*np.cos(P.flatten())
            N_a = N_t # number of angles (number of particles per energy slice)

        elif dist_type =='beam':
            # code for evenly distributing points over the area of a circle
            # The transverse components are direction cosines, so their
            # maximum radius is sin(maximum polar divergence).
            y_dash,z_dash = circular_pattern(N_t*N_phi,R=np.sin(div_rad))
            N_a = len(y_dash) # number of angles
            YD,G = np.meshgrid(y_dash,g)
            ZD,_ = np.meshgrid(z_dash,g)
            r_dash = np.sqrt(YD**2+ZD**2).flatten()
            b = g2b(G.flatten())
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
        self.N_t = N_t

        return p_6d_0

    def model_spectrometer(self,screens=None,dx=1e-3,xLims=None,N_max=10000,pLims=None,
                           ppc=32,calc_dispersion=True,show_tracks=True):
        """Run tracking, screen extraction, optional dispersion, and plotting.

        ``dx`` sets a spatially motivated upper time step using the initial
        maximum particle speed. ``ppc`` is the target number of steps per
        relativistic gyroperiod. Strong acceleration or rapidly changing
        fields may require a smaller ``dx``. ``xLims`` and ``pLims`` stop integration once
        every particle is outside the configured bounds or ``N_max`` pushes
        have run. Screen dictionaries require ``origin``, ``end`` and ``side``
        3-vectors; add a ``label`` for plot titles. Results are stored on this
        object and in the screen dictionaries.
        """
        self.show_tracks=show_tracks
        self.screens = screens
        
        self.xLims=xLims
        
        if self.p_6d_0 is None:
            self.initialise_particles()

        if not np.isfinite(dx) or dx <= 0:
            raise ValueError('dx must be finite and positive')
        b  = np.sqrt(1-1/self.G**2)

        v_max = np.max(b)*c
        dt_max = dx/v_max if v_max > 0 else dx/c
        P_tracker = pp3d.ParticlePusher(
            self.p_6d_0, self.EM_function, dtMin=0, dtMax=dt_max,
            ppc=ppc, q=self.q, m=self.m)
        self.t,self.p_6d_t = P_tracker.track_particles(xLims=self.xLims,pLims=pLims,N_max=N_max)
        if show_tracks:
            self.plot_tracks()
        if screens is not None:
            self.find_screen_particles()
        if calc_dispersion and screens is not None:
            self.calc_dispersion()

def load_radia_field(file_path, position_unit='m'):
    """Load a regularly ordered Radia CSV field grid.

    The CSV must contain ``x,y,z,Bx,By,Bz`` columns, with one row per grid
    point in x-major, y-middle, z-minor order. Radia coordinates are assumed
    to be in millimetres and are returned in metres by default; choose
    ``position_unit='mm'`` to preserve the file's coordinate units. Magnetic
    field components are returned in tesla as arrays shaped ``(Nx, Ny, Nz)``.
    """
    df = pd.read_csv(file_path)
    position_unit = str(position_unit).lower()
    if position_unit not in {'m', 'mm'}:
        raise ValueError("position_unit must be 'm' or 'mm'")
    x_mm = np.unique(df['x'].values)
    y_mm = np.unique(df['y'].values)
    z_mm = np.unique(df['z'].values)
    Nx, Ny, Nz = len(x_mm), len(y_mm), len(z_mm)
    if len(df) != Nx*Ny*Nz:
        raise ValueError('Radia CSV does not contain one row for every point on a complete 3D grid')
        
    X = np.reshape(df['x'].values,[Nx,Ny,Nz])
    Y = np.reshape(df['y'].values,[Nx,Ny,Nz])
    Z = np.reshape(df['z'].values,[Nx,Ny,Nz])
    Bx = np.reshape(df['Bx'].values,[Nx,Ny,Nz])
    By = np.reshape(df['By'].values,[Nx,Ny,Nz])
    Bz = np.reshape(df['Bz'].values,[Nx,Ny,Nz])

    expected_X, expected_Y, expected_Z = np.meshgrid(x_mm, y_mm, z_mm, indexing='ij')
    if not (np.array_equal(X, expected_X) and np.array_equal(Y, expected_Y) and np.array_equal(Z, expected_Z)):
        raise ValueError('Radia CSV rows must be ordered on a regular x-major, y-middle, z-minor grid')

    x = X[:,0,0]
    y = Y[0,:,0]
    z = Z[0,0,:]
    if position_unit == 'm':
        x, y, z = x*1e-3, y*1e-3, z*1e-3
    return x,y,z,Bx,By,Bz
