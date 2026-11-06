# EMspecPy

EMspecPy is a Python package for tracking relativistic particles through user-defined electric and magnetic fields, finding their intersections with finite detector screens, and estimating a screen's energy calibration and dispersion. It is intended for magnetic spectrometer simulation and experimental planning.

## Installation

Install from a local checkout with:

```bash
python -m pip install -e .
```

The package requires NumPy, SciPy, pandas, and Matplotlib. The worked example is [Examples/e_spec_example.ipynb](Examples/e_spec_example.ipynb).

## Units and particle state

Use SI units throughout the field callback and geometry:

- Position: metres
- Time: seconds
- Electric field: volts per metre
- Magnetic field: tesla
- Particle momentum: normalized to the particle rest mass times the speed of light, `m*c`
- Charge and mass passed to `EMspec`: multiples of the elementary charge magnitude and electron mass

The six-dimensional state is arranged as `(6, N)`: rows 0 to 2 are x, y, z positions, and rows 3 to 5 are normalized momentum components. The default `q=-1, m=1` describes electrons.

## Minimal example

The field callback receives the current time and a `(3, N)` position array. It must return electric and magnetic field arrays of the same shape.

```python
import numpy as np
from EMspecPy.spectrometer import EMspec

def em_field(time_s, position_m):
    electric = np.zeros_like(position_m)
    magnetic = np.zeros_like(position_m)
    inside = (position_m[0] >= 0.2) & (position_m[0] <= 0.5)
    magnetic[2, inside] = 0.8
    return electric, magnetic

spectrometer = EMspec(em_field)
initial_state = spectrometer.initialise_particles(
    N_g=31,                 # number of Lorentz-factor samples
    N_t=5,                  # number of directions in the fan or beam
    dist_type="fan",        # "fan" or "beam"
    g_lims=(100, 4000),     # minimum and maximum Lorentz factor gamma
    div_rad=2e-3,            # maximum divergence in radians
    phi=0.0,
)

screen = {
    "label": "Exit screen",
    "origin": np.array([1.0, -0.1, -0.05]),
    "end":    np.array([1.0,  0.1, -0.05]),
    "side":   np.array([1.0, -0.1,  0.05]),
}

spectrometer.model_spectrometer(
    screens=[screen],
    dx=1e-3,
    xLims=[[-0.1, 2.0], [-1.0, 1.0], [-1.0, 1.0]],
    calc_dispersion=True,
    show_tracks=False,
)

time_s = spectrometer.t
trajectory = spectrometer.p_6d_t  # (time, six-state components, particles)
hit_x_m = screen["screen_x"]
hit_energy_mev = screen["screen_E"]
energy_mev_per_mm = screen["spec_MeV_per_mm"]
```

The screen is a finite rectangle defined by three 3D points: `origin`, `end` (the other end of its length), and `side` (a point across its width). The screen axes are formed from those points. Screen-hit coordinates are measured from `origin` along the length and width axes. Hits outside the rectangle are treated as misses and recorded as NaN.

`g_lims` sets Lorentz factors, not kinetic energies. For a particle with mass `m` in electron-mass units, kinetic energy in MeV is `(gamma - 1) * m * 0.51099895`. The `div_rad` argument is in radians; for example, `0.01` rad is about 10 mrad.

## Tracking and outputs

`model_spectrometer()` stores the sampled times in `spectrometer.t` and the state history in `spectrometer.p_6d_t`. The history includes the initial state at `t=0`, followed by one state per push. `N_max` limits the number of pushes. Tracking stops when every particle leaves the supplied position or momentum bounds.

Each screen dictionary is populated with:

- `screen_x`, `screen_y`: local screen coordinates in metres
- `screen_g`, `screen_E`: Lorentz factor and kinetic energy in MeV
- `screen_eAng`: angle relative to the screen normal in radians
- `p_6d_screen`: position and normalized momentum at the crossing
- `spec_x_mm`, `spec_MeV`, `spec_MeV_min`, `spec_MeV_max`: energy calibration and angular-spread envelope
- `spec_MeV_per_mm`: derivative of calibrated energy with respect to screen position
- `spec_percentage_err`: half envelope width relative to the mean energy, in percent

The dispersion is calculated from the mean screen position at each energy. It requires hits at two or more distinct energies and a monotonic position-energy calibration. If a screen has too few hits, EMspecPy emits a warning and stores NaN dispersion arrays. Values outside the calibrated screen-position range are also NaN rather than extrapolated.

`show_tracks=True` plots the x-y trajectories, field image, and dispersion summary. For custom figures, use `show_tracks=False` and call `spectrometer.plot_tracks()` or plot the arrays stored on each screen directly.

## Particle pusher choice

`ParticlePusher` can be used directly from `EMspecPy.particle_pusher`. Its `pushMethod` accepts `"boris"` or `"boris_hc"` (also `"hc"` and `"higuera_cary"`). The latter applies the Higuera–Cary gamma correction to the relativistic magnetic rotation. The `EMspec.model_spectrometer()` convenience method currently uses the standard relativistic Boris method.

## Radia field grids

`load_radia_field(path)` reads a Radia CSV with `x`, `y`, `z`, `Bx`, `By`, and `Bz` columns. It expects one row per point in x-major, y-middle, z-minor order, and returns coordinate axes in metres by default with field arrays in tesla. Pass `position_unit="mm"` to return the original millimetre coordinate axes. The loader validates grid completeness and row ordering; it does not interpolate the field data. Build a field callback around the returned grid using an interpolation method appropriate for the data.

## Example notebook

The notebook demonstrates a vectorized hard-edged magnet callback, screen setup, particle initialization, trajectory plotting, and energy-dispersion plots. It is a workflow example rather than a validated magnet model; use a measured or simulated field map for quantitative predictions.
