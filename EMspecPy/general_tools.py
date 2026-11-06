# Author Matthew Streeter 2020
import numpy as np
import math
import pickle
from glob import glob

# pickle wrappers
def save_object(obj, filename):
    """Pickle ``obj`` to ``filename`` using the highest protocol."""
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    """Load and return a Python object from a pickle file."""
    with open(filename, 'rb') as fid:
        return pickle.load(fid)


def glob_path(p):
    """Return paths matching pattern ``p`` as strings (also accepts Path)."""
    return glob(str(p))


def d(x):
    """Estimate the absolute spacing of an equally spaced 1D array."""
    return np.abs(np.mean(np.diff(x)))


def smooth_gauss(x,y,sigma_x):
    """Gaussian-weight smooth ``y`` over coordinate ``x`` with width ``sigma_x``."""
    X1,X2 = np.meshgrid(x,x)
    W = np.exp(-(X1-X2)**2/(2*sigma_x**2))
    y_smooth = np.nansum(W*y,axis=1)/np.nansum(W,axis=1)
    return y_smooth

def factorial(n):
    """Return ``n!`` as a NumPy double; ``n`` must be a non-negative integer."""
    return np.double(math.factorial(n))

def normalise(x):
    """Scale an array so its largest absolute element is one."""
    scale = np.max(np.abs(x))
    if scale == 0:
        raise ValueError('Cannot normalise an all-zero array')
    return x/scale


def circular_pattern(N, R=1):
    """Return exactly ``N`` approximately uniform points in a disk of radius R."""
    if int(N) != N or N < 0:
        raise ValueError('N must be a non-negative integer')
    if not np.isfinite(R) or R < 0:
        raise ValueError('R must be finite and non-negative')
    if N == 0:
        return np.empty(0), np.empty(0)
    if N == 1:
        return np.zeros(1), np.zeros(1)

    # Vogel spiral: deterministic, avoids global RNG state, and has exact size.
    i = np.arange(N, dtype=float)
    radius = R*np.sqrt((i + 0.5)/N)
    angle = i*np.pi*(3-np.sqrt(5))
    return radius*np.sin(angle), radius*np.cos(angle)
