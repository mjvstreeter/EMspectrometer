# Author Matthew Streeter 2020
import matplotlib.pyplot as plt
import numpy as np
import pickle, os
from datetime import datetime
from glob import glob
import pandas as pd

# pickle wrappers
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as fid:
        return pickle.load(fid)


def glob_path(p):
    ''' Shortcut for using glob on a pathlib Path object
    '''
    return glob(str(p))


def d(x):
    ''' Function for calculating step size of assumed equally spaced array
    '''
    return np.abs(np.mean(np.diff(x)))


def smooth_gauss(x,y,sigma_x):
    X1,X2 = np.meshgrid(x,x)
    W = np.exp(-(X1-X2)**2/(2*sigma_x**2))
    y_smooth = np.nansum(W*y,axis=1)/np.nansum(W,axis=1)
    return y_smooth

def factorial(n):
    return np.double(np.math.factorial(n))

def normalise(x):
    return x/np.max(np.abs(x))