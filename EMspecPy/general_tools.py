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


def circular_pattern(N,R=1):

    N_r = round(np.pi*np.sqrt(N)/6) # change divider to change radial vs azimuthal density
    print(N_r)
    p_a = np.arange(1,N_r+1)
    r_a = (p_a/(N_r))**1*R
    p_a = p_a**1 # change exponent to change distribution

    # normalise and make integers with some randomness
    p_a = N/np.sum(p_a)*p_a
    p_a = (np.floor(p_a)+ (np.random.rand(len(p_a))<=np.mod(p_a,1))).astype(int)
    p_a = p_a.astype(int)
    # initial angle for each layer
    t_a = np.linspace(0,2*np.pi,N_r,endpoint=False)*1
    t_a = np.random.choice(t_a,N_r,replace=False)*1.0 # spiral, random or fixed pattern

    # build particles
    r = []
    x = np.zeros(np.sum(p_a)+1)
    y = np.zeros(np.sum(p_a)+1)
    k = 0
    for n in range(N_r):
        r =r +[r_a[n]]*p_a[n]
        
        theta = np.linspace(0,2*np.pi,p_a[n],endpoint=False)+t_a[n]
        x[k:(k+p_a[n])]=r_a[n]*np.sin(theta)
        y[k:(k+p_a[n])]=r_a[n]*np.cos(theta)
        k = k +p_a[n]
    x = np.array(x)
    y = np.array(y)
    return x,y