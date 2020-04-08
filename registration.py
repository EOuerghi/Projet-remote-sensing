"""
registration tools

Copyright (C) 2017-2018, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
"""

from __future__ import print_function
import numpy as np

import cv2


def symmetrize(x):
    xx = np.hstack((x[:,0:], x[:,::-1]))
    return np.vstack((xx[0:,:], xx[::-1,:]))
    
def unsymmetrize(x):
    sz=x.shape
    return x[0:int(sz[0]/2), 0:int(sz[1]/2)]


def fft_shift(im, dx, dy):
    '''
    Subpixel shift using Fourier interpolation
    Uses symmetrized periodization to avoid boundary artifacts 
    
    Args:
        im   : numpy array containing the input image
        dx,dy: subpixel shift to be applied
    Returns:
        a numpy array the resampled shifted image 
    '''
    import pyfftw
    import numpy as np

    im = symmetrize(im)
    imF = np.fft.fft2(im)
    X, Y = np.meshgrid(np.fft.fftfreq(im.shape[1]), np.fft.fftfreq(im.shape[0]))
    phi =  2j*np.pi*( X*dx + Y*dy ) 
    imFshift = imF * np.exp(-phi)
    return  unsymmetrize( np.real(np.fft.ifft2(imFshift)))

def hamming_win(im):
    '''
    generates the hamming window with the shape of im
    '''
    ny,nx = im.shape[0:2]
    return im*np.sqrt(np.outer(np.hamming(ny),np.hamming(nx)))


def hanning_win(im):
    '''
    generates the hanning window with the shape of im
    '''
    ny,nx = im.shape[0:2]
    return im*np.sqrt(np.outer(np.hanning(ny),np.hanning(nx)))

def cloud_simulator(im, std=50, M=.3,alpha=1):
    '''
    Add clouds to your image
    
    Arguments:
        im : numpy array containing the input image
        std: standard deviation of the gaussian filter used to produce the clouds
        M: saturation level 
        alpha: strenght of the cloud
        
    Return:
        a numpy array the cloudy image
    '''
    import scipy.ndimage
    import numpy as np
    r = np.random.randn(*im.shape)
    blob = scipy.ndimage.filters.gaussian_filter(r,std)
    blob = np.clip(blob, 0, np.max(blob)*M)
    blob /= (np.max(blob))
    blob = blob*alpha
    return im*(1.-blob) + np.max(im)*blob


def local_equalization (ref , tgt, sz=35):
    '''
    local equalization
    '''
    import cv2 
        
    if not (ref.shape == tgt.shape):
        return tgt
    
    muref = cv2.GaussianBlur(ref,(sz,sz),sz*.8)
    mutgt = cv2.GaussianBlur(tgt,(sz,sz),sz*.8)

    varref = cv2.GaussianBlur((ref-muref)**2,(sz,sz),sz*.8)
    vartgt = cv2.GaussianBlur((tgt-mutgt)**2,(sz,sz),sz*.8)
    
    sigref = np.sqrt(varref)
    sigtgt = np.sqrt(vartgt)
    
    return (tgt - mutgt)/sigtgt*sigref + muref 



def gaussian_filter(r,std):
    '''
    gaussian filter the image r
    '''
    import scipy.ndimage
    return scipy.ndimage.filters.gaussian_filter(r,std)



def zoom(img, z):
    """
    zoom image by factor z
    """
    from scipy import ndimage
    import warnings
    with warnings.catch_warnings():  # noisy warnings may occur here
        warnings.filterwarnings("ignore", category=UserWarning)
        return ndimage.interpolation.zoom(img, [z, z], output=None, order=5,
                                          mode='constant', cval=0.0, prefilter=True)
