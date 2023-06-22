import matplotlib

import numpy as np
import matplotlib.pyplot as plt
from lenstronomy.LensModel.lens_model import LensModel

def plot_contour(ax, lens_model_list, window_center1, window_center2, window_length, kwargs_lens_list, beta0, beta1, Img_ra, Img_dec,
                 T0 = 0, Tfac = 1, micro = False):
    """
    Given a square window, plot the time delay contour and the positions of lensed images on the lens plane

    :param ax: matplotlib axis instance
    :param lens_model_list: a list of lens models
    :param window_center1: right ascension of the center of the window
    :param window_center2: declination of the center of the window
    :param window_length: length of the window
    :param kwargs_lens_list: kwargs_lens_list
    :param beta0: right ascension of the source 
    :param beta1: declination of the source
    :param Img_ra: right ascension of the images
    :param Img_dec: declination of the images
    :param T0: the time delay at the window center
    :param Tfac: factor multiplying the fermat potential 
    :param micro: boolean; if True, plot the microimages
    :return: a plot of time delay contour and images around the center
    """

    print(Img_ra, Img_dec, 'Img_pos')
    lens_model_complete = LensModel(lens_model_list=lens_model_list)

    # define the window
    win_low1 = window_center1 - window_length / 2
    win_low2 = window_center2 - window_length / 2
    win_hi1 = win_low1 + window_length
    win_hi2 = win_low2 + window_length

    # Compute the time delay of points in the window
    num = 1000 # number of grids between the limits, total number of pixels = num*num
    x1s = np.linspace(win_low1, win_hi1, num)
    x2s = np.linspace(win_low2, win_hi2, num)
    X1s, X2s = np.meshgrid(x1s,x2s)
    Ts = Tfac * lens_model_complete.fermat_potential(X1s,X2s,kwargs_lens_list, beta0, beta1)
    Ts -= T0

    # Plot the figure
    CS = ax.contour(X1s, X2s, Ts, 50)
    ax.clabel(CS, CS.levels)
    ax.scatter(window_center1, window_center2)
    ax.scatter(Img_ra[:], Img_dec[:])
    return ax
