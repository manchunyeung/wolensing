import matplotlib

import numpy as np
import matplotlib.pyplot as plt
from lenstronomy.LensModel.lens_model import LensModel
from matplotlib.patches import FancyArrowPatch

from wolensing.lensmodels.derivative import Gradient_Td


def plot_contour(ax, lens_model_list, window_center1, window_center2, window_length, kwargs_lens_list, beta0, beta1, Img_ra=None, Img_dec=None, crit_ra=None, crit_dec=None, caustic_ra=None, caustic_dec=None,
                 T0 = 0, Tfac = 1, contour = 50, micro = False, gradient=False):
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
    CS = ax.contour(X1s, X2s, Ts, contour)
    ax.clabel(CS, CS.levels)
    if gradient:
        
        kwargs_macro = {'source_pos_x': beta0, 'source_pos_y': beta1}
        td_x, td_y = Gradient_Td(lens_model_list, X1s, X2s, kwargs_lens_list, kwargs_macro)
        
        norm = np.linalg.norm(np.array((td_x, td_y)), axis=0)
        u = td_x / norm
        v = td_y / norm
        
        print(u, v, 'arrow')
        step = 20
        scale = 5e10
        ax.quiver(X1s[::step, ::step], X2s[::step, ::step], u[::step, ::step], v[::step, ::step], units='xy', scale=scale, color='gray')
        # arrow = FancyArrowPatch((35, 35), (35+34*0.2, 35+0), arrowstyle='simple',
                            # color='r', mutation_scale=10)  
        # ax.add_patch(arrow)
    if crit_ra is not None and crit_dec is not None: 
        ax.scatter(crit_ra, crit_dec, markersize=2, color='red')
    ax.scatter(window_center1, window_center2)
    if Img_ra is not None:
        ax.scatter(Img_ra[:], Img_dec[:])
    return ax
