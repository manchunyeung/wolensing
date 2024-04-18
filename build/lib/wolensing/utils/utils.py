import numpy as np
from scipy.fft import fftfreq
from scipy.fftpack import fft
import jax.numpy as jnp
from jax import jit
from functools import partial

from wolensing.lensmodels.hessian import Hessian_Td

def fitfuncF0(t, F0, a, c):
    '''
    Fitting function of power law

    :param t: independent variable.
    :param F0: costant that the power law converges to.
    :param a: parameter multiplying the variable.
    :param c: exponent parameter to the variable.
    :return: fitted power law function. 
    '''
    return (F0 + 1 / (a * t ** 1 + c)) # .5

def fitfunc(t, a, b, c):
    '''
    Fitting function of power law

    :param t: independent variable.
    :param a: parameter multiplying the variable.
    :param c: exponent parameter to the variable.
    :return: fitted power law function. 
    '''
    return (1 + 1 / (a * t ** b + c))

def gridfromcorn(x1corn, x2corn, dx, num1, num2):
    '''
    Construct blocks for integration.
    
    :param x1corn: x-coordinate of the left bottom corner of the block.
    :param x2corn: y-coordinate of the left bottom corner of the block.
    :param dx: steps of integration window.
    :param num1: number of points on the horizontal side of the box.
    :param num2: number of points on the vertical side of the box.
    :return: numpy meshgrid of the box.
    '''
    x1s = np.linspace(x1corn, x1corn + dx * (num1 - 1), num=num1)
    x2s = np.linspace(x2corn, x2corn + dx * (num2 - 1), num=num2)
    X1, X2 = np.meshgrid(x1s, x2s)
    return X1, X2

def coswindowback(data, percent):
    """
    Cosine apodization function for a percentage of points at
    the end of a timeseries of length len(data)

    :param data: data to apodize.
    :param percent: percentage of data being apodized.
    :return: apodized results.
    """
    xback = np.linspace(0., -1., int(len(data) * percent / 100))
    back = [np.cos(np.pi * x / 2.) for x in xback]
    front = [1 for i in range(len(data) - len(back))]
    return np.concatenate((front, back)) * data

def F_tilde_extend(ts, F_tilde, kwargs_macro, kwargs):
    '''
    Extend the function with fitted power law.

    :param ts: time series.
    :param F_tilde: data.
    :param kwargs: arguments for integration.
    :return: extended ts and F_tilde.
    '''
    extend_to_t = kwargs['TExtend']
    Tscale = kwargs['Tscale']
    TimeLength = kwargs['TimeLength']
    TimeStep = kwargs['TimeStep']
    expected_num = TimeLength / TimeStep
    num = len(ts)
    residual = 0
    if num != expected_num:
        residual = num - expected_num
        residual = int(residual)
    fit_start = kwargs['LastImageT'] + kwargs['Tbuffer']
    dt = TimeStep 
    extend_num = int(extend_to_t / dt) + 0 - residual
    extension = extend_to_t - ts[-1]
    ts_extension = np.linspace(ts[-1] + dt, ts[-1] - dt + extension, extend_num)

    from bisect import bisect_left
    i = bisect_left(ts, fit_start)
    F0 = np.sqrt(kwargs_macro['mu'])
    # F0 = np.sqrt(1)

    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(lambda t, a, c: fitfuncF0(t, F0, a, c), ts[i:], F_tilde[i:], p0=(.1, .1))
    F_tilde_extension = np.array([fitfuncF0(t, F0, *popt) for t in ts_extension])
    F_tilde_extended = np.concatenate((F_tilde, F_tilde_extension))
    ts_extended = np.concatenate((ts, ts_extension))
    
    return ts_extended, F_tilde_extended

def iwFourier(ts, Ft, dt=1e-6):
    '''
    Fourier transform the time series data.

    :param ts: time series
    :param Ft: data
    :param type2: boolean, if True, use the appropriate time step for type 2 image.
    :return: sampling frequency and transformed data in frequency domain.
    '''

    num = len(ts)
    ws = 2 * np.pi * fftfreq(num, dt)[:num // 2]
    Fw = np.conjugate(fft(Ft)[:num // 2 ] * (1.j) * ws * dt)
    print('total time', num*dt)
    return ws, Fw

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def Morse_indices(lens_model_list, xs, ys, kwargs):
    '''
    :param lens_model_list: list of lens models.
    :param xs: x-coordinates of position on lens plane.
    :param ys: y-coordinates of position on lens plane.
    :kwargs: arguemnts for the lens models.
    :return: morse indices of the input positions.
    '''
    
    ns = np.zeros(xs.shape)
    
    for i, (x, y) in enumerate(zip(xs, ys)):
        hessian = Hessian_Td(lens_model_list, x, y, kwargs)
        detH = hessian[0]*hessian[1] - hessian[2]**2
        
        if detH<0:
            ns[i] = 0.5
        elif detH>0 and hessian[0]>0 and hessian[1]>0:
            ns[i] = 0
        elif detH>0 and hessian[0]<0 and hessian[1]<1:
            ns[i] = 1
        else:
            raise Exception('Inconclusive Hessian Matrix.')
             
    return ns
