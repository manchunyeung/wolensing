import numpy as np
from scipy.fft import fftfreq
from scipy.fftpack import fft

def fitfuncF0(t, F0, a, c):
    return (F0 + 1 / (a * t ** 1 + c)) # .5

def fitfunc(t, a, b, c):
    return (1 + 1 / (a * t ** b + c))

def gridfromcorn(x1corn, x2corn, dx, num1, num2):
    x1s = np.linspace(x1corn, x1corn + dx * (num1 - 1), num=num1)
    x2s = np.linspace(x2corn, x2corn + dx * (num2 - 1), num=num2)
    X1, X2 = np.meshgrid(x1s, x2s)
    return X1, X2

def coswindowback(data, percent):
    """
    cosine apodization function for a percentage of points at
    the end of a timeseries of length len(data)
    
    """
    xback = np.linspace(0., -1., int(len(data) * percent / 100))
    back = [np.cos(np.pi * x / 2.) for x in xback]
    front = [1 for i in range(len(data) - len(back))]
    return np.concatenate((front, back)) * data

def F_tilde_extend(ts, F_tilde, **kwargs):

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
    dt = TimeStep/Tscale
    extend_num = int(extend_to_t / dt) + 0 - residual
    extension = extend_to_t - ts[-1]
    ts_extension = np.linspace(ts[-1] + dt, ts[-1] - dt + extension, extend_num)

    from bisect import bisect_left
    i = bisect_left(ts, fit_start)
    F0 = np.sqrt(kwargs['mu'])

    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(lambda t, a, c: fitfuncF0(t, F0, a, c), ts[i:], F_tilde[i:], p0=(.1, .1))
    F_tilde_extension = np.array([fitfuncF0(t, F0, *popt) for t in ts_extension])
    F_tilde_extended = np.concatenate((F_tilde, F_tilde_extension))
    ts_extended = np.concatenate((ts, ts_extension))
    
    return ts_extended, F_tilde_extended

def iwFourier(ts, Ft, type2=None): # has to be removed hardcoded part
    num = len(ts)
    dt = ts[1] - ts[0]
    # if type2:
    #     dt = 1e-6
    # else:
    #     dt = 1e-5
    ws = 2 * np.pi * fftfreq(num, dt)[:num // 2]
    Fw = np.conjugate(fft(Ft)[:num // 2] * (1.j) * ws * dt)
    return ws, Fw

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
