import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from scipy.optimize import curve_fit
from fast_histogram import histogram1d
from scipy.fft import fftfreq
from scipy.fftpack import fft
import lensinggw.constants.constants as const
from tqdm import trange, tqdm

import sys
import os
path = os.getcwd()
dir = os.path.abspath(os.path.join(path, os.pardir))
sys.path.append(dir)

from utils.utils import *
# from wolensing.utils.utils import *

G = const.G  # gravitational constant [m^3 kg^-1 s^-2]
c = const.c  # speed of light [m/s]
M_sun = const.M_sun  # Solar mass [Kg]

def histogram_routine(lens_model_complete, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock, binnum,
                      binmin, binmax, Scale, kwargs_lens, y0, y1, dx):
    bincount = np.zeros(binnum)
    T = lens_model_complete.fermat_potential
    with tqdm(total = (Numblocks + 1)**2, desc = 'Integrating...') as pbar:
        for i in range(Numblocks + 1):
            for j in range(Numblocks + 1):
                if i in macroimindx[:,0] and j in macroimindx[:,1]:
                    pbar.update(1)
                    continue
                Nblock1 = Nblock
                Nblock2 = Nblock
                if i == Numblocks:
                    Nblock1 = Nresidue
                    if Nblock1 == 0:
                        pbar.update(1)
                        continue
                if j == Numblocks:
                    Nblock2 = Nresidue
                    if Nblock2 == 0:
                        pbar.update(1)
                        continue
                x1blockcorn = x1corn + i * Lblock
                x2blockcorn = x2corn + j * Lblock
                X1, X2 = gridfromcorn(x1blockcorn, x2blockcorn, dx, Nblock1, Nblock2)
                Ts = Scale ** (-2) * T(X1, X2, kwargs_lens, y0, y1)
                bincount += histogram1d(Ts, binnum, (binmin, binmax)) * dx ** 2
                pbar.update(1)
                del X1, X2, Ts
    return bincount

def integrator(lens_model_complete, kwargs_lens, **kwargs):
    kwargs_integrator = {'TimeStep': 0.0005,
                         'TimeMax': 100,
                         'TimeMin': -50,
                         'TimeLength': 10, # length in time considered after initial signal
                         'WindowSize': 15,
                         'PixelNum': 10000,
                         'PixelBlockMax': 2000,  # max number of pixels in a block
                         'WindowCenterX': 0,
                         'WindowCenterY': 0,
                         'InputScaled': True,
                         'ImageRa': np.array([]),
                         'ImageDec': np.array([])
                        }
    for key in kwargs_integrator.keys():
        if key in kwargs:
            value = kwargs[key]
            kwargs_integrator.update({key: value})

    kwargs_wolensing = {'LastImageT': 3,
                        'TExtend': 100,
                        'source_pos_x': .1,
                        'source_pos_y': 0.,
                        'theta_E': 1e-10}
    for key in kwargs_wolensing.keys():
        if key in kwargs:
            value = kwargs[key]
            kwargs_wolensing.update({key: value})
    T = lens_model_complete.fermat_potential
    binmax0 = kwargs_integrator['TimeMax']
    binmin = kwargs_integrator['TimeMin']
    binlength = kwargs_integrator['TimeLength']
    binwidth = kwargs_integrator['TimeStep']
    N = kwargs_integrator['PixelNum']
    Nblock = kwargs_integrator['PixelBlockMax']
    binnum = int((binmax0 - binmin) / binwidth) + 1
    binnumlength = int(binlength / binwidth)

    thetaE = kwargs_wolensing['theta_E']
    if kwargs_integrator['InputScaled']:
        Scale = kwargs_wolensing['theta_E']
    else:
        Scale = 1.
    y0 = Scale * kwargs_wolensing['source_pos_x']
    y1 = Scale * kwargs_wolensing['source_pos_y']

    binmax = binmin + binwidth * (binnum + 1)
    bins = np.linspace(binmin, binmax, binnum)

    x1cen = kwargs_integrator['WindowCenterX']
    x2cen = kwargs_integrator['WindowCenterY']
    L = Scale * kwargs_integrator['WindowSize']
    dx = L / (N - 1)

    x1corn = x1cen - L / 2
    x2corn = x2cen - L / 2
    Lblock = Nblock * dx
    Numblocks = N // Nblock
    Nresidue = N % Nblock
    imagesra = kwargs_integrator['ImageRa']
    imagesdec = kwargs_integrator['ImageDec']
    imindx1 = (imagesra - x1corn) // Lblock
    imindx2 = (imagesdec - x2corn) // Lblock
    imindxzip = np.unique(np.column_stack((imindx1, imindx2)), axis = 0)
    zoomblockcorn1 = imindx1*Lblock + x1corn
    zoomblockcorn2 = imindx2*Lblock + x2corn
    zoomblockcornzip = np.unique(np.column_stack((zoomblockcorn1, zoomblockcorn2)), axis = 0)

    bincount = histogram_routine(lens_model_complete, Numblocks, np.array([[None, None]]), Nblock, Nresidue, x1corn, x2corn, Lblock, binnum,
                      binmin, binmax, thetaE, kwargs_lens, y0, y1, dx)

    zoomN = N
    zoomdx = Lblock/(N+1)
    zoomNumblocks = zoomN // Nblock
    zoomNresidue = zoomN % Nblock
    zoomLblock = Nblock * zoomdx
    for (zoomblockcorn1, zoomblockcorn2) in zoomblockcornzip:
        bincount += histogram_routine(lens_model_complete, zoomNumblocks, np.array([[None, None]]), Nblock, zoomNresidue, zoomblockcorn1, zoomblockcorn2, zoomLblock, binnum,
                      binmin, binmax, thetaE, kwargs_lens, y0, y1, zoomdx)

    bincountback = np.trim_zeros(bincount, 'f')
    bincountfront = np.trim_zeros(bincount, 'b')
    fronttrimmed = len(bincount) - len(bincountback)
    backtrimmed = len(bincount) - len(bincountfront) + 1
    F_tilde = bincount[fronttrimmed:-backtrimmed] / (2 * np.pi * binwidth) / thetaE ** 2
    ts = bins[fronttrimmed:-backtrimmed] - bins[fronttrimmed]
    return ts[:binnumlength], F_tilde[:binnumlength]  # , Tmax

def amplification_factor_fd(lens_model_list, args, kwargs_lens, **kwargs):
    kwargs_wolensing = {'FrequencyScaled': True}
    for key in kwargs_wolensing.keys():
        if key in kwargs:
            value = kwargs[key]
            kwargs_wolensing.update({key: value})
    Tscale = kwargs['Tscale']
    lens_model_complete = LensModel(lens_model_list = lens_model_list)
    ts, F_tilde= integrator(lens_model_complete, kwargs_lens, **kwargs)
    if args.type2:
        ws, Fw = iwFourier(ts * Tscale, F_tilde, args) 
        fs = ws/(2*np.pi)
        peak = np.where(F_tilde == np.amax(F_tilde))
        index = int(peak[0])
        Tds = 5
        tdiff = ts[index]*Tscale-5
        overall_phase = np.exp(-1 * 2 * np.pi * 1j * (Tds+tdiff) * fs)
        Fw *= overall_phase
    else:
        dt = 1e-5
        print(ts, F_tilde)
        ts_extended, F_tilde_extended = F_tilde_extend(ts, F_tilde, args, **kwargs)
        F_tilde_apodized = coswindowback(F_tilde_extended, 50)
    # np.savetxt('./data/test/test_ws_100.00000_1.00000.txt', ts_extended*Tscale)
    # np.savetxt('./data/test/test_Fws_100.00000_1.00000.txt', F_tilde_apodized)
    ws, Fw = iwFourier(ts_extended*Tscale, F_tilde_apodized, args)
    print(ws, Fw)

    from bisect import bisect_left

    i = bisect_left(ws, 2*np.pi*2000)

    return ws[:i], Fw[:i]
