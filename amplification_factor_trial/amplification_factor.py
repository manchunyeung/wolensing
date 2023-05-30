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
from utils.histogram import histogram_routine

G = const.G  # gravitational constant [m^3 kg^-1 s^-2]
c = const.c  # speed of light [m/s]
M_sun = const.M_sun  # Solar mass [Kg]

class amplification_factor_fd(object):

    def __init__(self, lens_model_list=None, kwargs_lens=None, kwargs_initial=None, kwargs_integrator=None, embedded=False, type2=False):
        """

        :param lens_model_list: list of lens models 
        :param args: the macroimage in which microlensing takes place
        :param kwargs_lens: arguments for integrating the diffraction integral
        """



        # kwargs_wolensing = {'FrequencyScaled': True}
        # for key in kwargs_wolensing.keys():
        #     if key in kwargs:
        #         value = kwargs[key]
        #         kwargs_wolensing.update({key: value})
        # Tscale = kwargs['Tscale']

        self._kwargs_lens = kwargs_lens
        self._kwargs_initial = kwargs_initial
        self._kwargs_integrator = kwargs_integrator
        if lens_model_list != None:
            self._lens_model_complete = LensModel(lens_model_list = lens_model_list)
        self._embedded=embedded
        self._type2=type2        

        # ts, F_tilde = integrator(lens_model_complete, kwargs_lens, **kwargs)
        # if args.type2:
        #     ws, Fw = iwFourier(ts * Tscale, F_tilde, args) 
        #     fs = ws/(2*np.pi)
        #     peak = np.where(F_tilde == np.amax(F_tilde))
        #     index = int(peak[0])
        #     Tds = 5
        #     tdiff = ts[index]*Tscale-5
        #     overall_phase = np.exp(-1 * 2 * np.pi * 1j * (Tds+tdiff) * fs)
        #     Fw *= overall_phase
        # else:
        #     dt = 1e-5
        #     print(ts, F_tilde)
        #     ts_extended, F_tilde_extended = F_tilde_extend(ts, F_tilde, args, **kwargs)
        #     F_tilde_apodized = coswindowback(F_tilde_extended, 50)
        # ws, Fw = iwFourier(ts_extended*Tscale, F_tilde_apodized, args)
        # print(ws, Fw)

        # from bisect import bisect_left

        # i = bisect_left(ws, 2*np.pi*2000)

        # return ws[:i], Fw[:i]

    def integrator(self):
        
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
            if key in self._kwargs_integrator:
                value = self._kwargs_integrator[key]
                kwargs_integrator.update({key: value})

        T = self._lens_model_complete.fermat_potential
        binmax0 = kwargs_integrator['TimeMax']
        binmin = kwargs_integrator['TimeMin']
        binlength = kwargs_integrator['TimeLength']
        binwidth = kwargs_integrator['TimeStep']
        N = kwargs_integrator['PixelNum']
        Nblock = kwargs_integrator['PixelBlockMax']
        binnum = int((binmax0 - binmin) / binwidth) + 1
        binnumlength = int(binlength / binwidth)

        thetaE = self._kwargs_initial['theta_E']

        y0 = thetaE * self._kwargs_initial['source_pos_x']
        y1 = thetaE * self._kwargs_initial['source_pos_y']

        binmax = binmin + binwidth * (binnum + 1)
        bins = np.linspace(binmin, binmax, binnum)

        x1cen = kwargs_integrator['WindowCenterX']
        x2cen = kwargs_integrator['WindowCenterY']
        L = thetaE * kwargs_integrator['WindowSize']
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

        bincount = histogram_routine(self._lens_model_complete, Numblocks, np.array([[None, None]]), Nblock, Nresidue, x1corn, x2corn, Lblock, binnum,
                        binmin, binmax, thetaE, self._kwargs_initial, y0, y1, dx)

        zoomN = N
        zoomdx = Lblock/(N+1)
        zoomNumblocks = zoomN // Nblock
        zoomNresidue = zoomN % Nblock
        zoomLblock = Nblock * zoomdx
        for (zoomblockcorn1, zoomblockcorn2) in zoomblockcornzip:
            bincount += histogram_routine(self._lens_model_complete, zoomNumblocks, np.array([[None, None]]), Nblock, zoomNresidue, zoomblockcorn1, zoomblockcorn2, zoomLblock, binnum,
                        binmin, binmax, thetaE, self.kwargs_initial, y0, y1, zoomdx)

        bincountback = np.trim_zeros(bincount, 'f')
        bincountfront = np.trim_zeros(bincount, 'b')
        fronttrimmed = len(bincount) - len(bincountback)
        backtrimmed = len(bincount) - len(bincountfront) + 1
        F_tilde = bincount[fronttrimmed:-backtrimmed] / (2 * np.pi * binwidth) / thetaE ** 2
        ts = bins[fronttrimmed:-backtrimmed] - bins[fronttrimmed]

    def importor(self, ws, Fws):
        self._ws = ws
        self._Fws = Fws

    def plot(self, abs=True, pha=False, save=False, savefile=None):

        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots()

        ws = self._ws
        Fws = self._Fws

        fs=ws/(2*np.pi)

        # smoothen the curve(s)
        from scipy.signal import savgol_filter
        Fa_fil = savgol_filter(np.abs(Fws), 51, 3)
        Fp_fil = savgol_filter(np.angle(Fws), 51, 3)

        from bisect import bisect_left
        i = bisect_left(fs, 2000) ###hardcoded

        if abs:
            ax.plot(fs[:i], Fa_fil[:i], linewidth=1)
        elif pha:
            ax.plot(fs[:i], Fp_fil[:i], linewidth=1)

        ax.set_xlabel(r'Frequency (Hz)', fontsize = 14)
        if abs:
            ax.set_ylabel(r'$|F|/\sqrt{\mu}$', fontsize = 14)
        elif pha:
            ax.set_ylabel(r'$args(F)$', fontsize = 14)
        ax.tick_params(axis='x', labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(which = 'both', alpha = 0.5)
        fig.tight_layout()

        if plt.save:
            plt.savefig(savefile)

        plt.show()
