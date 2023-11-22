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
from utils.histogram import *
from lensmodels.potential import potential

G = const.G  # gravitational constant [m^3 kg^-1 s^-2]
c = const.c  # speed of light [m/s]
M_sun = const.M_sun  # Solar mass [Kg]

class amplification_factor(object):

    def __init__(self, lens_model_list=None, kwargs_lens=None, kwargs_macro=None, **kwargs):
        """

        :param lens_model_list: list of lens models 
        :param kwargs_lens: arguments for integrating the diffraction integral
        :param kwargs_macro: arguments of the macromodel
        """

        kwargs_integrator = {
                'TimeStep': 1e-5,
                'TimeMax': 100,
                'TimeMin': -50,
                'TimeLength': 10, # length in time considered after initial signal
                'TExtend': 10,
                'T0': 0,
                'Tscale': 0.,
                'WindowSize': 15,
                'PixelNum': 10000,
                'PixelBlockMax': 2000,  # max number of pixels in a block
                'WindowCenterX': 0,
                'WindowCenterY': 0,
                'LastImageT': .02,
                'Tbuffer':0
                }
        
        for key in kwargs_integrator.keys():
            if key in kwargs:
                value = kwargs[key]
                kwargs_integrator.update({key: value})

        self._Tscale = kwargs_integrator['Tscale']
        self._kwargs_lens = kwargs_lens
        self._kwargs_macro = kwargs_macro
        self._kwargs_integrator = kwargs_integrator
        self._lens_model_list = lens_model_list
        if lens_model_list != None:
            self._lens_model_complete = LensModel(lens_model_list = lens_model_list)

    def integrator(self, gpu=False):
        """
        Computes the amplification facator F(f) by constructing the histogram in time domain. Defines the integration window of lens plane first.        

        :param gpu: boolean, if True, use gpu computing for integration.
        :return: amplification factor in time domain.
        """


        # details of the lens model and source
        thetaE = self._kwargs_macro['theta_E']
        y0 = self._kwargs_macro['source_pos_x']
        y1 = self._kwargs_macro['source_pos_y']

        # defines the time integration
        binmax0 = self._kwargs_integrator['TimeMax']
        binmin = self._kwargs_integrator['TimeMin']
        binlength = self._kwargs_integrator['TimeLength']
        binwidth = self._kwargs_integrator['TimeStep']

        binnum = int((binmax0 - binmin) / binwidth) + 1
        binnumlength = int(binlength / binwidth)
        binmax = binmin + binwidth * (binnum + 1)
        bins = np.linspace(binmin, binmax, binnum)

        # dividing the lens plane into grid
        N = self._kwargs_integrator['PixelNum']
        Nblock = self._kwargs_integrator['PixelBlockMax']

        x1cen = self._kwargs_integrator['WindowCenterX'] # The positions where the window centered at, usually the lens or the macroimage in embedded lens case
        x2cen = self._kwargs_integrator['WindowCenterY']
        L = 1. * self._kwargs_integrator['WindowSize'] # Size of the integration window
        dx = L / (N - 1)

        x1corn = x1cen - L / 2
        x2corn = x2cen - L / 2
        Lblock = Nblock * dx
        Numblocks = N // Nblock
        Nresidue = N % Nblock


        if gpu:
            bincount = histogram_routine_gpu(self._lens_model_list, Numblocks, np.array([[None, None]]), Nblock, Nresidue, x1corn, x2corn, Lblock, binnum,
                            binmin, binmax, thetaE, self._kwargs_lens, y0, y1, dx)
        else:
            bincount = histogram_routine_cpu1(self._lens_model_complete, Numblocks, np.array([[None, None]]), Nblock, Nresidue, x1corn, x2corn, Lblock, binnum,
                            binmin, binmax, thetaE, self._kwargs_lens, y0, y1, dx)

            
        # trimming the array
        bincountback = np.trim_zeros(bincount, 'f')
        bincountfront = np.trim_zeros(bincount, 'b')
        fronttrimmed = len(bincount) - len(bincountback)
        backtrimmed = len(bincount) - len(bincountfront) + 1
        self._F_tilde = bincount[fronttrimmed:-backtrimmed] / (2 * np.pi * binwidth) / thetaE ** 2
        self._ts = bins[fronttrimmed:-backtrimmed] - bins[fronttrimmed]
        if binnumlength > len(ts):
            self._ts, self._F_tilde = ts[:binnumlength], F_tilde[:binnumlength]
        # # qVVvuit()
        # # if not binnumlength > len(ts):
        # #     ts, F_tilde = ts, F_tilde
        # # else:
        # print(binlength, binwidth, binnumlength)
        # print('done')
        # self._ts, self._F_tilde = ts[:binnumlength], F_tilde[:binnumlength]  # , Tmax
 
        return self._ts, self._F_tilde

    def fourier(self, freq_end=2000, type2=False):
        """
        Compute the amplification factor in frequency domain

        :param freq_end: higher end of the frequency series. Default to be 2000.
        :param type2: boolean, if True, switch to fourier transform of  microlensing of a type 2 image.
        :return: amplification factor in frequency domain.
        """
        
        if type2:
            ws, Fw = iwFourier(self._ts * self._Tscale, self._F_tilde, type2) 
            fs = ws/(2*np.pi)
            peak = np.where(self._F_tilde == np.amax(self._F_tilde))
            index = int(peak[0])
            Tds = 5 # in dimension time
            tdiff = self._ts[index]*self._Tscale-5 
            # tdiff = ts[index]*self._Tscale-Tds 
            overall_phase = np.exp(-1 * 2 * np.pi * 1j * (Tds+tdiff) * fs)
            Fw *= overall_phase
        else:
            ts_extended, F_tilde_extended = F_tilde_extend(self._ts, self._F_tilde, self._kwargs_integrator)
            ws, Fw = iwFourier(ts_extended*self._Tscale, F_tilde_extended)

        from bisect import bisect_left
        i = bisect_left(ws, 2*np.pi*2000)

        self._ws, self._Fws = ws, Fw
        return ws[:i], Fw[:i] 

    def importor(self, ts=None, F_tilde=None, ws=None, Fws=None, time=False, freq=False):
        """
        Imports the amplification factor

        :param ts: time series in dimensionless unit
        :param F_tilde: time domain amplification factor
        :param ws: sampling frequency in unit of angular frequency
        :param Fws: frequency domain amplification factor 
        :param time: boolean, if True, plot time domain amplification factor
        :param freq: boolean, if True, plot frequency domain amplification factor
        """
        if time:
            self._ts = ts
            self._F_tilde = F_tilde
        elif freq:
            self._ws = ws
            self._Fws = Fws
        else:
            raise Exception('Please choose either time domain or frequency domain to import.')

    def plot_time(self, saveplot=None):
        """
        Plots the amplification factor in time domain

        :param saveplot: where the plot is saved.
        :return: axes class of matplotlib representing the plot.
        """

        try:
            self._ts
        except NameError:
            raise Exception('Time data is empty. Either integrate or import time data.')

        import matplotlib.pyplot as plt
        plt.clf()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots()

        ts = self._ts
        F_tilde = self._F_tilde

        # smoothen the curve(s)
        from scipy.signal import savgol_filter
        F_smooth = savgol_filter(F_tilde, 51, 3)

        ax.plot(ts, F_smooth, linewidth=1)

        ax.set_xlabel(r'Time (1s/Tscale)', fontsize = 14)
        ax.set_ylabel(r'$F(t)$', fontsize = 14)
        ax.tick_params(axis='x', labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid(which = 'both', alpha = 0.5)
        fig.tight_layout()

        if saveplot != None:
            plt.savefig(saveplot)

        plt.show()
        return ax 

    def plot_freq(self, freq_end = 2000, saveplot=None, abs=True, pha=False):
        """
        Plots the amplification factor against frequency in semilogx

        :param freq_end: higher end of the frequency range 
        :param abs: boolean, if True, compute the absolute value of the amplification.
        :param pha: boolean, if True, compute the phase of the amplification.
        :param saveplot: where the plot is saved.
        :return: axes class of matplotlib representing the plot.
        """

        try:
            self._ws
        except NameError:
            raise Exception('Frequency data is empty. Either integrate or import frequency data.')


        # Either plot the absolute value or the argument
        if pha:
            abs=False

        import matplotlib.pyplot as plt
        plt.clf()
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
        i = bisect_left(fs, freq_end) 

        if abs:
            ax.semilogx(fs[:i], Fa_fil[:i], linewidth=1)
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

        if saveplot != None:
            plt.savefig(saveplot)

        plt.show()
        return ax 
