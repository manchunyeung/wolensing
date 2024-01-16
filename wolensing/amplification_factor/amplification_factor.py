import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from scipy.optimize import curve_fit
from fast_histogram import histogram1d
from scipy.fft import fftfreq
from scipy.fftpack import fft
import lensinggw.constants.constants as const
from tqdm import trange, tqdm

from wolensing.utils.utils import *
from wolensing.utils.histogram import *
from wolensing.lensmodels.potential import potential

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
            bincount = histogram_routine_cpu(self._lens_model_complete, Numblocks, np.array([[None, None]]), Nblock, Nresidue, x1corn, x2corn, Lblock, binnum,
                            binmin, binmax, thetaE, self._kwargs_lens, y0, y1, dx)

        
        
        # trimming the array
        bincountback = np.trim_zeros(bincount, 'f')
        bincountfront = np.trim_zeros(bincount, 'b')
        fronttrimmed = len(bincount) - len(bincountback)
        backtrimmed = len(bincount) - len(bincountfront) + 1
        self._F_tilde = bincount[fronttrimmed:-backtrimmed] / (2 * np.pi * binwidth) / thetaE ** 2
        self._ts = bins[fronttrimmed:-backtrimmed] - bins[fronttrimmed]
        if binnumlength < len(self._ts):
            self._ts, self._F_tilde = self._ts[:binnumlength], self._F_tilde[:binnumlength]
        return self._ts, self._F_tilde

    def fourier(self, freq_end=2000, type2=False):
        """
        Compute the amplification factor in frequency domain

        :param freq_end: higher end of the frequency series. Default to be 2000.
        :param type2: boolean, if True, switch to fourier transform of  microlensing of a type 2 image.
        :return: frequency array and amplification factor F(f) of wave optics.
        """
        dt = self._kwargs_integrator['TimeStep']*self._Tscale # precise timestep for fourier transform
        
        if type2:
            ws, Fw = iwFourier(self._ts * self._Tscale, self._F_tilde, dt) 
            fs = ws/(2*np.pi)
            peak = np.where(self._F_tilde == np.amax(self._F_tilde))
            index = int(peak[0])
            Tds = (self._kwargs_integrator['T0'] - self._kwargs_integrator['TimeMin']) * self._Tscale  # in dimension time
            tdiff = self._ts[index]*self._Tscale - Tds
            overall_phase = np.exp(-1 * 2 * np.pi * 1j * (Tds+tdiff) * fs)
            Fw *= overall_phase
        else:
            ts_extended, F_tilde_extended = F_tilde_extend(self._ts, self._F_tilde, self._kwargs_macro, self._kwargs_integrator)
            F_tilde_apodized = coswindowback(F_tilde_extended, 50)
            ws, Fw = iwFourier(ts_extended*self._Tscale, F_tilde_apodized, dt)

        from bisect import bisect_left
        i = bisect_left(ws, 2*np.pi*freq_end)

        self._fs, self._Fws = ws/(2*np.pi), Fw
        return self._fs[:i], self._Fws[:i] 

    def importor(self, ts=None, F_tilde=None, fs=None, Fws=None, time=False, freq=False):
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
            self._fs = fs
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

    def plot_freq(self, macromu = 1, freq_end = 2000, saveplot=None, abs=True, pha=False, smooth=True):
        """
        Plots the amplification factor against frequency in semilogx

        :param macromu: macro magnification of the strong lensed image. Default to be one.
        :param freq_end: higher end of the frequency range 
        :param abs: boolean, if True, compute the absolute value of the amplification.
        :param pha: boolean, if True, compute the phase of the amplification.
        :param saveplot: where the plot is saved.
        :return: axes class of matplotlib representing the plot.
        """

        try:
            self._fs
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

        fs = self._fs
        Fws = self._Fws
        
        from bisect import bisect_left
        i = bisect_left(fs, freq_end) 

        # smoothen the curve(s)
        if smooth:
            from scipy.signal import savgol_filter
            Fa_fil = savgol_filter(np.abs(Fws), 51, 3)
            Fp_fil = savgol_filter(np.angle(Fws), 51, 3)
            if abs:
                ax.semilogx(fs[:i], Fa_fil[:i], linewidth=1)
            elif pha:
                ax.semilogx(fs[:i], Fp_fil[:i], linewidth=1)

        else:
            if abs:
                ax.semilogx(fs[:i], np.abs(Fws[:i]), linewidth=1)
            elif pha:
                ax.semilogx(fs[:i], np.angle(Fws[:i]), linewidth=1)

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

    def geometrical_optics(self, mus, tds, Img_ra, Img_dec, upper_lim = 3000):
        """
        :param mus: magnifications of images.
        :param tds: time delays of images.
        :param Img_ra: right ascension of images relative to the center of lens plane.
        :param Img_dec: declination of images relative to the center of lens plane.
        :param upper_lim: desired upper limit of freqeuncy range of geometrical optics.
        :return: frequency array and amplification factor F(f) of geometrical optics.
        """
        fs = self._fs
        fs_grid = fs[1]-fs[0]
        
        num_interp = int((upper_lim-fs[0])/fs_grid) 
        self._geofs = np.linspace(fs[0], upper_lim, num_interp)
        
        ns = Morse_indices(self._lens_model_list, Img_ra, Img_dec, self._kwargs_lens)
        from lensinggw.amplification_factor.amplification_factor import amplification_from_data
        self._geoFws = amplification_from_data(self._geofs, mus, tds, ns)
        
        return self._geofs, self._geoFws
    
    def concatenate(self, transfreq = 1000):
        """
        :param transfreq: transitional frequency of wave optics to geometrical optics.
        :return: concatenated frequency array and amplification factor F(f).
        """
        index = (np.abs(self._fs-transfreq)).argmin()
        
        self._fullfs = np.concatenate((self._fs[:index], self._geofs[index:]))
        self._fullFws = np.concatenate((self._Fws[:index], self._geoFws[index:]))
        
        return self._fullfs, self._fullFws
