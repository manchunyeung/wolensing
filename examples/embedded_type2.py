## has to be removed hardcoded part !/users/man-chun.yeung/microlensing/env/bin/python3

import sys
import os
path = os.getcwd()
dir = os.path.abspath(os.path.join(path, os.pardir))
sys.path.append(dir)

from jax import config
config.update("jax_enable_x64", True)

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from lenstronomy.LensModel.lens_model import LensModel
import lensinggw.constants.constants as const
from lensinggw.utils.utils import TimeDelay, magnifications, getMinMaxSaddle
from lensinggw.amplification_factor.amplification_factor import geometricalOpticsMagnification

import wolensing.amplification_factor.amplification_factor as af

G = const.G  # gravitational constant [m^3 kg^-1 s^-2]
c = const.c  # speed of light [m/s]
M_sun = const.M_sun  # Solar mass [Kg]

imindex = 0

# coordinates in scaled units [x (radians) /thetaE_tot]
y0, y1 = 0.1, 0 # source position
l0, l1 = 0.05, 0 # lens position

ym = 100
angle = np.radians(float(0.))
zS = 1.0
zL = 0.5

# masses
mL1 = 1 * 1e10
mL2 = 100
mtot = mL1 + mL2

# convert to radians
from lensinggw.utils.utils import param_processing

thetaE1 = param_processing(zL, zS, mL1)
thetaE2 = param_processing(zL, zS, mL2)
thetaE = param_processing(zL, zS, mtot)


beta0, beta1 = y0 * thetaE, y1 * thetaE
eta10, eta11 = 0 * l0 * thetaE, 0 * l1 * thetaE

lens_model_list = ['SIS']
kwargs_sis_1 = {'center_x': eta10, 'center_y': eta11, 'theta_E': thetaE1}
kwargs_lens_list = [kwargs_sis_1]

print('thetaE1 and thetaE', thetaE1, thetaE)
kwargs_sis_1_scaled = {'center_x': eta10 / thetaE, 'center_y': eta11 / thetaE, 'theta_E': thetaE1 / thetaE}
kwargs_lens_list_scaled = [kwargs_sis_1_scaled]
from lensinggw.solver.images import microimages

solver_kwargs = {'SearchWindowMacro': 10 * thetaE1,
                 'SearchWindow': 5 * thetaE2,
                 'OverlapDistMacro': 1e-17,
                 'OnlyMacro': True}

MacroImg_ra, MacroImg_dec, pixel_width = microimages(source_pos_x=beta0,
                                                     source_pos_y=beta1,
                                                     lens_model_list=lens_model_list,
                                                     kwargs_lens=kwargs_lens_list,
                                                     **solver_kwargs)

Macromus = magnifications(MacroImg_ra, MacroImg_dec, lens_model_list, kwargs_lens_list)
T01 = TimeDelay(MacroImg_ra, MacroImg_dec,
                beta0, beta1,
                zL, zS,
                lens_model_list, kwargs_lens_list)


imindex = np.nonzero(T01)[0][0]

# # lens model
eta20, eta21 = MacroImg_ra[imindex] + np.cos(angle)*ym*thetaE2, MacroImg_dec[imindex] + np.sin(angle)*ym*thetaE2

lens_model_list = ['SIS', 'POINT_MASS']
kwargs_sis_1 = {'center_x': eta10, 'center_y': eta11, 'theta_E': thetaE1}
kwargs_point_mass_2 = {'center_x': eta20, 'center_y': eta21, 'theta_E': thetaE2}
kwargs_lens_list = [kwargs_sis_1, kwargs_point_mass_2]

Img_ra, Img_dec = MacroImg_ra, MacroImg_dec

# time delays, magnifications, Morse indices and amplification factor
from lensinggw.utils.utils import TimeDelay, magnifications, getMinMaxSaddle
from lensinggw.amplification_factor.amplification_factor import geometricalOpticsMagnification

tds = TimeDelay(Img_ra, Img_dec,
               beta0, beta1,
               zL, zS,
               lens_model_list, kwargs_lens_list)
mus = magnifications(Img_ra, Img_dec, lens_model_list, kwargs_lens_list)
ns = getMinMaxSaddle(Img_ra, Img_dec, lens_model_list, kwargs_lens_list, diff = None)

print('Time delays (seconds): ', tds)
print('magnifications: ', mus)
print('Morse indices: ', ns)

lens_model_complete = LensModel(lens_model_list=lens_model_list)
T = lens_model_complete.fermat_potential
T0 = thetaE ** (-2) * T(Img_ra[0], Img_dec[0], kwargs_lens_list, beta0, beta1)#[0]
if not isinstance(T0, float):
    T0 = T0[0]
Tscale = 4 * (1 + zL) * mtot * M_sun * G / c ** 3
print('T0 = {}'.format(T0))
print('Tscale = {}'.format(Tscale))

mL3 = 10
thetaE3 = param_processing(zL, zS, mL3)

kwargs_macro = {'source_pos_x': beta0,
                'source_pos_y': beta1,
                'theta_E': thetaE,
                'mu': np.abs(Macromu[imindex]),
               }

kwargs_integrator = {'InputScaled': False,
                     'PixelNum': int(args.pixel),
                     'PixelBlockMax': 2000,
                     'WindowSize': 10.*200*thetaE3,
                     'WindowCenterX': MacroImg_ra[imindex],
                     'WindowCenterY': MacroImg_dec[imindex],
                     'TimeStep': 1e-6/Tscale, 
                     'TimeMax': T0 + 7/Tscale,
                     'TimeMin': T0 - 5/Tscale,
                     'TimeLength': 12/Tscale,
                     'LastImageT': 4e-7/Tscale,
                     'Tbuffer':0., 
                     'T0': T0,
                     'Tscale': Tscale}    

amplification = af.amplification_factor(lens_model_list=lens_model_list, kwargs_lens=kwargs_lens_list, kwargs_macro=kwargs_macro, **kwargs_integrator)
ts, Ft = amplification.integrator(gpu=True)
ws, Fws = amplification.fourier(type2=True)
