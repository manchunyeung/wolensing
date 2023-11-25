#!/users/man-chun.yeung/microlensing/env/bin/python3

import sys
import os
path = os.getcwd()
dir = os.path.abspath(os.path.join(path, os.pardir))
sys.path.append(dir)

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from lenstronomy.LensModel.lens_model import LensModel
import lensinggw.constants.constants as const
from lensinggw.utils.utils import TimeDelay, magnifications, getMinMaxSaddle
from lensinggw.amplification_factor.amplification_factor import geometricalOpticsMagnification

from plot.plot import plot_contour
import amplification_factor_trial.amplification_factor as af
import lensmodels.morse_indices as morse

G = const.G  # gravitational constant [m^3 kg^-1 s^-2]
c = const.c  # speed of light [m/s]
M_sun = const.M_sun  # Solar mass [Kg]


THIS_DIR = os.getcwd()
DATA_DIR = os.path.join(THIS_DIR,'data')


# coordinates in scaled units [x (radians) /thetaE_tot]
y0, y1 = 0.5, 0 # source position
l0, l1 = 0., 0 # lens position

zS = 1.0
zL = 0.5

mL1 = 1e10
mL3 = 10
mtot = mL1
# convert to radians
from lensinggw.utils.utils import param_processing

thetaE1 = param_processing(zL, zS, mL1)
thetaE2 = param_processing(zL, zS, 100)
thetaE3 = param_processing(zL, zS, mL3)

thetaE =thetaE1
beta0, beta1 = y0 * thetaE, y1 * thetaE
eta10, eta11 = 0 * l0 * thetaE, 0 * l1 * thetaE

lens_model_list = ['SIS']
kwargs_sis_1 = {'center_x': eta10, 'center_y': eta11, 'theta_E': thetaE1}
kwargs_lens_list = [kwargs_sis_1]

kwargs_sis_1_scaled = {'center_x': eta10 / thetaE, 'center_y': eta11 / thetaE, 'theta_E': thetaE1 / thetaE}
kwargs_lens_list_scaled = [kwargs_sis_1_scaled]
from lensinggw.solver.images import microimages

solver_kwargs = {'SearchWindowMacro': 10 * thetaE1,
                 'SearchWindow': 5*thetaE2,
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

ns = getMinMaxSaddle(MacroImg_ra, MacroImg_dec, lens_model_list, kwargs_lens_list, diff = None)

Img_ra, Img_dec = MacroImg_ra, MacroImg_dec

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
T0 = thetaE ** (-2) * T(eta10, eta11, kwargs_lens_list, beta0, beta1)#[0]
Tscale = 4 * (1 + zL) * mtot * M_sun * G / c ** 3
print('T0 = {}'.format(T0))
print('Tscale = {}'.format(Tscale))

scale = 1000

fig, ax = plt.subplots()
plot_contour(ax, lens_model_list, eta10, eta11, 1000*210*thetaE3, kwargs_lens_list, beta0, beta1, Img_ra, Img_dec,
                    T0 = T0, Tfac = (thetaE)**(-2), micro=False)
plt.show()
# quit()
print(scale, 'scale')
scalet = 600000

kwargs_macro = {'source_pos_x': beta0,
                'source_pos_y': beta1,
                'theta_E': thetaE,
                # 'mu': args.mu,
                'T01': T0
               }

kwargs_integrator = {'InputScaled': True,
                     'PixelNum': int(20000),
                     'PixelBlockMax': 2000,
                     'WindowSize': scale * 210*thetaE3,
                     'WindowCenterX': eta10,
                     'WindowCenterY': eta11,
                     'TimeStep': scale*1e-5/Tscale, 
                     'TimeMax': T0 + scalet*1/Tscale,
                     'TimeMin': T0 - scalet*1/Tscale,
                     'TimeLength': scalet*4/Tscale,
                     'LastImageT': 0/Tscale,
                     'TExtend': scalet*10/Tscale,
                     'Tbuffer':0., 
                     'T0': T0,
                     'Tscale': Tscale}

# kwargs_integrator = {'InputScaled': False,
#                     'PixelNum': int(args.pixel),
#                     'PixelBlockMax': 2000,
#                     'WindowSize': args.Winfac*210*thetaE3,
#                     'WindowCenterX': 0.,
#                     'WindowCenterY': 0.,
#                     'TimeStep': args.TimeStep/Tscale, 
#                     'TimeMax': T0 + 5/Tscale,
#                     'TimeMin': T0 - 5/Tscale,
#                     'TimeLength': 10/Tscale,
#                     'LastImageT': args.LastImageT/Tscale,
#                     'TExtend': args.TExtend/Tscale,
#                     'Tbuffer':0., 
#                     'T0': T0,
#                     'Tscale': Tscale}

amplification = af.amplification_factor_fd(lens_model_list=lens_model_list, kwargs_lens=kwargs_lens_list, kwargs_macro=kwargs_macro, **kwargs_integrator)
ws, Fws = amplification.integrator(embedded=False)
amplification.plot()

# np.savetxt('./data/{0}/{0}_ws_{1:1.5f}_{2:1.5f}.txt'.format('test', mL1, 1), ws[::1])
# np.savetxt('./data/{0}/{0}_Fws_{1:1.5f}_{2:1.5f}.txt'.format('test', mL1, 1), Fws[::1])
