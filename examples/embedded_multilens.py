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

# imindex = 0

# coordinates in scaled units [x (radians) /thetaE_tot]
y0, y1 = 0.1, 0 # source position
l0, l1 = 0.05, 0 # lens position

ym = 100
angle = np.radians(float(0.))
zS = 1.0
zL = 0.5

mL4=10

# masses
mL1 = 1 * 1e10
mL2 = 100
mtot = mL1 + mL2

masses = [0.0875411729158816, 0.1380156891218021, 0.37757290272476596, 0.08326002905591416, 0.11751088559578819, 0.0878737809276154, 0.14164220540181344, 0.32080191454865087, 0.1432894584167837, 0.2607705005317567, 0.09098854113286821, 0.636857294268748, 0.10244447554800015, 0.11147041258795794, 0.2124085350914183, 0.6266784519859449, 0.42496685401191886, 1.1390594002155605, 0.4254607239653152, 0.1671246164756932, 0.17585315448008126, 0.36911502207552227]

mass_desnity = 100 # M/pc2

def conversion(masses, mass_density, cosmo, zL=0.5):
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=69.7, Om0=0.306, Tcmb0=2.725)
    D_l = cosmo.angular_diameter_distance(zL) #MPc

    total_mass = np.sum(masses)
    area =  total_mass/mass_density # pc2
    radius = np.sqrt(area/np.pi) # pc radius in distant galaxy
    angle_rad = radius / (D_l*1e6/u.Mpc)
    angle_ac = angle_rad / ac2rad
    return angle_rad

def critical_density(zL, zS):
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=69.7, Om0=0.306, Tcmb0=2.725)
    DL       = cosmo.angular_diameter_distance(zL)
    DS       = cosmo.angular_diameter_distance(zS)
    DLS      = cosmo.angular_diameter_distance_z1z2(zL, zS)

    from wolensing.utils import constants as const
    return const.c**2 * DL * 1e-6 / (4*np.pi*(const.G / 3.08567758128e16)*DLS*DS)

# convert to radians
from lensinggw.utils.utils import param_processing

thetaE1 = param_processing(zL, zS, mL1)
thetaE2 = param_processing(zL, zS, mL2)
thetaE = param_processing(zL, zS, mtot)
thetaE4 = param_processing(zL, zS, mL4)


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



num_points = len(masses)

angle = np.random.uniform(0, 2*np.pi, size=num_points)
ym = np.random.uniform(0, 2, size=num_points)

thetaEl = []
eta20, eta21 = [], []

microtype = imindex

for j in range(0,num_points):
    thetaEl.append(param_processing(zL, zS, masses[j]))# lens model
    #positioning the microlenses
    eta20.append(MacroImg_ra[microtype] + ym[j]*np.cos(angle[j]) * thetaEl[j])
    eta21.append(MacroImg_dec[microtype] + ym[j] * np.sin(angle[j]) * thetaEl[j])
    lens_model_list.append('POINT_MASS')
    kwargs_lens_list.append({'center_x': eta20[j], 'center_y': eta21[j], 'theta_E': thetaEl[j]})
    
from lensinggw.solver.images import microimages
solver_kwargs = {'SearchWindowMacro': 10 * thetaE1,
                 'SearchWindow': 10 * thetaE4,
                 'Pixels': 1e3,
                 'OverlapDist': 1e-18,
                 'OverlapDistMacro': 1e-17}
solver_kwargs.update({'Improvement' : 0.1})
solver_kwargs.update({'MinDist' : 10**(-17)})

from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.lens_model import LensModel


lensmodel = LensModel(lens_model_list=lens_model_list)

solver = LensEquationSolver(
    lensModel = lensmodel
)

# Img_ra, Img_dec = solver.image_position_stochastic(
#     source_x = beta0, 
#     source_y = beta1,
#     kwargs_lens = kwargs_lens_list,
#     search_window = 10 * thetaE4,
#     precision_limit=10**(-17),
#     x_center = MacroImg_ra[microtype],
#     y_center= MacroImg_dec[microtype],
# )



# # # lens model
# eta20, eta21 = MacroImg_ra[imindex] + np.cos(angle)*ym*thetaE2, MacroImg_dec[imindex] + np.sin(angle)*ym*thetaE2

# lens_model_list = ['SIS', 'POINT_MASS']
# kwargs_sis_1 = {'center_x': eta10, 'center_y': eta11, 'theta_E': thetaE1}
# kwargs_point_mass_2 = {'center_x': eta20, 'center_y': eta21, 'theta_E': thetaE2}
# kwargs_lens_list = [kwargs_sis_1, kwargs_point_mass_2]

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
                'mu': np.abs(Macromus[imindex]),
               }

kwargs_integrator = {'InputScaled': False,
                     'PixelNum': int(40000),
                     'PixelBlockMax': 2000,
                     'WindowSize': 1.*210*thetaE3,
                     'WindowCenterX': MacroImg_ra[imindex],
                     'WindowCenterY': MacroImg_dec[imindex],
                     'TimeStep': 1e-5/Tscale, 
                     'TimeMax': T0 + 1/Tscale,
                     'TimeMin': T0 - .1/Tscale,
                     'TimeLength': 2/Tscale,
                     'TExtend': 10/Tscale,
                     'LastImageT': .02/Tscale,
                     'Tbuffer':0., 
                     'T0': T0,
                     'Tscale': Tscale}    

amplification = af.amplification_factor(lens_model_list=lens_model_list, kwargs_lens=kwargs_lens_list, kwargs_macro=kwargs_macro, **kwargs_integrator)
ts, Ft = amplification.integrator(gpu=True)
ws, Fws = amplification.fourier()

np.savetxt('./ws.txt', ws)
np.savetxt('./Fws.txt', Fws)
amplification.plot_freq(freq_end=4000, saveplot='./try.pdf')
