#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


get_ipython().run_line_magic('env', 'JAX_ENABLE_X64=True')


# In[3]:


from lenstronomy.LensModel.lens_model import LensModel
import lensinggw.constants.constants as const
from lensinggw.utils.utils import TimeDelay, magnifications, getMinMaxSaddle


# In[4]:


import sys
import os
path = os.getcwd()
dir = os.path.abspath(os.path.join(path, os.pardir))
sys.path.append(dir)


# In[5]:


from plot.plot import plot_contour
import amplification_factor.amplification_factor as af
from lensmodels.potential import potential


# The macroimage where the microlens is placed around.

# In[6]:


type2 = False


# In[7]:


ym = 0.8
angle = np.radians(float(0))


# Importing constants

# In[8]:


G = const.G  # gravitational constant [m^3 kg^-1 s^-2]
c = const.c  # speed of light [m/s]
M_sun = const.M_sun  # Solar mass [Kg]


# In[9]:


y0, y1 = 0.5, 0 # source position
l0, l1 = 0., 0 # lens position

zS = 1.0 # source redshift
zL = 0.5 # lens redshift


# In[10]:


mL1 = 1e10
mL2 = 20
mtot = mL1 + mL2

# convert to radians
from lensinggw.utils.utils import param_processing
thetaE1 = param_processing(zL, zS, mL1)
thetaE2 = param_processing(zL, zS, mL2)
thetaE = param_processing(zL, zS, mtot)


# In[11]:


thetaE


# In[12]:


Mpc=3.085677581491367e+22
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=69.7, Om0=0.306, Tcmb0=2.725)
DL       = cosmo.angular_diameter_distance(zL)
DS       = cosmo.angular_diameter_distance(zS)
DLS      = cosmo.angular_diameter_distance_z1z2(zL, zS)
D        = DLS/(DL*DS)
print(np.float64(D/Mpc))


# In[13]:


from utils.utils import Einstein_radius


# In[14]:


Einstein_radius(zL, zS, mtot)


# In[15]:


beta0, beta1 = y0 * thetaE, y1 * thetaE
eta10, eta11 = 0 * l0 * thetaE, 0 * l1 * thetaE
lens_model_list = ['SIS']
kwargs_sis_1 = {'center_x': eta10, 'center_y': eta11, 'theta_E': thetaE1}
kwargs_lens_list = [kwargs_sis_1]

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

Td = TimeDelay(MacroImg_ra, MacroImg_dec,
                beta0, beta1,
                zL, zS,
                lens_model_list, kwargs_lens_list)
Mus = magnifications(MacroImg_ra, MacroImg_dec, lens_model_list, kwargs_lens_list)
if type2:
    microtype = np.nonzero(Td)[0][0]
else:
    microtype = np.where(Td==0)[0][0]

masses = [60, 40, 20, 30]
angle = [0, 150, 260]
ym = [1., .4, 4.]

mass_density = 100
from utils.lensing import *
radius = injection_radius(masses, mass_density) #rad
lens_model_list, kwargs_lens_list = field_injection(zL, zS, MacroImg_ra[microtype], MacroImg_dec[microtype], masses, radius, lens_model_list, kwargs_lens_list, seed=1)

lens_model_list.append('POINT_MASS')
kwargs_lens_list.append({'center_x': 1.2081120649948832e-06, 'center_y': 2.3590773305795136e-10, 'theta_E': 8.052888061582411e-12})

from lensinggw.solver.images import microimages
solver_kwargs = {'SearchWindowMacro': 10 * thetaE1,
                 'SearchWindow': 40 * thetaE2,
                 'Pixels': 10*1e3,
                 'ÍmgIndex': 1,
                 'OverlapDist': 1e-18,
                 'OverlapDistMacro': 1e-17}
solver_kwargs.update({'Improvement' : 0.1})
solver_kwargs.update({'MinDist' : 10**(-7)})

Img_ra, Img_dec, MacroImg_ra, MacroImg_dec, pixel_width = microimages(source_pos_x=beta0,
                                                                      source_pos_y=beta1,
                                                                      lens_model_list=lens_model_list,
                                                                      kwargs_lens=kwargs_lens_list,
                                                                      **solver_kwargs)

# time delays, magnifications, Morse indices 
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
filename = __file__
np.savetxt(f'{filename}_tds.txt', tds)


lens_model_complete = LensModel(lens_model_list=lens_model_list)
T = lens_model_complete.fermat_potential
T0 = thetaE ** (-2) * T(Img_ra[microtype], Img_dec[microtype], kwargs_lens_list, beta0, beta1)#[0]
if not isinstance(T0, float):
    T0 = T0[0]
Tscale = 4 * (1 + zL) * mtot * M_sun * G / c ** 3
print('T0 = {}'.format(T0))
print('Tscale = {}'.format(Tscale))


# In[ ]:


# plot only the microimages around the desired macroimage
Img_ra = np.delete(Img_ra, [0])
Img_dec = np.delete(Img_dec, [0])


# In[ ]:


# from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
# from lenstronomy.LensModel.lens_model import LensModel
# lensmodel = LensModel(lens_model_list=lens_model_list)
# solver = LensEquationSolver(
#     lensModel = lensmodel
# )
# Img_ra, Img_dec = solver.image_position_stochastic(
#     source_x = beta0,
#     source_y = beta1,
#     kwargs_lens = kwargs_lens_list,
#     # min_distance = 1e-13,
#     # solver='lenstronomy'
#     search_window = 20*thetaE2,
#     # verbose=True,
#     precision_limit=10**(-17),
#     x_center = MacroImg_ra[microtype],
#     y_center= MacroImg_dec[microtype]
# )


# In[ ]:


Img_ra, Img_dec


# In[ ]:


kwargs_lens_list


# In[ ]:


lens_model_complete = LensModel(lens_model_list=lens_model_list)
T = lens_model_complete.fermat_potential
T0 = thetaE ** (-2) * T(MacroImg_ra[microtype], MacroImg_dec[microtype], kwargs_lens_list, beta0, beta1)#[0]


# In[ ]:


import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plot_contour(ax, lens_model_list, eta10, eta11, 4*thetaE1, kwargs_lens_list, beta0, beta1, MacroImg_ra, MacroImg_dec,
                    T0 = T0, Tfac = (thetaE)**(-2), micro=False)


# In[ ]:


fig1, ax1 = plt.subplots()
plot_contour(ax1, lens_model_list, MacroImg_ra[microtype], MacroImg_dec[microtype], 20*thetaE2, kwargs_lens_list, beta0, beta1, Img_ra, Img_dec,
                    T0 = T0, Tfac = (thetaE)**(-2), micro=True)


# In[ ]:


lens_model_list, kwargs_lens_list


# In[ ]:


Tscale = 4 * (1 + zL) * mtot * M_sun * G / c ** 3


# In[ ]:


# Define the characteristic WindowSize
mL3 = 10
thetaE3 = param_processing(zL, zS, mL3)


# In[ ]:


import time


# In[ ]:


kwargs_macro = {'source_pos_x': beta0,
                'source_pos_y': beta1,
                'theta_E': thetaE,
                'mu': np.abs(Mus[microtype]),
               }

kwargs_integrator = {'PixelNum': int(40000),
                     'PixelBlockMax': 2000,
                     'WindowSize': 2.*210*thetaE3,
                     'WindowCenterX': MacroImg_ra[microtype],
                     'WindowCenterY': MacroImg_dec[microtype],
                     'T0': T0,
                     'TimeStep': 1e-5/Tscale, 
                     'TimeMax': T0 + .7/Tscale,
                     'TimeMin': T0 - .5/Tscale,
                     'TimeLength': 1./Tscale,
                     'TExtend': 10/Tscale,
                     'LastImageT': .02/Tscale,
                     'Tbuffer': .1/Tscale,
                     'Tscale': Tscale}


amplification = af.amplification_factor(lens_model_list=lens_model_list, kwargs_lens=kwargs_lens_list, kwargs_macro=kwargs_macro, **kwargs_integrator)
ts, Ft = amplification.integrator(gpu=False)
ws, Fws = amplification.fourier()
amplification.plot_freq()


# In[ ]:


plt.plot(ts, Ft)


# In[ ]:


amplification.plot_freq(smooth=False)


# In[ ]:


amplification.plot_freq(pha=True, smooth=False)


# # Geometrical optics
# To get the geometrical optics around type 1 image, we need the magnifications, time delay and the image positions of the microimages

# In[ ]:


tds = TimeDelay(Img_ra, Img_dec,
               beta0, beta1,
               zL, zS,
               lens_model_list, kwargs_lens_list)
mus = magnifications(Img_ra, Img_dec, lens_model_list, kwargs_lens_list)
ns = getMinMaxSaddle(Img_ra, Img_dec, lens_model_list, kwargs_lens_list, diff = None)


# In[ ]:


from wolensing.utils.utils import *
Morce_indices()


# In[ ]:


Img_ra, Img_dec


# In[ ]:


geofs, geoFws = amplification.geometrical_optics(mus, tds, Img_ra, Img_dec, upper_lim=2000)


# In[ ]:


plt.semilogx(geofs, np.abs(geoFws))


# In[ ]:


plt.semilogx(geofs, np.angle(geoFws))


# # Full amplification factor concatenated with geometrical optics
# Given the computaional cost of the diffraction integral, we concatenate the result of wave optics at low frequency and geometrical optics at high frequency to obtain a full amplification factor with default upper bound of 3000Hz.

# In[ ]:


fullfs, fullFws = amplification.concatenate()
plt.semilogx(fullfs, np.abs(fullFws))


# In[ ]:


plt.semilogx(fullfs, np.angle(fullFws))


# In[ ]:


fac = np.sqrt(np.abs(Mus[microtype]))


# In[ ]:


plt.clf()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# In[ ]:


from bisect import bisect_left
low_index = bisect_left(geofs, 1)
geofs = geofs[low_index:]
geoFws = geoFws[low_index:]
ws = ws[low_index:]
Fws = Fws[low_index:]


# In[ ]:


ww = geofs
plt.semilogx(geofs, (fac*np.exp(-ww*np.pi/188) + (1-np.exp(-ww*np.pi/188))*np.abs(geoFws))/fac, label='Approximation')
plt.semilogx(geofs, np.abs(geoFws)/fac,label=r'$F_{geo}$')
plt.semilogx(ws, np.abs(Fws)/fac, label=r'$F_{wave}$')
plt.ylabel(r'$|F|/\sqrt{\mu}$')
plt.xlabel(r'$f$')
plt.grid(which='both', alpha=0.5)
plt.legend()
plt.xlim(1, 2000)
# plt.savefig('./approx.png',bbox_inches='tight', dpi=300)


# In[ ]:


mus, tds


# In[ ]:


model = (fac*np.exp(-ww*np.pi/188) + (1-np.exp(-ww*np.pi/188))*np.abs(geoFws))


# In[ ]:


np.savetxt('data/4mt_ws.txt', ws)
np.savetxt('data/4mt_Fws.txt', Fws)
np.savetxt('data/4mt_geofs.txt', geofs)
np.savetxt('data/4mt_model.txt', model)


# In[ ]:


model_ang = (1-np.exp(-ww*np.pi/37.5))*np.angle(geoFws)
np.savetxt('data/4mt_model_ang.txt', model_ang)


# In[ ]:


plt.semilogx(geofs, (1-np.exp(-ww*np.pi/37.5))*np.angle(geoFws), label='Approximation')
plt.semilogx(geofs, np.angle(geoFws),label=r'$F_{geo}$')
plt.semilogx(ws, np.angle(Fws), label=r'$F_{wave}$')
plt.ylabel(r'$args(F)$')
plt.xlabel(r'$f$')
plt.grid(which='both', alpha=0.5)
plt.legend()
plt.xlim(1, 2000)
# plt.savefig('./approx_ang.png',bbox_inches='tight', dpi=300)


# In[ ]:


Abs = (fac*np.exp(-ww*np.pi/1000) + (1-np.exp(-ww*np.pi/1000))*np.abs(geoFws))/fac
Pha = (1-np.exp(-ww*np.pi/1000))*np.angle(geoFws)
complete_approx = Abs * np.exp(1j * Pha)


# In[ ]:


amplification.plot_freq()


# In[ ]:


amplification.plot_freq(smooth=False)


# In[ ]:


plt.plot(ws[:720], np.abs(Fws[:720])/np.sqrt(3))


# In[ ]:


from bisect import bisect_left
i = bisect_left(ws, 60)


# In[ ]:


i


# In[ ]:


plt.semilogx(ww,(1-np.exp(-ww*np.pi/1000)))


# In[ ]:


plt.semilogx(ww, np.exp(-ww*np.pi/20))


# In[ ]:


import pycbc.psd as psd
from pycbc.filter.matchedfilter import overlap, match

hp, hc = get_fd_waveform(approximant = 'IMRPhenomXPHM',
                         mass1 = 30,
                         mass2 = 30,
                         delta_f = .04,
                         f_lower = 10,
                         f_final = 2000)
dfnoise = 0.04
noiselen = int(2001/dfnoise)+1
noise2 = psd.analytical.aLIGOaLIGODesignSensitivityT1800044(noiselen, dfnoise, 1)
noisef = np.linspace(1, 1 + dfnoise*(noiselen-1), num = noiselen)


# In[ ]:


np.savetxt('gf.txt', geofs)
np.savetxt('complete.txt', complete_approx)
np.savetxt('ws.txt', ws)
np.savetxt('Fws.txt', Fws)


# In[ ]:


np.savetxt('gwoF.txt', geoFws)


# In[ ]:


mus, tds, ns


# In[ ]:


Img_ra


# In[ ]:


from wolensing.utils.utils import *
Morse_indices(amplification._lens_model_list, Img_ra, Img_dec, amplification._kwargs_lens)


# In[ ]:


plt.plot(geofs, -0.000435*geofs + np.pi - 23/geofs)


# In[ ]:




