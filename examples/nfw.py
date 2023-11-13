#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
path = os.getcwd()
dir = os.path.abspath(os.path.join(path, os.pardir))
sys.path.append(dir)


# In[2]:


import numpy as np
import time


# In[3]:


from lenstronomy.LensModel.lens_model import LensModel
import lensinggw.constants.constants as const
from lensinggw.utils.utils import TimeDelay, magnifications, getMinMaxSaddle


# In[4]:


from plot.plot import plot_contour
import amplification_factor.amplification_factor as af
from lensmodels.potential import potential


import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# The macroimage where the microlens is placed around.

# In[5]:


# type2 = True


# In[6]:


ym = 0.1
angle = np.radians(float(0))


# Importing constants

# In[7]:


G = const.G  # gravitational constant [m^3 kg^-1 s^-2]
c = const.c  # speed of light [m/s]
M_sun = const.M_sun  # Solar mass [Kg]


# In[8]:
df = 0.25
textendmax = 1/df
tlength = .13
textend = textendmax-tlength

y0, y1 = 0.5, 0 # source position
l0, l1 = 0., 0 # lens position

zS = 1.0 # source redshift
zL = 0.5 # lens redshift


# In[9]:


mL1 = 1 * 1e3
# mL2 = 100 
# mtot = mL1 + mL2

# convert to radians
from lensinggw.utils.utils import param_processing
thetaE1 = param_processing(zL, zS, mL1)
thetaE = thetaE1
thetaE2 = param_processing(zL, zS, 10)
# thetaE = param_processing(zL, zS, mtot)
# print(mtot, thetaE)

# In[10]:

beta0, beta1 = y0 * thetaE, y1 * thetaE
eta10, eta11 = 0 * l0 * thetaE, 0 * l1 * thetaE
lens_model_list = ['NFW']
kwargs_sis_1 = {'center_x': eta10, 'center_y': eta11, 'theta_E': thetaE1}
kwargs_lens_list = [kwargs_sis_1]

kwargs_sis_1_scaled = {'center_x': eta10 / thetaE, 'center_y': eta11 / thetaE, 'theta_E': thetaE1 / thetaE}
kwargs_lens_list_scaled = [kwargs_sis_1_scaled]

from lensinggw.solver.images import microimages
solver_kwargs = {'SearchWindowMacro': 10 * thetaE1,
                 'SearchWindow': 5 * thetaE2,
                 'OverlapDistMacro': 1e-17,
                 'OnlyMacro': True}
print(beta0, beta1, lens_model_list, kwargs_lens_list)
# MacroImg_ra, MacroImg_dec, pixel_width = microimages(source_pos_x=beta0,
#                                                      source_pos_y=beta1,
#                                                      lens_model_list=lens_model_list,
#                                                      kwargs_lens=kwargs_lens_list,
#                                                      **solver_kwargs)

# Td = TimeDelay(MacroImg_ra, MacroImg_dec,
#                 beta0, beta1,
#                 zL, zS,
#                 lens_model_list, kwargs_lens_list)
# Mus = magnifications(MacroImg_ra, MacroImg_dec, lens_model_list, kwargs_lens_list)
# if type2:
#     microtype = np.nonzero(Td)[0][0]
# else:
#     microtype = np.where(Td==0)[0][0]

# print(MacroImg_ra[microtype], MacroImg_dec[microtype])
# print(MacroImg_ra[microtype] + np.cos(angle)*ym*thetaE2, MacroImg_dec[microtype] + np.sin(angle)*ym*thetaE2)
# print(angle, ym, thetaE2)
# Injecting microlens around desired macroimage
# eta20, eta21 = MacroImg_ra[microtype] + np.cos(angle)*ym*thetaE2, MacroImg_dec[microtype] + np.sin(angle)*ym*thetaE2
# eta21 = 1.2705494208814505e-21
# lens_model_list = ['SIS', 'POINT_MASS']
# kwargs_sis_1 = {'center_x': eta10, 'center_y': eta11, 'theta_E': thetaE1}
# kwargs_point_mass_2 = {'center_x': eta20, 'center_y': eta21, 'theta_E': thetaE2}
# kwargs_lens_list = [kwargs_sis_1, kwargs_point_mass_2]
# print(kwargs_lens_list)


# from lensinggw.solver.images import microimages
# solver_kwargs = {'SearchWindowMacro': 10 * thetaE1,
#                  'SearchWindow': 10 * thetaE2,
#                  'Pixels': 1e3,
#                  'OverlapDist': 1e-18,
#                  'OverlapDistMacro': 1e-17}
# solver_kwargs.update({'Improvement' : 0.1})
# solver_kwargs.update({'MinDist' : 10**(-7)})

# Img_ra, Img_dec, MacroImg_ra, MacroImg_dec, pixel_width = microimages(source_pos_x=beta0,
#                                                                       source_pos_y=beta1,
#                                                                       lens_model_list=lens_model_list,
#                                                                       kwargs_lens=kwargs_lens_list,
#                                                                       **solver_kwargs)
# Images_dict = {'Source_ra': beta0,
#                'Source_dec': beta1,
#                'Img_ra': Img_ra,
#                'Img_dec': Img_dec,
#                'MacroImg_ra': MacroImg_ra,
#                'MacroImg_dec': MacroImg_dec,
#                'Microlens_ra': [eta20],
#                'Microlens_dec': [eta21],
#                'thetaE': thetaE}

# # time delays, magnifications, Morse indices 
# from lensinggw.utils.utils import TimeDelay, magnifications, getMinMaxSaddle
# from lensinggw.amplification_factor.amplification_factor import geometricalOpticsMagnification
# tds = TimeDelay(Img_ra, Img_dec,
#                beta0, beta1,
#                zL, zS,
#                lens_model_list, kwargs_lens_list)
# mus = magnifications(Img_ra, Img_dec, lens_model_list, kwargs_lens_list)
# ns = getMinMaxSaddle(Img_ra, Img_dec, lens_model_list, kwargs_lens_list, diff = None)
# print('Time delays (seconds): ', tds)
# print('magnifications: ', mus)
# print('Morse indices: ', ns)

# print(Img_ra[microtype], Img_dec[microtype])

lens_model_complete = LensModel(lens_model_list=lens_model_list)
T = lens_model_complete.fermat_potential


# y = jnp.array([beta0,beta1], dtype=jnp.float64)
y = np.array([beta0,beta1], dtype=np.float64)
# T0 = thetaE ** (-2) * T(beta0, beta1, kwargs_lens_list, beta0, beta1)#[0]
T0 = thetaE ** (-2) * potential(lens_model_list, beta0, beta1, y, kwargs_lens_list)#[0]
if not isinstance(T0, float):
    T0 = T0[0]
Tscale = 4 * (1 + zL) * mL1 * M_sun * G / c ** 3
print('T0 = {}'.format(T0))
print('Tscale = {}'.format(Tscale))


# In[11]:


# plot only the microimages around the desired macroimage
# Img_ra = np.delete(Img_ra, [0])
# Img_dec = np.delete(Img_dec, [0])


# In[12]:


# import matplotlib.pyplot as plt

# fig, ax = plt.subplots()
# plot_contour(ax, lens_model_list, eta10, eta11, 4*thetaE1, kwargs_lens_list, beta0, beta1, MacroImg_ra, MacroImg_dec,
#                     T0 = T0, Tfac = (thetaE)**(-2), micro=False)


# # In[13]:


# fig1, ax1 = plt.subplots()
# plot_contour(ax1, lens_model_list, MacroImg_ra[microtype], MacroImg_dec[microtype], 8*thetaE2, kwargs_lens_list, beta0, beta1, Img_ra, Img_dec,
#                     T0 = T0, Tfac = (thetaE)**(-2), micro=True)


# In[14]:


# Define the characteristic WindowSize
mL3 = 10
thetaE3 = param_processing(zL, zS, mL3)

print(kwargs_lens_list)

# In[15]:


kwargs_macro = {'source_pos_x': beta0,
                'source_pos_y': beta1,
                'theta_E': thetaE,
                'mu': 1,
               }

# kwargs_integrator = {'PixelNum': int(300000),
#                      'PixelBlockMax': 2000,
#                      'WindowSize': 10.*210*thetaE3,
#                      'WindowCenterX': MacroImg_ra[microtype],
#                      'WindowCenterY': MacroImg_dec[microtype],
#                      'T0': T0,
#                      'TimeStep': 1e-6/Tscale, 
#                      'TimeMax': T0 + 7./Tscale,
#                      'TimeMin': T0 - 5./Tscale,
#                      'TimeLength': 12/Tscale,
#                      'TExtend': 7./Tscale,
#                      'LastImageT': 4e-7/Tscale,
#                      'Tbuffer': 0,
#                      'Tscale': Tscale}

kwargs_integrator = {'PixelNum': int(20000),
                     'PixelBlockMax': 2000,
                     'WindowSize': 1.*210*thetaE3,
                     'WindowCenterX': beta0,
                     'WindowCenterY': beta1,
                     'T0': T0,
                     'TimeStep': 1e-5/Tscale, 
                     'TimeMax': T0 + 1./Tscale,
                     'TimeMin': T0 - .1/Tscale,
                     'TimeLength': tlength/Tscale,
                     'TExtend': textend/Tscale,
                     'LastImageT': .02/Tscale,
                     'Tbuffer': 0,
                     'Tscale': Tscale}

amplification = af.amplification_factor_fd(lens_model_list=lens_model_list, kwargs_lens=kwargs_lens_list, kwargs_macro=kwargs_macro, **kwargs_integrator)
start = time.time()
ws, Fws = amplification.integrator(gpu=True, plot=True)
end = time.time()
print(end - start)
amplification.plot(saveplot='./nfw.pdf')


# np.savetxt('./data/{0}/{0}_ws_{1:1.5f}_{2:1.5f}_{3}.txt'.format('test', mL2, ym, 'cpu'), ws[::1])
# np.savetxt('./data/{0}/{0}_Fws_{1:1.5f}_{2:1.5f}_{3}.txt'.format('test', mL2, ym, 'cpu'), Fws[::1])
# In[ ]:




