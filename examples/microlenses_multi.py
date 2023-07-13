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


# The macroimage where the microlens is placed around.

# In[5]:


type2 = False


# In[6]:


ym = 0.5
angle = np.radians(float(0))


# Importing constants

# In[7]:


G = const.G  # gravitational constant [m^3 kg^-1 s^-2]
c = const.c  # speed of light [m/s]
M_sun = const.M_sun  # Solar mass [Kg]


# In[8]:


y0, y1 = 0.5, 0 # source position
l0, l1 = 0., 0 # lens position

zS = 1.0 # source redshift
zL = 0.5 # lens redshift


# In[9]:


mL1 = 1 * 1e10
# mL1 = 0.0001

mL2 = 1
mL3 = 1.2
mL4 = 0.8
mL5 = 3
mtot = mL1 + mL2

# convert to radians
from lensinggw.utils.utils import param_processing
thetaE1 = param_processing(zL, zS, mL1)
thetaE2 = param_processing(zL, zS, mL2)
thetaE3 = param_processing(zL, zS, mL3)
thetaE4 = param_processing(zL, zS, mL4)
thetaE5 = param_processing(zL, zS, mL5)
thetaE = param_processing(zL, zS, mtot)


# In[10]:


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

# Injecting microlens around desired macroimage
eta20, eta21 = MacroImg_ra[microtype] + 0.2*thetaE2, MacroImg_dec[microtype] + 0.*thetaE2
eta30, eta31 = MacroImg_ra[microtype] + -0.3*thetaE3, MacroImg_dec[microtype] + 0.*thetaE3
eta40, eta41 = MacroImg_ra[microtype] + 0.*thetaE4, MacroImg_dec[microtype] + -0.1*thetaE4
eta50, eta51 = MacroImg_ra[microtype] + 0.*thetaE5, MacroImg_dec[microtype] + 0.1*thetaE5
lens_model_list = ['SIS', 'POINT_MASS', 'POINT_MASS', 'POINT_MASS', 'POINT_MASS']
kwargs_sis_1 = {'center_x': eta10, 'center_y': eta11, 'theta_E': thetaE1}
kwargs_point_mass_2 = {'center_x': eta20, 'center_y': eta21, 'theta_E': thetaE2}
kwargs_point_mass_3 = {'center_x': eta30, 'center_y': eta31, 'theta_E': thetaE3}
kwargs_point_mass_4 = {'center_x': eta40, 'center_y': eta41, 'theta_E': thetaE4}
kwargs_point_mass_5 = {'center_x': eta50, 'center_y': eta51, 'theta_E': thetaE5}
kwargs_lens_list = [kwargs_sis_1, kwargs_point_mass_2, kwargs_point_mass_3, kwargs_point_mass_4, kwargs_point_mass_5]
print(kwargs_lens_list)


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
    
lens_model_complete = LensModel(lens_model_list=lens_model_list)
T = lens_model_complete.fermat_potential
T0 = thetaE ** (-2) * T(MacroImg_ra[microtype], MacroImg_dec[microtype], kwargs_lens_list, beta0, beta1)#[0]
if not isinstance(T0, float):
    T0 = T0[0]
Tscale = 4 * (1 + zL) * mtot * M_sun * G / c ** 3
print('T0 = {}'.format(T0))
print('Tscale = {}'.format(Tscale))


# In[11]:


# plot only the microimages around the desired macroimage
# Img_ra = np.delete(Img_ra, [0])
# Img_dec = np.delete(Img_dec, [0])


# In[12]:


import matplotlib.pyplot as plt

# fig, ax = plt.subplots()
# plot_contour(ax, lens_model_list, eta10, eta11, 4*thetaE1, kwargs_lens_list, beta0, beta1, MacroImg_ra, MacroImg_dec,
#                     T0 = T0, Tfac = (thetaE)**(-2), micro=False)


# In[13]:


# fig1, ax1 = plt.subplots()
# plot_contour(ax1, lens_model_list, MacroImg_ra[microtype], MacroImg_dec[microtype], 8*thetaE2, kwargs_lens_list, beta0, beta1, Img_ra, Img_dec,
#                     T0 = T0, Tfac = (thetaE)**(-2), micro=True)


# In[14]:


# Define the characteristic WindowSize
mL3 = 10
thetaE3 = param_processing(zL, zS, mL3)


# In[15]:


kwargs_macro = {'source_pos_x': beta0,
                'source_pos_y': beta1,
                'theta_E': thetaE,
                'mu': Mus[microtype],
               }

kwargs_integrator = {'PixelNum': int(50000),
                     'PixelBlockMax': 2000,
                     'WindowSize': 600*thetaE3,
                     'WindowCenterX': MacroImg_ra[microtype],
                     'WindowCenterY': MacroImg_dec[microtype],
                     'T0': T0,
                     'TimeStep': 1e-5/Tscale, 
                     'TimeMax': T0 + 1/Tscale,
                     'TimeMin': T0 - .1/Tscale,
                     'TimeLength': 2/Tscale,
                     'TExtend': 10/Tscale,
                     'LastImageT': .02/Tscale,
                     'Tbuffer': 0,
                     'Tscale': Tscale}


amplification = af.amplification_factor_fd(lens_model_list=lens_model_list, kwargs_lens=kwargs_lens_list, kwargs_macro=kwargs_macro, **kwargs_integrator)
start = time.time()
ws, Fws = amplification.integrator()
end = time.time()
print(end - start)
amplification.plot(saveplot='./suc.pdf')


# In[ ]:




