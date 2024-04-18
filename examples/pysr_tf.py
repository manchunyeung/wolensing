import sympy
import numpy as np
from matplotlib import pyplot as plt
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split

# from lenstronomy.LensModel.lens_model import LensModel
# import lensinggw.constants.constants as const
# from lensinggw.utils.utils import TimeDelay, magnifications, getMinMaxSaddle
# from wolensing.plot.plot import plot_contour
# import wolensing.amplification_factor.amplification_factor as af

# ym = 0.5
# angle = np.radians(float(90))
# G = const.G  # gravitational constant [m^3 kg^-1 s^-2]
# c = const.c  # speed of light [m/s]
# M_sun = const.M_sun  # Solar mass [Kg]

# y0, y1 = 0.5, 0 # source position
# l0, l1 = 0., 0 # lens position

# zS = 1.0 # source redshift
# zL = 0.5 # lens redshift

# mL1 = 1e10
# mL2 = 20
# mtot = mL1 + mL2

# # convert to radians
# from lensinggw.utils.utils import param_processing
# thetaE1 = param_processing(zL, zS, mL1)
# thetaE2 = param_processing(zL, zS, mL2)
# thetaE = param_processing(zL, zS, mtot)

# beta0, beta1 = y0 * thetaE, y1 * thetaE
# eta10, eta11 = 0 * l0 * thetaE, 0 * l1 * thetaE
# lens_model_list = ['SIS']
# kwargs_sis_1 = {'center_x': eta10, 'center_y': eta11, 'theta_E': thetaE1}
# kwargs_lens_list = [kwargs_sis_1]

# kwargs_sis_1_scaled = {'center_x': eta10 / thetaE, 'center_y': eta11 / thetaE, 'theta_E': thetaE1 / thetaE}
# kwargs_lens_list_scaled = [kwargs_sis_1_scaled]

# from lensinggw.solver.images import microimages
# solver_kwargs = {'SearchWindowMacro': 10 * thetaE1,
#                  'SearchWindow': 5 * thetaE2,
#                  'OverlapDistMacro': 1e-17,
#                  'OnlyMacro': True}
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

# # Injecting microlens around desired macroimage
# eta20, eta21 = MacroImg_ra[microtype] + np.cos(angle)*ym*thetaE2, MacroImg_dec[microtype] + np.sin(angle)*ym*thetaE2
# lens_model_list = ['SIS', 'POINT_MASS']
# kwargs_sis_1 = {'center_x': eta10, 'center_y': eta11, 'theta_E': thetaE1}
# kwargs_point_mass_2 = {'center_x': eta20, 'center_y': eta21, 'theta_E': thetaE2}
# kwargs_lens_list = [kwargs_sis_1, kwargs_point_mass_2]

def geometrical_optics(f):
    """
    :param mus: magnifications of images.
    :param tds: time delays of images.
    :param Img_ra: right ascension of images relative to the center of lens plane.
    :param Img_dec: declination of images relative to the center of lens plane.
    :param upper_lim: desired upper limit of freqeuncy range of geometrical optics.
    :return: frequency array and amplification factor F(f) of geometrical optics.
    """

    mu = array([ 2.28571429,  2.31428572, -0.79998707, -0.80001293])
    tds = array([0.        , 0.00034243, 0.00052051, 0.00052052])
    ns = [0, 0, 0.5, 0.5]

    for i in range(len(mu)):
        Fmag += np.sqrt(np.abs(mu[i]))* np.exp(1j*np.pi*(2.*f*tds[i] - ns[i]))

    # from lensinggw.amplification_factor.amplification_factor import amplification_from_data
    # geoFws = amplification_from_data(self._geofs, mus, tds, ns)
    
    return np.abs(Fmag)



Fg = np.loadtxt('./gwoF.txt', dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})
gf = np.loadtxt('./gf.txt')

Fws = np.loadtxt('./Fws.txt', dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})
ws = np.loadtxt('./ws.txt')

Fg = np.abs(Fg)
Fws = np.abs(Fws)

Y = 10 * np.random.randn(22400, 1)
# gf.reshape(len(gf), 1)


# print(X.shape)
X = np.linspace(10,2000,22400)
# print(X, X.shape[0])
# print(X, Y.shape[0])
X = X.reshape(X.shape[0], 1)

default_pysr_params = dict(
    populations=30,
    model_selection="best",
)

model = PySRRegressor(
    niterations=30,
    unary_operators=["exp"],
    binary_operators=["+", "-", "*", "/"],
    **default_pysr_params,
    extra_sympy_mappings={
        "Fgeo": lambda x: geometrical_optics(x)
    }
)

model.fit(X, y=Fws)
print(model.latex())
