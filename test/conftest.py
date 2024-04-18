import pytest
import numpy as np

from lenstronomy.LensModel.lens_model import LensModel
import wolensing.utils.constants as const
from wolensing.utils.utils import Einstein_radius
import wolensing.amplification_factor.amplification_factor as af

@pytest.fixture
def sis_amp():
    """
    Testing standard SIS model
    """

    zS, zL = 1., 0.5
    mL1 = 1e3

    G = const.G  # gravitational constant [m^3 kg^-1 s^-2]
    c = const.c  # speed of light [m/s]
    M_sun = const.M_sun  # Solar mass [Kg]
    df = 0.25
    textendmax = 1/df
    tlength = .13
    textend = textendmax-tlength
    thetaE = Einstein_radius(zL, zS, mL1)
    beta0, beta1 = 0.1 * thetaE, 0 * thetaE
    eta10, eta11 = 0 * thetaE, 0 * thetaE
    eta0, eta1 = 0., 0.
    lens_model_list = ['SIS']
    kwargs_sis_1 = {'center_x': eta10, 'center_y': eta11, 'theta_E': thetaE}
    kwargs_lens_list = [kwargs_sis_1]


    lens_model_complete = LensModel(lens_model_list=lens_model_list)
    T = lens_model_complete.fermat_potential
    T0 = thetaE ** (-2) * T(0, 0, kwargs_lens_list, beta0, beta1)#[0]
    Tscale = 4 * (1 + zL) * mL1 * M_sun * G / c ** 3

    mL3 = 10
    thetaE3 = Einstein_radius(zL, zS, mL3)

    kwargs_macro = {'source_pos_x': beta0,
                    'source_pos_y': beta1,
                    'theta_E': thetaE,
                    'mu': 1,
                   }

    kwargs_integrator = {'PixelNum': int(20000),
                         'PixelBlockMax': 2000,
                         'WindowSize': 1.*210*thetaE3,
                         'WindowCenterX': 0,
                         'WindowCenterY': 0,
                         'T0': T0,
                         'TimeStep': 1e-5/Tscale, 
                         'TimeMax': T0 + 1./Tscale,
                         'TimeMin': T0 - .1/Tscale,
                         'TimeLength': tlength/Tscale,
                         'TExtend': 10/Tscale,
                         'LastImageT': .02/Tscale,
                         'Tbuffer': 0,
                         'Tscale': Tscale}
    amplification = af.amplification_factor(lens_model_list=lens_model_list, kwargs_lens=kwargs_lens_list, kwargs_macro=kwargs_macro, **kwargs_integrator)
    
    return amplification
