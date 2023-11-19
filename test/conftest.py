import amplification.amplificaiton_factor as af

@pytest.fixture
def sis_amp():
    """
    Testing standard SIS model
    """
    zS, zL = 1., 0.5
    m = 1e3

    thetaE = param_processing(zL, zS, m)
    beta0, beta1 = y0 * thetaE, y1 * thetaE
    eta0, eta1 = 0., 0.
    lens_model_list = ['SIS']
    kwargs_sis_1 = {'center_x': eta10, 'center_y': eta11, 'theta_E': thetaE}


    lens_model_complete = LensModel(lens_model_list=lens_model_list)
    T = lens_model_complete.fermat_potential
    T0 = thetaE ** (-2) * potential(lens_model_list, 0, 0, y, kwargs_lens_list)#[0]
    Tscale = 4 * (1 + zL) * mL1 * M_sun * G / c ** 3

    mL3 = 10
    thetaE3 = param_processing(zL, zS, mL3)
    kwargs_macro = {'source_pos_x': beta0,
                    'source_pos_y': beta1,
                    'theta_E': thetaE,
                    'mu': 1,
                   }
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
                         'TExtend': 10/Tscale,
                         'LastImageT': .02/Tscale,
                         'Tbuffer': 0,
                         'Tscale': Tscale}

    amplification = af.amplification_factor(lens_model_list=lens_model_list, kwargs_lens=kwargs_lens_list, kwargs_macro=kwargs_macro, **kwargs_integrator)
    
    return amplification
