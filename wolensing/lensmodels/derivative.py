import numpy as np

def Gradient_Td(lens_model_list, x, y, kwargs_lens, kwargs_macro, matrix=False):
    '''
    :param lens_model_list: list of lens models.
    :param x: x-coordinates of position on lens plane.
    :param y: y-coordinates of position on lens plane.
    :kwargs: arguemnts for the lens models.
    :return: gradient of time delay at the input position.
    '''
    
    source_x = kwargs_macro['source_pos_x']
    source_y = kwargs_macro['source_pos_y']

    td_x = x - source_x
    td_y = y - source_y

    for lens_type, lens_kwargs in zip(lens_model_list, kwargs_lens):
        thetaE = lens_kwargs['theta_E']
        x_center = lens_kwargs['center_x']
        y_center = lens_kwargs['center_y']

        x_shift, y_shift = x-x_center, y-y_center

        if lens_type == 'SIS':
            f_x, f_y = Gradient_SIS(x_shift, y_shift, thetaE)
            td_x -= f_x
            td_y -= f_y
        elif lens_type == 'POINT_MASS':
            f_x, f_y = Gradient_PM(x_shift, y_shift, thetaE)
            td_x -= f_x
            td_y -= f_y
        elif lens_type == 'SIE':
            e1 = lens_kwargs['e1']
            e2 = lens_kwargs['e2']
            f_x, f_y = Gradient_SIE(x_shift, y_shift, thetaE, e1, e2)
            td_x -= f_x
            td_y -= f_y
    
    if matrix:
        return np.array([td_x, td_y])
    
    return td_x, td_y
    
def Gradient_SIS(x, y, thetaE):
    '''
    :param x: x-coordinates of position on lens plane with respect to the lens position.
    :param y: y-coordinates of position on lens plane with respect to the lens position.
    :param thetaE: Einstein radius of the lens.
    :return: independent components of hessian matrix of SIS profile.    
    '''
    
    prefactor = thetaE / np.sqrt(x**2 + y**2)
    f_x = x * prefactor
    f_y = y * prefactor

    return f_x, f_y

def Gradient_PM(x, y, thetaE):
    '''
    :param x: x-coordinates of position on lens plane with respect to the lens position.
    :param y: y-coordinates of position on lens plane with respect to the lens position.
    :param thetaE: Einstein radius of the lens.
    :return: independent components of hessian matrix of PM profile.    
    '''
    
    prefactor = thetaE**2 / (x**2 + y**2)
    f_x = x * prefactor
    f_y = y * prefactor

    return f_x, f_y

def _ellipticity2phi_q(e1, e2):
    phi = np.arctan2(e2, e1) / 2.0
    c = np.sqrt(e1**2 + e2**2)
    c = np.minimum(c, 0.9999)
    q = (1.0 - c) / (1.0 + c)
    return phi, q


def _rotate(xcoords, ycoords, angle):
    return (
        xcoords * np.cos(angle) + ycoords * np.sin(angle),
        -xcoords * np.sin(angle) + ycoords * np.cos(angle),
    )


def Gradient_SIE(x, y, theta_E, e1, e2):
    """
    Deflection angles (alpha_x, alpha_y) for the SIE model.

    This mirrors the parameterization used in `wolensing.lensmodels.lens.Psi_SIE`,
    i.e. the SIE is specified by (theta_E, e1, e2).
    """
    phi_G, q = _ellipticity2phi_q(e1, e2)

    # Match the normalization used in `Psi_SIE`
    theta_E = theta_E / np.sqrt((1.0 + q**2) / (2.0 * q))
    b = theta_E * np.sqrt((1.0 + q**2) / 2.0)

    s_scale = 1e-10
    s = s_scale * np.sqrt((1.0 + q**2) / (2.0 * q**2))

    if q >= 1.0:
        q = 0.99999999

    x_rotate, y_rotate = _rotate(x, y, phi_G)
    psi = np.sqrt(q**2 * (s**2 + x_rotate**2) + y_rotate**2)

    sq = np.sqrt(1.0 - q**2)
    alpha_x_r = b / sq * np.arctan(sq * x_rotate / (psi + s))
    alpha_y_r = b / sq * np.arctanh(sq * y_rotate / (psi + q**2 * s))

    # Rotate gradients back: grad transforms with R^T
    alpha_x = alpha_x_r * np.cos(phi_G) - alpha_y_r * np.sin(phi_G)
    alpha_y = alpha_x_r * np.sin(phi_G) + alpha_y_r * np.cos(phi_G)
    return alpha_x, alpha_y
