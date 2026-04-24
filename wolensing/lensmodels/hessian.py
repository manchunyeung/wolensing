import numpy as np

from .derivative import Gradient_SIE


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


def _alpha_sie(x, y, theta_E, e1, e2):
    """SIE deflection angles (alpha_x, alpha_y) in global coordinates."""
    phi_G, q = _ellipticity2phi_q(e1, e2)

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

    alpha_x = alpha_x_r * np.cos(phi_G) - alpha_y_r * np.sin(phi_G)
    alpha_y = alpha_x_r * np.sin(phi_G) + alpha_y_r * np.cos(phi_G)
    return alpha_x, alpha_y


def Hessian_Td(lens_model_list, x, y, kwargs, matrix=False):
    '''
    :param lens_model_list: list of lens models.
    :param x: x-coordinates of position on lens plane.
    :param y: y-coordinates of position on lens plane.
    :kwargs: arguemnts for the lens models.
    :param matrix: return hessian matrix if True.
    :return: independent components of hessian matrix of time delay function.    
    '''
    
    hessian = np.array([1.,1.,0.])
    
    for lens_type, lens_kwargs in zip(lens_model_list, kwargs):
        thetaE = lens_kwargs['theta_E']
        x_center = lens_kwargs['center_x']
        y_center = lens_kwargs['center_y']

        x_shift, y_shift = x-x_center, y-y_center

        if lens_type == 'SIS':
            hessian -= Hessian_SIS(x_shift, y_shift, thetaE)
        elif lens_type == 'POINT_MASS':
            hessian -= Hessian_PM(x_shift, y_shift, thetaE)
        elif lens_type == 'SIE':
            e1 = lens_kwargs['e1']
            e2 = lens_kwargs['e2']
            hessian -= Hessian_SIE(x_shift, y_shift, thetaE, e1, e2)
    
    if matrix:
        return np.array([[hessian[0], hessian[2]], [hessian[2], hessian[1]]])

    return hessian
    
def Hessian_SIS(x, y, thetaE):
    '''
    :param x: x-coordinates of position on lens plane with respect to the lens position.
    :param y: y-coordinates of position on lens plane with respect to the lens position.
    :param thetaE: Einstein radius of the lens.
    :return: independent components of hessian matrix of SIS profile.    
    '''
    
    prefactor = thetaE * np.sqrt(x**2 + y**2)**(-3.)
    f_xx = y**2 * prefactor
    f_yy = x**2 * prefactor
    f_xy = -x * y * prefactor

    return f_xx, f_yy, f_xy

def Hessian_PM(x, y, thetaE):
    '''
    :param x: x-coordinates of position on lens plane with respect to the lens position.
    :param y: y-coordinates of position on lens plane with respect to the lens position.
    :param thetaE: Einstein radius of the lens.
    :return: independent components of hessian matrix of PM profile.    
    '''
    
    prefactor = thetaE**2 * (x**2 + y**2)**(-2.)
    f_xx = (-x**2 + y**2) * prefactor
    f_yy = -1 * f_xx
    f_xy = (-2 * x * y) * prefactor
    
    return f_xx, f_yy, f_xy
    
def Hessian_SIE(x, y, theta_E, e1, e2, diff=1e-6):
    """
    Independent components (f_xx, f_yy, f_xy) of the Hessian of the SIE potential.
    Computed numerically from deflection angles for stability/consistency.
    """
    ax_p, ay_p = _alpha_sie(x + diff, y, theta_E, e1, e2)
    ax_m, ay_m = _alpha_sie(x - diff, y, theta_E, e1, e2)
    ax_py, ay_py = _alpha_sie(x, y + diff, theta_E, e1, e2)
    ax_my, ay_my = _alpha_sie(x, y - diff, theta_E, e1, e2)

    f_xx = (ax_p - ax_m) / (2.0 * diff)
    f_yy = (ay_py - ay_my) / (2.0 * diff)

    f_xy_from_ax = (ax_py - ax_my) / (2.0 * diff)
    f_yx_from_ay = (ay_p - ay_m) / (2.0 * diff)
    f_xy = 0.5 * (f_xy_from_ax + f_yx_from_ay)
    return f_xx, f_yy, f_xy
