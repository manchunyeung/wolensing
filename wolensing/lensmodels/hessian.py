import numpy as np

<<<<<<< HEAD
import sys
import os
dir = '/home/manchun.yeung/microlensing/wolensing/wolensing'
sys.path.append(dir)

from lensmodels.derivative import Gradient_SIE as sie_d

=======
>>>>>>> 05c6cb0c90922e0b5a54961674e74efb6e9368dc
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
<<<<<<< HEAD
        elif lens_type == 'SIE':
            hessian -= Hessian_SIE(x_shift, y_shift, thetaE)
=======
>>>>>>> 05c6cb0c90922e0b5a54961674e74efb6e9368dc
    
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
    
<<<<<<< HEAD
def Hessian_SIE(x, y, b, s, q):
    """Returns Hessian matrix of function d^2f/dx^2, d^2/dxdy, d^2/dydx,
    d^f/dy^2."""
    alpha_ra, alpha_dec = sie_d(x, y, b, s, q)
    diff = 0.0000000001
    alpha_ra_dx, alpha_dec_dx = sie_d(x + diff, y, b, s, q)
    alpha_ra_dy, alpha_dec_dy = sie_d(x, y + diff, b, s, q)

    f_xx = (alpha_ra_dx - alpha_ra) / diff
    f_xy = (alpha_ra_dy - alpha_ra) / diff
    f_yx = (alpha_dec_dx - alpha_dec) / diff
    f_yy = (alpha_dec_dy - alpha_dec) / diff
    return f_xx, f_xy, f_yx, f_yy
=======
>>>>>>> 05c6cb0c90922e0b5a54961674e74efb6e9368dc
