import numpy as np
import warnings

def Hessian_Td(lens_model_list, x, y, kwargs):
    '''
    :param lens_model_list: list of lens models.
    :param x: x-coordinates of position on lens plane.
    :param y: y-coordinates of position on lens plane.
    :kwargs: arguemnts for the lens models.
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
            hessian -= Hessian_PM(x_shift, y_shift, thetaE)  # Make sure Psi_PM is JAX-compatible
    
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
    
