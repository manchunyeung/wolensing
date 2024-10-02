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
    
