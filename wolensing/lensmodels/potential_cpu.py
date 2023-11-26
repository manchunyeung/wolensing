import numpy as np
# from numba import jit
from .lens import *
from jax import jit
from functools import partial, wraps

def geometrical(x1, x2, y):
    '''
    :param x1: x-coordinates of position on lens plane with respect to the window center.
    :param x2: y-coordinates of position on lens plane with respect to the window center.
    :param y: numpy array, source positions.
    :return: geometrical part of the time delay.
    '''
    geometry = ((x1 - y[0])**2 + (x2 - y[1])**2) / 2.
    # x = jnp.array([x1, x2], dtype=jnp.float64)
    # # if isinstance(x1, np.ndarray): 
    # geo = (1/2) * jnp.linalg.norm(x-y[:, jnp.newaxis, jnp.newaxis], axis=0)**2
    # else:
    #     geo = (1/2) * jnp.linalg.norm(x-y, axis=0)**2
    return geometry

def potential(lens_model_list, x1, x2, y, kwargs):
    '''
    :param lens_model_list: list of lens models.
    :param x1: x-coordinates of position on lens plane with respect to the window center.
    :param x2: y-coordinates of position on lens plane with respect to the window center.
    :param y: numpy array, source positions.
    :kwargs: arguemnts for the lens models.
    :return: time delay function.
    '''
    potential = np.float64(0.0)

    for lens_type, lens_kwargs in zip(lens_model_list, kwargs):
        thetaE = lens_kwargs['theta_E']
        x_center = lens_kwargs['center_x']
        y_center = lens_kwargs['center_y']

        if lens_type == 'SIS':
            potential += Psi_SIS(x1, x2, x_center, y_center, thetaE)  # Make sure Psi_SIS is JAX-compatible
        elif lens_type == 'POINT_MASS':
            potential += Psi_PM(x1, x2, x_center, y_center, thetaE)  # Make sure Psi_PM is JAX-compatible
        elif lens_type == 'NFW':
            potential += Psi_NFW(x1, x2, x_center, y_center, thetaE, kappa=2)
        elif lens_type == 'SIE':

            e1 = lens_kwargs['e1']
            e2 = lens_kwargs['e2']
            potential += Psi_SIE(x1, x2, x_center, y_center, thetaE, e1, e2)
    geo = geometrical(x1, x2, y)
    # if geo<potential:
    #     fermat_potential = geo
    # else:
    fermat_potential = geo - potential
    print(geo, 'geo')
    print(potential, 'pot')
    # print(fermat_potential, 'fermat')
    return fermat_potential

