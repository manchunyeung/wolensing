import numpy as np
from wolensing.lensmodels.lens import *
from jax import jit

@jit
def geometrical(x1, x2, y):
    '''
    :param x1: x-coordinates of position on lens plane with respect to the window center.
    :param x2: y-coordinates of position on lens plane with respect to the window center.
    :param y: numpy array, source positions.
    :return: geometrical part of the time delay.
    '''
    x = jnp.array([x1, x2], dtype=jnp.float64)
    geo = (1/2) * jnp.linalg.norm(x-y[:, jnp.newaxis, jnp.newaxis], axis=0)**2
    return geo

def potential(lens_model_list, x1, x2, y, kwargs):
    '''
    :param lens_model_list: list of lens models.
    :param x1: x-coordinates of position on lens plane with respect to the window center.
    :param x2: y-coordinates of position on lens plane with respect to the window center.
    :param y: numpy array, source positions.
    :kwargs: arguemnts for the lens models.
    :return: time delay function.
    '''
    potential = jnp.float64(0.0)

    for lens_type, lens_kwargs in zip(lens_model_list, kwargs):
        thetaE = lens_kwargs['theta_E']
        x_center = lens_kwargs['center_x']
        y_center = lens_kwargs['center_y']

        if lens_type == 'SIS':
            potential += Psi_SIS(x1, x2, x_center, y_center, thetaE)  # Make sure Psi_SIS is JAX-compatible
        elif lens_type == 'POINT_MASS':
            potential += Psi_PM(x1, x2, x_center, y_center, thetaE)  # Make sure Psi_PM is JAX-compatible
        elif lens_type == 'NFW':
            potential += Psi_NFW(x1, x2, x_center, y_center, thetaE, kappa=3)
        elif lens_type == 'SIE':
            e1 = lens_kwargs['e1']
            e2 = lens_kwargs['e2']
            potential += Psi_SIE(x1, x2, x_center, y_center, thetaE, e1, e2)

    geo = geometrical(x1, x2, y)
    fermat_potential = geo - potential
    return fermat_potential

