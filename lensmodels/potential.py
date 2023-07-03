import numpy as np
from numba import jit
from .lens import *
from jax import jit

@jit
def geometrical(x1, x2, y):
    # return (1/2) * ((x1 - y[0])**2 + (x2 - y[1])**2)
    x = jnp.array([x1, x2], dtype=jnp.float64)
    return (1/2) * jnp.linalg.norm(x-y[:, jnp.newaxis, jnp.newaxis], axis=0)**2
# @jit
def potential(lens_model_list, x1, x2, y, kwargs):
    potential = jnp.float64(0.0)
    # geometrical = (1/2) * ((x1 - y[0])**2 + (x2 - y[1])**2)

    for lens_type, lens_kwargs in zip(lens_model_list, kwargs):
        thetaE = lens_kwargs['theta_E']
        x_center = lens_kwargs['center_x']
        y_center = lens_kwargs['center_y']

        if lens_type == 'SIS':
            potential += Psi_SIS(x1, x2, x_center, y_center, thetaE)  # Make sure Psi_SIS is JAX-compatible
        elif lens_type == 'POINT_MASS':
            potential += Psi_PM(x1, x2, x_center, y_center, thetaE)  # Make sure Psi_PM is JAX-compatible

    geo = geometrical(x1, x2, y)
    fermat_potential = geo - potential
    return fermat_potential

