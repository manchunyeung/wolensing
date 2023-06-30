import numpy as np
from numba import jit
from .lens import *
import jax

# @jit
def potential(lens_model_list, x1, x2, y, kwargs):
    potential = np.float32(0.0)
    geometrical = (1/2) * ((x1 - y[0])**2 + (x2 - y[1])**2)

    for lens_type, lens_kwargs in zip(lens_model_list, kwargs):
        thetaE = lens_kwargs['theta_E']
        x_center = lens_kwargs['center_x']
        y_center = lens_kwargs['center_y']

        if lens_type == 'SIS':
            potential += Psi_SIS(x1, x2, x_center, y_center, thetaE)  # Make sure Psi_SIS is JAX-compatible
        elif lens_type == 'POINT_MASS':
            potential += Psi_PM(x1, x2, x_center, y_center, thetaE)  # Make sure Psi_PM is JAX-compatible

    fermat_potential = geometrical - potential
    return fermat_potential

