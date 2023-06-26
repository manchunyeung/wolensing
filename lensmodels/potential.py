import jax.numpy as jnp
from jax import jit
from .lens import *

@jit
def potential(x, y, kwargs):

    for lens_type in lens_model_list:
        if lens_type == 'SIS':
            potential += Psi_SIS(x)
        elif lens_type == 'POINT_MASS':
            potential += Psi_PM(x)
    
    geometrical = (jnp.linalg.norm(x-y)**2)/2
    fermat_potential = geometrical - potential
    return fermat_potential