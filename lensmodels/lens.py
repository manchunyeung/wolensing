import jax.numpy as jnp
import numpy as np
from jax import jit
# from numba import jit, njit

# @jit
def Psi_SIS(X1, X2, x_center, y_center, thetaE):
    # thetaE = kwargs['theta_E']
    # x_center = kwargs['center_x']
    # y_center = kwargs['center_y']
    x_shift = X1-x_center
    y_shift = X2-y_center
    shifted = np.array([x_shift, y_shift], dtype=jnp.float64)

    # return thetaE * jnp.sqrt(x_shift*x_shift + y_shift*y_shift)

    # np.savetxt('./sis1.txt', results)
    return thetaE * jnp.linalg.norm(shifted, axis=0)

@jit
def Psi_PM(X1, X2,x_center, y_center, thetaE):
    # thetaE = kwargs['theta_E']
    # x_center = kwargs['center_x']
    # y_center = kwargs['center_y']
    x_shift = X1-x_center
    y_shift = X2-y_center

    # r = jnp.sqrt(x_shift*x_shift + y_shift*y_shift)
    shifted = jnp.array([x_shift, y_shift], dtype=jnp.float64)
    # np.savetxt('./pm1.txt', thetaE**2 * jnp.log(jnp.linalg.norm(shifted, axis=0)))
    # phi = thetaE**2*jnp.log(r)
    # return phi
    return thetaE**2 * jnp.log(jnp.linalg.norm(shifted, axis=0))
