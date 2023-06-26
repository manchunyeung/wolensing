import jax.numpy as jnp
from jax import jit

@jit
def Psi_SIS(x):
    return jnp.linalg.norm(x)

@jit
def Psi_PM(x):
    return jnp.log(jnp.linalg.norm(x))
