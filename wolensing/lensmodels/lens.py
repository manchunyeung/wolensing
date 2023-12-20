import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from scipy.special import hyp2f1

def Psi_SIS(X1, X2, x_center, y_center, thetaE):
    """
    Return the Psi of SIS model.
    
    :param X1: x-coordinate in image plane relative to center
    :param X2: y-coordinate in image plane relative to center
    :param x_center: x_coordinate of the window center
    :param y_center: y_coordinate of the window center
    :param thetaE: Einstein radius of the given lens model
    :return: deflecetion potential of SIS model
    """
    x_shift = X1-x_center
    y_shift = X2-y_center
    shifted = np.array([x_shift, y_shift], dtype=jnp.float64)

    Psi = thetaE * jnp.linalg.norm(shifted, axis=0)
    return Psi
    
@jit
def Psi_PM(X1, X2, x_center, y_center, thetaE): 
    """
    Return the Psi of point mass model.
    
    :param X1: x-coordinate in image plane relative to center
    :param X2: y-coordinate in image plane relative to center
    :param x_center: x_coordinate of the window center
    :param y_center: y_coordinate of the window center
    :param thetaE: Einstein radius of the given lens model
    :return: deflection potential of point mass model
    """
    x_shift = X1-x_center
    y_shift = X2-y_center
    shifted = jnp.array([x_shift, y_shift], dtype=jnp.float64)
    
    Psi = thetaE**2 * jnp.log(jnp.linalg.norm(shifted, axis=0))
    return Psi

@jit
def derivatives(x, y, b, s, q):
    """Returns df/dx and df/dy of the function."""
    psi = jnp.sqrt(q**2 * (s**2 + x**2) + y**2)
    f_x = (b / jnp.sqrt(1.0 - q**2) * jnp.arctan(jnp.sqrt(1.0 - q**2) * x / (psi + s)))
    f_y = (b/ jnp.sqrt(1.0 - q**2) * jnp.arctanh(jnp.sqrt(1.0 - q**2) * y / (psi + q**2 * s)))
    return f_x, f_y

@jit
def ellipticity2phi_q(e1, e2):
    """Transforms complex ellipticity moduli in orientation angle and axis ratio.

    :param e1: eccentricity in x-direction
    :param e2: eccentricity in xy-direction
    :return: angle in radian, axis ratio (minor/major)
    """
    phi = jnp.arctan2(e2, e1) / 2
    c = jnp.sqrt(e1**2 + e2**2)
    c = jnp.minimum(c, 0.9999)
    q = (1 - c) / (1 + c)
    return phi, q

@jit
def rotate(xcoords, ycoords, angle):
    """

    :param xcoords: x points
    :param ycoords: y points
    :param angle: angle in radians
    :return: x points and y points rotated ccw by angle theta
    """
    return xcoords * jnp.cos(angle) + ycoords * jnp.sin(angle), -xcoords * jnp.sin(angle) + ycoords * jnp.cos(angle)

def Psi_SIE(X1, X2, x_center, y_center, theta_E, e1, e2):
    """
    Return the Psi of SIE model.
    
    :param X1: x-coordinate in image plane relative to center
    :param X2: y-coordinate in image plane relative to center
    :param x_center: x_coordinate of the window center
    :param y_center: y_coordinate of the window center
    :param thetaE: Einstein radius of the given lens model
    :param e1: ellipticity
    :param e2: ellipticity
    :return: deflecetion potential of SIE model
    """
    gamma = 2
    t = gamma-1
    phi_G, q = ellipticity2phi_q(e1, e2)
    theta_E = theta_E / (jnp.sqrt((1.+q**2) / (2. * q)))
    b = theta_E * jnp.sqrt((1+q**2)/2)
    s_scale = 0.0000000001
    s = s_scale * jnp.sqrt((1 + q**2) / (2*q**2))

    x_shift = X1-x_center
    y_shift = X2-y_center   
    x_rotate, y_rotate = rotate(x_shift, y_shift, phi_G)
    
    psi = jnp.sqrt(q**2 * (s**2 + x_rotate**2) + y_rotate**2)
    if q>=1:
        q = 0.99999999
    alpha_x, alpha_y = derivatives(x_rotate, y_rotate, b, s, q)
    
    f_ = (x_rotate * alpha_x + y_rotate * alpha_y - b * s * 1.0 / 2.0 * jnp.log((psi + s) ** 2 + (1.0 - q**2) * x_rotate**2))
    return f_


def Psi_NFW(X1, X2, x_center, y_center, thetaE, kappa): 
    """

    :param xcoords: x points
    :param ycoords: y points
    :param angle: angle in radians
    :return: x points and y points rotated ccw by angle theta
    """
    
    x_shift = X1-x_center
    y_shift = X2-y_center
    shifted = np.array([x_shift, y_shift], dtype=np.float64) 
    x_norm = np.linalg.norm(shifted, axis=0)
    
    if x_norm<1:
        if x_norm<1e-7:
            print('True')
            y = np.sqrt(1-x_norm**2)
            print(((1/2) * (np.log(1+y)+y)))
            Psi = kappa / 2 * (1 - ((1/2) * (np.log(1+y)+y))) *  thetaE
            print(Psi, 'si')
        else:
            Psi = kappa / 2 * (np.log(x_norm/2)**2 - np.arctanh(np.sqrt(1-x_norm**2))**2) *  thetaE
    else:
        Psi = kappa / 2 * (np.log(x_norm/2)**2 + np.arctan(np.sqrt(x_norm**2 - 1))**2) * thetaE
    # x_safe_low = jnp.where(x_norm<1, x, 0.5*dim_1)
    # x_safe_hi = jnp.where(x_norm<1, 2*dim_1, x)
    # x_safe_low_norm = jnp.linalg.norm(x_safe_low)
    # x_safe_hi_norm = jnp.linalg.norm(x_safe_hi)
    # Psi = jnp.where(x_norm<1,
    #     kappa / 2 * (jnp.log(x_safe_low_norm/2)**2 - jnp.arctanh(jnp.sqrt(1-x_safe_low_norm**2))**2),
    #     kappa / 2 * (jnp.log(x_safe_hi_norm/2)**2 + jnp.arctan(jnp.sqrt(x_safe_hi_norm**2 - 1))**2))
    return Psi
