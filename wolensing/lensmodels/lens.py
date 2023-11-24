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

def derivatives(x, y, b, t, q):
    """Returns the deflection angles.

    :param x: x-coordinate in image plane relative to center (major axis)
    :param y: y-coordinate in image plane relative to center (minor axis)
    :param b: critical radius
    :param t: projected power-law slope
    :param q: axis ratio
    :return: f_x, f_y
    """
    # elliptical radius, eq. (5)
    Z = np.empty(np.shape(x), dtype=complex)
    Z.real = q * x
    Z.imag = y
    R = np.abs(Z)
    R = np.maximum(R, 0.000000001)

    # angular dependency with extra factor of R, eq. (23)
    R_omega = Z * hyp2f1(1, t / 2, 2 - t / 2, -(1 - q) / (1 + q) * (Z / Z.conj()))

    # deflection, eq. (22)
    alpha = 2 / (1 + q) * (b / R) ** t * R_omega

    # return real and imaginary part
    alpha_real = np.nan_to_num(alpha.real, posinf=10**10, neginf=-(10**10))
    alpha_imag = np.nan_to_num(alpha.imag, posinf=10**10, neginf=-(10**10))

    return alpha_real, alpha_imag

def ellipticity2phi_q(e1, e2):
    """Transforms complex ellipticity moduli in orientation angle and axis ratio.

    :param e1: eccentricity in x-direction
    :param e2: eccentricity in xy-direction
    :return: angle in radian, axis ratio (minor/major)
    """
    phi = np.arctan2(e2, e1) / 2
    c = np.sqrt(e1**2 + e2**2)
    c = np.minimum(c, 0.9999)
    q = (1 - c) / (1 + c)
    return phi, q

def rotate(xcoords, ycoords, angle):
    """

    :param xcoords: x points
    :param ycoords: y points
    :param angle: angle in radians
    :return: x points and y points rotated ccw by angle theta
    """
    return xcoords * np.cos(angle) + ycoords * np.sin(angle), -xcoords * np.sin(angle) + ycoords * np.cos(angle)

def Psi_SIE(X1, X2, x_center, y_center, theta_E, e1, e2):
    gamma = 2
    t = gamma-1
    phi_G, q = ellipticity2phi_q(e1, e2)
    b = theta_E * jnp.sqrt(q)

    x_shift = X1-x_center
    y_shift = X2-y_center
       
    x_rotate, y_rotate = rotate(x_shift, y_shift, phi_G)
    alpha_x, alpha_y = derivatives(x_rotate, y_rotate, b, t, q)
    Psi = (x_rotate * alpha_x + y_rotate * alpha_y) / (2 - t)
    return Psi

def Psi_NFW(X1, X2, x_center, y_center, thetaE, kappa): 
    """

    :param xcoords: x points
    :param ycoords: y points
    :param angle: angle in radians
    :return: x points and y points rotated ccw by angle theta
    """
    
    x_shift = X1-x_center
    y_shift = X2-y_center
    # if isinstance(x_shift, ):
    # if x_shift == 0:
    #     if x_shift == y_shift:
    #         x_shift = 0.01
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
