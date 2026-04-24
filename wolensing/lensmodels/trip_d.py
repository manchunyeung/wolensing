import numpy as np
from scipy.special import airy as Air
from wolensing.lensmodels.hessian import Hessian_Td
from wolensing.utils.lensing import Einstein_radius
from wolensing.utils.constants import c
from astropy.cosmology import FlatLambdaCDM

Mpc = 3.085677581491367e+22


def analytic_fold(lens_model_list, x, y, source, kwargs, zL, zS, mL, fs):
    prefactor = (
        2 ** (5 / 6)
        * np.pi ** (1 / 2)
        * np.exp(1j * np.pi * (3 / 2)) ** (5 / 2)
    )

    triple_d = total_triple_d(lens_model_list, x, y, kwargs)
    phi_yyy = triple_d[0]

    second_d = Hessian_Td(lens_model_list, x, y, kwargs, matrix=False)
    phi_xx = second_d[0]

    y1, y2 = source

    cosmo = FlatLambdaCDM(H0=69.7, Om0=0.306, Tcmb0=2.725)
    DL = cosmo.angular_diameter_distance(zL)
    DS = cosmo.angular_diameter_distance(zS)
    DLS = cosmo.angular_diameter_distance_z1z2(zL, zS)
    D = np.float64((DS / (DL * DLS)) / Mpc)

    ws = (1 + zL) * Einstein_radius(zL, zS, mL) ** 2 * D * 2 * np.pi / c * fs
    ai, _aip, _bi, _bip = Air(2 ** (1 / 3) * y2 * ws ** (2 / 3) / abs(phi_yyy) ** (1 / 3))

    function = (
        ws ** (1 / 6)
        / (abs(phi_xx) ** (1 / 2) * abs(phi_yyy) ** (1 / 3))
        * ai
        * np.exp(-1j * ws * y1**2 / (2 * phi_xx))
    )
    return prefactor * function


def total_triple_d(lens_model_list, x, y, kwargs):
    """
    Third derivatives of the time delay function.

    Returns a vector ordered as (phi_yyy, phi_xxx, phi_xxy, phi_yyx).
    The geometrical term contributes no third derivatives, so this is minus
    the third derivatives of the lens potential.
    """
    triple_d = np.zeros(4, dtype=np.float64)

    for lens_type, lens_kwargs in zip(lens_model_list, kwargs):
        thetaE = np.float64(lens_kwargs['theta_E'])
        x_center = np.float64(lens_kwargs['center_x'])
        y_center = np.float64(lens_kwargs['center_y'])

        x_shift, y_shift = np.float64(x-x_center), np.float64(y-y_center)

        if lens_type == 'SIS':
            triple_d -= TripD_SIS(x_shift, y_shift, thetaE)
        elif lens_type == 'POINT_MASS':
            triple_d -= TripD_PM(x_shift, y_shift, thetaE)
        elif lens_type == 'SIE':
            e1 = lens_kwargs['e1']
            e2 = lens_kwargs['e2']
            triple_d -= TripD_SIE(x_shift, y_shift, thetaE, e1, e2)
    return triple_d
    
def TripD_SIS(x, y, thetaE):
    prefac = thetaE * np.power(np.sqrt((x**2+y**2)), -5)
    
    f_yyy = -3*x*x*y*prefac
    f_xxx = -3*y*y*x*prefac
    f_xxy = -y*(-2*x**2+y**2) * prefac
    f_yyx = -x*(-2*y**2+x**2) * prefac
    return np.array([f_yyy, f_xxx, f_xxy, f_yyx], dtype=np.float64)

def TripD_PM(x, y, thetaE):
    prefac = thetaE**2 * np.power((x**2 + y**2), -3)
    
    f_xxx = 2*(x**3-3*x*y**2) * prefac
    f_yyy = 2*(y**3-3*y*x**2) * prefac
    f_xxy = -2*y*(-3*x**2+y**2) * prefac
    f_yyx = -2*x*(-3*y**2+x**2) * prefac
    
    return np.array([f_yyy, f_xxx, f_xxy, f_yyx], dtype=np.float64)


def TripD_SIE(x, y, theta_E, e1, e2, diff=1e-4):
    """
    Numerical third derivatives of the SIE potential, returned as
    (psi_yyy, psi_xxx, psi_xxy, psi_yyx).
    """

    def alpha(x0, y0):
        # Gradient_SIE returns (alpha_x, alpha_y)
        from .derivative import Gradient_SIE

        return Gradient_SIE(x0, y0, theta_E, e1, e2)

    def psi_xx(x0, y0):
        ax_p, _ay_p = alpha(x0 + diff, y0)
        ax_m, _ay_m = alpha(x0 - diff, y0)
        return (ax_p - ax_m) / (2.0 * diff)

    def psi_yy(x0, y0):
        _ax_p, ay_p = alpha(x0, y0 + diff)
        _ax_m, ay_m = alpha(x0, y0 - diff)
        return (ay_p - ay_m) / (2.0 * diff)

    psi_xxx = (psi_xx(x + diff, y) - psi_xx(x - diff, y)) / (2.0 * diff)
    psi_xxy = (psi_xx(x, y + diff) - psi_xx(x, y - diff)) / (2.0 * diff)
    psi_yyx = (psi_yy(x + diff, y) - psi_yy(x - diff, y)) / (2.0 * diff)
    psi_yyy = (psi_yy(x, y + diff) - psi_yy(x, y - diff)) / (2.0 * diff)

    return np.array([psi_yyy, psi_xxx, psi_xxy, psi_yyx], dtype=np.float64)
