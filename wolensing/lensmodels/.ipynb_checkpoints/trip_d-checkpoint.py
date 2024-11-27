import numpy as np
<<<<<<< HEAD
from scipy.special import airy as Air
from wolensing.lensmodels.hessian import Hessian_Td
from wolensing.utils.utils import Einstein_radius
from wolensing.utils.constants import c
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
Mpc = 3.085677581491367e+22


import sys
import os
dir = '/home/manchun.yeung/microlensing/wolensing/wolensing'
sys.path.append(dir)

from lensmodels.derivative import Gradient_SIE as sie_d

def analytic_fold(lens_model_list, x, y, source, kwargs, zL, zS, mL, fs):
    prefactor = 2**(5/6) * np.pi **(1/2) * np.exp(1j * np.pi * (3/2)) ** (5/2)
    # prefactor = 2**(5/6) * np.pi **(1/2)
    
    triple_d = total_triple_d(lens_model_list, x, y, kwargs)
    print(triple_d)
    phi_yyy = triple_d[0]
    print(phi_yyy, 'yyy')
    
    second_d = Hessian_Td(lens_model_list, x, y, kwargs, matrix=False)
    phi_xx = second_d[0]
    print(phi_xx, 'xx')

    y1, y2 = source

    
    cosmo = FlatLambdaCDM(H0=69.7, Om0=0.306, Tcmb0=2.725)
    DL       = cosmo.angular_diameter_distance(zL)
    DS       = cosmo.angular_diameter_distance(zS)
    DLS      = cosmo.angular_diameter_distance_z1z2(zL, zS)
    D = DS/(DL*DLS)
    print(D)
    D = np.float64(D/Mpc)
    print(D)
    
    ws = (1+zL) * Einstein_radius(zL, zS, mL) **2 * D * 2 * np.pi / c * fs
    # ws = fs * 8 * np.pi * (1+zL) * mL 
    ai, aip, bi, bip = Air(2**(1/3) * y2 * ws **(2/3) / abs(phi_yyy)**(1/3))
    
    function = ws**(1/6) / (abs(phi_xx) ** (1/2) * abs(phi_yyy)**(1/3)) * ai * np.exp(-1j * ws * y1**2/(2*phi_xx))
    return prefactor * function


def total_triple_d(lens_model_list, x, y, kwargs):
    # triple_d = np.float64(0)
    triple_d = np.array([0., 0., 0., 0.])TEMA
=======

def total_triple_d(lens_model_list, x, y, kwargs):
    triple_d = np.float64(0)
>>>>>>> 05c6cb0c90922e0b5a54961674e74efb6e9368dc
    
    for lens_type, lens_kwargs in zip(lens_model_list, kwargs):
        thetaE = np.float64(lens_kwargs['theta_E'])
        x_center = np.float64(lens_kwargs['center_x'])
        y_center = np.float64(lens_kwargs['center_y'])

        x_shift, y_shift = np.float64(x-x_center), np.float64(y-y_center)

        if lens_type == 'SIS':
<<<<<<< HEAD
            triple_d -= TripD_SIS(x_shift, y_shift, thetaE)
        elif lens_type == 'POINT_MASS':
            triple_d -= TripD_PM(x_shift, y_shift, thetaE)
        elif lens_type == 'SIE':
            triple_d -= TripD_SIE(x, y, b, s, q)
=======
            triple_d += TripD_SIS(x_shift, y_shift, thetaE)
        elif lens_type == 'POINT_MASS':
            triple_d += TripD_PM(x_shift, y_shift, thetaE)
>>>>>>> 05c6cb0c90922e0b5a54961674e74efb6e9368dc
    return triple_d
    
def TripD_SIS(x, y, thetaE):
    prefac = thetaE * np.power(np.sqrt((x**2+y**2)), -5)
    
    f_yyy = -3*x*x*y*prefac
    f_xxx = -3*y*y*x*prefac
    f_xxy = -y*(-2*x**2+y**2) * prefac
    f_yyx = -x*(-2*y**2+x**2) * prefac
<<<<<<< HEAD

    # total = f_yyy * y**3 + f_xxx * x**3 + 3 * f_xxy * (x**2 * y) + 3 * f_yyx * (y**2 * x)
    return f_xxx, f_xxy, f_yyx, f_yyy
=======
    
    total = f_yyy * y**5 + f_xxx * x**3 + 3 * f_xxy * (x**2 * y) + 3 * f_yyx * (y**2 * x)
    return total
>>>>>>> 05c6cb0c90922e0b5a54961674e74efb6e9368dc

def TripD_PM(x, y, thetaE):
    prefac = thetaE**2 * np.power((x**2 + y**2), -3)
    
    f_xxx = 2*(x**3-3*x*y**2) * prefac
    f_yyy = 2*(y**3-3*y*x**2) * prefac
    f_xxy = -2*y*(-3*x**2+y**2) * prefac
    f_yyx = -2*x*(-3*y**2+x**2) * prefac
    
<<<<<<< HEAD
    # total = f_yyy * y**3 + f_xxx * x**3 + 3 * f_xxy * (x**2 * y) + 3 * f_yyx * (y**2 * x)
    # return total
    return f_xxx, f_xxy, f_yyx, f_yyy

def TripD_SIE(x, y, b, s, q):
    
    def major_axis(x, y, b, s, q):
        f_x, f_y = sie_d(x, y, b, s, q)
        diff = 0.0000000001
        fx_pdx, fy_pdx = sie_d(x + diff, y, b, s, q)
        fx_mdx, fy_mdx = sie_d(x - diff, y, b, s, q)
    
        fx_pdy, fy_pdy = sie_d(x, y + diff, b, s, q)
        fx_mdy, fy_mdy = sie_d(x, y - diff, b, s, q)
    
        
        f_xxx = (-fx_mdx+2*f_x-fx_pdx) / diff**2
        f_xxy = (-fy_mdx+2*f_y-fy_pdx) / diff**2
        f_yyx = (-fx_mdy+2*f_x-fx_pdy) / diff**2
        f_yyy = (-fy_mdy+2*f_x-fy_pdy) / diff**2
        return f_xxx, f_xxy, f_yyx, f_yyy

    f__xxx, f__xxy, f__yyx, f__yyy = major_axis(x, y, b, s, q)
    
    f_xxx = np.cos(theta)**3 * f__yyy + 3 * np.sin(theta) * np.cos(theta)**2 * f__yyx + 3 * np.cos(theta) * np.sin(theta) * f__xxy + np.sin(theta) * f__xxx
    f_yyy = np.sin(theta)**3 * f__yyy - 3 * np.cos(theta) * np.sin(theta)**2 * f__yyx - 3 *  np.cos(theta) * np.sin(theta) * f__xxy + np.cos(theta) * f__xxx
    f_xxy = np.cos(theta)**2 * np.sin(theta) * f__yyy - (1/4) * (np.cos(theta) + 3 * np.cos(3 * theta)) * f__yyx - (1/2) * np.sin(theta) * ((1 + 3 * np.cos(2 * theta)) * f__xxy + np.sin(2 * theta) * f__xxx) 
    f_yyx = np.sin(theta)**2 * np.cos(theta) * f__yyy - (1/4) * (np.sin(theta) - 3 * np.sin(3 * theta)) * f__yyx - 2 * np.cos(theta) * ((-1 + 3 * np.cos(2 * theta)) * f__xxy + np.sin(2 * theta) * f__xxx) 

    return f_xxx, f_xxy, f_yyx, f_yyy
=======
    total = f_yyy * y**3 + f_xxx * x**3 + 3 * f_xxy * (x**2 * y) + 3 * f_yyx * (y**2 * x)
    return total
>>>>>>> 05c6cb0c90922e0b5a54961674e74efb6e9368dc
