import numpy as np

def total_triple_d(lens_model_list, x, y, kwargs):
    triple_d = np.float64(0)
    
    for lens_type, lens_kwargs in zip(lens_model_list, kwargs):
        thetaE = np.float64(lens_kwargs['theta_E'])
        x_center = np.float64(lens_kwargs['center_x'])
        y_center = np.float64(lens_kwargs['center_y'])

        x_shift, y_shift = np.float64(x-x_center), np.float64(y-y_center)

        if lens_type == 'SIS':
            triple_d += TripD_SIS(x_shift, y_shift, thetaE)
        elif lens_type == 'POINT_MASS':
            triple_d += TripD_PM(x_shift, y_shift, thetaE)
    return triple_d
    
def TripD_SIS(x, y, thetaE):
    prefac = thetaE * np.power(np.sqrt((x**2+y**2)), -5)
    
    f_yyy = -3*x*x*y*prefac
    f_xxx = -3*y*y*x*prefac
    f_xxy = -y*(-2*x**2+y**2) * prefac
    f_yyx = -x*(-2*y**2+x**2) * prefac
    
    total = f_yyy * y**5 + f_xxx * x**3 + 3 * f_xxy * (x**2 * y) + 3 * f_yyx * (y**2 * x)
    return total

def TripD_PM(x, y, thetaE):
    prefac = thetaE**2 * np.power((x**2 + y**2), -3)
    
    f_xxx = 2*(x**3-3*x*y**2) * prefac
    f_yyy = 2*(y**3-3*y*x**2) * prefac
    f_xxy = -2*y*(-3*x**2+y**2) * prefac
    f_yyx = -2*x*(-3*y**2+x**2) * prefac
    
    total = f_yyy * y**3 + f_xxx * x**3 + 3 * f_xxy * (x**2 * y) + 3 * f_yyx * (y**2 * x)
    return total