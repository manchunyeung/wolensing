import numpy as np
import os
import sys
path = os.getcwd()
dir = os.path.abspath(os.path.join(path, os.pardir))
sys.path.append(dir)
from lensmodels.SIS import SIS

def morse_indices(Img_ra, Img_dec, kwargs_lens_list):
    center_x = kwargs_lens_list['center_x']
    center_y = kwargs_lens_list['center_y']
    theta_E = kwargs_lens_list['theta_E']

    for i in range(len(Img_ra)):
        # SIS = SIS(Img_ra[i], Img_dec[i], kwargs_lens_list)
        # f_xx, f_x, f_yx, f_yy = SIS.hessian(Img_ra[i], Img_dec[i], kwargs_lens_list)
        print(Img_ra[i], Img_dec[i], 'Img_ra and dec')
        f_xx, f_xy, f_yx, f_yy = SIS.hessian(Img_ra[i], Img_dec[i], theta_E, center_x, center_y)
        print(f_xx, f_xy, f_yx, f_yy)
        td_xx = 1-f_xx
        td_yy = 1-f_yy
        D = f_xx*f_yy - f_yx*f_xy
        print(D, 'D')
