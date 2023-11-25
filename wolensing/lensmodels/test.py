import numpy as np

import sys
import os
path = os.getcwd()
dir = os.path.abspath(os.path.join(path, os.pardir))
sys.path.append(dir)

from lenstronomy.LensModel.lens_model import LensModel
from lensmodels.lens import Psi_SIE

theta_E = 1.0
x = np.array([2])
y = np.array([0])
q = 0.9
phi_G = 1.0
import lenstronomy.Util.param_util as param_util
e1, e2 = param_util.phi_q2_ellipticity(1., 0.9)
values = Psi_SIE(x, y, 0, 0, theta_E, e1, e2)

lens_model_complete = LensModel(lens_model_list=['SIE'])
T = lens_model_complete.fermat_potential

kwargs_sis_1 = [{'center_x': 0, 'center_y': 0, 'theta_E': theta_E, 'e1':e1, 'e2':e2}]

# x,y image position
values1 = T(2, 0, kwargs_sis_1, 0, 0)
print(values, values1)
