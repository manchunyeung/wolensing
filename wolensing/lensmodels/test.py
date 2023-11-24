import numpy as np

import sys
import os
path = os.getcwd()
dir = os.path.abspath(os.path.join(path, os.pardir))
sys.path.append(dir)

from lenstronomy.LensModel.Profiles.sie import SIE
from lensmodels.lens import Psi_SIE

x = np.array([1])
y = np.array([2])
theta_E = 1.0
q = 0.9
phi_G = 1.0
import lenstronomy.Util.param_util as param_util
e1, e2 = param_util.phi_q2_ellipticity(1., 0.9)
values = Psi_SIE(x, y, 0, 0, theta_E, e1, e2)
values1 = SIE.function(x=x, y=y, theta_E=theta_E, e1=e1, e2=e2)
gamma = 2
assert values==values1

