import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from wolensing.utils import constants as const
import lensinggw.constants.constants as const1

Mpc = const1.Mpc
ac2rad = np.pi/648000

cosmo = FlatLambdaCDM(H0=69.7, Om0=0.306, Tcmb0=2.725)
masses = [0.0875411729158816, 0.1380156891218021, 0.37757290272476596, 0.08326002905591416, 0.11751088559578819, 0.0878737809276154, 0.14164220540181344, 0.32080191454865087, 0.1432894584167837, 0.2607705005317567, 0.09098854113286821, 0.636857294268748, 0.10244447554800015, 0.11147041258795794, 0.2124085350914183, 0.6266784519859449, 0.42496685401191886, 1.1390594002155605, 0.4254607239653152, 0.1671246164756932, 0.17585315448008126, 0.36911502207552227]

mass_density = 30 # M/pc2
def conversion(masses, mass_density, cosmo, zL=0.5):
    D_l = cosmo.angular_diameter_distance(zL) #MPc

    total_mass = np.sum(masses)
    area =  total_mass/mass_density # pc2
    radius = np.sqrt(area/np.pi) # pc radius in distant galaxy
    angle_rad = radius / (D_l*1e6/u.Mpc)
    angle_ac = angle_rad / ac2rad
    return angle_ac, angle_rad

print(conversion(masses, mass_density, cosmo))
angle_ac, angle_rad = conversion(masses, mass_density, cosmo)

mL = np.sum(masses)

zS = 1.
zL = 0.5

DL       = cosmo.angular_diameter_distance(zL)
DS       = cosmo.angular_diameter_distance(zS)
DLS      = cosmo.angular_diameter_distance_z1z2(zL, zS)
D        = DLS/(DL*DS)
D        = np.float64(D/(Mpc))
theta_E2 = (4*const.G*mL*const.M_sun*D)/const.c**2
theta_E  = np.sqrt(theta_E2)
from lensinggw.utils.utils import param_processing
param = param_processing(0.5, 1., mL)

print(param, theta_E, mL)

def convert_thetaE_to_M(angle_ac):
    return angle_ac**2 * const.c**2 / (4*const.G*D*const.M_sun)

print(convert_thetaE_to_M(angle_rad))



