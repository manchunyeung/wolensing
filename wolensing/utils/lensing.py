import numpy as np
import astropy.units as u
import lensinggw.constants.constants as const

Mpc = const.Mpc

def Einstein_radius(zL, zS, mL):
    '''
    :param zL: redshift where the lens locates
    :param zS: redshift where the source locates
    :param mL: lens mass
    :return: Einstein radius of the lens system
    '''

    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    cosmo = FlatLambdaCDM(H0=69.7, Om0=0.306, Tcmb0=2.725)

    DL       = cosmo.angular_diameter_distance(zL)
    DS       = cosmo.angular_diameter_distance(zS)
    DLS      = cosmo.angular_diameter_distance_z1z2(zL, zS)
    D        = DLS/(DL*DS)
    D        = np.float64(D/(Mpc))
    theta_E2 = (4*const.G*mL*const.M_sun*D)/const.c**2
    theta_E  = np.sqrt(theta_E2)

    return theta_E

def injection_radius(masses, mass_density, cosmo=None, zL=0.5):
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=69.7, Om0=0.306, Tcmb0=2.725)
    D_l = cosmo.angular_diameter_distance(zL) #MPc

    total_mass = np.sum(masses)
    area =  total_mass/mass_density # pc2
    radius = np.sqrt(area/np.pi) # pc radius in distant galaxy
    angle_rad = radius / (D_l*1e6/u.Mpc)
    # angle_ac = angle_rad / ac2rad
    return angle_rad

def field_injection(zL, zS, source_ra, source_dec, masses, radius, lens_model_list, kwargs_lens_list, seed=0):
    np.random.seed(seed)
    num_points = len(masses)
    angle_list = np.random.uniform(0, 2*np.pi, num_points)
    ym_list = np.random.uniform(0, radius, num_points)

    for i in range(num_points):
        thetaE = Einstein_radius(zL, zS, masses[i])
        eta0, eta1 = source_ra + ym_list[i]*np.cos(angle_list[i]), source_dec + ym_list[i]*np.sin(angle_list[i])
        lens_model_list.append('POINT_MASS')
        kwargs_lens_list.append({'center_x': eta0, 'center_y': eta1, 'theta_E': thetaE})
    return lens_model_list, kwargs_lens_list
