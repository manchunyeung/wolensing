import numpy as np
import astropy.units as u
import wolensing.utils.constants as const
from lenstronomy.LensModel.lens_model import LensModel

Mpc = const.Mpc

def Einstein_radius(zL, zS, mL):
    '''
    Computes the Einstein radius of a given lens system

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

def Magnifications(Img_ra, Img_dec, lens_model_list, kwargs_lens_list, diff=None):

    r"""     
    Computes image magnifications for a given lens model 
     
    :param Img_ra: images right ascensions (arbitrary units)
    :type Img_ra: array
    :param Img_dec: images declinations (arbitrary units)
    :type Img_dec: array
    :param lens_model_list: names of the lens profiles to be considered for the lens model
    :type lens_model_list: list of strings
    :param kwargs_lens_list: keyword arguments of the lens parameters matching each lens profile in *lens_model_list*
    :type kwargs_lens_list: list of dictionaries
    :param diff: If set, computes the deflection as a finite numerical differential of the lensing
     potential. This differential is only applicable in the single lensing plane where the form of the lensing
     potential is analytically known. *Optional*, default is *None*
    :type diff: None or float
    
    :returns: images magnifications
    :rtype: array

    ref: lensinggw.utils.utils.magnifications
    """   
    # instantiate the lens model
    lens_model = LensModel(lens_model_list=lens_model_list)
    
    # magnifications
    mu = lens_model.magnification(Img_ra, Img_dec, kwargs_lens_list, diff=diff)
    
    return mu

def Time_delay(Img_ra, Img_dec, source_pos_x, source_pos_y, zL, zS, lens_model_list, kwargs_lens_list, scaled=False, scale_factor=None, cosmo=None):
    r"""    
    Computes image time delays for a given lens model 

    :param Img_ra: images right ascensions (arbitrary units)
    :type Img_ra: array
    :param Img_dec: images declinations (arbitrary units)
    :type Img_dec: array
    :param source_pos_x: source position x-coordinate (same units as Img_ra)
    :type source_pos_x: float
    :param source_pos_y: source position y-coordinate (same units as Img_dec)
    :type source_pos_y: float
    :param zL: lens redshift
    :type zL: float
    :param zS: source redshift
    :type zS: float
    :param lens_model_list: names of the lens profiles to be considered for the lens model
    :type lens_model_list: list of strings
    :param kwargs_lens_list: keyword arguments of the lens parameters matching each lens profile in *lens_model_list*
    :type kwargs_lens_list: list of dictionaries
    :param scaled: specifies if the input is given in arbitrary units, *optional*. If not specified, the input is assumed to be in radians
    :type scaled: bool
    :param scale_factor: scale factor, *optional*. Used to account for the proper conversion factor in the time delays when coordinates are given in arbitrary units, as per :math:`x_{a.u.} = x_{radians}/scale\_factor`. Only considered when *scaled* is *True*
    :type scale_factor: float
    :param cosmo: cosmology used to compute angular diameter distances, *optional*. If not specified, a FlatLambdaCDM instance with H0=69.7, Om0=0.306, Tcmb0=2.725 is considered
    :type cosmo: instance of the astropy cosmology class
    
    :returns: time delay in seconds
    :rtype: array

    """
    
    # set a default cosmology if not specified
    if cosmo is None:
        from astropy.cosmology import FlatLambdaCDM              
        cosmo = FlatLambdaCDM(H0=69.7, Om0=0.306, Tcmb0=2.725)
    
    # instantiate the lens model
    lens_model = LensModel(lens_model_list=lens_model_list)
    
    # transform the redshift to angular diameter distance
    DL         = cosmo.angular_diameter_distance(zL)
    DS         = cosmo.angular_diameter_distance(zS)
    DLS        = cosmo.angular_diameter_distance_z1z2(zL, zS)
    D          = DLS/(DL*DS)
    D          = np.float64(D/(Mpc))
    prefactor  = (1+zL)/(D*c)
    shift2     = (Img_ra-source_pos_x)**2+(Img_dec-source_pos_y)**2
    potential  = 0.0

    # compute the time delay, Eq. 6 of Diego et al https://www.aanda.org/articles/aa/pdf/2019/07/aa35490-19.pdf
    potential = lens_model.potential(Img_ra, Img_dec, kwargs_lens_list)    
    td        = prefactor*(0.5*shift2 - potential)
    
    # if scaled, account for it
    if (scaled):
        if scale_factor is None:
            raise ValueError("Must specify scale_factor when scaled=True")
        else:
            td *= scale_factor**2
            
    # normalize the time delays so that one of the images has delay '0'
    td = td - np.min(td)
        
    return td

def injection_radius(zL, masses, mass_density, cosmo=None):
    r'''
    Computes the injection radius for a given mass distribution

    :param masses: masses of the objects
    :type masses: array
    :param mass_density: mass density of the objects in M/pc2
    :type mass_density: float
    :param cosmo: cosmology used to compute angular diameter distances, *optional*. If not specified, a FlatLambdaCDM instance with H0=69.7, Om0=0.306, Tcmb0=2.725 is considered
    :type cosmo: instance of the astropy cosmology class
    :param zL: lens redshift
    :type zL: float

    :returns: injection radius in radians
    :rtype: float
 
    '''
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
    '''
    Computes the field injection for a given mass distribution

    :param zL: lens redshift
    :type zL: float
    :param zS: source redshift
    :type zS: float
    :param source_ra: source right ascension
    :type source_ra: float
    :param source_dec: source declination
    :type source_dec: float
    :param masses: masses of the objects
    :type masses: array
    :param radius: radius of the injection
    :type radius: float
    :param lens_model_list: names of the lens profiles to be considered for the lens model
    :type lens_model_list: list of strings
    :param kwargs_lens_list: keyword arguments of the lens parameters matching each lens profile in *lens_model_list*
    :type kwargs_lens_list: list of dictionaries
    :param seed: seed for the random number generator
    :type seed: int

    :returns: lens_model_list, kwargs_lens_list
    :rtype: list, list
    '''
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

def distance_to_angle(radius, redshift):
    '''
    Computes the angle from the distance
    :param radius: radius in Mpc
    :type radius: float
    :param redshift: redshift
    :type redshift: float

    :returns: angle in radians
    :rtype: float
    '''
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=69.7, Om0=0.306, Tcmb0=2.725)
    D = cosmo.angular_diameter_distance(redshift) #MPc
    
    angle_rad = radius / (D*1e6/u.Mpc)
    return angle_rad
    