import bilby
import numpy as np
from scipy.interpolate import griddata

def BBH_SLML_lens(freq, mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, luminosity_distance, theta_jn, phase, ra, dec, psi, M_lz, y, **kwargs):
    """
    Compatible with freqeuncy model of bilby WaveformGenerator. Creates a lal binary black hole waveform and compute the lensed waveform based on the lens parameters.
    """
    waveform_kwargs.update(kwargs)

    waveform_approximant = waveform_kwargs["waveform_approximant"]
    reference_frequency = waveform_kwargs["reference_frequency"]
    minimum_frequency = waveform_kwargs["minimum_frequency"]

    base_waveform = bilby.gw.source.lal_binary_black_hole(freq, mass_1=mass_1, mass_2=mass_2, 
                                                          a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, 
                                                          phi_12=phi_12, phi_jl=phi_jl, 
                                                          luminosity_distance=luminosity_distance, theta_jn=theta_jn, 
                                                          phase=phase, waveform_approximant=waveform_approximant, 
                                                          reference_frequency=reference_frequency, minimum_frequency=minimum_frequency,
                                                          ra=ra, dec=dec, psi=psi)        
    
    #import the pre-computed amplification factor
    ws = np.loadtxt('/mnt/c/users/simon/microlensing/wolensing/main/data/test/test_f4ws_20.00000_0.50000.txt')
    Fws = np.loadtxt('/mnt/c/users/simon/microlensing/wolensing/main/data/test/test_f4Fws_20.00000_0.50000.txt', dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})

    fs = ws/(2*np.pi)

    Finterpolated = griddata(fs, Fws, freq) 
    Finterpolated = np.array(Finterpolated)

    lens_waveform=dict()

    for i in base_waveform:
        lens_waveform[i] = base_waveform[i] * Finterpolated 
    return(lens_waveform)

