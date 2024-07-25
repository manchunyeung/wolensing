import numpy as np

tds = np.loadtxt('./tds.txt')[1:]
mus = np.loadtxt('./mus.txt')[1:]
Img_ra= np.loadtxt('./ra.txt')[1:]
Img_dec = np.loadtxt('./dec.txt')[1:]

geofs, geoFws = amplification.geometrical_optics(mus, tds, Img_ra, Img_dec, upper_lim=2000)

from wolensing.utils.utils import *
ns = Morse_indices(amplification._lens_model_list, Img_ra, Img_dec, amplification._kwargs_lens)
