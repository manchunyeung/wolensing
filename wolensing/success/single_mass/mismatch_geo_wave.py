from pycbc.waveform import get_fd_waveform
import numpy as np
from scipy.integrate import simps
import pycbc.psd as psd
from scipy.interpolate import griddata
from bisect import bisect_left
from pycbc.filter.matchedfilter import overlap

import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')



hp, hc = get_fd_waveform(approximant = 'IMRPhenomD',
                         mass1 = 30,
                         mass2 = 30,
                         delta_f = .04,
                         f_lower = 10,
                         f_final = 2000)

dfnoise = 0.04
noiselen = int(2001/dfnoise)+1
noise2 = psd.analytical.aLIGOaLIGODesignSensitivityT1800044(noiselen, dfnoise, 1)
noisef = np.linspace(1, 1 + dfnoise*(noiselen-1), num = noiselen)

fs = hp.sample_frequencies
i = bisect_left(fs, 10)
j = bisect_left(fs, 1950)
fs = fs[i:j]
hp = hp[i:j]
hc = hc[i:j]
noise2interpolated = griddata(noisef, noise2, fs)

run = '201201d'
ylist = np.linspace(0.4, 1.7, 53)
ylist = np.array([f"{y:.3f}".rstrip('0').rstrip('.') for y in ylist], dtype=float)
matchlist = np.zeros([len(ylist)])
# print(matchlist2D)
for i, y in enumerate(ylist):
    Fw = np.loadtxt('./data/fold_[{y}]_Fw.txt', dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})
    ws = np.loadtxt('./data/fold_[{y}]_ws.txt')
    Finterpolated = griddata(ws, F, fs)
    # plt.semilogx(ws/(2*np.pi),np.angle(F), label = 'm = {}, y = {}'.format(mass, y))
    # plt.show()
    # print(Finterpolated)
    hpL = Finterpolated*hp
    match_wave = hpL.match(hp, psd = noise2, low_frequency_cutoff=10, high_frequency_cutoff=2000)

    geo_array = np.loadtxt(f'[{y}]_geo.txt')
    tds = geo_array[0]
    mus = geo_array[1]
    ns = geo_array[2]
    from wolensing.utils.utils import compute_geometrical
    geoF = compute_geometrical(ws, mus, tds, ns)
    Finterpolated = griddata(ws, geoF, fs)

    hpL = Finterpolated*hp
    match_geo = hpL.match(hp, psd = noise2, low_frequency_cutoff=10, high_frequency_cutoff=2000)

    matchlist[i] = [match_wave[0], match_geo[0]]

np.savetxt('./matchlist.txt', matchlist)

plt.plot(ylist, matchlist[:,0], label = 'Wave')
plt.plot(ylist, matchlist[:,1], label = 'Geometrical', linestyle = '--')
plt.savefig('mismatch_geo_wave.pdf')

plt.show()
