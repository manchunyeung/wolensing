from pycbc.waveform import get_fd_waveform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.integrate import simps
import pycbc.psd as psd
from scipy.interpolate import griddata
from bisect import bisect_left
from pycbc.filter.matchedfilter import overlap, match

plt.rc('text', usetex=False)
plt.rc('font', family='serif')


def fmt(x):
    return rf"{x} \%" if plt.rcParams["text.usetex"] else f"{x} %"

hp, hc = get_fd_waveform(approximant = 'IMRPhenomXPHM',
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

# tdiff=-0.00003
mass = 10
y = 1
# F = np.loadtxt('./cluster_data/test20/test20_type1normFws_{0:1.5f}_{1:1.5f}.txt'.format(mass, y), dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})
# ws = np.loadtxt('./cluster_data/test20/test20_type1normws_{0:1.5f}_{1:1.5f}.txt'.format(mass, y))
# F1 = np.loadtxt('./cluster_data/test15/test15_normFws_{0:1.5f}_{1:1.5f}.txt'.format(mass, y), dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})
# ws1 = np.loadtxt('./cluster_data/test15/test15_normws_{0:1.5f}_{1:1.5f}.txt'.format(mass, y))
# F = np.loadtxt('../main/data/bilby/bilby_Fws_{0:1.5f}_{1:1.5f}.txt'.format(mass, y), dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})
# ws = np.loadtxt('../main/data/bilby/bilby_ws_{0:1.5f}_{1:1.5f}.txt'.format(mass, y))
# F = np.loadtxt('./cluster_data/201130/201130_Fws_{0:1.5f}_{1:1.5f}.txt'.format(mass, y), dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})
# ws = np.loadtxt('./cluster_data/201130/201130_ws_{0:1.5f}_{1:1.5f}.txt'.format(mass, y))
# fs = ws/(2*np.pi)


# Finterpolated = griddata(ws/(2*np.pi), F, fs)
# h1norm = Finterpolated*hp
# match = hp1norm.match(h1norm, psd = noise2, low_frequency_cutoff=10, high_frequency_cutoff=2000)
# print('match', match)
# quit()

# Tds = 5
# overall_phase = np.exp(-1 * 2 * np.pi * 1j * (Tds) * fs)
# hp *= overall_phase

# run='dthe'
# mass=100
# y=1

# thelist=[60]
matchlist =[]

# for i, the in enumerate(thelist):
F1 = np.loadtxt('./complete.txt', dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})
gf = np.loadtxt('./gf.txt')
Finterpolated = griddata(gf, F1, fs)

F2 = np.loadtxt('./Fws.txt', dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})
ws = np.loadtxt('./ws.txt')
Finterpolated1 = griddata(ws, F2, fs)

hpL = Finterpolated*hp
hpA = Finterpolated1*hp
# match = hpL.match(hp, psd = noise2, low_frequency_cutoff=10, high_frequency_cutoff=2000)
match1 = match(hpL, hpA, psd = noise2, low_frequency_cutoff=10, high_frequency_cutoff = 2000)
matchlist.append(100*(1-match1[0]))

print(matchlist)

# run = 'test'
# mass = 100
# y=1
# F = np.loadtxt('./data/{0}/{0}_dmFws_{1:1.5f}_{2:1.5f}.txt'.format(run, mass, y, the), dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})
# ws = np.loadtxt('./data/{0}/{0}_dmws_{1:1.5f}_{2:1.5f}.txt'.format(run, mass, y, the))
# Finterpolated = griddata(ws/(2*np.pi), F, fs)
# hpL = Finterpolated*hp
# # match = hpL.match(hp, psd = noise2, low_frequency_cutoff=10, high_frequency_cutoff=2000)
# match1 = match(hpL, hp, psd = noise2, low_frequency_cutoff=10, high_frequency_cutoff = 2000)
# print(1-match1[0])

# print(np.angle(F1), 'F1')
# print(np.angle(F))
# print(np.angle(F1)-np.angle(F))

# fig, ax = plt.subplots()

# ax.plot(thelist, matchlist)
# # plt.xlabel(u'm $(M_\u2609)$')
# plt.xlabel(u'$\phi (degrees)$', fontsize = 12)
# plt.ylabel(r'$Mismatch (\%)$', fontsize = 12)
# # plt.clabel(CS2)
# plt.savefig('mismatch_angle_3.pdf')
# plt.show()
