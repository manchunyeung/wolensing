import numpy as np
from bisect import bisect_left
import pycbc.psd as psd
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Tkagg')
import matplotlib.cm as cm
import lensinggw.constants.constants as const
import sys
from mpmath import hyp1f1, gamma
from argparse import ArgumentParser
from wolensing.utils.utils import smooth

GCSM = 2.47701878*1.98892*10**(-6) # G/c^3 *solar mass

#----------------------------------------------------
### command line options

parser = ArgumentParser()

# choose to plot absolute value or phase
parser.add_argument('-p', '--phase', action = 'store_true', default = False, help = 'plot the phase')
parser.add_argument('-a', '--absolute', action = 'store_true', default = False, help = 'plot the absolute value')
parser.add_argument('-b', '--both', action = 'store_true', default = False, help = 'plot both the absolute and phase')
parser.add_argument('-s', '--save', action = 'store_true', default = False, help = 'save the graph')

args = parser.parse_args()

#----------------------------------------------------
### settings

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

run = 'bilby'
code = 'f4'

G = const.G  # gravitational constant [m^3 kg^-1 s^-2]
c = const.c  # speed of light [m/s]
M_sun = const.M_sun  # Solar mass [Kg]

mtot = 100
zL = 0.5

freqfac = 8 * (1 + zL) * np.pi * mtot * M_sun * G / c ** 3

# Normalize the amplificaiton factor with respecet to the macromagnifcation
mu = 6
scale = np.sqrt(mu)

#----------------------------------------------------
# Import 

masslist = [50.]
distancelist = [1.]
colorlist = ['red', 'blue', 'green', 'purple', 'orange']

fig, ax1 = plt.subplots()

psdfs = np.linspace(5,2005,10000)
df = psdfs[1] - psdfs[0]
noise = np.sqrt(psd.analytical.aLIGOaLIGODesignSensitivityT1800044(len(psdfs), df, psdfs[0]))

minima = np.log10(1)
maxima = np.log10(30)

norm = matplotlib.colors.Normalize(vmin = minima, vmax = maxima, clip = True)
mapper = cm.ScalarMappable(norm = norm, cmap = cm.gist_rainbow)


n = 0

# for code in codelist:
for distance in distancelist:
    for mass in masslist:
        if code == 'None':
            ws = np.loadtxt('./data/{0}/{0}_ws_{1:1.5f}_{2:1.5f}.txt'.format(run, mass, distance))
            Fws = np.loadtxt('./data/{0}/{0}_Fws_{1:1.5f}_{2:1.5f}.txt'.format(run, mass, distance), dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})/scale

            # fs = np.loadtxt('./data/ws_lookup.txt'.format(run, mass, distance))
            # Fws = np.loadtxt('./data/lookup.txt'.format(run, mass, distance), dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})/scale
        elif code == 'arbitrary':
            # if n==0:
            #     ws = np.loadtxt('./data/{0}/{0}_1nearfield50ws_field1.00000.txt'.format(run, mass, distance))
            #     Fws = np.loadtxt('./data/{0}/{0}_1nearfield50Fws_field1.00000.txt'.format(run, mass, distance), dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})/scale
            # elif n==1: 
            #     ws = np.loadtxt('./data/{0}/{0}_true50ws_50.00000_1.00000.txt'.format(run, mass, distance))
            #     Fws = np.loadtxt('./data/{0}/{0}_true50Fws_50.00000_1.00000.txt'.format(run, mass, distance), dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})/scale
            ws = np.loadtxt('./data/{0}/{0}_type1normws_{1:1.5f}_{2:1.5f}.txt'.format(run, mass, distance))
            Fws = np.loadtxt('./data/{0}/{0}_type1normFws_{1:1.5f}_{2:1.5f}.txt'.format(run, mass, distance), dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})/scale
            # ws = np.loadtxt('./data/{0}/test14_ws_{1:1.5f}_{2:1.5f}.txt'.format(run, mass, distance))
            # Fws = np.loadtxt('./data/{0}/test14_Fws_{1:1.5f}_{2:1.5f}.txt'.format(run, mass, distance), dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})/scale
        else:
            ws = np.loadtxt('./data/{0}/{0}_{1}ws_{2:1.5f}_{3:1.5f}.txt'.format(run, code, mass, distance))
            Fws = np.loadtxt('./data/{0}/{0}_{1}Fws_{2:1.5f}_{3:1.5f}.txt'.format(run, code, mass, distance), dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})/scale
            # ws = np.loadtxt('./data/fs_100lookup.txt'.format(run, mass, distance)) * 2* np.pi
            # Fws = np.loadtxt('./data/100lookup.txt'.format(run, mass, distance), dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})/scale
            # ws = np.loadtxt('./data/{0}/{1}fs.txt'.format(run, code, mass, distance))
            # Fws = np.loadtxt('./data/{0}/{1}Fws.txt'.format(run, code, mass, distance), dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})/scale
 
        # smoothen the curve(s)
        from scipy.signal import savgol_filter
        Fa_fil = savgol_filter(np.abs(Fws), 51, 3)
        Fp_fil = savgol_filter(np.angle(Fws), 51, 3)


        # set the upper limit of the curve 
        i = bisect_left(fs, 2000)
        if args.absolute:
            ax1.plot(fs[:i], Fa_fil[:i], linewidth=1, label = r'$m = {} M_\odot$'.format(mass, distance), c = colorlist[n])
        elif args.phase:
            ax1.plot(fs[:i], Fp_fil[:i], linewidth=1, label = r'$\eta_m = {}\theta_m$'.format(mass, distance), c = colorlist[n])
        elif args.both:
            ax1.plot(fs, np.abs(Fws), linewidth=1, label = r'$m = {}, ym = {}$'.format(mass, distance), c = colorlist[n])
            ax1.plot(fs, smooth(phase,500), linewidth=1, label = r'$m = {}, ym = {}$'.format(mass, distance), c = colorlist[n])

        n += 1

ax1.set_xlim(10,2000)

ax2 = ax1.twinx()
ax2.loglog(psdfs, noise, ':', color = 'gray', alpha = 0.8)#, label = r'aLIGO PSD')

ax1.set_xlabel(r'Frequency (Hz)', fontsize = 14)
if args.absolute:
    ax1.set_ylabel(r'$|F|/\sqrt{\mu}$', fontsize = 14)
    yup = ax1.get_ylim()[1]
    ax1.semilogx(fs, [np.sqrt(mu)/scale for f in fs], color = 'black')
elif args.phase:
    ax1.set_ylabel(r'$args(F)$', fontsize = 14)
    ax1.semilogx(fs, [-np.pi/2 for f in fs], color = 'black')
    # ax1.semilogx(fs, [0 for f in fs], color = 'black')
ax1.tick_params(axis='x', labelsize=11)
ax1.tick_params(axis='y', labelsize=11)
ax1.grid(which = 'both', alpha = 0.5)
ax2.tick_params(axis='x', labelsize=11)
ax2.set_ylabel(r'strain (Hz$^{-1/2}$)', fontsize = 14)
fig.tight_layout()
h1, l1 = ax1.get_legend_handles_labels()
ax1.legend(h1, l1, ncol = 1, fontsize = 13, loc = 3)

if args.save:
    plt.savefig('./plots/{}_{}_{}_{}Fs.pdf'.format(run, code, mass, distance), bbox_inches = 'tight')

plt.show()
