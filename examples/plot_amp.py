import sys
import os
path = os.getcwd()
dir = os.path.abspath(os.path.join(path, os.pardir))
sys.path.append(dir)
import amplification_factor.amplification_factor as af
import numpy as np

scale = np.sqrt(1)
Fws = np.loadtxt('./test_sis_Fws.txt', dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})/scale
ws = np.loadtxt('./test_sis_ws.txt')

amplification = af.amplification_factor()
amplification.importor(freq=True, ws=ws, Fws=Fws)
amplification.plot_freq(saveplot = './abs.pdf')
