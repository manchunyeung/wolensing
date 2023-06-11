import sys
import os
path = os.getcwd()
dir = os.path.abspath(os.path.join(path, os.pardir))
sys.path.append(dir)
import amplification_factor_trial.amplification_factor as af
import numpy as np

scale = np.sqrt(11)
Fws = np.loadtxt('./data/test/test_Fws_100.00000_1.00000.txt', dtype=complex, converters={0: lambda s: complex(s.decode().replace('+-', '-'))})/scale
ws = np.loadtxt('./data/test/test_ws_100.00000_1.00000.txt')

amplification = af.amplification_factor_fd()
amplification.importor(ws, Fws)
amplification.plot(saveplot = './abs.pdf')
