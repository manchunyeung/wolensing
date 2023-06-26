import numpy as np
from fast_histogram import histogram1d
from tqdm import trange, tqdm
from utils.utils import gridfromcorn
from numba import jit
from lensmodels.potential import potential

def histogram_routine(lens_model_complete, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock, binnum,
                      binmin, binmax, Scale, kwargs_lens, y0, y1, dx):
    bincount = np.zeros(binnum)
    # T = lens_model_complete.fermat_potential
    with tqdm(total = (Numblocks + 1)**2, desc = 'Integrating...') as pbar:
        for i in range(Numblocks + 1):
            for j in range(Numblocks + 1):
                if i in macroimindx[:,0] and j in macroimindx[:,1]:
                    pbar.update(1)
                    continue
                Nblock1 = Nblock
                Nblock2 = Nblock
                if i == Numblocks:
                    Nblock1 = Nresidue
                    if Nblock1 == 0:
                        pbar.update(1)
                        continue
                if j == Numblocks:
                    Nblock2 = Nresidue
                    if Nblock2 == 0:
                        pbar.update(1)
                        continue
                x1blockcorn = x1corn + i * Lblock
                x2blockcorn = x2corn + j * Lblock
                X1, X2 = gridfromcorn(x1blockcorn, x2blockcorn, dx, Nblock1, Nblock2)
                x = np.array([X1, X2])
                y = np.array([y0, y1])
                Ts = Scale ** (-2) * potential(x, y, kwargs_lens)
                bincount += histogram1d(Ts, binnum, (binmin, binmax)) * dx ** 2
                pbar.update(1)
                del X1, X2, Ts
    return bincount
