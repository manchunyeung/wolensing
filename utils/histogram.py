import numpy as np
from fast_histogram import histogram1d
from tqdm import trange, tqdm
from utils.utils import gridfromcorn
from numba import jit
import numba as nb
from lensmodels.potential import potential
import multiprocessing as mp

# def histogram_routine(lens_model_complete, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock, binnum,
#                       binmin, binmax, Scale, kwargs_lens, y0, y1, dx):
#     bincount = np.zeros(binnum)
#     # T = lens_model_complete.fermat_potential
#     k = 0
#     y = np.array([y0, y1])
#     print('start')
#     with tqdm(total = (Numblocks + 1)**2, desc = 'Integrating...') as pbar:
#         for i in range(Numblocks + 1):
#             for j in range(Numblocks + 1):
#                 if i in macroimindx[:,0] and j in macroimindx[:,1]:
#                     pbar.update(1)
#                     continue
#                 Nblock1 = Nblock
#                 Nblock2 = Nblock
#                 if i == Numblocks:
#                     Nblock1 = Nresidue
#                     if Nblock1 == 0:
#                         pbar.update(1)
#                         continue
#                 if j == Numblocks:
#                     Nblock2 = Nresidue
#                     if Nblock2 == 0:
#                         pbar.update(1)
#                         continue
#                 x1blockcorn = x1corn + i * Lblock
#                 x2blockcorn = x2corn + j * Lblock
#                 X1, X2 = gridfromcorn(x1blockcorn, x2blockcorn, dx, Nblock1, Nblock2)
#                 # x = np.array([X1, X2])
#                 # np.savetxt('./X1.txt', X1)
#                 # np.savetxt('./X2.txt', X2)
#                 Ts = Scale ** (-2) * potential(lens_model_complete, X1, X2, y, kwargs_lens)
#                 # Ts = Scale ** (-2) * T(X1, X2, kwargs_lens, y0, y1)
#                 bincount += histogram1d(Ts, binnum, (binmin, binmax)) * dx ** 2
#                 # if k==50:
#                 #     print(binnum, binmin, binmax, dx)
#                 # np.savetxt('./Ts.txt', Ts)
#                 # exit()
#                 pbar.update(1)
#                 del X1, X2, Ts
#                 k+=1
#     # np.savetxt('./bin.txt', bincount)
#     return bincount


def process_block(args):
    # lens_model_complete, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock, binnum, \
    #     binmin, binmax, Scale, kwargs_lens, y, dx, i, j = args

    T, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock, binnum, \
        binmin, binmax, Scale, kwargs_lens, y0, y1, dx, i, j = args

    if i in macroimindx[:, 0] and j in macroimindx[:, 1]:
        return np.zeros(binnum)

    Nblock1 = Nblock
    Nblock2 = Nblock
    if i == Numblocks:
        Nblock1 = Nresidue
        if Nblock1 == 0:
            return np.zeros(binnum)
    if j == Numblocks:
        Nblock2 = Nresidue
        if Nblock2 == 0:
            return np.zeros(binnum)

    x1blockcorn = x1corn + i * Lblock
    x2blockcorn = x2corn + j * Lblock
    X1, X2 = gridfromcorn(x1blockcorn, x2blockcorn, dx, Nblock1, Nblock2)
    # Ts = Scale ** (-2) * potential(lens_model_complete, X1, X2, y, kwargs_lens)
    Ts = Scale ** (-2) * T(X1, X2, kwargs_lens, y0, y1)
    bincounts = histogram1d(Ts, binnum, (binmin, binmax)) * dx ** 2

    return bincounts

def histogram_routine(lens_model_complete, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock, binnum,
                      binmin, binmax, Scale, kwargs_lens, y0, y1, dx):
    bincount = np.zeros(binnum)
    y = np.array([y0, y1])
    print('start')
    T = lens_model_complete.fermat_potential

    with mp.Pool() as pool:
        total_blocks = (Numblocks + 1) ** 2

        # args = [(lens_model_complete, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock, binnum,
        #          binmin, binmax, Scale, kwargs_lens, y, dx, i, j)
        #         for i in range(Numblocks + 1) for j in range(Numblocks + 1)]

        args = [(T, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock, binnum,
            binmin, binmax, Scale, kwargs_lens, y0, y1, dx, i, j)
        for i in range(Numblocks + 1) for j in range(Numblocks + 1)]

        results = pool.map(process_block, args)
        for result in results:
            bincount += result

    return bincount

