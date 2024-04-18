import numpy as np
from fast_histogram import histogram1d
from tqdm import trange, tqdm
import numba as nb
import multiprocessing as mp
import jax.numpy as jnp
from jax import pmap, vmap, jit
import multiprocessing
from wolensing.utils.utils import gridfromcorn
from wolensing.lensmodels.potential import potential

def histogram_routine_gpu(lens_model_complete, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock, binnum,
                      binmin, binmax, Scale, kwargs_lens, y0, y1, dx):
    '''
    Obtain F(t) by constructing histogram of time delay function within the given window.

    :param lens_model_complete: lens models of the system.
    :param Numblocks: number of boxes with Nblock inside.
    :param macroimindx:
    :param Nblock: number of pixels in a box.
    :param Nredisude: number of pixels not in a box.
    :param x1corn: x-coordinate of the lower side of the window.
    :param x2corn: y-coordinate of the left side of the window.
    :param Lblock: size of the integration block.
    :param binnum: total number of steps.
    :param binmin: lower bound of the time integration.
    :param binmax: upper bound of the time integration.
    :param Scale: Einstein radius of the system.
    :param kwargs_lens: arguments of the lens models.
    :param y0: x-coordinate of source position.
    :param y1: y-coordinate of source position.
    :param dx: integration step of the window covering lens place.
    :return: histogram of F(t).
    '''
    bincount = jnp.zeros(binnum, dtype=jnp.float64)
    k = 0
    y = jnp.array([y0, y1], dtype=jnp.float64)
    print('start')
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
                Ts = Scale ** (-2) * potential(lens_model_complete, X1, X2, y, kwargs_lens)
                bincount += jnp.histogram(Ts, binnum, (binmin, binmax))[0] * dx ** 2
                pbar.update(1)
                del X1, X2, Ts
                k+=1
    return bincount

def histogram_routine_cpu(lens_model_complete, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock, binnum,
                      binmin, binmax, Scale, kwargs_lens, y0, y1, dx):
    '''
    Obtain F(t) by constructing histogram of time delay function within the given window.

    :param lens_model_complete: lens models of the system.
    :param Numblocks: number of boxes with Nblock inside.
    :param macroimindx:
    :param Nblock: number of pixels in a box.
    :param Nredisude: number of pixels not in a box.
    :param x1corn: x-coordinate of the lower side of the window.
    :param x2corn: y-coordinate of the left side of the window.
    :param Lblock: size of the integration block.
    :param binnum: total number of steps.
    :param binmin: lower bound of the time integration.
    :param binmax: upper bound of the time integration.
    :param Scale: Einstein radius of the system.
    :param kwargs_lens: arguments of the lens models.
    :param y0: x-coordinate of source position.
    :param y1: y-coordinate of source position.
    :param dx: integration step of the window covering lens place.
    :return: histogram of F(t).
    '''
    bincount = np.zeros(binnum, dtype=np.float64)
    T = lens_model_complete.fermat_potential
    k = 0
    y = np.array([y0, y1], dtype=np.float64)
    print('start')
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
                # Ts = Scale ** (-2) * potential(lens_model_complete, X1, X2, y, kwargs_lens)
                Ts = Scale ** (-2) * T(X1, X2, kwargs_lens, y0, y1)
                bincount += histogram1d(Ts, binnum, (binmin, binmax)) * dx ** 2
                pbar.update(1)
                del X1, X2, Ts
                k+=1
    return bincount

