import numpy as np
from fast_histogram import histogram1d
from tqdm import trange, tqdm
from utils.utils import gridfromcorn
import numba as nb
from lensmodels.potential import potential
import multiprocessing as mp
import jax.numpy as jnp
from jax import pmap, vmap, jit
import multiprocessing

def histogram_routine_gpu(lens_model_complete, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock, binnum,
                      binmin, binmax, Scale, kwargs_lens, y0, y1, dx):
    bincount = jnp.zeros(binnum, dtype=jnp.float64)
    # T = lens_model_complete.fermat_potential
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
                # Ts = Scale ** (-2) * T(X1, X2, kwargs_lens, y0, y1)
                bincount += jnp.histogram(Ts, binnum, (binmin, binmax))[0] * dx ** 2
                pbar.update(1)
                del X1, X2, Ts
                k+=1
    return bincount

def histogram_routine_cpu1(lens_model_complete, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock, binnum,
                      binmin, binmax, Scale, kwargs_lens, y0, y1, dx):
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

# def process_block(args):
#     i, j, lens_model_complete, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock, binnum, binmin, \
#     binmax, Scale, kwargs_lens, y, dx = args

#     if i in macroimindx[:, 0] and j in macroimindx[:, 1]:
#         return None

#     Nblock1 = Nblock
#     Nblock2 = Nblock
#     if i == Numblocks:
#         Nblock1 = Nresidue
#         if Nblock1 == 0:
#             return None
#     if j == Numblocks:
#         Nblock2 = Nresidue
#         if Nblock2 == 0:
#             return None

#     x1blockcorn = x1corn + i * Lblock
#     x2blockcorn = x2corn + j * Lblock
#     X1, X2 = gridfromcorn(x1blockcorn, x2blockcorn, dx, Nblock1, Nblock2)
#     Ts = Scale ** (-2) * T(lens_model_complete, X1, X2, y, kwargs_lens)
#     bincounts = np.histogram(Ts, binnum, (binmin, binmax))[0] * dx ** 2

#     return bincounts

# def histogram_routine_cpu(lens_model_complete, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock, binnum,
#                       binmin, binmax, Scale, kwargs_lens, y0, y1, dx):
#     bincount = np.zeros(binnum, dtype=np.float64)
#     y = np.array([y0, y1], dtype=np.float64)
#     print('start')
#     T = lens_model_complete.fermat_potential

#     total_iterations = (Numblocks + 1) ** 2

#     with multiprocessing.Pool() as pool:
#         args_list = [(i, j, T, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock,
#                       binnum, binmin, binmax, Scale, kwargs_lens, y, dx)
#                      for i in range(Numblocks + 1) for j in range(Numblocks + 1)]
#         with tqdm(total=total_iterations, desc='Integrating...') as pbar:
#             results = []
#             for result in pool.imap(process_block, args_list):
#                 if result is not None:
#                     bincount += result
#                 pbar.update(1)

#     return bincount

# def histogram_routine(lens_model_complete, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock, binnum,
#                       binmin, binmax, Scale, kwargs_lens, y0, y1, dx):
#     bincount = jnp.zeros(binnum, dtype=jnp.float64)
#     # T = lens_model_complete.fermat_potential
#     k = 0
#     y = jnp.array([y0, y1], dtype=jnp.float64)
#     print('start')

#     @jit
#     def process_block(i, j):
#         if i in macroimindx[:, 0] and j in macroimindx[:, 1]:
#             return jnp.zeros(binnum)

#         Nblock1 = Nblock
#         Nblock2 = Nblock
#         if i == Numblocks:
#             Nblock1 = Nresidue
#             if Nblock1 == 0:
#                 return jnp.zeros(binnum)
#         if j == Numblocks:
#             Nblock2 = Nresidue
#             if Nblock2 == 0:
#                 return jnp.zeros(binnum)

#         x1blockcorn = x1corn + i * Lblock
#         x2blockcorn = x2corn + j * Lblock
#         X1, X2 = gridfromcorn(x1blockcorn, x2blockcorn, dx, Nblock1, Nblock2)
#         Ts = Scale ** (-2) * potential(lens_model_complete, X1, X2, y, kwargs_lens)
#         bincounts = jnp.histogram(Ts, binnum, (binmin, binmax))[0] * dx ** 2

#         return bincounts

#     process_block_vmap = vmap(process_block)

#     with tqdm(total=(Numblocks + 1) ** 2, desc='Integrating...') as pbar:
#         for i in range(Numblocks + 1):
#             results = process_block_vmap(i, jnp.arange(Numblocks + 1))
#             bincount += jnp.sum(results, axis=0)
#             pbar.update(Numblocks + 1)

#     return bincount




# def process_block(args):
#     # lens_model_complete, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock, binnum, \
#     #     binmin, binmax, Scale, kwargs_lens, y, dx, i, j = args

#     T, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock, binnum, \
#         binmin, binmax, Scale, kwargs_lens, y0, y1, dx, i, j = args

#     if i in macroimindx[:, 0] and j in macroimindx[:, 1]:
#         return jnp.zeros(binnum, dtype=jnp.float64)

#     Nblock1 = Nblock
#     Nblock2 = Nblock
#     if i == Numblocks:
#         Nblock1 = Nresidue
#         if Nblock1 == 0:
#             return jnp.zeros(binnum, dtype=jnp.float64)
#     if j == Numblocks:
#         Nblock2 = Nresidue
#         if Nblock2 == 0:
#             return jnp.zeros(binnum, dtype=jnp.float64)

#     x1blockcorn = x1corn + i * Lblock
#     x2blockcorn = x2corn + j * Lblock
#     X1, X2 = gridfromcorn(x1blockcorn, x2blockcorn, dx, Nblock1, Nblock2)
#     # Ts = Scale ** (-2) * potential(lens_model_complete, X1, X2, y, kwargs_lens)
#     Ts = Scale ** (-2) * T(X1, X2, kwargs_lens, y0, y1)
#     bincounts = histogram1d(Ts, binnum, (binmin, binmax)) * dx ** 2

#     return bincounts

# def process_blocks(args):
#     T, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock, binnum, binmin, binmax, Scale, kwargs_lens, y, dx, i, j = args
#     args = (T, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock, binnum,
#              binmin, binmax, Scale, kwargs_lens, y, dx, i, j)
#     return process_block(args)
# def histogram_routine_cpu(lens_model_complete, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock, binnum,
#                       binmin, binmax, Scale, kwargs_lens, y0, y1, dx):

#     bincount = jnp.zeros(binnum , dtype=jnp.float64)
#     y = jnp.array([y0, y1], dtype=jnp.float64)
#     print('start')
#     total_blocks = (Numblocks + 1) ** 2

#     T = lens_model_complete.fermat_potential
#     # @pmap
#     # def process_blocks(i, j):
#     #     args = (lens_model_complete, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock, binnum,
#     #              binmin, binmax, Scale, kwargs_lens, y, dx, i, j)
#     #     return process_block(args)

#     # results = process_blocks(jnp.arange(Numblocks + 1), jnp.arange(Numblocks + 1))


#     pool = multiprocessing.Pool()
#     results = pool.starmap(process_blocks, [(T, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock, binnum, binmin, binmax, Scale, kwargs_lens, y, dx, i, j) for i in range(Numblocks + 1) for j in range(Numblocks + 1)])
#     pool.close()
#     pool.join()

#     bincount = jnp.sum(results, axis=0)

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

def histogram_routine_cpu(lens_model_complete, Numblocks, macroimindx, Nblock, Nresidue, x1corn, x2corn, Lblock, binnum,
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
