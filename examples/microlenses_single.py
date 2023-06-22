#!/users/man-chun.yeung/microlensing/env/bin/python3

import sys
import os
path = os.getcwd()
dir = os.path.abspath(os.path.join(path, os.pardir))
sys.path.append(dir)

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from lenstronomy.LensModel.lens_model import LensModel
import lensinggw.constants.constants as const
from lensinggw.utils.utils import TimeDelay, magnifications, getMinMaxSaddle
from lensinggw.amplification_factor.amplification_factor import geometricalOpticsMagnification

from plot.plot import plot_contour
import amplification_factor_trial.amplification_factor as af
import lensmodels.morse_indices as morse
#----------------------------------------------------
### command line options

parser = ArgumentParser()

# usual arguments
parser.add_argument('ym', type = float, help = 'ym')
parser.add_argument('mass', type = float, help = 'mass of the microlens')
parser.add_argument('angle', type = float, help = 'angular position of the microlense with respect to the macro')
parser.add_argument('pixel', type = float, help = 'Pixel of the window')
parser.add_argument('run', help = 'run name')

# plotting mode (contours)
parser.add_argument('-src', '--src', type = float, default = 0.1, help = 'source position')
parser.add_argument('-p', '--plot', action = 'store_true', default = False, help = 'Plot out the images with contour lines')
parser.add_argument('-t2', '--type2', action = 'store_true', default = False)
# parser.add_argument('-c', '--code', metavar = 'filename', type = str, help = 'special code for the run')
parser.add_argument('-sf', '--sampling_frequency', type = float, default = 0.25, help = 'choosing a specific sampling frequency (Hz)')

args = parser.parse_args()

# Update args with a respecetive kwargs set for the type of macroimage
kwargs_type = vars(args)
if args.type2:
    kwargs_type.update({'LastImageT': 4e-7, 'TExtend': 7., 'mu': 9, 'TimeMax': 7., 'TimeMin': 5., 'TimeLength': 12, 'Timg': 59015, 'TimeStep': 1e-6, 'Winfac': 10.})
else:
    df = args.sampling_frequency
    textendmax = 1/df
    tlength = .13
    textend = textendmax-tlength
    textend = 10
    print(textend, 'textend')
    kwargs_type.update({'LastImageT': .02, 'TExtend': textend, 'mu': 11., 'TimeMax': 1, 'TimeMin': .1, 'TimeLength': tlength, 'Timg': 118211.81107161, 'TimeStep': 1e-5, 'Winfac': 1.})

#----------------------------------------------------

G = const.G  # gravitational constant [m^3 kg^-1 s^-2]
c = const.c  # speed of light [m/s]
M_sun = const.M_sun  # Solar mass [Kg]

if not args.type2: #wanna delete, contradicts with .plot
    imindex = 1
else:
    imindex = 0

THIS_DIR = os.getcwd()
DATA_DIR = os.path.join(THIS_DIR,'data')
run = args.run

if args.plot:
    contourtxtname = os.path.join(DATA_DIR,'{}_contour.png'.format(run))
    contourtxtnamemicro = os.path.join(DATA_DIR,'{0}_contourmicro_{1}.png'.format(run, imindex))
    contourtxtnamemicromicro = os.path.join(DATA_DIR,'{0}_contourmicromicro_{1}.png'.format(run, imindex))
    imagesnpzname = os.path.join(DATA_DIR,'{}_images.npz'.format(run))

# coordinates in scaled units [x (radians) /thetaE_tot]
y0, y1 = args.src, 0 # source position
l0, l1 = 0.05, 0 # lens position

ym = args.ym
angle = np.radians(float(args.angle))
zS = 1.0
zL = 0.5

# mlist = np.logspace(0,2,num = int(50))
# masses
mlist = [args.mass]
for mL2 in mlist:
    mL1 = 1 * 1e10
    mL1 = 0.00001
    mL3 = 10
    mtot = mL1 + mL2

    # convert to radians
    from lensinggw.utils.utils import param_processing

    thetaE1 = param_processing(zL, zS, mL1)
    thetaE2 = param_processing(zL, zS, mL2)
    thetaE3 = param_processing(zL, zS, mL3)
    thetaE = param_processing(zL, zS, mtot)


    beta0, beta1 = y0 * thetaE, y1 * thetaE
    eta10, eta11 = 0 * l0 * thetaE, 0 * l1 * thetaE

    lens_model_list = ['SIS']
    kwargs_sis_1 = {'center_x': eta10, 'center_y': eta11, 'theta_E': thetaE1}
    kwargs_lens_list = [kwargs_sis_1]
    
    print('thetaE1 and thetaE', thetaE1, thetaE)
    kwargs_sis_1_scaled = {'center_x': eta10 / thetaE, 'center_y': eta11 / thetaE, 'theta_E': thetaE1 / thetaE}
    kwargs_lens_list_scaled = [kwargs_sis_1_scaled]
    from lensinggw.solver.images import microimages

    solver_kwargs = {'SearchWindowMacro': 10 * thetaE1,
                     'SearchWindow': 5 * thetaE2,
                     'OverlapDistMacro': 1e-17,
                     'OnlyMacro': True}

    MacroImg_ra, MacroImg_dec, pixel_width = microimages(source_pos_x=beta0,
                                                         source_pos_y=beta1,
                                                         lens_model_list=lens_model_list,
                                                         kwargs_lens=kwargs_lens_list,
                                                         **solver_kwargs)

    Macromus = magnifications(MacroImg_ra, MacroImg_dec, lens_model_list, kwargs_lens_list)
    T01 = TimeDelay(MacroImg_ra, MacroImg_dec,
                    beta0, beta1,
                    zL, zS,
                    lens_model_list, kwargs_lens_list)

    ns = getMinMaxSaddle(MacroImg_ra, MacroImg_dec, lens_model_list, kwargs_lens_list, diff = None)
    ns_1 = morse.morse_indices(MacroImg_ra, MacroImg_dec, kwargs_sis_1)
    print(T01, ns, ns_1)

    if args.type2:
        imindex = np.nonzero(T01)[0][0]
    else:
        imindex = np.where(T01==0)[0][0]
    args.mu = Macromus[imindex]

    # # lens model
    eta20, eta21 = MacroImg_ra[imindex] + np.cos(angle)*ym*thetaE2, MacroImg_dec[imindex] + np.sin(angle)*ym*thetaE2

    lens_model_list = ['SIS', 'SIS']
    kwargs_sis_1 = {'center_x': eta10, 'center_y': eta11, 'theta_E': thetaE1}
    kwargs_point_mass_2 = {'center_x': eta20, 'center_y': eta21, 'theta_E': thetaE2}
    kwargs_lens_list = [kwargs_sis_1, kwargs_point_mass_2]

    from lensinggw.solver.images import microimages

    solver_kwargs = {'SearchWindowMacro': 10 * thetaE1,
                     'SearchWindow': 10 * thetaE3,
                     'Pixels': 1e3,
                     'OverlapDist': 1e-18,
                     # 'PrecisionLimit': 1e-21,
                     'OverlapDistMacro': 1e-17}
    solver_kwargs.update({'Improvement' : 0.1})
    solver_kwargs.update({'MinDist' : 10**(-7)})

    # if not args.type2:
    Img_ra, Img_dec, MacroImg_ra, MacroImg_dec, pixel_width = microimages(source_pos_x=beta0,
                                                                          source_pos_y=beta1,
                                                                          lens_model_list=lens_model_list,
                                                                          kwargs_lens=kwargs_lens_list,
                                                                          **solver_kwargs)

    Images_dict = {'Source_ra': beta0,
                   'Source_dec': beta1,
                   'Img_ra': Img_ra,
                   'Img_dec': Img_dec,
                   'MacroImg_ra': MacroImg_ra,
                   'MacroImg_dec': MacroImg_dec,
                   'Microlens_ra': [eta20],
                   'Microlens_dec': [eta21],
                   'thetaE': thetaE}
    # else:
    Img_ra, Img_dec = MacroImg_ra, MacroImg_dec
    
    # time delays, magnifications, Morse indices and amplification factor
    from lensinggw.utils.utils import TimeDelay, magnifications, getMinMaxSaddle
    from lensinggw.amplification_factor.amplification_factor import geometricalOpticsMagnification

    tds = TimeDelay(Img_ra, Img_dec,
                   beta0, beta1,
                   zL, zS,
                   lens_model_list, kwargs_lens_list)
    mus = magnifications(Img_ra, Img_dec, lens_model_list, kwargs_lens_list)
    ns = getMinMaxSaddle(Img_ra, Img_dec, lens_model_list, kwargs_lens_list, diff = None)

    print('Time delays (seconds): ', tds)
    print('magnifications: ', mus)
    print('Morse indices: ', ns)

    # hardcoded
    if not args.type2:
        # minidx = np.where(tds == 0.)
        minidx = 1
    else:
        minidx = 0
        minidx=imindex
    lens_model_complete = LensModel(lens_model_list=lens_model_list)
    T = lens_model_complete.fermat_potential
    # T0 = thetaE ** (-2) * T(Img_ra[minidx], Img_dec[minidx], kwargs_lens_list, beta0, beta1)#[0]
    T0 = thetaE ** (-2) * T(eta10, eta11, kwargs_lens_list, beta0, beta1)#[0]
    if not isinstance(T0, float):
        T0 = T0[0]
    Tscale = 4 * (1 + zL) * mtot * M_sun * G / c ** 3
    print('T0 = {}'.format(T0))
    print('Tscale = {}'.format(Tscale))

    print('TimeStep', args.TimeStep)

    if args.plot:
        fig, ax = plt.subplots()
        # plot_contour(ax, lens_model_list, y0*thetaE, y1*thetaE, 4*thetaE, kwargs_lens_list, beta0, beta1, Img_ra, Img_dec,
        #                     T0 = T0, Tfac = (thetaE)**(-2), micro=True, test = args.type2, savename = contourtxtname)
        # plt.show()
        print(MacroImg_ra[imindex], MacroImg_dec[imindex])
        plot_contour(ax, lens_model_list, MacroImg_ra[imindex], MacroImg_dec[imindex], 210*thetaE3, kwargs_lens_list, beta0, beta1, Img_ra, Img_dec,
                            T0 = T0, Tfac = (thetaE)**(-2), micro=True)
        plt.show()
        quit()


    kwargs_macro = {'source_pos_x': beta0,
                    'source_pos_y': beta1,
                    'theta_E': thetaE,
                    'mu': args.mu,
                    'T01': T01
                   }

    kwargs_integrator = {'InputScaled': False,
                         'PixelNum': int(args.pixel),
                         'PixelBlockMax': 2000,
                         'WindowSize': args.Winfac*200*thetaE3,
                         'WindowCenterX': MacroImg_ra[imindex],
                         'WindowCenterY': MacroImg_dec[imindex],
                         'TimeStep': args.TimeStep/Tscale, 
                         'TimeMax': T0 + args.TimeMax/Tscale,
                         'TimeMin': T0 - args.TimeMin/Tscale,
                         'TimeLength': 6/Tscale,
                         'LastImageT': args.LastImageT/Tscale,
                         'TExtend': args.TExtend/Tscale,
                         'Tbuffer':0., 
                         'T0': T0,
                         'Tscale': Tscale}
    
    # kwargs_integrator = {'InputScaled': False,
    #                     'PixelNum': int(args.pixel),
    #                     'PixelBlockMax': 2000,
    #                     'WindowSize': args.Winfac*210*thetaE3,
    #                     'WindowCenterX': 0.,
    #                     'WindowCenterY': 0.,
    #                     'TimeStep': args.TimeStep/Tscale, 
    #                     'TimeMax': T0 + 5/Tscale,
    #                     'TimeMin': T0 - 5/Tscale,
    #                     'TimeLength': 10/Tscale,
    #                     'LastImageT': args.LastImageT/Tscale,
    #                     'TExtend': args.TExtend/Tscale,
    #                     'Tbuffer':0., 
    #                     'T0': T0,
    #                     'Tscale': Tscale}

    amplification = af.amplification_factor_fd(lens_model_list=lens_model_list, kwargs_lens=kwargs_lens_list, kwargs_macro=kwargs_macro, **kwargs_integrator)
    ws, Fws = amplification.integrator()
    amplification.plot()

    # np.savetxt('./data/{0}/{0}_ws_{1:1.5f}_{2:1.5f}.txt'.format(run, mL2, ym), ws[::1])
    # np.savetxt('./data/{0}/{0}_Fws_{1:1.5f}_{2:1.5f}.txt'.format(run, mL2, ym), Fws[::1])
