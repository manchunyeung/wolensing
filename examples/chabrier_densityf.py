###Importing relevant libraries

import sys
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
path = os.getcwd()
dir = os.path.abspath(os.path.join(path, os.pardir))
sys.path.append(dir)
import numpy as np
import matplotlib.pyplot as plt
import argparse
from argparse import ArgumentParser

from lenstronomy.LensModel.lens_model import LensModel
import lensinggw.constants.constants as const
from lensinggw.utils.utils import TimeDelay, magnifications, getMinMaxSaddle
from lensinggw.amplification_factor.amplification_factor import geometricalOpticsMagnification

from plot.plot import plot_contour
import amplification_factor.amplification_factor as af
from lensinggw.utils.utils import param_processing



#----------------------------------------------------


### command line options

parser = ArgumentParser()

# usual arguments
parser.add_argument('-src', '--src', type = float, default = 0.1, help = 'source position')
parser.add_argument('-p', '--plot', action = 'store_true', default = False, help = 'Plot out the images with contour lines')
parser.add_argument('-t2', '--type2', action = 'store_true', default = False)
# parser.add_argument('-c', '--code', metavar = 'filename', type = str, help = 'special code for the run')
parser.add_argument('-sf', '--sampling_frequency', type = float, default = 0.25, help = 'choosing a specific sampling frequency (Hz)')
args = parser.parse_args()

#----------------------------------------------------

#mass= np.loadtxt('masses from Relevant IMF')  #loading relevant masses from IMF
#mass=np.array(mass)
mass=[0.0875411729158816, 0.1380156891218021, 0.37757290272476596, 0.08326002905591416, 0.11751088559578819, 0.0878737809276154, 0.14164220540181344, 0.32080191454865087, 0.1432894584167837, 0.2607705005317567, 0.09098854113286821, 0.636857294268748, 0.10244447554800015, 0.11147041258795794, 0.2124085350914183, 0.6266784519859449, 0.42496685401191886, 1.1390594002155605, 0.4254607239653152, 0.1671246164756932, 0.17585315448008126, 0.36911502207552227, 0.1331235211932984, 0.6016395397587949, 0.7933744286210441, 0.42747170785219685, 0.22261658642110216, 0.10932507755278971, 0.1406444799904107, 0.1366232557128965, 0.14884236414318858, 0.19626458656021137, 0.2373647216421461, 0.1201145697485532, 0.48826716295221956, 0.3841401957379758, 0.5897595410342606, 0.1323094711391549, 0.28192039014606574, 0.3112735462148643, 0.16801321143762243, 0.1764578067996469, 0.5067566574473692, 0.27182604134636307, 0.7353778482113479, 0.20962530951686645, 0.5425268135812605, 0.6138969891002811, 0.46906093727165266, 0.6983169787970537, 0.35381175525186964, 0.1938289323334855, 0.14266664101347107, 0.5257788195498755, 0.10368483753262311, 0.17314382503980863, 0.1233372690718119, 0.2652130808997028, 0.693759954642086, 0.14899995995742654, 0.4966398131976725, 0.4840946701128482, 0.3223672452008741, 0.1264395622913355, 0.524341364391967, 0.19948743938870198, 0.3990684024548484, 0.19648096784404456, 0.6277253786806841, 0.3572078802112069, 0.10935381094455206, 0.11170719423786252, 0.5912482920999221, 0.16068614513260024, 0.16602006569289895, 0.0863316957830875, 0.29633174301563786, 0.49586795267650824, 0.7316548600797632, 0.11504195200461981, 0.6098861137208111, 0.11117674322961363, 0.16347501499569306, 0.16644621587575756, 0.2526870128594504, 0.21982758583882406, 0.22905730367878838, 0.22597567330824966, 0.8803804310615377, 0.12704243529981776, 0.11804539955670462, 0.5178623013082213, 0.14208960520058941, 0.33109270801906865, 0.8399961332019935, 0.09105580937273003, 0.4584255547405802, 0.08536227566048099, 0.21880300363881566, 0.18742408308006153, 0.3454639600005885, 0.4545124934526552, 0.08729912612402385, 0.09183479195577761, 0.27489259269705335, 0.368014432730991, 0.10586941408543118, 0.28187652197795104, 0.10588462111911362, 0.19694624181432896, 1.2348604537851748, 0.5760460254475586, 0.3468365328612529, 0.1974048180908095, 0.8088380039765515, 0.3910930527213851, 0.08031104298737239, 0.3288420759275759, 0.257701347189583, 0.204555234994582, 0.18635543058276188, 0.16009619756433785, 0.13667689840964123, 0.4436185629577214, 0.37181495338742027, 0.10829841323996174, 0.42740463344560164, 0.4648421533639838, 0.39844677434656256, 0.12276751474732411, 0.1497385231567059, 0.13455048432141956, 0.7326858566883386, 0.8469258102015667, 0.2846315707426615, 0.10361468351216026, 0.10327067036387486, 0.16093428418527544, 0.09909750488908187, 0.22017575864813146, 0.13022562568867713, 0.09817901855619601, 0.11359917921111046, 0.14787772530428508, 0.32223115967091887, 0.14438638912022622, 0.22636375822763366, 0.1508102244228382, 0.09421420103439823, 0.09420560142302992, 0.12845475317801583, 0.12031189641012763, 0.08983500253328633, 0.4522590981297488, 0.2125050533362804, 0.13405687278522366, 0.5061555804486358, 0.15199874366191218, 0.11522721689032857, 0.15331833888265486, 0.3286972499978612, 0.1364015379458153, 0.13182972564697468, 0.6773894027560546, 0.23447094849939093, 0.40490970130136555, 0.7334462880075556, 0.13398111358089174, 0.2313822106459586, 0.1024354466422358, 1.067538897792573, 0.5110531294744058, 0.3101525067706349, 0.11669327335860222, 1.1420841471405525, 0.08543067392655455, 0.1813159502352595, 0.9498566204768994, 0.14648348335645942, 0.45094492503946776, 1.1819822423988344, 0.35697005803592974, 0.1417142224960117, 1.123058038013592, 0.17114162740018618, 0.9929003645417682, 0.10171192962042469, 0.9078721119947383, 0.4224542630680143, 0.08711997693949222, 0.31989612048306554, 0.12224316374740339, 0.22403812429852143, 0.2195971752795328, 0.09764311068908389, 0.516004745947273, 1.0721607995296687, 0.26418399695334466, 0.17421612931856542, 0.31389494679546465]

#mass=[0.18227444396346612, 0.5150875701189991, 0.39451094374917256, 0.638530592860076, 0.16682502548206227, 1.2262562342191987, 0.7170802934115559, 0.11836383462950102, 0.4024622171972135, 0.3201323382597446, 0.6955054180256542, 0.6116181534032235, 0.41198903710272067, 0.6445820936909867, 0.11330703399559386, 0.14835338583460556, 0.16327943745712856, 0.477754167915732, 0.10592626831205769, 0.10976056110934611, 0.09296822458874164, 0.14245787601395915, 0.4608114236049782, 0.10831377245160481, 0.0911693534358726, 0.3560783737580413, 0.11207419139325903, 0.27084204048044097, 0.1782890231385507, 0.2210494214228313, 0.12086518112336643, 0.3828492400074023, 0.5901047625912305, 0.8652863505195562, 0.39517207393554626, 0.2034659681410833, 0.2109690632540115, 0.2678943945080466, 0.0920866470598544, 0.09946950731517248, 1.1044181460491322, 0.26059870835135507, 0.09493283699383315, 0.08204011676247981, 0.5610772493635268, 0.8002605040140375, 0.3597309667665003, 0.16807929780035447, 0.3608943714908111, 0.8383708345229695, 0.13775653420802, 0.3391696393543725, 0.2244708088974414, 0.6657851359471343, 0.8410830674469748, 0.20925069263239177, 0.26902414081441145, 0.13147624038235803, 1.3739904083832013, 0.11851816301823809, 0.2606292186841815, 0.3080809933457567, 0.7842543044979021, 0.10024922520384755, 0.3314871071620338, 0.4192648917156002, 0.7824653770338412, 0.1258495132510028, 1.1869055858531854, 0.14205102227881686, 0.7436938616506263, 0.09712016184313724, 0.3290153720272783, 0.6428707787858263, 0.2469169483093679, 0.1994949784221582, 0.08226151401037264, 0.17025519434444, 0.11850224161110501, 0.11831928626309945, 0.1710742541605977, 1.2396099536617458, 0.11489882603396359, 0.5905801696157583, 0.1960531355770249, 0.11456714005515325, 0.11003474930887704, 0.19643243723172385, 0.12101152215664218, 0.22145171592160678, 0.3777836866339093, 0.994199767274014, 0.08541188056748024, 1.262906446316184, 0.39944335441658424, 0.11254459928455045, 0.5468327622893654, 0.21962566337485206, 0.3650981581471056, 0.38531792943533816, 0.9400046473159323, 0.16978016883590583, 0.9199153994162644, 0.20372884920373255, 0.7554952194916494, 0.4315348751007218, 0.1730262114608116, 0.09139151595245515, 0.17397199490051632, 0.2452999258506441, 0.2434343027333158, 0.11845699487115335, 0.9400269640634529, 0.28694746330755266, 0.36232337293342826, 0.3792277209820595, 0.9661452729449458, 0.14430533466537684, 0.0982856635348884, 0.38534051340831715, 0.14503014313727694, 0.1502955727667225, 0.08934642973077939, 0.11239403276663729, 0.12323878133179647, 0.08531776422367972, 0.29407017267554175, 0.1216848574278479, 0.32501903518230507, 0.2881212309965703, 0.3857041031370626, 0.5544255356090366, 0.09868506672484845, 0.10652612295959447, 0.0956553862159841, 0.09845643872045687, 0.19107000218371661, 0.896140985652719, 0.5696813187176105, 0.6542900698649184, 0.5798607995599345, 0.10693296484419057, 0.875056201322954, 0.7743186307660792, 0.1580449609371039, 0.23013893603465327, 0.25479826387436144, 0.11111302658805121, 0.26560771384379256, 1.1405856377197754, 1.4527878229041746, 0.09526848225341668, 0.08435723524467703, 0.12302194811574003, 0.32600308772101505, 0.120370335399331, 0.5134369308330896, 0.13235887481790815, 0.3505594674980581, 0.41414642468059526, 0.2571198632139568, 0.17447122470822246, 0.3543786765264015, 0.2245965115568999, 0.4238722055251769, 1.007620251779934, 1.0196260118441662, 0.22892048455520217, 0.08537092821487975, 0.30886614426077463, 0.46664307969940494, 0.6665872005569925, 0.09197361018713233, 0.10760904908275892, 0.23687678466736245, 0.10031291020988832, 0.22910387896093962, 0.23200642634310417, 0.9484660523842493, 0.26825595600724395, 0.19662767576587137, 0.13397826387485717, 0.32289839560480266, 0.5461870845030385, 0.11632368478222276, 0.08690668095809968, 0.3140155528815352, 0.08176051551662149, 0.9003770080934416, 0.3000665157108845, 0.21124609895853608, 0.4789666197704082, 1.072435422676039, 0.41900776071469203, 0.8901525474513555, 0.6196241963589431, 0.20210629582881542, 0.1035316340644515, 0.10107408643224151, 0.09095316820547471]

#Getting parameters for the distribution of microlenses around type 1 macroimage

density = 107 #(solar mass/pc^2)      
radius = np.sqrt(sum(mass) / np.pi / density)
R=1261.825281709447 * 10**6 #pc , angular distance to the lens plane 

num_points=len(mass) #no. of lenses
np.random.seed(1)
seed=1

angle1 = np.random.uniform(0, 2*np.pi, size=num_points)
ym1 = np.sqrt(np.random.uniform(0, radius**2, size=num_points))
ym1=ym1/R

args = parser.parse_args()



#----------------------------------------------------

# Update args with a respecetive kwargs set for the type of macroimage
kwargs_type = vars(args)
if args.type2:
    kwargs_type.update({'LastImageT': 4e-7, 'TExtend': 7., 'mu': 9, 'TimeMax': 7., 'TimeMin': 5., 'TimeLength': 12, 'Timg': 59015, 'TimeStep': 1e-6, 'Winfac': 10.})
else: #why 
    df = args.sampling_frequency
    textendmax = 1/df
    tlength = 3
    textend = 4
    kwargs_type.update({'LastImageT': 4e-5, 'TExtend': textend, 'mu': 11., 'TimeMax': 1, 'TimeMin':2, 'TimeLength':3, 'Timg': 118211.81107161, 'TimeStep': 1e-5, 'Winfac': 1})
    
G = const.G  # gravitational constant [m^3 kg^-1 s^-2]
c = const.c  # speed of light [m/s]
M_sun = const.M_sun  # Solar mass [Kg]

if not args.type2:   #identification of the type 1 macroimage 
    imindex = 1
else:
    imindex = 0



run = 'test'

if args.plot:
    contourtxtname = os.path.join(DATA_DIR,'{}_contour.png'.format(run))
    contourtxtnamemicro = os.path.join(DATA_DIR,'{0}_contourmicro_{1}.png'.format(run, imindex))
    contourtxtnamemicromicro = os.path.join(DATA_DIR,'{0}_contourmicromicro_{1}.png'.format(run, imindex))
    imagesnpzname = os.path.join(DATA_DIR,'{}_images.npz'.format(run))

#----------------------------------------------------


# coordinates in scaled units [x (radians) /thetaE_tot]\


y0, y1 = 0.1, 0 	#source position
l0, l1 = 0, 0 		# lens position
zS,zL  = 1.0, 0.5 	#redshift source and lens
mL = 1 * 1e10 		#mass of the macrolens


#some required parameters for computation
mtot=0
thetaE=[]
thetaEl=[]
eta20=[]
eta21=[]

k=[]
l=[]
ws1=[]
Fws1=[]
tds=[]
mus=[]
ns=[]
ym=[]
angle=[]




#######################################collecting relevant parmeters for the microlenses####################
for i in range (0,num_points):
	angle.append(float(angle1[i]))
	ym.append(float(ym1[i]))
	mtot=sum(mass[:i+1]) + mL #total mass
	thetaEl.append(param_processing(zL, zS, mass[i]))# lens model


thetaE=(param_processing(zL,zS,mtot)) #einstein radius for the total mass
thetaE1 =(param_processing(zL,zS,mL)) ##einstein angle for the macrolens

beta0=(y0 * thetaE) 	#source position in angular size
beta1=(y1 * thetaE)     #source position in angular size

eta10, eta11 = 0 * l0 * thetaE, 0 * l1 * thetaE	#macrolens position
kwargs_sis_1 = {'center_x': eta10, 'center_y': eta11, 'theta_E': thetaE1}

##################solving for the macromodel##############################################################
lens_model_list = ['SIS']
kwargs_sis_1 = {'center_x': eta10, 'center_y': eta11, 'theta_E': thetaE1}
kwargs_lens_list = [kwargs_sis_1]

from lensinggw.solver.images import microimages

solver_kwargs ={'SearchWindowMacro': 10 * thetaE1,'SearchWindow': 5 * thetaE1,'OverlapDistMacro': 1e-17,'OnlyMacro': True,'Optimization':True}
MacroImg_ra, MacroImg_dec, pixel_width = microimages(source_pos_x=beta0,source_pos_y=beta1,lens_model_list=lens_model_list,kwargs_lens=kwargs_lens_list,**solver_kwargs) 

#magnification for macromodel
Macromus=(magnifications(MacroImg_ra, MacroImg_dec, lens_model_list, kwargs_lens_list))
print(Macromus,"macromus")
				
#Timedelay for macromodel
T01 = (TimeDelay(MacroImg_ra, MacroImg_dec,beta0, beta1,zL, zS,lens_model_list, kwargs_lens_list))
print(T01,"time delay for macromodel")

if args.type2:
	imindex = np.nonzero(T01)[0]
else:
	imindex = np.where(T01==0)[0][0]
	args.mu=(Macromus[imindex])
	print(imindex)

##################for the microlenses #####################################################

for j in range(0,num_points):
	thetaEl.append(param_processing(zL, zS, mass[j]))# lens model
	#positioning the microlenses
	eta20.append(MacroImg_ra[imindex] + (ym[j])*np.cos((angle[j])))
	eta21.append(MacroImg_dec[imindex] + (ym[j]) * np.sin((angle[j])))
	lens_model_list.append('POINT_MASS')
	kwargs_lens_list.append({'center_x': eta20[j], 'center_y': eta21[j], 'theta_E': thetaEl[j]})
	k.append(eta20[j])
	l.append(eta21[j])


cmap = plt.get_cmap('viridis')
plt.scatter(eta20,eta21,c=mass, cmap=cmap) 
cbar = plt.colorbar()
cbar.set_label('Mass')
filename0 = '/home/astha/slensing/lensinggw/scatterplot-{}-{}.png'.format(str(density), str(seed))
plt.show()

####################### Calculating the amplification factor############################################################

# hardcoded --by Simon Yeung, with my contribution being the selection of parameter 'WindowSize'
if not args.type2:
	minidx = 1
else:
	minidx = 0
	minidx=imindex

lens_model_complete = LensModel(lens_model_list=lens_model_list)
T = lens_model_complete.fermat_potential
T0 = thetaE ** (-2) * T(MacroImg_ra[minidx], MacroImg_dec[minidx], kwargs_lens_list, beta0, beta1)#[0]
print(T0,"hshbh")
if not isinstance(T0, float):
	T0 = T0[0]
Tscale = 4 * (1 + zL) * mtot * M_sun * G / c ** 3
print('T0 = {}'.format(T0))
print('Tscale = {}'.format(Tscale))
print('TimeStep', args.TimeStep)
if args.plot:
	fig, ax = plt.subplots()
	print(MacroImg_ra[imindex], MacroImg_dec[imindex])
	plot_contour(ax, lens_model_list, MacroImg_ra[imindex], MacroImg_dec[imindex], 13*thetaE2, kwargs_lens_list, beta0, beta1, Img_ra, Img_dec, T0 = T0, Tfac = (thetaE)**(-2), micro=True)
	plt.xlabel('RA(radians)')
	plt.ylabel('DEC(radians)')
	plt.title("Plotting time delay contours")
	plt.show()



kwargs_wolensing = {'source_pos_x': beta0,'source_pos_y': beta1,'theta_E': thetaE,'LastImageT': args.LastImageT/Tscale,'TExtend': args.TExtend/Tscale,'Tbuffer':0.,'mu': args.mu}

kwargs_integrator = {'InputScaled': False,'PixelNum': int(5e4),'PixelBlockMax': 2000,'WindowSize': args.Winfac*600*max(thetaEl),'WindowCenterX': MacroImg_ra[imindex],'WindowCenterY': MacroImg_dec[imindex],'TimeStep': args.TimeStep/Tscale,'TimeMax': T0 + args.TimeMax/Tscale,'TimeMin': T0 - args.TimeMin/Tscale,'TimeLength': args.TimeLength/Tscale,'ImageRa': [],'ImageDec': [],'T0': T0,'Tscale': Tscale}#why window size
dx=(600*max(thetaEl))
print(dx,"wins")

ws, Fws,ts, Fs  = af.amplification_factor_fd(lens_model_list, args, kwargs_lens_list, **kwargs_wolensing,**kwargs_integrator) #calculation of amplification factor 


#################################Getting Relavnt Outputs for Amplification Factor###########################################


filename1 = '/home/astha/slensing/lensinggw/output/mass-{}-{}.txt'.format(str(density), str(seed))
np.savetxt(filename1,mass)

filename2 = '/home/astha/slensing/lensinggw/output/ws-{}-{}.txt'.format(str(density), str(seed))
np.savetxt(filename2,ws[::1])

filename3 = '/home/astha/slensing/lensinggw/output/Fws-{}-{}.txt'.format(str(density), str(seed))
np.savetxt(filename3,Fws[::1])

filename4 = '/home/astha/slensing/lensinggw/output/macromus-{}-{}.txt'.format(str(density), str(seed))
argmu=np.array(args.mu)
np.savetxt(filename4,argmu.reshape(1,))


plt.semilogx(ts, Fs)
filename5 = '/home/astha/slensing/lensinggw/output/Ft-{}-{}.png'.format(str(density), str(seed))
plt.savefig(filename5)

filename6 = '/home/astha/slensing/lensinggw/output/Ft-{}-{}.txt'.format(str(density), str(seed))
np.savetxt(filename6, Fs)

filename7 = '/home/astha/slensing/lensinggw/output/ts-{}-{}.txt'.format(str(density), str(seed))
np.savetxt(filename7, ts)


