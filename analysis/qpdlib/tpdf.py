import sys,os
import numpy as np
import copy

#--matplotlib
import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import pylab as py

import pandas as pd

#--from scipy stack 
from scipy.integrate import quad, cumtrapz

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config

#--from fitlib
from fitlib.resman import RESMAN

#--from local
from analysis.qpdlib.qpdcalc import QPDCALC
from analysis.corelib import core
from analysis.corelib import classifier

import kmeanconf as kc

import warnings
warnings.filterwarnings("ignore")

import lhapdf

FLAV=[]
FLAV.append('u')
FLAV.append('d')
FLAV.append('uv')
FLAV.append('dv')
FLAV.append('ub')
FLAV.append('db')

cmap = matplotlib.cm.get_cmap('plasma')

def gen_xf(wdir,Q2):
    
    print('\ngenerating tpdf at Q2 = %s from %s'%(Q2,wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if 'tpdf' not in conf['steps'][istep]['active distributions']:
        if 'tpdf' not in conf['steps'][istep]['passive distributions']:
                print('tpdf is not an active or passive distribution')
                return 

    if Q2==None: Q2 = conf['Q20']

    conf['SofferBound'] = True
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    tpdf=conf['tpdf']
    #--setup kinematics
    X=10**np.linspace(-6,-1,200)
    X=np.append(X,np.linspace(0.1,0.99,200))

    Q2 = Q2 * np.ones(len(X))

    tpdf.evolve(Q2)

    #--compute XF for all replicas        
    XF={}
    cnt=0
    for par in replicas:
        core.mod_conf(istep,core.get_replicas(wdir)[cnt])   
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)

        for flav in FLAV:
            if flav not in XF:  XF[flav]=[]
            if flav=='uv':
                 func=lambda x: tpdf.get_xF(x,Q2,'uv')
            elif flav=='dv':
                 func=lambda x: tpdf.get_xF(x,Q2,'dv')
            elif flav=='d/u':
                 func=lambda x: tpdf.get_xF(x,Q2,'d')/tpdf.get_xF(x,Q2,'u')
            elif flav=='db+ub':
                 func=lambda x: tpdf.get_xF(x,Q2,'db') + tpdf.get_xF(x,Q2,'ub')
            elif flav=='db-ub':
                 func=lambda x: tpdf.get_xF(x,Q2,'db') - tpdf.get_xF(x,Q2,'ub')
            elif flav=='db/ub':
                 func=lambda x: tpdf.get_xF(x,Q2,'db') / tpdf.get_xF(x,Q2,'ub')
            elif flav=='s+sb':
                 func=lambda x: tpdf.get_xF(x,Q2,'s') + tpdf.get_xF(x,Q2,'sb')
            elif flav=='s-sb':
                 func=lambda x: tpdf.get_xF(x,Q2,'s') - tpdf.get_xF(x,Q2,'sb')
            elif flav=='Rs':
                 func=lambda x: (tpdf.get_xF(x,Q2,'s') + tpdf.get_xF(x,Q2,'sb'))\
                                /(tpdf.get_xF(x,Q2,'db') + tpdf.get_xF(x,Q2,'ub'))
            else:
                 func=lambda x: tpdf.get_xF(x,Q2,flav) 

            #XF[flav].append(np.array([func(x) for x in X]))
            XF[flav].append(func(X))


    print()

    #--also save SB
    tpdf.setup_SB(Q2=Q2[0])
    SB      = {}
    SB['X']  = tpdf.X
    SB['u']  = tpdf.X*(tpdf.SB['mean']['u']  + tpdf.SB['std']['u'])
    SB['d']  = tpdf.X*(tpdf.SB['mean']['d']  + tpdf.SB['std']['d'])
    SB['ub'] = tpdf.X*(tpdf.SB['mean']['ub'] + tpdf.SB['std']['ub'])
    SB['db'] = tpdf.X*(tpdf.SB['mean']['db'] + tpdf.SB['std']['db'])
    SB['uv'] = tpdf.X*(tpdf.SB['mean']['uv'] + tpdf.SB['std']['uv'])
    SB['dv'] = tpdf.X*(tpdf.SB['mean']['dv'] + tpdf.SB['std']['dv'])
 
    checkdir('%s/data'%wdir)
    filename='%s/data/tpdf-Q2=%3.5f.dat'%(wdir,Q2[0])

    save({'X':X,'Q2':Q2,'XF':XF,'SB':SB},filename)
    print ('Saving data to %s'%filename)

def plot_xf_main(wdir,Q2,mode=0):
    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas 


    load_config('%s/input.py'%wdir)
    istep=core.get_istep()

    if 'tpdf' not in conf['steps'][istep]['active distributions']:
        if 'tpdf' not in conf['steps'][istep]['passive distributions']:
                print('tpdf is not an active or passive distribution')
                return 

    nrows,ncols=2,2
    N = nrows*ncols
    fig = py.figure(figsize=(ncols*9,nrows*5))
    axs,axLs = {},{}
    for i in range(N):
        axs[i+1] = py.subplot(nrows,ncols,i+1)
        divider = make_axes_locatable(axs[i+1])
        axLs[i+1] = divider.append_axes("right",size=3.50,pad=0,sharey=axs[i+1])
        axLs[i+1].set_xlim(0.1,0.9)
        axLs[i+1].spines['left'].set_visible(False)
        axLs[i+1].yaxis.set_ticks_position('right')
        py.setp(axLs[i+1].get_xticklabels(),visible=True)

        axs[i+1].spines['right'].set_visible(False)

    hand = {}
    thy  = {}

    if Q2==None: Q2 = conf['Q20']

    filename='%s/data/tpdf-Q2=%3.5f.dat'%(wdir,Q2)
    #--load data if it exists
    try:
        data=load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_xf(wdir,Q2)
        data=load(filename)
        
    replicas=core.get_replicas(wdir)
    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
    best_cluster=cluster_order[0]

    X=data['X']

    for flav in data['XF']:
        mean = np.mean(data['XF'][flav],axis=0)
        std  = np.std (data['XF'][flav],axis=0)

        if   flav=='uv': ax,axL = axs[1],axLs[1]
        elif flav=='dv': ax,axL = axs[2],axLs[2]
        elif flav=='ub': ax,axL = axs[3],axLs[3]
        elif flav=='db': ax,axL = axs[4],axLs[4]
        else: continue

        #--plot each replica
        if mode==0:
            for i in range(len(data['XF'][flav])):
                hand['thy'] ,= ax.plot(X,np.array(data['XF'][flav][i]),color='red',alpha=0.3)
                axL              .plot(X,np.array(data['XF'][flav][i]),color='red',alpha=0.3)
 
        #--plot average and standard deviation
        if mode==1:
            hand['thy']  = ax.fill_between(X,(mean-std),(mean+std),color='red',alpha=0.9,zorder=6)
            axL              .fill_between(X,(mean-std),(mean+std),color='red',alpha=0.9,zorder=6)


    for i in range(N):
        axs[i+1].set_xlim(5e-3,0.1)
        axs[i+1].semilogx()

        axs[i+1].tick_params(axis='both', which='major', top=True, direction='in',labelsize=30,length=8)
        axs[i+1].tick_params(axis='both', which='minor', top=True, direction='in',labelsize=30,length=4)
        axs[i+1].tick_params(axis='x',    which='major', pad = 8)
        axs[i+1].set_xticks([0.01,0.1])
        axs[i+1].set_xticklabels([r'$10^{-2}$',r'$0.1$'])

        axLs[i+1].set_xlim(0.1,0.9)

        axLs[i+1].tick_params(axis='both', which='major', top=True, right=True, left=False, labelright=False, direction='in',labelsize=30,length=8)
        axLs[i+1].tick_params(axis='both', which='minor', top=True, right=True, left=False, labelright=False, direction='in',labelsize=30,length=4)
        axLs[i+1].tick_params(axis='x',    which='major', pad = 8)
        axLs[i+1].set_xticks([0.3,0.5,0.7])
        #axLs[i+1].set_xticklabels([r'$0.2$',r'$0.3$'])
        minorLocator = MultipleLocator(0.05)
        axLs[i+1].xaxis.set_minor_locator(minorLocator)

    axs[1].set_ylim(-0.1,0.7)
    axs[1].set_yticks([0.0,0.2,0.4,0.6])
    minorLocator = MultipleLocator(0.05)
    axs[1].yaxis.set_minor_locator(minorLocator)
    
    axs[2].set_ylim(-0.25,0.15)
    axs[2].set_yticks([-0.2,-0.1,0.0,0.1])
    minorLocator = MultipleLocator(0.05)
    axs[2].yaxis.set_minor_locator(minorLocator)

    axs[3].set_ylim(-0.08,0.08)
    axs[3].set_yticks([-0.05,0,0.05])
    minorLocator = MultipleLocator(0.01)
    axs[3].yaxis.set_minor_locator(minorLocator)

    axs[4].set_ylim(-0.08,0.08)
    axs[4].set_yticks([-0.05,0,0.05])
    minorLocator = MultipleLocator(0.01)
    axs[4].yaxis.set_minor_locator(minorLocator)



    for i in range(N):
        axs [i+1].axvline(0.1,color='k',linestyle=':' ,alpha=0.5)
        axLs[i+1].axvline(0.1,color='k',linestyle=':' ,alpha=0.5)
        axs [i+1].axhline(0.0,ls='--',color='black',alpha=0.5,zorder=5)
        axLs[i+1].axhline(0.0,ls='--',color='black',alpha=0.5,zorder=5)

    axs[1] .tick_params(labelbottom=False)
    axLs[1].tick_params(labelbottom=False)
    axs[2] .tick_params(labelbottom=False)
    axLs[2].tick_params(labelbottom=False)

    axLs[3].set_xlabel(r'\boldmath$x$',size=40)
    axLs[4].set_xlabel(r'\boldmath$x$',size=40)   
    axLs[3].xaxis.set_label_coords(0.92,0.00)
    axLs[4].xaxis.set_label_coords(0.92,0.00)

    axs[1].set_ylabel(r'\boldmath$x h_1(x)$',size=40)
    axs[3].set_ylabel(r'\boldmath$x h_1(x)$',size=40)
    axs[1].yaxis.set_label_coords(-0.25,0.50)
    axs[3].yaxis.set_label_coords(-0.25,0.50)

    axs[1] .text(0.15 ,0.80  ,r'\boldmath{$u_v$}'     , transform=axs[1] .transAxes,size=50)
    axLs[2].text(0.60 ,0.15  ,r'\boldmath{$d_v$}'     , transform=axLs[2].transAxes,size=50)
    axLs[3].text(0.60 ,0.80  ,r'\boldmath{$\bar{u}$}' , transform=axLs[3].transAxes,size=50)
    axLs[4].text(0.60 ,0.80  ,r'\boldmath{$\bar{d}$}' , transform=axLs[4].transAxes,size=50)

    axs[2].text(0.08,0.08,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$', transform=axs[2].transAxes,size=30)


    #--plot Soffer Bound
    X = data['SB']['X']
    uv  = data['SB']['uv']
    dv  = data['SB']['dv']
    ub  = data['SB']['ub']
    db  = data['SB']['db']
    hand['SB'] ,= axs [1].plot(X, uv ,color='blue',ls=':',alpha=1.0,zorder=7)
    hand['SB'] ,= axs [1].plot(X,-uv ,color='blue',ls=':',alpha=1.0,zorder=7)
    hand['SB'] ,= axs [2].plot(X, dv ,color='blue',ls=':',alpha=1.0,zorder=7)
    hand['SB'] ,= axs [2].plot(X,-dv ,color='blue',ls=':',alpha=1.0,zorder=7)
    hand['SB'] ,= axLs[1].plot(X, uv ,color='blue',ls=':',alpha=1.0,zorder=7)
    hand['SB'] ,= axLs[1].plot(X,-uv ,color='blue',ls=':',alpha=1.0,zorder=7)
    hand['SB'] ,= axLs[2].plot(X, dv ,color='blue',ls=':',alpha=1.0,zorder=7)
    hand['SB'] ,= axLs[2].plot(X,-dv ,color='blue',ls=':',alpha=1.0,zorder=7)
    hand['SB'] ,= axs [3].plot(X, ub ,color='blue',ls=':',alpha=1.0,zorder=7)
    hand['SB'] ,= axs [3].plot(X,-ub ,color='blue',ls=':',alpha=1.0,zorder=7)
    hand['SB'] ,= axLs[3].plot(X, ub ,color='blue',ls=':',alpha=1.0,zorder=7)
    hand['SB'] ,= axLs[3].plot(X,-ub ,color='blue',ls=':',alpha=1.0,zorder=7)
    hand['SB'] ,= axs [4].plot(X, db ,color='blue',ls=':',alpha=1.0,zorder=7)
    hand['SB'] ,= axs [4].plot(X,-db ,color='blue',ls=':',alpha=1.0,zorder=7)
    hand['SB'] ,= axLs[4].plot(X, db ,color='blue',ls=':',alpha=1.0,zorder=7)
    hand['SB'] ,= axLs[4].plot(X,-db ,color='blue',ls=':',alpha=1.0,zorder=7)

    #--plot JAM3D
    os.environ['LHAPDF_DATA_PATH'] = '%s/qcdlib/lhapdf'%(os.environ['FITPACK'])
    JAM3D = lhapdf.mkPDFs('JAM23-transversity_proton_lo')
    X = np.linspace(0.001,0.99,100)
    nrep = len(JAM3D)

    flavs = ['uv','dv']
    plot = {flav: [] for flav in flavs} 

    for i in range(nrep):
        #--skip mean value
        if i==0: continue
        dv =  np.array([JAM3D[i].xfxQ2( 1,x,Q2)-JAM3D[i].xfxQ2(-1,x,Q2) for x in X])
        uv =  np.array([JAM3D[i].xfxQ2( 2,x,Q2)-JAM3D[i].xfxQ2(-2,x,Q2) for x in X])
        plot['dv'].append(dv)
        plot['uv'].append(uv)

    mean_uv = np.mean(plot['uv'],axis=0)
    mean_dv = np.mean(plot['dv'],axis=0)
    std_uv  = np.std (plot['uv'],axis=0)
    std_dv  = np.std (plot['dv'],axis=0)
    
    hand['JAM3D']  = axs [1].fill_between(X,(mean_uv-std_uv),(mean_uv+std_uv),color='blue',alpha=0.5,zorder=5)
    hand['JAM3D']  = axLs[1].fill_between(X,(mean_uv-std_uv),(mean_uv+std_uv),color='blue',alpha=0.5,zorder=5)
    hand['JAM3D']  = axs [2].fill_between(X,(mean_dv-std_dv),(mean_dv+std_dv),color='blue',alpha=0.5,zorder=5)
    hand['JAM3D']  = axLs[2].fill_between(X,(mean_dv-std_dv),(mean_dv+std_dv),color='blue',alpha=0.5,zorder=5)

    #--plot JAM3D (no LQCD)
    os.environ['LHAPDF_DATA_PATH'] = '%s/qcdlib/lhapdf'%(os.environ['FITPACK'])
    JAM3D = lhapdf.mkPDFs('JAM23-transversity_proton_lo_nolat')
    X = np.linspace(0.001,0.99,100)
    nrep = len(JAM3D)

    flavs = ['uv','dv']
    plot = {flav: [] for flav in flavs} 

    for i in range(nrep):
        #--skip mean value
        if i==0: continue
        dv =  np.array([JAM3D[i].xfxQ2( 1,x,Q2)-JAM3D[i].xfxQ2(-1,x,Q2) for x in X])
        uv =  np.array([JAM3D[i].xfxQ2( 2,x,Q2)-JAM3D[i].xfxQ2(-2,x,Q2) for x in X])
        plot['dv'].append(dv)
        plot['uv'].append(uv)

    mean_uv = np.mean(plot['uv'],axis=0)
    mean_dv = np.mean(plot['dv'],axis=0)
    std_uv  = np.std (plot['uv'],axis=0)
    std_dv  = np.std (plot['dv'],axis=0)
    
    hand['JAM3D_noLQCD']  = axs [1].fill_between(X,(mean_uv-std_uv),(mean_uv+std_uv),color='green',alpha=0.5,zorder=5)
    hand['JAM3D_noLQCD']  = axLs[1].fill_between(X,(mean_uv-std_uv),(mean_uv+std_uv),color='green',alpha=0.5,zorder=5)
    hand['JAM3D_noLQCD']  = axs [2].fill_between(X,(mean_dv-std_dv),(mean_dv+std_dv),color='green',alpha=0.5,zorder=5)
    hand['JAM3D_noLQCD']  = axLs[2].fill_between(X,(mean_dv-std_dv),(mean_dv+std_dv),color='green',alpha=0.5,zorder=5)

    #--plot Pavia
    #radici = pd.read_excel('./plots/models/xh1Radici.xlsx')
    #x = radici.x
    #xh1u = radici.xh1u
    #xh1d = radici.xh1d
    #hand['Radici'] ,= axs[1] .plot(x,xh1u,'--' ,lw=2,color='black',alpha=0.8,zorder=10)            
    #hand['Radici'] ,= axLs[1].plot(x,xh1u,'--' ,lw=2,color='black',alpha=0.8,zorder=10)            
    #hand['Radici'] ,= axs[2] .plot(x,xh1d,'--' ,lw=2,color='black',alpha=0.8,zorder=10)            
    #hand['Radici'] ,= axLs[2].plot(x,xh1d,'--' ,lw=2,color='black',alpha=0.8,zorder=10)            



    handles,labels = [],[]
    handles.append(hand['thy'])
    handles.append(hand['JAM3D'])
    handles.append(hand['JAM3D_noLQCD'])
    handles.append(hand['SB'])

    labels.append(r'\textrm{\textbf{JAM}}')
    labels.append(r'\textrm{\textbf{JAM3D}}')
    labels.append(r'\textrm{\textbf{JAM3D (no LQCD)}}')
    labels.append(r'\textrm{\textbf{SB}}')

    axs[3].legend(handles,labels,loc='upper left',fontsize=20,frameon=0,handletextpad=0.3,handlelength=0.9,ncol=1,columnspacing=1.0)

    #axs [4].axis('off')   
    #axLs[4].axis('off')   
 

    py.tight_layout()
    py.subplots_adjust(hspace = 0.05, wspace = 0.15, left=0.10)

    filename = '%s/gallery/tpdfs-Q2=%3.5f'%(wdir,Q2)
    if mode==1: filename += '-bands'

    filename+='.png'

    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)

def plot_xf(wdir,Q2=None,mode=0):

    plot_xf_main(wdir,Q2,mode=mode)


#--tpdf moments (truncated)
def gen_moments_trunc(wdir, Q2 = 4, flavors = ['uv','dv','u','d','ub','db'], mom = 1, xmin = 0.006, xmax = 0.99):
    load_config('%s/input.py' % wdir)
    istep = core.get_istep()

    replicas = core.get_replicas(wdir)

    if 'tpdf' not in conf['steps'][istep]['active distributions']:
        if 'tpdf' not in conf['steps'][istep]['passive distributions']:
            print('tpdf not an active or passive distribution')
            return

    conf['SofferBound'] = False
    resman = RESMAN(nworkers = 1, parallel = False, datasets = False, load_lhapdf=False)
    parman = resman.parman
    parman.order = replicas[0]['order'][istep]

    tpdf = conf['tpdf']

    ## setup kinematics
    xs = np.geomspace(xmin,0.1,100)
    xs = np.append(xs, np.linspace(0.1, xmax, 100))
    if Q2 == None: Q2 = conf['Q20']
    Q2 = Q2 * np.ones(len(xs))
    print('\ngenerating moment %d for tpdf from %s at Q2 = %3.2f from %3.2f to %3.2f' % (mom, wdir, Q2[0], xmin, xmax))

    ## compute moments for all replicas
    moments = {}
    n_replicas = len(replicas)

    power = mom - 2

    for i in range(n_replicas): ## using 'scipy.integrate.cumtrapz' takes about 9.984 seconds for 100 x points, 4 flavors and 516 replicas
        lprint('%d/%d' % (i + 1, n_replicas))

        parman.order = copy.copy(replicas[i]['order'][istep])
        parman.set_new_params(replicas[i]['params'][istep], initial = True)

        for flavor in flavors:
            if flavor not in moments:
                moments[flavor] = []
            if flavor == 'uv':
                func = lambda x:  x**power*(tpdf.get_xF(x, Q2, 'u') - tpdf.get_xF(x, Q2, 'ub'))
            elif flavor == 'dv':
                func = lambda x:  x**power*(tpdf.get_xF(x, Q2, 'd') - tpdf.get_xF(x, Q2, 'db'))
            elif flavor == 'db-ub':
                func = lambda x:  x**power*(tpdf.get_xF(x, Q2, 'db') - tpdf.get_xF(x, Q2, 'ub'))
            else:
                func = lambda x:  x**power*tpdf.get_xF(x, Q2, flavor)

            function_values = func(xs)
            moment_temp = cumtrapz(function_values, xs, initial = 0.0)
            moment_temp = np.array(moment_temp)
            moment_max = moment_temp[-1]
            moments[flavor].append(moment_max - moment_temp)
    print()

    MOM = {}
    #--get moment for xmin
    for flavor in flavors:
        MOM[flavor] = []
        for i in range(n_replicas):
            MOM[flavor].append(moments[flavor][i][0])


    checkdir('%s/data' % wdir)
    filename = '%s/data/tpdf-truncmom-xmin=%3.5f-xmax=%3.5f-Q2=%3.5f'%(wdir,xmin,xmax,Q2[0])
    save({'X': xs, 'Q2': Q2, 'XF': moments, 'moments': MOM}, filename)

#--tpdf moments (full)
def gen_moments(wdir, Q2 = 4, flavors = ['uv','dv','u','d','ub','db'], mom = 1):
    load_config('%s/input.py' % wdir)
    istep = core.get_istep()

    replicas = core.get_replicas(wdir)

    if 'tpdf' not in conf['steps'][istep]['active distributions']:
        if 'tpdf' not in conf['steps'][istep]['passive distributions']:
            print('tpdf not an active or passive distribution')
            return

    conf['SofferBound'] = False
    resman = RESMAN(nworkers = 1, parallel = False, datasets = False, load_lhapdf=False)
    parman = resman.parman
    parman.order = replicas[0]['order'][istep]

    tpdf = conf['tpdf-mom']

    ## setup kinematics
    if Q2 == None: Q2 = conf['Q20']
    print('\ngenerating full moment %d for tpdf from %s at Q2 = %3.2f' % (mom, wdir, Q2))

    ## compute moments for all replicas
    moments = {}
    n_replicas = len(replicas)

    for i in range(n_replicas):
        lprint('%d/%d' % (i + 1, n_replicas))

        parman.order = copy.copy(replicas[i]['order'][istep])
        parman.set_new_params(replicas[i]['params'][istep], initial = True)

        tpdf.evolve([Q2])
        storage = tpdf.storage[Q2]

        for flavor in flavors:
            if flavor not in moments:
                moments[flavor] = []
            if flavor == 'quark':
                func = storage['u'] + storage['ub'] + storage['d'] + storage['db'] + \
                       storage['s'] + storage['sb'] + storage['c'] + storage['cb']
            elif flavor == 'uv':
                func = storage['u'] - storage['ub']
            elif flavor == 'dv':
                func = storage['d'] - storage['db']
            elif flavor == 'db-ub':
                func = storage['db'] - storage['ub']
            elif flavor == 'sp':
                func = storage['s']  + storage['sb']
            else:
                func = storage[flavor]

            func = np.real(func[mom-1])

            moments[flavor].append(func)

    print()


    checkdir('%s/data' % wdir)
    save({'Q2': Q2, 'moments': moments}, '%s/data/tpdf-moment-%d-Q2=%3.5f.dat' % (wdir,mom,Q2))







