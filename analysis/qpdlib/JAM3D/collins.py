#!/usr/bin/env python
import os,sys
import subprocess
import numpy as np
import scipy as sp
import pandas as pd
import copy
import random
import itertools as it


#--matplotlib
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.rcParams['text.usetex'] = True
from qcdlib.ff0 import FF
from qcdlib.pdf0 import PDF

from scipy import interpolate
from matplotlib.lines import Line2D
import pylab as py

from analysis.corelib import core,classifier

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config
from tools.inputmod  import INPUTMOD
from tools.randomstr import id_generator
#--from fitlib
from fitlib.resman import RESMAN

FLAV=[]
FLAV.append('u')   # 1
FLAV.append('d')   # 3

def gen_xf(wdir):
    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
    if 'collinspi' not in conf['steps'][istep]['active distributions']: return
    replicas=core.get_replicas(wdir)
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    parman.order=replicas[0]['order'][istep]
    ffpion=conf['collinspi']

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    #--setup kinematics
    X1=10**np.linspace(-4,-2,100)
    X2=np.linspace(0.011,0.99,100)
    X=np.concatenate([X1,X2])
    #Q2=conf['aux'].Q02
    Q2array=[2,4,10,100]

    #--compute XF for all replicas
    for Q2 in Q2array:
        XF={}
        cnt=0
        for replica in replicas:
            core.mod_conf(istep,core.get_replicas(wdir)[cnt])   
            cnt+=1
            lprint('%d/%d'%(cnt,len(replicas)))
            
            parman.set_new_params(replica,initial=True)
            for flav in FLAV:
                if flav not in XF:  XF[flav]=[]
                if  flav=='u':
                    func=lambda x: ffpion.get_C(x,Q2)[1]
                elif flav=='ub':
                    func=lambda x: ffpion.get_C(x,Q2)[2]
                elif flav=='d':
                    func=lambda x: ffpion.get_C(x,Q2)[3]
                elif flav=='db':
                    func=lambda x: ffpion.get_C(x,Q2)[4]
                elif flav=='s':
                    func=lambda x: ffpion.get_C(x,Q2)[5]
                elif flav=='sb':
                    func=lambda x: ffpion.get_C(x,Q2)[6]

                #XF[flav].append([2*0.135*x**2*func(x) for x in X])
                XF[flav].append([x*func(x) for x in X])
        print()
        checkdir('%s/data'%wdir)
        save({'X':X,'Q2':Q2,'XF':XF},'%s/data/collinspi-%d-%d.dat'%(wdir,istep,int(Q2)))

def plot_xf(wdir):
    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
    if 'collinspi' not in conf['steps'][istep]['active distributions']: return

    #cluster,colors,nc,order = classifier.get_clusters(wdir,istep)
    #data=load('%s/data/collinspi-%d.dat'%(wdir,istep))
    
    replicas=core.get_replicas(wdir)

    Q2array=[2,4,10,100]
    for Q2 in Q2array:
        if Q2==2:
            data1=load('%s/data/collinspi-%d-%d.dat'%(wdir,istep,int(Q2)))
            #save(data1,'%s/npdata/collinspi_reps-Q2-%d.dat'%(wdir,int(Q2)))
            #np.save('%s/nparrays/collinspi-Q2-%d.npy'%(wdir,int(Q2)),data1)
        elif Q2==10:
            data2=load('%s/data/collinspi-%d-%d.dat'%(wdir,istep,int(Q2)))
            #save(data2,'%s/npdata/collinspi_reps-Q2-%d.dat'%(wdir,int(Q2)))
            #np.save('%s/nparrays/collinspi-Q2-%d.npy'%(wdir,int(Q2)),data2)
        elif Q2==100:
            data3=load('%s/data/collinspi-%d-%d.dat'%(wdir,istep,int(Q2)))
            #save(data3,'%s/npdata/collinspi_reps-Q2-%d.dat'%(wdir,int(Q2)))
            #np.save('%s/nparrays/collinspi-Q2-%d.npy'%(wdir,int(Q2)),data3)
        elif Q2==4:
            data4=load('%s/data/collinspi-%d-%d.dat'%(wdir,istep,int(Q2)))
            #save(data4,'%s/npdata/collinspi_reps-Q2-%d.dat'%(wdir,int(Q2)))
            #np.save('%s/nparrays/collinspi-Q2-%d.npy'%(wdir,int(Q2)),data3)

    X=data1['X']

    ncols=2
    nrows=len(FLAV)/ncols
    if len(FLAV)%ncols>0: nrows+=1
    nrows = int(nrows)
    fig = py.figure(figsize=(ncols*3,nrows*2))

    cnt=0
    rand_list=[]
    for j in range(100):
        rand_list.append(random.randint(0, 950))
    for flav in FLAV:
        cnt+=1
        ax=py.subplot(nrows,ncols,cnt)
        _data1=data1['XF'][flav]
        _data2=data2['XF'][flav]
        _data3=data3['XF'][flav]
        _data4=data4['XF'][flav]
        for Q2 in Q2array:
            if Q2==4:
                for i in range(len(replicas)):
                    ax.plot(X,data4['XF'][flav][i],color='y',zorder= 0,alpha=0.3)
                c='r'
                meanXF2 = np.mean(_data4,axis=0)
                stdXF2 = np.std(_data4,axis=0)

                XFarray=[X,meanXF2,stdXF2]
                #save(XFarray,'%s/npdata/collinspi-%s-Q2-%d.dat'%(wdir,flav,int(Q2)))

                lower2=meanXF2-stdXF2
                upper2=meanXF2+stdXF2
                ax.plot(X,meanXF2,'%s-'%c,zorder=10,alpha=0.3)
                ax.fill_between(X, lower2, upper2,color=c,alpha=0.3)
            elif Q2==10:
                c='b'
                meanXF2 = np.mean(_data2,axis=0)
                #stdXF2 = np.std(_data2,axis=0)

                #XFarray=[X,meanXF2,stdXF2]
                #save(XFarray,'%s/npdata/sivers-%s-Q2-%d.dat'%(wdir,flav,int(Q2)))

                #lower2=meanXF2-stdXF2
                #upper2=meanXF2+stdXF2
                ax.plot(X,meanXF2,'%s-'%c,zorder=10,alpha=0.5)
                #ax.fill_between(X, lower2, upper2,color=c,alpha=0.5)
        
            elif Q2==100:
                c='g'
                meanXF3 = np.mean(_data3,axis=0)
                #stdXF3 = np.std(_data3,axis=0)
                #lower3=meanXF3-stdXF3
                #upper3=meanXF3+stdXF3
                ax.plot(X,meanXF3,'%s-'%c,zorder=10,alpha=0.5)
                #ax.fill_between(X, lower3, upper3,color=c,alpha=0.5)
                #for i in rand_list:
                #    ax.plot(X,data['XF'][flav][i],color='b',zorder= 0,alpha=0.1)

        #ax.semilogx()
        #ax.set_ylim(-0.04,0.04)
        ax.set_xlim(0.15, 1.0)
        if flav=='u': ax.set_ylim(-0.1,1)
        if flav=='d': ax.set_ylim(-1,0.1)
        if flav=='ub': ax.set_ylim(-0.055,0)
        if flav=='db': ax.set_ylim(0,0.03)
        if flav=='s' : ax.set_ylim(-0.055,0.03)
        if flav=='sb': ax.set_ylim(-0.055,0.03)
        #if flav=='s+sb/db+ub': ax.set_ylim(0,4)
        #if flav=='s+sb': ax.set_ylim(0,2)
        #if flav=='u':ax.set_ylabel('$2M_h\,z^2\,H_1^{\perp(1)\,fav}(z)$')
        #else: ax.set_ylabel('$2M_h\,z^2\,H_1^{\perp(1)\,unf}(z)$')
        if flav=='u':ax.set_ylabel('$z\,H_1^{\perp(1)\,fav}(z)$')
        else: ax.set_ylabel('$z\,H_1^{\perp(1)\,unf}(z)$')
        ax.set_xlabel('$z$')

    py.tight_layout()
    checkdir('%s/gallery'%wdir)
    py.savefig('%s/gallery/collinspi-%d.pdf'%(wdir,istep))
    py.close()

def plot_xf_relerr(wdir):
    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
    if 'collinspi' not in conf['steps'][istep]['active distributions']: return

    cluster,colors,nc,order = self.get_clusters(wdir,istep)
    #return

    Q2=10
    data=load('%s/data/collinspi-%d-%d.dat'%(wdir,istep,int(Q2)))

    X=data['X']

    ncols=2
    nrows=len(self.FLAV)/ncols
    if len(self.FLAV)%ncols>1: nrows+=1
    nrows = int(nrows)
    fig = py.figure(figsize=(ncols*3,nrows*2))

    cnt=0

    for flav in self.FLAV:
        cnt+=1
        ax=py.subplot(nrows,ncols,cnt)
        _data=data['XF'][flav]
        c='r'
        meanXF = np.mean(_data,axis=0)
        stdXF = np.std(_data,axis=0)

        _interpXF=interpolate.splrep(X,meanXF)
        interpXF=lambda x: interpolate.splev(x,_interpXF)

        _dinterpXF=interpolate.splrep(X,meanXF+stdXF)
        dinterpXF=lambda x: interpolate.splev(x,_dinterpXF)

        momXF=sp.integrate.quad(interpXF,0,1)[0]
        dmomXF=sp.integrate.quad(dinterpXF,0,1)[0]

        mom_err = np.abs((dmomXF-momXF)/momXF)

        print('collinspi %s moment rel err=%0.2f'%(flav,mom_err))

        ax.plot(X,np.abs(stdXF/meanXF),'%s-'%c,zorder=10,alpha=0.5)

        if flav=='u': ax.set_ylim(0,0.5)
        #if flav=='ub': ax.set_ylim(-0.05,0.05)
        if flav=='d': ax.set_ylim(0,0.75)
        #if flav=='db': ax.set_ylim(-0.05,0.05)
        #if flav=='s' : ax.set_ylim(-0.05,0.05)
        #if flav=='sb': ax.set_ylim(-0.05,0.05)
        #ax.set_ylim(-0.1,0.1)
        ax.set_xlim(0.2, 0.75)
        if flav=='u':ax.set_ylabel('$|\Delta H_1^{\perp(1)\,fav}/H_1^{\perp(1)\,fav}|$')
        else: ax.set_ylabel('$|\Delta H_1^{\perp(1)\,unf}/H_1^{\perp(1)\,unf}|$')
        ax.set_xlabel('$z$')

    py.tight_layout()
    checkdir('%s/gallery'%wdir)
    #py.savefig('%s/gallery/pdf-%d.pdf'%(wdir,istep))
    py.savefig('%s/gallery/collinspi-relerr-%d.pdf'%(wdir,istep))
    py.close()










