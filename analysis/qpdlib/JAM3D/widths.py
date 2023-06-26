#!/usr/bin/env python
import os,sys
import subprocess
import numpy as np
import scipy as sp
import pandas as pd
import copy
import random
import itertools as it

from analysis.corelib import core

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


#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config
from tools.inputmod  import INPUTMOD
from tools.randomstr import id_generator
#--from fitlib
from fitlib.resman import RESMAN




def pdf_par_plot(wdir):

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    par=[]
    par.append(['widths1_uv', 'widths1_sea'])

    kind=1
    tag='pdf'

    labels   = load('%s/data/labels-%d.dat'%(wdir,istep))
    chi2dof  = labels['chi2dof']
    cluster   = labels['cluster']

    clusters=np.unique(cluster)
    data={}
    for i in clusters:
        data[i]={}
        for row in par:
            for _ in row:
                if _!=None: data[i][_]=[]


    replicas=core.get_replicas(wdir)
    for j in range(len(replicas)):
        #for replica in replicas:
        replica=replicas[j]
        #if cluster[j]!=0: continue
        params=replica['params'][istep]
        order=replica['order'][istep]
        for i in range(len(order)):
            if kind != order[i][0]:continue
            if tag  != order[i][1]:continue
            for _ in data[cluster[j]]:
                if  _ ==order[i][2]:
                    data[cluster[j]][_].append(params[i])

    data=pd.DataFrame(data)
    widths1_uvMean = np.mean(np.array(data[0]['widths1_uv']))
    widths1_uvSTD = np.std(np.array(data[0]['widths1_uv']))
    meanuv = widths1_uvMean
    stduv=widths1_uvSTD

    widths1_seaMean = np.mean(np.array(data[0]['widths1_sea']))
    widths1_seaSTD = np.std(np.array(data[0]['widths1_sea']))
    meansea = widths1_seaMean
    stdsea=widths1_seaSTD

    #data=data[data['s1 a'] >-1]#data.query("'s1 a'>-0.9")
    nrows,ncols=len(par),3
    fig = py.figure(figsize=(ncols*3,nrows*1.5))
    cnt=0
    for row in par:
        for _ in row:
            cnt+=1
            if _==None: continue
            ax=py.subplot(nrows,ncols,cnt)
            for i in data:
                if i==0: c='r'
                if i==1: c='b'
                if i==2: c='g'
                if i==3: c='m'
                ax.hist(data[i][_],bins=50,color=c)
            if '_' in _: _ = _.replace("_", " ")
            ax.set_xlabel(_)
            
    fig.text(1.2,0.7,'$Mean Valence: %f$'%meanuv,transform=ax.transAxes)
    fig.text(1.2,0.5,'$Error Valence: %f$'%stduv,transform=ax.transAxes)
    fig.text(1.2,0.3,'$Mean Sea: %f$'%meansea,transform=ax.transAxes)
    fig.text(1.2,0.1,'$Error Sea: %f$'%stdsea,transform=ax.transAxes)

    py.tight_layout()
    checkdir('%s/gallery'%wdir)
    py.savefig('%s/gallery/par-plot-pdf-%d.pdf'%(wdir,istep))
    py.close()

def pdfpi_par_plot(wdir):

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    par=[]
    par.append(['widths1_ubv', 'widths1_sea'])

    kind=1
    tag='pdfpi-'

    labels   = load('%s/data/labels-%d.dat'%(wdir,istep))
    chi2dof  = labels['chi2dof']
    cluster   = labels['cluster']

    clusters=np.unique(cluster)
    data={}
    for i in clusters:
        data[i]={}
        for row in par:
            for _ in row:
                if _!=None: data[i][_]=[]

    replicas=core.get_replicas(wdir)
    for j in range(len(replicas)):
        #for replica in replicas:
        replica=replicas[j]
        #if cluster[j]!=0: continue
        params=replica['params'][istep]
        order=replica['order'][istep]
        for i in range(len(order)):
            if kind != order[i][0]:continue
            if tag  != order[i][1]:continue
            for _ in data[cluster[j]]:
                if  _ ==order[i][2]:
                    data[cluster[j]][_].append(params[i])

    data=pd.DataFrame(data)
    widths1_ubvMean = np.mean(data[0]['widths1_ubv'])
    widths1_ubvSTD = np.std(data[0]['widths1_ubv'])
    meanubv = str(widths1_ubvMean)
    stdubv=str(widths1_ubvSTD)

    widths1_seaMean = np.mean(data[0]['widths1_sea'])
    widths1_seaSTD = np.std(data[0]['widths1_sea'])
    meansea = str(widths1_seaMean)
    stdsea=str(widths1_seaSTD)

    #data=data[data['s1 a'] >-1]#data.query("'s1 a'>-0.9")
    nrows,ncols=len(par),3
    fig = py.figure(figsize=(ncols*3,nrows*1.5))
    cnt=0
    for row in par:
        for _ in row:
            cnt+=1
            if _==None: continue
            ax=py.subplot(nrows,ncols,cnt)
            for i in data:
                if i==0: c='r'
                if i==1: c='b'
                if i==2: c='g'
                if i==3: c='m'
                ax.hist(data[i][_],bins=50,color=c)
                
            if '_' in _: _ = _.replace("_", " ")
            ax.set_xlabel(_)
            
    fig.text(1.2,0.7,'$Mean Valence: %s$'%meanubv,transform=ax.transAxes)
    fig.text(1.2,0.5,'$Error Valence: %s$'%stdubv,transform=ax.transAxes)
    fig.text(1.2,0.3,'$Mean Sea: %s$'%meansea,transform=ax.transAxes)
    fig.text(1.2,0.1,'$Error Sea: %s$'%stdsea,transform=ax.transAxes)

    py.tight_layout()
    checkdir('%s/gallery'%wdir)
    py.savefig('%s/gallery/par-plot-pion-pdf-%d.pdf'%(wdir,istep))
    py.close()

def kaon_par_plot(wdir):

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    par=[]
    par.append(['widths1_fav','widths1_ufav'])

    kind=1
    tag='ffk'
    #name='c1 N'

    labels   = load('%s/data/labels-%d.dat'%(wdir,istep))
    chi2dof  = labels['chi2dof']
    cluster   = labels['cluster']

    clusters=np.unique(cluster)
    data={}
    for i in clusters:
        data[i]={}
        for row in par:
            for _ in row:
                if _!=None: data[i][_]=[]

    replicas=core.get_replicas(wdir)
    #data={_:[] for _ in self.FLAV}

    for j in range(len(replicas)):
        #for replica in replicas:
        replica=replicas[j]
        #if cluster[j]!=0: continue
        params=replica['params'][istep]
        order=replica['order'][istep]
        for i in range(len(order)):
            if kind != order[i][0]:continue
            if tag  != order[i][1]:continue
            for _ in data[cluster[j]]:
                if  _ ==order[i][2]:
                    data[cluster[j]][_].append(params[i])

    data=pd.DataFrame(data)

    widths1_favMean = np.mean(data[0]['widths1_fav'])
    widths1_favSTD = np.std(data[0]['widths1_fav'])
    meanfav = str(widths1_favMean)
    stdfav=str(widths1_favSTD)

    widths1_ufavMean = np.mean(data[0]['widths1_ufav'])
    widths1_ufavSTD = np.std(data[0]['widths1_ufav'])
    meanufav = str(widths1_ufavMean)
    stdufav=str(widths1_ufavSTD)


    #data=data[data['s1 a'] >-1]#data.query("'s1 a'>-0.9")
    nrows,ncols=len(par),3
    fig = py.figure(figsize=(ncols*3,nrows*1.5))
    cnt=0
    for row in par:
        for _ in row:
            cnt+=1
            if _==None: continue
            ax=py.subplot(nrows,ncols,cnt)
            for i in data:
                if i==0: c='r'
                if i==1: c='b'
                if i==2: c='g'
                if i==3: c='m'
                ax.hist(data[i][_],bins=50,color=c)
                
            if '_' in _: _ = _.replace("_", " ")
            ax.set_xlabel(_)

    fig.text(1.2,0.7,'$Mean Fav: %s$'%meanfav,transform=ax.transAxes)
    fig.text(1.2,0.5,'$Error Fav: %s$'%stdfav,transform=ax.transAxes)
    fig.text(1.2,0.3,'$Mean Unfav: %s$'%meanufav,transform=ax.transAxes)
    fig.text(1.2,0.1,'$Error Unfav: %s$'%stdufav,transform=ax.transAxes)

    py.tight_layout()
    checkdir('%s/gallery'%wdir)
    py.savefig('%s/gallery/par-plot-k-%d.pdf'%(wdir,istep))
    py.close()

def pion_par_plot(wdir):

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    par=[]
    par.append(['widths1_fav', 'widths1_ufav'])

    kind=1
    tag='ffpi'

    labels   = load('%s/data/labels-%d.dat'%(wdir,istep))
    chi2dof  = labels['chi2dof']
    cluster   = labels['cluster']

    clusters=np.unique(cluster)
    data={}
    for i in clusters:
        data[i]={}
        for row in par:
            for _ in row:
                if _!=None: data[i][_]=[]


    replicas=core.get_replicas(wdir)
    for j in range(len(replicas)):
        #for replica in replicas:
        replica=replicas[j]
        #if cluster[j]!=0: continue
        params=replica['params'][istep]
        order=replica['order'][istep]
        for i in range(len(order)):
            if kind != order[i][0]:continue
            if tag  != order[i][1]:continue
            for _ in data[cluster[j]]:
                if  _ ==order[i][2]:
                    data[cluster[j]][_].append(params[i])

    data=pd.DataFrame(data)

    widths1_favMean = np.mean(data[0]['widths1_fav'])
    widths1_favSTD = np.std(data[0]['widths1_fav'])
    meanfav = str(widths1_favMean)
    stdfav=str(widths1_favSTD)

    widths1_ufavMean = np.mean(data[0]['widths1_ufav'])
    widths1_ufavSTD = np.std(data[0]['widths1_ufav'])
    meanufav = str(widths1_ufavMean)
    stdufav=str(widths1_ufavSTD)

    #data=data[data['s1 a'] >-1]#data.query("'s1 a'>-0.9")
    nrows,ncols=len(par),3
    fig = py.figure(figsize=(ncols*3,nrows*1.5))
    cnt=0
    for row in par:
        for _ in row:
            cnt+=1
            if _==None: continue
            ax=py.subplot(nrows,ncols,cnt)
            for i in data:
                if i==0: c='r'
                if i==1: c='b'
                if i==2: c='g'
                if i==3: c='m'
                ax.hist(data[i][_],bins=50,color=c)
                
            if '_' in _: _ = _.replace("_", " ")
            ax.set_xlabel(_)

    fig.text(1.2,0.7,'$Mean Fav: %s$'%meanfav,transform=ax.transAxes)
    fig.text(1.2,0.5,'$Error Fav: %s$'%stdfav,transform=ax.transAxes)
    fig.text(1.2,0.3,'$Mean Unfav: %s$'%meanufav,transform=ax.transAxes)
    fig.text(1.2,0.1,'$Error Unfav: %s$'%stdufav,transform=ax.transAxes)

    py.tight_layout()
    checkdir('%s/gallery'%wdir)
    py.savefig('%s/gallery/par-plot-pi-%d.pdf'%(wdir,istep))
    py.close()









