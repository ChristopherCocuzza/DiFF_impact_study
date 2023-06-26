#!/usr/bin/env python
import sys,os
import numpy as np
import pandas as pd
import copy
from subprocess import Popen, PIPE, STDOUT

#--matplotlib
import matplotlib
matplotlib.use('Agg')
import pylab as py
from matplotlib.ticker import MultipleLocator

#--from scipy stack 
from scipy.integrate import fixed_quad
from scipy import interpolate

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config

#-- from qcdlib
from qcdlib import aux

#--from local
from analysis.corelib import core
from analysis.corelib import classifier

#--from obslib
from obslib.dihadron_pp.reader import READER

import kmeanconf as kc

reaction = 'dihadron_pp'

def plot_obs(wdir,mode=1):

    print('\ngenerating STAR plots from %s'%(wdir))
    plot_star_RS200               (wdir,mode,angle=0.2)
    plot_star_RS200               (wdir,mode,angle=0.3)
    plot_star_RS200               (wdir,mode,angle=0.4)
    plot_star_RS500_M             (wdir,mode)
    plot_star_RS500_PhT           (wdir,mode)
    plot_star_RS500_eta           (wdir,mode)

    plot_star_RS200_ang03_M       (wdir,mode)
    plot_star_RS200_ang03_PhT     (wdir,mode)
    plot_star_RS200_ang03_eta     (wdir,mode)

    plot_star_RS200_xsec_M        (wdir,mode)

#--published 200 GeV asymmetry
def plot_star_RS200(wdir,mode,angle=0.4):
    nrows,ncols=1,3
    fig = py.figure(figsize=(ncols*9,nrows*5))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)
    ax13=py.subplot(nrows,ncols,3)

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if reaction not in predictions['reactions']: return
    if angle==0.2:
        if 2000 not in predictions['reactions'][reaction]: return
        if 2001 not in predictions['reactions'][reaction]: return
        if 2002 not in predictions['reactions'][reaction]: return
        if 2003 not in predictions['reactions'][reaction]: return
        if 2004 not in predictions['reactions'][reaction]: return
    if angle==0.3:
        if 2100 not in predictions['reactions'][reaction]: return
        if 2101 not in predictions['reactions'][reaction]: return
        if 2102 not in predictions['reactions'][reaction]: return
        if 2103 not in predictions['reactions'][reaction]: return
        if 2104 not in predictions['reactions'][reaction]: return
    if angle==0.4:
        if 2200 not in predictions['reactions'][reaction]: return
        if 2201 not in predictions['reactions'][reaction]: return
        if 2202 not in predictions['reactions'][reaction]: return
        if 2203 not in predictions['reactions'][reaction]: return
        if 2204 not in predictions['reactions'][reaction]: return

    filters = conf['datasets'][reaction]['filters']

    conf['aux']=aux.AUX()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    data = predictions['reactions'][reaction]

    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)

    #--get theory by seperating solutions and taking mean
    for idx in data:
        predictions = copy.copy(data[idx]['prediction-rep'])
        for ic in range(nc):
            predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
            data[idx]['thy-%d'%ic]  = np.mean(predictions_ic,axis=0)
            data[idx]['dthy-%d'%ic] = np.std(predictions_ic,axis=0)

    hand = {}

    ic = 0
    #--plot z and Mh
    for idx in data:

        if 'max_open_angle' not in data[idx]: continue
        max_angle = data[idx]['max_open_angle'][0]
        if max_angle != angle: continue

        binned = data[idx]['binned'][0]

        if binned == 'M':   ax = ax11
        if binned == 'PhT': ax = ax12
        if binned == 'eta': ax = ax13

        eta = data[idx]['eta']

        if binned != 'eta' and eta[0] < 0: color = 'darkblue'
        if binned != 'eta' and eta[0] > 0: color = 'firebrick'
        if binned == 'eta':                color = 'firebrick'

        M   = data[idx]['M']
        pT  = data[idx]['PhT']
        eta = data[idx]['eta']
        value = data[idx]['value']
        alpha = data[idx]['alpha']        
        thy = data[idx]['thy-%d'%ic]
        std = data[idx]['dthy-%d'%ic]

        if binned=='M':
            hand[idx]    = ax.errorbar(M  ,value,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
        if binned=='PhT':
            hand[idx]    = ax.errorbar(pT ,value,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
        if binned=='eta':
            hand[idx]    = ax.errorbar(eta,value,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)

        if binned=='M':
            if mode==0:
                hand['thy'] ,= ax.plot(M,thy,color=color)
            if mode==1:
                thy_plot ,= ax.plot(M,thy,color=color)
                thy_band  = ax.fill_between(M,thy-std,thy+std,color=color,alpha=0.4)
        if binned=='PhT':
            if mode==0:
                hand['thy'] ,= ax.plot(pT,thy,color=color)
            if mode==1:
                thy_plot ,= ax.plot(pT,thy,color=color)
                thy_band  = ax.fill_between(pT,thy-std,thy+std,color=color,alpha=0.4)
        if binned=='eta':
            if mode==0:
                hand['thy'] ,= ax.plot(eta,thy,color=color)
            if mode==1:
                thy_plot ,= ax.plot(eta,thy,color=color)
                thy_band  = ax.fill_between(eta,thy-std,thy+std,color=color,alpha=0.4)


    ax11.set_xlim(0.3,1.3)
    ax11.set_xticks([0.4,0.6,0.8,1.0,1.2])
    minorLocator = MultipleLocator(0.1)
    ax11.xaxis.set_minor_locator(minorLocator)

    ax12.set_xlim(3,11)
    ax12.set_xticks([4,6,8,10])
    minorLocator = MultipleLocator(1)
    ax12.xaxis.set_minor_locator(minorLocator)


    ax13.set_xlim(-0.9,0.9)
    ax13.set_xticks([-0.8,-0.4,0,0.4,0.8])
    minorLocator = MultipleLocator(0.1)
    ax13.xaxis.set_minor_locator(minorLocator)
 
    ax11.set_ylim(-0.1,0.20)
    ax11.set_yticks([-0.05,0,0.05,0.10,0.15])
    minorLocator = MultipleLocator(0.01)
    ax11.yaxis.set_minor_locator(minorLocator)

    ax12.set_ylim(-0.08,0.08)
    ax12.set_yticks([-0.05,0,0.05])
    minorLocator = MultipleLocator(0.01)
    ax12.yaxis.set_minor_locator(minorLocator)

    ax13.set_ylim(-0.01,0.04)
    ax13.set_yticks([0,0.01,0.02,0.03])
    minorLocator = MultipleLocator(0.005)
    ax13.yaxis.set_minor_locator(minorLocator)

    ax11.set_xlabel(r'\boldmath$M_h~[{\rm GeV}]$',size=30)
    ax12.set_xlabel(r'\boldmath$P_{hT}~[{\rm GeV}]$',size=30)
    ax13.set_xlabel(r'\boldmath$\eta$',size=30)

    for ax in [ax11,ax12,ax13]:
        ax .tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax .tick_params(axis='both',which='minor',size=4)
        ax .tick_params(axis='both',which='major',size=8)
        ax .axhline(0,0,1,color='black',ls='--',alpha=0.5)


    ax12.text(0.05,0.85,r'\textrm{\textbf{STAR \boldmath$A_{UT}$}}' , transform=ax12.transAxes, size=25)
    ax13.text(0.05,0.85,r'\textrm{\textbf{Max opening angle: %s}}'%angle, transform=ax13.transAxes, size=25)
    ax13.text(0.05,0.70,r'\boldmath$\sqrt{s} = 200~{\rm GeV}$' ,      transform=ax13.transAxes, size=25)

    minorLocator = MultipleLocator(0.1)
    #ax11.xaxis.set_minor_locator(minorLocator)

    fs = 30

    handles,labels = [], []
    if angle==0.2:
        handles.append(hand[2000])
        handles.append(hand[2001])
    if angle==0.3:
        handles.append(hand[2100])
        handles.append(hand[2101])
    if angle==0.4:
        handles.append(hand[2200])
        handles.append(hand[2201])
    labels.append(r'\boldmath$\eta<0$')
    labels.append(r'\boldmath$\eta>0$')
    ax11.legend(handles,labels,loc='upper left',fontsize=fs,frameon=False, handlelength = 1.0, handletextpad = 0.1, ncol = 1, columnspacing = 1.0)


    py.tight_layout()
    py.subplots_adjust(wspace=0.14,left=0.03)
    filename='%s/gallery/star-ang%s'%(wdir,angle)
    if mode==1: filename+= '-bands'
    filename+='.png'

    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()


#--published 500 GeV asymmetry
def plot_star_RS500_M(wdir,mode):
    nrows,ncols=2,3
    fig = py.figure(figsize=(ncols*9,nrows*5))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)
    ax13=py.subplot(nrows,ncols,3)
    ax21=py.subplot(nrows,ncols,4)
    ax22=py.subplot(nrows,ncols,5)

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if reaction not in predictions['reactions']: return
    if 3002 not in predictions['reactions'][reaction]: return
    if 3003 not in predictions['reactions'][reaction]: return

    filters = conf['datasets'][reaction]['filters']

    conf['aux']=aux.AUX()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    data = predictions['reactions'][reaction]

    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)

    #--get theory by seperating solutions and taking mean
    for idx in data:
        predictions = copy.copy(data[idx]['prediction-rep'])
        for ic in range(nc):
            predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
            data[idx]['thy-%d'%ic]  = np.mean(predictions_ic,axis=0)
            data[idx]['dthy-%d'%ic] = np.std(predictions_ic,axis=0)

    ic = 0
    hand = {}

    for idx in data:

        if idx not in [3002,3003]: continue

        binned = data[idx]['binned'][0]
        bins   = np.unique(data[idx]['bin'])

        if data[idx]['eta'][0] > 0: color = 'firebrick'
        if data[idx]['eta'][0] < 0: color = 'darkblue'
        for i in bins:
            i = int(i)
            if i==1: ax = ax11
            if i==2: ax = ax12
            if i==3: ax = ax13
            if i==4: ax = ax21
            if i==5: ax = ax22
            ind = np.where(data[idx]['bin'] == i)
            M   = data[idx]['M'][ind]
            pT  = data[idx]['PhT'][ind]
            eta = data[idx]['eta'][ind]
            value = data[idx]['value'][ind]
            alpha = data[idx]['alpha'][ind]
            thy = data[idx]['thy-%d'%ic][ind]
            std = data[idx]['dthy-%d'%ic][ind]

            if idx==3003: M += 0.05

            hand[idx]    = ax.errorbar(M  ,value,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
            ax.text(0.05,0.85,r'\boldmath$\langle P_{hT} \rangle = %d~{\rm GeV}$'%pT[0] , transform=ax.transAxes, size=25)
            if mode==0:
                hand['thy'] ,= ax.plot(M,thy,color=color)
            if mode==1:
                thy_plot ,= ax.plot(M,thy,color=color)
                thy_band  = ax.fill_between(M,thy-std,thy+std,color=color,alpha=0.4)


    for ax in [ax11,ax12,ax13,ax21,ax22]:

        ax.set_xlim(0.2,2.4)
        ax.set_xticks([0.5,1.0,1.5,2])
        minorLocator = MultipleLocator(0.1)
        ax.xaxis.set_minor_locator(minorLocator)

        ax.set_ylim(-0.025,0.06)
        ax.set_yticks([-0.02,0,0.02,0.04])
        minorLocator = MultipleLocator(0.005)
        ax.yaxis.set_minor_locator(minorLocator)

        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.tick_params(axis='both',which='minor',size=4)
        ax.tick_params(axis='both',which='major',size=8)
        ax.axhline(0,0,1,color='black',ls='--',alpha=0.5)
 
    for ax in [ax11,ax12]:
        ax.tick_params(labelbottom = False)

    for ax in [ax12,ax13,ax22]:
        ax.tick_params(labelleft = False)

    for ax in [ax13,ax21,ax22]:
        ax.set_xlabel(r'\boldmath$M_h~[{\rm GeV}]$',size=30)

    ax22.text(1.10,0.60,r'\textrm{\textbf{STAR \boldmath$A_{UT}$}}' , transform=ax22.transAxes, size=30)
    ax22.text(1.10,0.50,r'\boldmath$\sqrt{s} = 500~{\rm GeV}$' ,      transform=ax22.transAxes, size=30)

    fs = 30

    handles,labels = [], []
    handles.append(hand[3002])
    handles.append(hand[3003])
    labels.append(r'\boldmath$-1<\eta<0$')
    labels.append(r'\boldmath$0<\eta<1$')
    ax11.legend(handles,labels,loc='upper right',fontsize=fs,frameon=False, handlelength = 1.0, handletextpad = 0.1, ncol = 1, columnspacing = 1.0)


    py.tight_layout()
    py.subplots_adjust(wspace=0.0,hspace=0.02,left=0.03)
    filename='%s/gallery/star-RS500-M'%(wdir)
    if mode==1: filename+= '-bands'
    filename+='.png'

    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()

def plot_star_RS500_PhT(wdir,mode):
    nrows,ncols=2,3
    fig = py.figure(figsize=(ncols*9,nrows*5))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)
    ax13=py.subplot(nrows,ncols,3)
    ax21=py.subplot(nrows,ncols,4)
    ax22=py.subplot(nrows,ncols,5)

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if reaction not in predictions['reactions']: return
    if 3001 not in predictions['reactions'][reaction]: return

    filters = conf['datasets'][reaction]['filters']

    conf['aux']=aux.AUX()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    data = predictions['reactions'][reaction]

    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)

    #--get theory by seperating solutions and taking mean
    for idx in data:
        predictions = copy.copy(data[idx]['prediction-rep'])
        for ic in range(nc):
            predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
            data[idx]['thy-%d'%ic]  = np.mean(predictions_ic,axis=0)
            data[idx]['dthy-%d'%ic] = np.std(predictions_ic,axis=0)

    ic = 0
    hand = {}

    for idx in data:

        if idx not in [3001]: continue

        binned = data[idx]['binned'][0]
        bins   = np.unique(data[idx]['bin'])

        if data[idx]['etamax'][0] >= 0: color = 'firebrick'
        if data[idx]['etamax'][0] <= 0: color = 'darkblue'
        for i in bins:
            i = int(i)
            if i==1: ax = ax11
            if i==2: ax = ax12
            if i==3: ax = ax13
            if i==4: ax = ax21
            if i==5: ax = ax22
            ind = np.where(data[idx]['bin'] == i)
            M   = data[idx]['M'][ind]
            pT  = data[idx]['PhT'][ind]
            value = data[idx]['value'][ind]
            alpha = data[idx]['alpha'][ind]
            thy = data[idx]['thy-%d'%ic][ind]
            std = data[idx]['dthy-%d'%ic][ind]

            hand[idx]    = ax.errorbar(pT  ,value,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
            ax.text(0.05,0.85,r'\boldmath$\langle M_{h} \rangle = %3.2f~{\rm GeV}$'%M[0] , transform=ax.transAxes, size=25)
            if mode==0:
                hand['thy'] ,= ax.plot(pT,thy,color=color)
            if mode==1:
                thy_plot ,= ax.plot(pT,thy,color=color)
                thy_band  = ax.fill_between(pT,thy-std,thy+std,color=color,alpha=0.4)


    for ax in [ax11,ax12,ax13,ax21,ax22]:

        ax.set_xlim(2,20)
        ax.set_xticks([4,8,12,16])
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)

        ax.set_ylim(-0.015,0.06)
        ax.set_yticks([0,0.02,0.04])
        minorLocator = MultipleLocator(0.005)
        ax.yaxis.set_minor_locator(minorLocator)

        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.tick_params(axis='both',which='minor',size=4)
        ax.tick_params(axis='both',which='major',size=8)
        ax.axhline(0,0,1,color='black',ls='--',alpha=0.5)
 
    for ax in [ax11,ax12]:
        ax.tick_params(labelbottom = False)

    for ax in [ax12,ax13,ax22]:
        ax.tick_params(labelleft = False)

    for ax in [ax13,ax21,ax22]:
        ax.set_xlabel(r'\boldmath$P_{hT}~[{\rm GeV}]$',size=30)

    ax22.text(1.10,0.60,r'\textrm{\textbf{STAR \boldmath$A_{UT}$}}' , transform=ax22.transAxes, size=30)
    ax22.text(1.10,0.50,r'\boldmath$\sqrt{s} = 500~{\rm GeV}$' ,      transform=ax22.transAxes, size=30)

    fs = 30

    handles,labels = [], []
    handles.append(hand[3001])
    labels.append(r'\boldmath$0<\eta<1$')
    ax11.legend(handles,labels,loc='upper right',fontsize=fs,frameon=False, handlelength = 1.0, handletextpad = 0.1, ncol = 1, columnspacing = 1.0)


    py.tight_layout()
    py.subplots_adjust(wspace=0.0,hspace=0.02,left=0.03)
    filename='%s/gallery/star-RS500-PhT'%(wdir)
    if mode==1: filename+= '-bands'
    filename+='.png'

    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()

def plot_star_RS500_eta(wdir,mode):
    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax11=py.subplot(nrows,ncols,1)

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if reaction not in predictions['reactions']: return
    if 3004 not in predictions['reactions'][reaction]: return

    filters = conf['datasets'][reaction]['filters']

    conf['aux']=aux.AUX()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    data = predictions['reactions'][reaction]

    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)

    #--get theory by seperating solutions and taking mean
    for idx in data:
        predictions = copy.copy(data[idx]['prediction-rep'])
        for ic in range(nc):
            predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
            data[idx]['thy-%d'%ic]  = np.mean(predictions_ic,axis=0)
            data[idx]['dthy-%d'%ic] = np.std(predictions_ic,axis=0)

    ic = 0
    hand = {}

    for idx in data:

        if idx not in [3004]: continue
        ax = ax11

        color = 'firebrick'
        M       = data[idx]['M']
        pT      = data[idx]['PhT']
        eta     = data[idx]['eta']
        value   = data[idx]['value']
        alpha = data[idx]['alpha']
        thy = data[idx]['thy-%d'%ic]
        std = data[idx]['dthy-%d'%ic]

        hand[idx]    = ax.errorbar(eta,value,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
        if mode==0:
            for k in range(len(predictions_ic)):
                thy = predictions_ic[k]
                hand['thy'] ,= ax.plot(eta,thy,color='black',lw=1.0,alpha=0.5)
        if mode==1:
            thy_plot ,= ax.plot(eta,thy,color=color)
            thy_band  = ax.fill_between(eta,thy-std,thy+std,color=color,alpha=0.4)


    for ax in [ax11]:

        ax.set_xlim(-1,1)
        ax.set_xticks([-0.5,0,0.5])
        minorLocator = MultipleLocator(0.1)
        ax.xaxis.set_minor_locator(minorLocator)

        ax.set_ylim(-0.005,0.04)
        ax.set_yticks([0,0.01,0.02,0.03])
        minorLocator = MultipleLocator(0.002)
        ax.yaxis.set_minor_locator(minorLocator)

        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.tick_params(axis='both',which='minor',size=4)
        ax.tick_params(axis='both',which='major',size=8)
        ax.axhline(0,0,1,color='black',ls='--',alpha=0.5)
        ax.set_xlabel(r'\boldmath$\eta$',size=40)
        ax.xaxis.set_label_coords(0.95,-0.01)

    ax11.text(0.05,0.85,r'\textrm{\textbf{STAR \boldmath$A_{UT}$}}'         ,transform=ax11.transAxes, size=25)
    ax11.text(0.05,0.75,r'\boldmath$\sqrt{s} = 500~{\rm GeV}$'              ,transform=ax11.transAxes, size=20)
    ax11.text(0.05,0.65,r'\boldmath$\langle P_{hT} \rangle = 13~{\rm GeV}$' ,transform=ax11.transAxes, size=20)
    ax11.text(0.05,0.55,r'\boldmath$\langle M_h \rangle = 1 ~ {\rm GeV}$'   ,transform=ax11.transAxes, size=20)

    #handles,labels = [], []
    #handles.append(hand[5020])
    #labels.append(r'\textrm{\textbf{STAR}}')
    #ax11.legend(handles,labels,loc='upper right',fontsize=fs,frameon=False, handlelength = 1.0, handletextpad = 0.1, ncol = 1, columnspacing = 1.0)

    py.tight_layout()
    py.subplots_adjust(wspace=0.0,hspace=0.00,left=0.10)
    filename='%s/gallery/star-RS500-eta'%wdir
    if mode==1: filename+= '-bands'
    filename+='.png'

    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()



#--preliminary 200 GeV asymmetry
def plot_star_RS200_ang03_M(wdir,mode):
    nrows,ncols=2,3
    fig = py.figure(figsize=(ncols*9,nrows*5))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)
    ax13=py.subplot(nrows,ncols,3)
    ax21=py.subplot(nrows,ncols,4)
    ax22=py.subplot(nrows,ncols,5)

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if reaction not in predictions['reactions']: return
    if 2300 not in predictions['reactions'][reaction]: return
    if 2301 not in predictions['reactions'][reaction]: return

    filters = conf['datasets'][reaction]['filters']

    conf['aux']=aux.AUX()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    data = predictions['reactions'][reaction]

    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)

    #--get theory by seperating solutions and taking mean
    for idx in data:
        predictions = copy.copy(data[idx]['prediction-rep'])
        for ic in range(nc):
            predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
            data[idx]['thy-%d'%ic]  = np.mean(predictions_ic,axis=0)
            data[idx]['dthy-%d'%ic] = np.std(predictions_ic,axis=0)

    ic = 0
    hand = {}

    for idx in data:

        if idx not in [2300,2301]: continue

        binned = data[idx]['binned'][0]
        bins   = np.unique(data[idx]['bin'])

        if data[idx]['eta'][0] >= 0: color = 'firebrick'
        if data[idx]['eta'][0] <= 0: color = 'darkblue'
        for i in bins:
            i = int(i)
            if i==1: ax = ax11
            if i==2: ax = ax12
            if i==3: ax = ax13
            if i==4: ax = ax21
            if i==5: ax = ax22
            ind = np.where(data[idx]['bin'] == i)
            M   = data[idx]['M'][ind]
            pT  = data[idx]['PhT'][ind]
            value = data[idx]['value'][ind]
            alpha = data[idx]['alpha'][ind]
            thy = data[idx]['thy-%d'%ic][ind]
            std = data[idx]['dthy-%d'%ic][ind]

            if idx==2301: M += 0.05

            hand[idx]    = ax.errorbar(M  ,value,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
            ax.text(0.05,0.85,r'\boldmath$\langle P_{hT} \rangle = %3.2f~{\rm GeV}$'%pT[0] , transform=ax.transAxes, size=25)
            if mode==0:
                hand['thy'] ,= ax.plot(M,thy,color=color)
            if mode==1:
                thy_plot ,= ax.plot(M,thy,color=color)
                thy_band  = ax.fill_between(M,thy-std,thy+std,color=color,alpha=0.4)


    for ax in [ax11,ax12,ax13,ax21,ax22]:

        ax.set_xlim(0.2,1.9)
        ax.set_xticks([0.5,1.0,1.5])
        minorLocator = MultipleLocator(0.1)
        ax.xaxis.set_minor_locator(minorLocator)

        minorLocator = MultipleLocator(0.005)
        ax.yaxis.set_minor_locator(minorLocator)

        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.tick_params(axis='both',which='minor',size=4)
        ax.tick_params(axis='both',which='major',size=8)
        ax.axhline(0,0,1,color='black',ls='--',alpha=0.5)

    for ax in [ax11,ax12,ax13]:
        ax.set_ylim(-0.015,0.04)
        ax.set_yticks([-0.01,0,0.01,0.02,0.03])

    for ax in [ax21,ax22]:
        ax.set_ylim(-0.015,0.10)
        ax.set_yticks([0,0.02,0.04,0.06,0.08])
 
    for ax in [ax11,ax12]:
        ax.tick_params(labelbottom = False)

    for ax in [ax12,ax13,ax22]:
        ax.tick_params(labelleft = False)

    for ax in [ax13,ax21,ax22]:
        ax.set_xlabel(r'\boldmath$M_h~[{\rm GeV}]$',size=30)

    ax22.text(1.10,0.60,r'\textrm{\textbf{STAR \boldmath$A_{UT}$}}'        , transform=ax22.transAxes, size=30)
    ax22.text(1.10,0.50,r'\boldmath$\sqrt{s} = 200~{\rm GeV}$'             , transform=ax22.transAxes, size=30)

    fs = 30

    handles,labels = [], []
    handles.append(hand[2300])
    handles.append(hand[2301])
    labels.append(r'\boldmath$-1<\eta<0$')
    labels.append(r'\boldmath$0<\eta<1$')
    ax11.legend(handles,labels,loc='upper right',fontsize=fs,frameon=False, handlelength = 1.0, handletextpad = 0.1, ncol = 1, columnspacing = 1.0)


    py.tight_layout()
    py.subplots_adjust(wspace=0.0,hspace=0.02,left=0.03)
    filename='%s/gallery/star-RS200-ang0.3-M'%(wdir)
    if mode==1: filename+= '-bands'
    filename+='.png'

    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()

def plot_star_RS200_ang03_PhT(wdir,mode):
    nrows,ncols=2,3
    fig = py.figure(figsize=(ncols*9,nrows*5))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)
    ax13=py.subplot(nrows,ncols,3)
    ax21=py.subplot(nrows,ncols,4)
    ax22=py.subplot(nrows,ncols,5)

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if reaction not in predictions['reactions']: return
    if 2302 not in predictions['reactions'][reaction]: return
    if 2303 not in predictions['reactions'][reaction]: return

    filters = conf['datasets'][reaction]['filters']

    conf['aux']=aux.AUX()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    data = predictions['reactions'][reaction]

    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)

    #--get theory by seperating solutions and taking mean
    for idx in data:
        predictions = copy.copy(data[idx]['prediction-rep'])
        for ic in range(nc):
            predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
            data[idx]['thy-%d'%ic]  = np.mean(predictions_ic,axis=0)
            data[idx]['dthy-%d'%ic] = np.std(predictions_ic,axis=0)

    ic = 0
    hand = {}

    for idx in data:

        if idx not in [2302,2303]: continue

        binned = data[idx]['binned'][0]
        bins   = np.unique(data[idx]['bin'])

        if data[idx]['eta'][0] >= 0: color = 'firebrick'
        if data[idx]['eta'][0] <= 0: color = 'darkblue'
        for i in bins:
            i = int(i)
            if i==1: ax = ax11
            if i==2: ax = ax12
            if i==3: ax = ax13
            if i==4: ax = ax21
            if i==5: ax = ax22
            ind = np.where(data[idx]['bin'] == i)
            M   = data[idx]['M'][ind]
            pT  = data[idx]['PhT'][ind]
            value = data[idx]['value'][ind]
            alpha = data[idx]['alpha'][ind]
            thy = data[idx]['thy-%d'%ic][ind]
            std = data[idx]['dthy-%d'%ic][ind]

            if idx==2303: pT += 0.05

            hand[idx]    = ax.errorbar(pT  ,value,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
            ax.text(0.05,0.85,r'\boldmath$\langle M_{h} \rangle = %3.2f~{\rm GeV}$'%M[0] , transform=ax.transAxes, size=25)
            if mode==0:
                hand['thy'] ,= ax.plot(pT,thy,color=color)
            if mode==1:
                thy_plot ,= ax.plot(pT,thy,color=color)
                thy_band  = ax.fill_between(pT,thy-std,thy+std,color=color,alpha=0.4)


    for ax in [ax11,ax12,ax13,ax21,ax22]:

        ax.set_xlim(2,11)
        ax.set_xticks([4,6,8,10])
        minorLocator = MultipleLocator(0.5)
        ax.xaxis.set_minor_locator(minorLocator)

        ax.set_ylim(-0.015,0.08)
        ax.set_yticks([0,0.02,0.04,0.06])
        minorLocator = MultipleLocator(0.005)
        ax.yaxis.set_minor_locator(minorLocator)

        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.tick_params(axis='both',which='minor',size=4)
        ax.tick_params(axis='both',which='major',size=8)
        ax.axhline(0,0,1,color='black',ls='--',alpha=0.5)
 
    for ax in [ax11,ax12]:
        ax.tick_params(labelbottom = False)

    for ax in [ax12,ax13,ax22]:
        ax.tick_params(labelleft = False)

    for ax in [ax13,ax21,ax22]:
        ax.set_xlabel(r'\boldmath$P_{hT}~[{\rm GeV}]$',size=30)

    ax22.text(1.10,0.60,r'\textrm{\textbf{STAR \boldmath$A_{UT}$}}' , transform=ax22.transAxes, size=30)
    ax22.text(1.10,0.50,r'\boldmath$\sqrt{s} = 200~{\rm GeV}$' ,      transform=ax22.transAxes, size=30)

    fs = 30

    handles,labels = [], []
    handles.append(hand[2302])
    handles.append(hand[2303])
    labels.append(r'\boldmath$-1<\eta<0$')
    labels.append(r'\boldmath$0<\eta<1$')
    ax11.legend(handles,labels,loc='upper right',fontsize=fs,frameon=False, handlelength = 1.0, handletextpad = 0.1, ncol = 1, columnspacing = 1.0)


    py.tight_layout()
    py.subplots_adjust(wspace=0.0,hspace=0.02,left=0.03)
    filename='%s/gallery/star-RS200-ang0.3-PhT'%(wdir)
    if mode==1: filename+= '-bands'
    filename+='.png'

    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()

def plot_star_RS200_ang03_eta(wdir,mode):
    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax11=py.subplot(nrows,ncols,1)

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if reaction not in predictions['reactions']: return
    if 2304 not in predictions['reactions'][reaction]: return

    filters = conf['datasets'][reaction]['filters']

    conf['aux']=aux.AUX()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    data = predictions['reactions'][reaction]

    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)

    #--get theory by seperating solutions and taking mean
    for idx in data:
        predictions = copy.copy(data[idx]['prediction-rep'])
        for ic in range(nc):
            predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
            data[idx]['thy-%d'%ic]  = np.mean(predictions_ic,axis=0)
            data[idx]['dthy-%d'%ic] = np.std(predictions_ic,axis=0)

    ic = 0
    hand = {}

    for idx in data:

        if idx not in [2304]: continue
        ax = ax11

        color = 'firebrick'
        M       = data[idx]['M']
        pT      = data[idx]['PhT']
        eta     = data[idx]['eta']
        value   = data[idx]['value']
        alpha = data[idx]['alpha']
        thy = data[idx]['thy-%d'%ic]
        std = data[idx]['dthy-%d'%ic]

        hand[idx]    = ax.errorbar(eta,value,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
        if mode==0:
            hand['thy'] ,= ax.plot(eta,thy,color=color)
        if mode==1:
            thy_plot ,= ax.plot(eta,thy,color=color)
            thy_band  = ax.fill_between(eta,thy-std,thy+std,color=color,alpha=0.4)


    for ax in [ax11]:

        ax.set_xlim(-1,1)
        ax.set_xticks([-0.5,0,0.5])
        minorLocator = MultipleLocator(0.1)
        ax.xaxis.set_minor_locator(minorLocator)

        ax.set_ylim(-0.003,0.035)
        ax.set_yticks([0,0.01,0.02,0.03])
        minorLocator = MultipleLocator(0.002)
        ax.yaxis.set_minor_locator(minorLocator)

        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.tick_params(axis='both',which='minor',size=4)
        ax.tick_params(axis='both',which='major',size=8)
        ax.axhline(0,0,1,color='black',ls='--',alpha=0.5)
        ax.set_xlabel(r'\boldmath$\eta$',size=40)
        ax.xaxis.set_label_coords(0.95,-0.01)

    ax11.text(0.05,0.85,r'\textrm{\textbf{STAR \boldmath$A_{UT}$}}'         ,transform=ax11.transAxes, size=25)
    ax11.text(0.05,0.75,r'\boldmath$\sqrt{s} = 200~{\rm GeV}$'              ,transform=ax11.transAxes, size=20)
    ax11.text(0.05,0.65,r'\boldmath$\langle P_{hT} \rangle = 5.8~{\rm GeV}$'  ,transform=ax11.transAxes, size=20)
    ax11.text(0.05,0.55,r'\boldmath$\langle M_h \rangle = 0.58 ~ {\rm GeV}$'   ,transform=ax11.transAxes, size=20)

    #handles,labels = [], []
    #handles.append(hand[5020])
    #labels.append(r'\textrm{\textbf{STAR}}')
    #ax11.legend(handles,labels,loc='upper right',fontsize=fs,frameon=False, handlelength = 1.0, handletextpad = 0.1, ncol = 1, columnspacing = 1.0)

    py.tight_layout()
    py.subplots_adjust(wspace=0.0,hspace=0.00,left=0.10)
    filename='%s/gallery/star-RS200-ang0.3-eta'%wdir
    if mode==1: filename+= '-bands'
    filename+='.png'

    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()


#--preliminary cross section 
def plot_star_RS200_xsec_M(wdir,mode):
    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*9,nrows*7))
    ax11=py.subplot(nrows,ncols,1)

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if reaction not in predictions['reactions']: return
    if 1000 not in predictions['reactions'][reaction]: return

    filters = conf['datasets'][reaction]['filters']

    conf['aux']=aux.AUX()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    data = predictions['reactions'][reaction]

    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)

    #--get theory by seperating solutions and taking mean
    for idx in data:
        predictions = copy.copy(data[idx]['prediction-rep'])
        for ic in range(nc):
            predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
            data[idx]['thy-%d'%ic]  = np.mean(predictions_ic,axis=0)
            data[idx]['dthy-%d'%ic] = np.std(predictions_ic,axis=0)

    ic = 0
    hand = {}

    for idx in data:

        if idx not in [1000]: continue
        ax = ax11

        color = 'firebrick'
        M       = data[idx]['M']
        value   = data[idx]['value']
        alpha = data[idx]['alpha']
        thy = data[idx]['thy-%d'%ic]
        std = data[idx]['dthy-%d'%ic]

        hand[idx]    = ax.errorbar(M,value,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
        if mode==0:
            hand['thy'] ,= ax.plot(M,thy,color=color)
        if mode==1:
            thy_plot ,= ax.plot(M,thy,color=color)
            thy_band  = ax.fill_between(M,thy-std,thy+std,color=color,alpha=0.4)

        pTmin = data[idx]['PhTmin'][0] 
        pTmax = data[idx]['PhTmax'][0] 
        etamin = data[idx]['etamin'][0]
        etamax = data[idx]['etamax'][0] 
        ax.text(0.05,0.12,r'\boldmath$%3.1f < P_{hT} < %3.1f~{\rm GeV}$'%(pTmin,pTmax)  ,transform=ax.transAxes, size=20)
        ax.text(0.05,0.05,r'\boldmath$%d < \eta < %d$'%(etamin,etamax)  ,transform=ax.transAxes, size=20)


    for ax in [ax11]:

        ax.set_xlim(0.30,1.80)
        ax.set_xticks([0.5,1.0,1.5])
        minorLocator = MultipleLocator(0.1)
        ax.xaxis.set_minor_locator(minorLocator)

        ax.semilogy()
        #ax.set_ylim(4e-5,2)
        #ax.set_yticks([0,0.01,0.02,0.03])
        #minorLocator = MultipleLocator(0.002)
        #ax.yaxis.set_minor_locator(minorLocator)


        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.tick_params(axis='both',which='minor',size=4)
        ax.tick_params(axis='both',which='major',size=8)
        #ax.axhline(0,0,1,color='black',ls='--',alpha=0.5)
        ax.set_xlabel(r'\boldmath$M_h [{\rm GeV}]$',size=30)
        #ax.xaxis.set_label_coords(0.95,-0.04)

    ax11.text(0.05,0.30,r'\textrm{\textbf{\boldmath$\frac{d \sigma_{UU}}{dM_h}$ [pb/GeV$^3$]}}' ,transform=ax11.transAxes, size=25)
    ax11.text(0.05,0.19,r'\boldmath$\sqrt{s} = 200~{\rm GeV},~R<0.7$'         ,transform=ax11.transAxes, size=20)

    handles,labels = [], []
    handles.append(hand[1000])
    handles.append((thy_plot,thy_band))
    labels.append(r'\textrm{\textbf{STAR}}')
    labels.append(r'\textrm{\textbf{JAMDiFF Prediction}}')
    ax11.legend(handles,labels,loc='upper right',fontsize=25,frameon=False, handlelength = 1.0, handletextpad = 0.3)

    py.tight_layout()
    py.subplots_adjust(wspace=0.0,hspace=0.00,left=0.10)
    filename='%s/gallery/star-RS200-xsec-M'%wdir
    if mode==1: filename+= '-bands'
    filename+='.png'

    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()









 
