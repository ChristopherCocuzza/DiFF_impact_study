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
from obslib.dihadron_sidis.reader import READER

import kmeanconf as kc

reaction = 'dihadron_sidis'

def plot_obs(wdir,mode=1):

    plot_sidis_x(wdir,mode)
    plot_sidis_M(wdir,mode)
    plot_sidis_z(wdir,mode)
    plot_mult(wdir,mode)

def plot_sidis_x(wdir,mode):

    print('\ngenerating dihadron SIDIS plot from %s'%(wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if reaction not in predictions['reactions']: return
    flag = True
    if 2000 in predictions['reactions'][reaction]: flag = False 
    if 2100 in predictions['reactions'][reaction]: flag = False 
    if 2110 in predictions['reactions'][reaction]: flag = False 
    if flag: return

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


    nrows,ncols=1,2
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax = {}
    for i in range(2):
        ax[i+1] = py.subplot(nrows,ncols,i+1)

    #######################
    #--plot absolute values
    #######################

    hand = {}
    #--plot data
    for idx in data:
        if   idx==2000: i,color = 1,'darkgreen'
        elif idx==2100: i,color = 2,'firebrick'
        elif idx==2110: i,color = 2,'darkblue'
        else: continue        
    
        x   = data[idx]['x']
        xdo = data[idx]['xdo']
        xup = data[idx]['xup']
        M = data[idx]['M']
        values = data[idx]['value']
        alpha  = data[idx]['alpha']
        thy = data[idx]['thy-%d'%ic]
        std = data[idx]['dthy-%d'%ic]
        xerr = [x-xdo,xup-x]

        hand[idx] = ax[i].errorbar(x,values,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
 
        if mode==0:
            for k in range(len(predictions_ic)):
                thy_plot ,= ax[i].plot(x,predictions_ic[k],color='black',lw=1.0,alpha=0.5)

        if mode==1:
            down = thy - std
            up   = thy + std
            thy_plot ,= ax[i].plot(x,thy,color=color)
            thy_band  = ax[i].fill_between(x,down,up,color=color,alpha=0.4)


    ax[1].set_xlim(0.00,0.19)
    ax[1].set_xticks([0.05,0.10,0.15])
    minorLocator = MultipleLocator(0.01)
    ax[1].xaxis.set_minor_locator(minorLocator)

    ax[2].semilogx()
    ax[2].set_xlim(2e-3,0.40)
    for i in range(2):
        ax[i+1].tick_params(axis='both',which='both',top=True,right=True,labelbottom=False,direction='in',labelsize=30)
        ax[i+1].xaxis.set_tick_params(which='major',length=6)
        ax[i+1].xaxis.set_tick_params(which='minor',length=3)
        ax[i+1].yaxis.set_tick_params(which='major',length=6)
        ax[i+1].yaxis.set_tick_params(which='minor',length=3)

    for i in range(2):
        ax[i+1].tick_params(labelbottom=True)
        ax[i+1].set_xlabel(r'\boldmath$x$',size=30)
        ax[i+1].xaxis.set_label_coords(0.95,-0.02)
        ax[i+1].tick_params(axis='x',which='major',pad=8)

    ax[1].set_ylim(-0.02,0.07)
    minorLocator = MultipleLocator(0.01)
    majorLocator = MultipleLocator(0.02)
    ax[1].yaxis.set_minor_locator(minorLocator)
    ax[1].yaxis.set_major_locator(majorLocator)


    ax[2].set_ylim(-0.19,0.11)
    ax[2].set_yticks([-0.15,-0.10,-0.05,0.00,0.05,0.10])
    minorLocator = MultipleLocator(0.01)
    ax[2].yaxis.set_minor_locator(minorLocator)


    ax[1] .text(0.05, 0.80, r'\boldmath$A_{U \perp}^{\sin(\phi_{R \perp} + \phi_S) \sin(\theta)}$',transform=ax[1].transAxes,size=35)

    ax[1].axhline(0,0,1,color='black',ls='--',alpha=0.5)
    ax[2].axhline(0,0,1,color='black',ls='--',alpha=0.5)

    handles,labels = [], []
    if 2000 in hand: handles.append(hand[2000])
    handles.append((thy_band,thy_plot))
    if 2000 in hand: labels.append(r'\textbf{\textrm{HERMES}}') 
    labels.append(r'\textbf{\textrm{JAM}}') 
    ax[1].legend(handles,labels,frameon=False,fontsize=22,loc='upper right',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)

    handles,labels = [], []
    if 2100 in hand: handles.append(hand[2100])
    if 2110 in hand: handles.append(hand[2110])
    if 2100 in hand: labels.append(r'\textbf{\textrm{COMPASS \boldmath$p$}}') 
    if 2110 in hand: labels.append(r'\textbf{\textrm{COMPASS \boldmath$D$}}') 
    ax[2].legend(handles,labels,frameon=False,fontsize=22,loc='lower left',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)

    py.tight_layout()
    py.subplots_adjust(hspace=0,wspace=0.20)


    checkdir('%s/gallery'%wdir)
    filename='%s/gallery/sidis-x'%wdir
    if mode==1: filename+='-bands'
    filename+='.png'

    py.savefig(filename)
    print('Saving SIDIS plot to %s'%filename)

def plot_sidis_M(wdir,mode):

    print('\ngenerating dihadron SIDIS plot from %s'%(wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if reaction not in predictions['reactions']: return
    flag = True
    if 2001 in predictions['reactions'][reaction]: flag = False 
    if 2101 in predictions['reactions'][reaction]: flag = False 
    if 2111 in predictions['reactions'][reaction]: flag = False 
    if flag: return

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


    nrows,ncols=1,2
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax = {}
    for i in range(2):
        ax[i+1] = py.subplot(nrows,ncols,i+1)

    #######################
    #--plot absolute values
    #######################

    hand = {}
    #--plot data
    for idx in data:
        if   idx==2001: i,color = 1,'darkgreen'
        elif idx==2101: i,color = 2,'firebrick'
        elif idx==2111: i,color = 2,'darkblue'
        else: continue        
    
        x   = data[idx]['x']
        xdo = data[idx]['xdo']
        xup = data[idx]['xup']
        M = data[idx]['M']
        values = data[idx]['value']
        alpha  = data[idx]['alpha']
        thy = data[idx]['thy-%d'%ic]
        std = data[idx]['dthy-%d'%ic]

        if idx==3012: M += 0.02

        hand[idx] = ax[i].errorbar(M,values,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
 
        if mode==0:
            for k in range(len(predictions_ic)):
                thy_plot ,= ax[i].plot(M,predictions_ic[k],color='black',lw=1.0,alpha=0.5)

        if mode==1:
            down = thy - std
            up   = thy + std
            thy_plot ,= ax[i].plot(M,thy,color=color)
            thy_band  = ax[i].fill_between(M,down,up,color=color,alpha=0.4)


    ax[1].set_xlim(0.00,2.00)
    #ax[1].set_xticks([0.05,0.10,0.15])
    #minorLocator = MultipleLocator(0.01)
    #ax[1].xaxis.set_minor_locator(minorLocator)

    ax[2].set_xlim(0.00,2.00)
    for i in range(2):
        ax[i+1].tick_params(axis='both',which='both',top=True,right=True,labelbottom=False,direction='in',labelsize=30)
        ax[i+1].xaxis.set_tick_params(which='major',length=6)
        ax[i+1].xaxis.set_tick_params(which='minor',length=3)
        ax[i+1].yaxis.set_tick_params(which='major',length=6)
        ax[i+1].yaxis.set_tick_params(which='minor',length=3)

    for i in range(2):
        ax[i+1].tick_params(labelbottom=True)
        ax[i+1].set_xlabel(r'\boldmath$M_h$',size=30)
        ax[i+1].xaxis.set_label_coords(0.95,-0.02)
        ax[i+1].tick_params(axis='x',which='major',pad=8)

    ax[1].set_ylim(-0.02,0.07)
    minorLocator = MultipleLocator(0.01)
    majorLocator = MultipleLocator(0.02)
    ax[1].yaxis.set_minor_locator(minorLocator)
    ax[1].yaxis.set_major_locator(majorLocator)


    ax[2].set_ylim(-0.19,0.11)
    ax[2].set_yticks([-0.15,-0.10,-0.05,0.00,0.05,0.10])
    minorLocator = MultipleLocator(0.01)
    ax[2].yaxis.set_minor_locator(minorLocator)


    ax[1] .text(0.05, 0.80, r'\boldmath$A_{U \perp}^{\sin(\phi_{R \perp} + \phi_S) \sin(\theta)}$',transform=ax[1].transAxes,size=35)

    ax[1].axhline(0,0,1,color='black',ls='--',alpha=0.5)
    ax[2].axhline(0,0,1,color='black',ls='--',alpha=0.5)

    handles,labels = [], []
    if 2001 in hand: handles.append(hand[2001])
    handles.append((thy_band,thy_plot))
    if 2001 in hand: labels.append(r'\textbf{\textrm{HERMES}}') 
    labels.append(r'\textbf{\textrm{JAM}}') 
    ax[1].legend(handles,labels,frameon=False,fontsize=22,loc='upper right',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)

    handles,labels = [], []
    if 2101 in hand: handles.append(hand[2101])
    if 2111 in hand: handles.append(hand[2111])
    if 2101 in hand: labels.append(r'\textbf{\textrm{COMPASS \boldmath$p$}}') 
    if 2111 in hand: labels.append(r'\textbf{\textrm{COMPASS \boldmath$D$}}') 
    ax[2].legend(handles,labels,frameon=False,fontsize=22,loc='lower left',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)

    py.tight_layout()
    py.subplots_adjust(hspace=0,wspace=0.20)


    checkdir('%s/gallery'%wdir)
    filename='%s/gallery/sidis-M'%wdir
    if mode==1: filename+='-bands'
    filename+='.png'

    py.savefig(filename)
    print('Saving SIDIS plot to %s'%filename)

def plot_sidis_z(wdir,mode):

    print('\ngenerating dihadron SIDIS plot from %s'%(wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if reaction not in predictions['reactions']: return
    flag = True
    if 2002 in predictions['reactions'][reaction]: flag = False 
    if 2102 in predictions['reactions'][reaction]: flag = False 
    if 2112 in predictions['reactions'][reaction]: flag = False 
    if flag: return

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


    nrows,ncols=1,2
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax = {}
    for i in range(2):
        ax[i+1] = py.subplot(nrows,ncols,i+1)

    #######################
    #--plot absolute values
    #######################

    hand = {}
    #--plot data
    for idx in data:
        if   idx==2002: i,color = 1,'darkgreen'
        elif idx==2102: i,color = 2,'firebrick'
        elif idx==2112: i,color = 2,'darkblue'
        else: continue        
    
        x   = data[idx]['x']
        z   = data[idx]['z']
        xdo = data[idx]['xdo']
        xup = data[idx]['xup']
        M = data[idx]['M']
        values = data[idx]['value']
        alpha  = data[idx]['alpha']
        thy = data[idx]['thy-%d'%ic]
        std = data[idx]['dthy-%d'%ic]

        hand[idx] = ax[i].errorbar(z,values,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
 
        if mode==0:
            for k in range(len(predictions_ic)):
                thy_plot ,= ax[i].plot(z,predictions_ic[k],color='black',lw=1.0,alpha=0.5)

        if mode==1:
            down = thy - std
            up   = thy + std
            thy_plot ,= ax[i].plot(z,thy,color=color)
            thy_band  = ax[i].fill_between(z,down,up,color=color,alpha=0.4)


    ax[1].set_xlim(0.00,1.00)
    #ax[1].set_xticks([0.05,0.10,0.15])
    #minorLocator = MultipleLocator(0.01)
    #ax[1].xaxis.set_minor_locator(minorLocator)

    ax[2].set_xlim(0.00,1.00)
    for i in range(2):
        ax[i+1].tick_params(axis='both',which='both',top=True,right=True,labelbottom=False,direction='in',labelsize=30)
        ax[i+1].xaxis.set_tick_params(which='major',length=6)
        ax[i+1].xaxis.set_tick_params(which='minor',length=3)
        ax[i+1].yaxis.set_tick_params(which='major',length=6)
        ax[i+1].yaxis.set_tick_params(which='minor',length=3)

    for i in range(2):
        ax[i+1].tick_params(labelbottom=True)
        ax[i+1].set_xlabel(r'\boldmath$z$',size=30)
        ax[i+1].xaxis.set_label_coords(0.95,-0.02)
        ax[i+1].tick_params(axis='x',which='major',pad=8)

    ax[1].set_ylim(-0.02,0.07)
    minorLocator = MultipleLocator(0.01)
    majorLocator = MultipleLocator(0.02)
    ax[1].yaxis.set_minor_locator(minorLocator)
    ax[1].yaxis.set_major_locator(majorLocator)


    ax[2].set_ylim(-0.19,0.11)
    ax[2].set_yticks([-0.15,-0.10,-0.05,0.00,0.05,0.10])
    minorLocator = MultipleLocator(0.01)
    ax[2].yaxis.set_minor_locator(minorLocator)


    ax[1] .text(0.05, 0.80, r'\boldmath$A_{U \perp}^{\sin(\phi_{R \perp} + \phi_S) \sin(\theta)}$',transform=ax[1].transAxes,size=35)

    ax[1].axhline(0,0,1,color='black',ls='--',alpha=0.5)
    ax[2].axhline(0,0,1,color='black',ls='--',alpha=0.5)

    handles,labels = [], []
    if 2002 in hand: handles.append(hand[2002])
    handles.append((thy_band,thy_plot))
    if 2002 in hand: labels.append(r'\textbf{\textrm{HERMES}}') 
    labels.append(r'\textbf{\textrm{JAM}}') 
    ax[1].legend(handles,labels,frameon=False,fontsize=22,loc='upper right',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)

    handles,labels = [], []
    if 2102 in hand: handles.append(hand[2102])
    if 2112 in hand: handles.append(hand[2112])
    if 2102 in hand: labels.append(r'\textbf{\textrm{COMPASS \boldmath$p$}}') 
    if 2112 in hand: labels.append(r'\textbf{\textrm{COMPASS \boldmath$D$}}') 
    ax[2].legend(handles,labels,frameon=False,fontsize=22,loc='lower left',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)

    py.tight_layout()
    py.subplots_adjust(hspace=0,wspace=0.20)


    checkdir('%s/gallery'%wdir)
    filename='%s/gallery/sidis-z'%wdir
    if mode==1: filename+='-bands'
    filename+='.png'

    py.savefig(filename)
    print('Saving SIDIS plot to %s'%filename)


#--for predictions
def plot_mult(wdir,mode):

    print('\ngenerating dihadron SIDIS plot from %s'%(wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if reaction not in predictions['reactions']: return
    flag = True
    if 7000 in predictions['reactions'][reaction]: flag = False 
    if 7001 in predictions['reactions'][reaction]: flag = False 
    if flag: return

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


    nrows,ncols=2,3
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax = {}
    for i in range(6):
        ax[i+1] = py.subplot(nrows,ncols,i+1)

    #######################
    #--plot absolute values
    #######################

    hand = {}
    #--plot data
    for idx in data:

        BIN = data[idx]['binning']
        bins = np.unique(BIN)
 
        Q2 = data[idx]['Q2']
        z  = data[idx]['z']
        x  = data[idx]['x']
        M  = data[idx]['M']
        values = data[idx]['value']
        alpha  = data[idx]['alpha']
        thy = data[idx]['thy-%d'%ic]
        std = data[idx]['dthy-%d'%ic]

        #hand[idx] = ax[i].errorbar(x,values,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
 
        #if mode==0:
        #    for k in range(len(predictions_ic)):
        #        thy_plot ,= ax[i].plot(z,predictions_ic[k],color='black',lw=1.0,alpha=0.5)

        if mode==1:
            down = thy - std
            up   = thy + std
            for j in bins:
                if idx==7001 and j == 1: i,color = 1, 'black'
                if idx==7001 and j == 2: i,color = 1, 'red'
                if idx==7001 and j == 3: i,color = 1, 'green'
                if idx==7001 and j == 4: i,color = 2, 'black'
                if idx==7001 and j == 5: i,color = 2, 'red'
                if idx==7001 and j == 6: i,color = 2, 'green'
                if idx==7001 and j == 7: i,color = 3, 'black'
                if idx==7001 and j == 8: i,color = 3, 'red'
                if idx==7001 and j == 9: i,color = 3, 'green'
                if idx==7000 and j == 1: i,color = 4, 'black'
                if idx==7000 and j == 2: i,color = 4, 'red'
                if idx==7000 and j == 3: i,color = 4, 'green'
                if idx==7000 and j == 4: i,color = 5, 'black'
                if idx==7000 and j == 5: i,color = 5, 'red'
                if idx==7000 and j == 6: i,color = 5, 'green'
                if idx==7000 and j == 7: i,color = 6, 'black'
                if idx==7000 and j == 8: i,color = 6, 'red'
                if idx==7000 and j == 9: i,color = 6, 'green'
                thy_plot ,= ax[i].plot(z[BIN==j],thy[BIN==j],color=color)
                thy_band  = ax[i].fill_between(z[BIN==j],down[BIN==j],up[BIN==j],color=color,alpha=0.4)


    for i in range(6):
        ax[i+1].tick_params(axis='both',which='both',top=True,right=True,labelbottom=False,direction='in',labelsize=30)
        ax[i+1].xaxis.set_tick_params(which='major',length=6)
        ax[i+1].xaxis.set_tick_params(which='minor',length=3)
        ax[i+1].yaxis.set_tick_params(which='major',length=6)
        ax[i+1].yaxis.set_tick_params(which='minor',length=3)

    for i in range(6):
        ax[i+1].tick_params(labelbottom=True)
        ax[i+1].set_xlabel(r'\boldmath$z$',size=30)
        ax[i+1].xaxis.set_label_coords(0.95,-0.02)
        ax[i+1].tick_params(axis='x',which='major',pad=8)
        ax[i+1].set_xlim(0.20,1.00)
        ax[i+1].set_xticks([0.4,0.6,0.8])
        minorLocator = MultipleLocator(0.1)
        ax[i+1].xaxis.set_minor_locator(minorLocator)
        ax[i+1].semilogy()
        ax[i+1].set_ylim(2e-3,5)

    for i in [2,3,5,6]:
        ax[i].tick_params(labelleft=False)

    for i in [1,2,3]:
        ax[i].tick_params(labelbottom=False)

    ax[1].set_ylabel(r'\boldmath$M_D^{\pi^+ \pi^-} ~ [1/{\rm GeV}]$',size=30)
    ax[4].set_ylabel(r'\boldmath$M_p^{\pi^+ \pi^-} ~ [1/{\rm GeV}]$',size=30)
    

    #minorLocator = MultipleLocator(0.01)
    #majorLocator = MultipleLocator(0.02)
    #ax[1].yaxis.set_minor_locator(minorLocator)
    #ax[1].yaxis.set_major_locator(majorLocator)

    #ax[1] .text(0.05, 0.80, r'\boldmath$A_{U \perp}^{\sin(\phi_{R \perp} + \phi_S) \sin(\theta)}$',transform=ax[1].transAxes,size=35)


    #handles,labels = [], []
    #if 3000 in hand: handles.append(hand[3000])
    #handles.append((thy_band,thy_plot))
    #if 3000 in hand: labels.append(r'\textbf{\textrm{HERMES}}') 
    #labels.append(r'\textbf{\textrm{JAM}}') 
    #ax[1].legend(handles,labels,frameon=False,fontsize=22,loc='upper right',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)

    #handles,labels = [], []
    #if 3001 in hand: handles.append(hand[3001])
    #if 3002 in hand: handles.append(hand[3002])
    #if 3001 in hand: labels.append(r'\textbf{\textrm{COMPASS \boldmath$p$}}') 
    #if 3002 in hand: labels.append(r'\textbf{\textrm{COMPASS \boldmath$D$}}') 
    #ax[2].legend(handles,labels,frameon=False,fontsize=22,loc='lower left',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)

    py.tight_layout()
    py.subplots_adjust(hspace=0.01,wspace=0.01)


    checkdir('%s/gallery'%wdir)
    filename='%s/gallery/sidis_mult'%wdir
    if mode==1: filename+='-bands'
    filename+='.png'

    py.savefig(filename)
    print('Saving SIDIS multiplicity plot to %s'%filename)




