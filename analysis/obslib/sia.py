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
from obslib.dihadron_sia.reader import READER

import kmeanconf as kc

reaction = 'dihadron_sia'

def plot_obs(wdir,mode=1):

    #-cross-section
    plot_unpolarized(wdir,mode)
    plot_unpolarized_log(wdir,mode)
    #if mode==1: 
    #    plot_unpolarized_ratio(wdir)
    #    plot_unpolarized_difference(wdir)

    #--asymmmetry
    plot_a12R_z1_M1(wdir,mode)
    plot_a12R_M1_M2(wdir,mode)
    plot_a12R_z1_z2(wdir,mode)

def plot_unpolarized(wdir,mode):

    print('\ngenerating unpolarized dihadron SIA plot from %s'%(wdir))
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


    nrows,ncols=4,4
    fig = py.figure(figsize=(ncols*5,nrows*3))
    ax = {}
    for i in range(16):
        ax[i+1] = py.subplot(nrows,ncols,i+1)

    #######################
    #--plot absolute values
    #######################

    hand = {}
    #--plot data
    for idx in data:
        if idx==1000: color = 'firebrick'
        else: continue        
   
        Z = data[idx]['z']
        nbins = len(np.unique(Z))+1
        Zmin = data[idx]['zdo']
        Zmax = data[idx]['zup']
        M = data[idx]['M']
        values = data[idx]['value']
        alpha = data[idx]['alpha']
        BIN = data[idx]['zbin']
        _thy = data[idx]['thy-%d'%ic]
        _std = data[idx]['dthy-%d'%ic]
        for i in range(nbins):
            lprint('generating bin: [%s/%s]'%(i+1,nbins))
            if (i+1) not in BIN: continue
            if mode==0:
                for k in range(len(predictions_ic)):
                    z,m,val,alp,thy,std = [],[],[],[],[],[]
                    for j in range(len(values)):
                        if BIN[j] != (i+1): continue
                        val.append(values[j])
                        alp.append(alpha[j])
                        z.append(Z[j])
                        m.append(M[j])
                        thy.append(predictions_ic[k][j])
                        std.append(_std[j])
                        zmin,zmax = Zmin[j],Zmax[j]
                        hand[idx] = ax[i+1].errorbar(m,val,yerr=alp,color=color,fmt='o',ms=2.0,capsize=3.0)
                    if k==0: ax[i+1].text(0.05,0.85,r'\boldmath$%s < z < %s$'%(zmin,zmax),transform=ax[i+1].transAxes,size=22)
                    #--plot mean and std of all replicas
                    thy = np.array(thy)
                    thy_plot ,= ax[i+1].plot(m,thy,color='black',lw=1.0,alpha=0.5)
            if mode==1:
                z,m,val,alp,thy,std = [],[],[],[],[],[]
                for j in range(len(values)):
                    if BIN[j] != (i+1): continue
                    val.append(values[j])
                    alp.append(alpha[j])
                    z.append(Z[j])
                    m.append(M[j])
                    thy.append(_thy[j])
                    std.append(_std[j])
                    zmin,zmax = Zmin[j],Zmax[j]
                    hand[idx] = ax[i+1].errorbar(m,val,yerr=alp,color=color,fmt='o',ms=2.0,capsize=3.0)
                ax[i+1].text(0.05,0.85,r'\boldmath$%s < z < %s$'%(zmin,zmax),transform=ax[i+1].transAxes,size=22)
                #--plot mean and std of all replicas
                thy = np.array(thy)
                std = np.array(std)
                down = thy - std
                up   = thy + std
                thy_plot ,= ax[i+1].plot(m,thy,color='black')
                thy_band  = ax[i+1].fill_between(m,down,up,color='gold',alpha=0.4)




    for i in range(16):
        ax[i+1].tick_params(axis='both',which='both',top=True,right=True,labelbottom=False,direction='in',labelsize=30)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(0.5)
        ax[i+1].xaxis.set_minor_locator(minorLocator)
        ax[i+1].xaxis.set_major_locator(majorLocator)
        ax[i+1].xaxis.set_tick_params(which='major',length=6)
        ax[i+1].xaxis.set_tick_params(which='minor',length=3)
        ax[i+1].set_xlim(0.30,2.30)
        ax[i+1].set_xticks([0.5,1.0,1.5,2.0])

    for i in [13,14,15,16]:
        ax[i].tick_params(labelbottom=True)
        ax[i].set_xlabel(r'\boldmath$M_h ~ [{\rm GeV}]$',size=30)
        #ax[i].xaxis.set_label_coords(0.95,-0.02)

    for i in [3,4,6,7,8,10,11,12,14,15,16]:
        ax[i].tick_params(labelleft=False)

    for i in [2,3,4]:
        ax[i].set_ylim(0.0,23)
        minorLocator = MultipleLocator(2)
        majorLocator = MultipleLocator(10)
        ax[i].yaxis.set_minor_locator(minorLocator)
        ax[i].yaxis.set_major_locator(majorLocator)
        ax[i].yaxis.set_tick_params(which='major',length=6)
        ax[i].yaxis.set_tick_params(which='minor',length=3)
        ax[i].set_yticks([10,20])

    for i in [5,6,7,8]:
        ax[i].set_ylim(0.0,13)
        minorLocator = MultipleLocator(1)
        majorLocator = MultipleLocator(5)
        ax[i].yaxis.set_minor_locator(minorLocator)
        ax[i].yaxis.set_major_locator(majorLocator)
        ax[i].yaxis.set_tick_params(which='major',length=6)
        ax[i].yaxis.set_tick_params(which='minor',length=3)
        ax[i].set_yticks([5,10])

    for i in [9,10,11,12]:
        ax[i].set_ylim(0.0,3.5)
        minorLocator = MultipleLocator(0.5)
        majorLocator = MultipleLocator(1)
        ax[i].yaxis.set_minor_locator(minorLocator)
        ax[i].yaxis.set_major_locator(majorLocator)
        ax[i].yaxis.set_tick_params(which='major',length=6)
        ax[i].yaxis.set_tick_params(which='minor',length=3)
        ax[i].set_yticks([1,2,3])

    for i in [13,14,15,16]:
        ax[i].set_ylim(0.0,0.7)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(0.2)
        ax[i].yaxis.set_minor_locator(minorLocator)
        ax[i].yaxis.set_major_locator(majorLocator)
        ax[i].yaxis.set_tick_params(which='major',length=6)
        ax[i].yaxis.set_tick_params(which='minor',length=3)
        ax[i].set_yticks([0.0,0.2,0.4,0.6])


    #ax[1] .text(0.05, 0.65, r'\boldmath${\rm d}^2 \sigma/{\rm d}z {\rm d}M_{h} ~ [{\rm nb}/{\rm GeV}]$',transform=ax[1].transAxes,size=25)
    ax[1] .text(0.05, 0.75, r'\boldmath$\frac{{\rm d}^2 \sigma}{{\rm d}z {\rm d}M_{h}} ~ [{\rm nb}/{\rm GeV}]$',transform=ax[1].transAxes,size=25)
    ax[1].text(0.05, 0.55, r'$\sqrt{s} = 10.58$'+' '+r'\textrm{GeV}', transform=ax[1].transAxes,size=20)

    thy ,= ax[2].plot([],[],color='black')
    thy_band = ax[2].fill_between([],[],[],color='gold',alpha=1.0)


    ax[1].axis('off')

    handles,labels = [], []
    handles.append(hand[1000])
    handles.append((thy_band,thy))
    labels.append(r'\textbf{\textrm{BELLE}}') 
    labels.append(r'\textbf{\textrm{JAM}}') 
    ax[1].legend(handles,labels,frameon=False,fontsize=22,loc='lower left',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)

    py.tight_layout()
    py.subplots_adjust(hspace=0,wspace=0)


    checkdir('%s/gallery'%wdir)
    filename='%s/gallery/belle-unpolarized'%wdir
    if mode==1: filename+='-bands'
    filename+='.png'

    py.savefig(filename)
    print('Saving Belle plot to %s'%filename)

def plot_unpolarized_log(wdir,mode):

    print('\ngenerating unpolarized dihadron SIA plot from %s'%(wdir))
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


    nrows,ncols=4,4
    fig = py.figure(figsize=(ncols*5,nrows*3))
    ax = {}
    for i in range(16):
        ax[i+1] = py.subplot(nrows,ncols,i+1)

    #######################
    #--plot absolute values
    #######################

    hand = {}
    #--plot data
    for idx in data:
        if idx==1000: color = 'firebrick'
        else: continue        
    
        Z = data[idx]['z']
        nbins = len(np.unique(Z))+1
        Zmin = data[idx]['zdo']
        Zmax = data[idx]['zup']
        M = data[idx]['M']
        values = data[idx]['value']
        alpha = data[idx]['alpha']
        BIN = data[idx]['zbin']
        _thy = data[idx]['thy-%d'%ic]
        _std = data[idx]['dthy-%d'%ic]
        for i in range(nbins):
            lprint('generating bin: [%s/%s]'%(i+1,nbins))
            if (i+1) not in BIN: continue
            if mode==0:
                for k in range(len(predictions_ic)):
                    z,m,val,alp,thy,std = [],[],[],[],[],[]
                    for j in range(len(values)):
                        if BIN[j] != (i+1): continue
                        val.append(values[j])
                        alp.append(alpha[j])
                        z.append(Z[j])
                        m.append(M[j])
                        thy.append(predictions_ic[k][j])
                        std.append(_std[j])
                        zmin,zmax = Zmin[j],Zmax[j]
                        hand[idx] = ax[i+1].errorbar(m,val,yerr=alp,color=color,fmt='o',ms=2.0,capsize=3.0)
                    if k==0: ax[i+1].text(0.05,0.85,r'\boldmath$%s < z < %s$'%(zmin,zmax),transform=ax[i+1].transAxes,size=22)
                    #--plot mean and std of all replicas
                    thy = np.array(thy)
                    thy_plot ,= ax[i+1].plot(m,thy,color='black',lw=1.0,alpha=0.5)
            if mode==1:
                z,m,val,alp,thy,std = [],[],[],[],[],[]
                for j in range(len(values)):
                    if BIN[j] != (i+1): continue
                    val.append(values[j])
                    alp.append(alpha[j])
                    z.append(Z[j])
                    m.append(M[j])
                    thy.append(_thy[j])
                    std.append(_std[j])
                    zmin,zmax = Zmin[j],Zmax[j]
                    hand[idx] = ax[i+1].errorbar(m,val,yerr=alp,color=color,fmt='o',ms=2.0,capsize=3.0)
                ax[i+1].text(0.05,0.85,r'\boldmath$%s < z < %s$'%(zmin,zmax),transform=ax[i+1].transAxes,size=22)
                #--plot mean and std of all replicas
                thy = np.array(thy)
                std = np.array(std)
                down = thy - std
                up   = thy + std
                thy_plot ,= ax[i+1].plot(m,thy,color='black')
                thy_band  = ax[i+1].fill_between(m,down,up,color='gold',alpha=0.4)




    for i in range(16):
        ax[i+1].tick_params(axis='both',which='both',top=True,right=True,labelbottom=False,direction='in',labelsize=30)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(0.5)
        ax[i+1].xaxis.set_minor_locator(minorLocator)
        ax[i+1].xaxis.set_major_locator(majorLocator)
        ax[i+1].xaxis.set_tick_params(which='major',length=6)
        ax[i+1].xaxis.set_tick_params(which='minor',length=3)
        ax[i+1].set_xlim(0.30,2.30)
        ax[i+1].set_xticks([0.5,1.0,1.5,2.0])
        ax[i+1].semilogy()

    for i in [13,14,15,16]:
        ax[i].tick_params(labelbottom=True)
        ax[i].set_xlabel(r'\boldmath$M_h ~ [{\rm GeV}]$',size=30)
        #ax[i].xaxis.set_label_coords(0.95,-0.02)

    for i in [3,4,6,7,8,10,11,12,14,15,16]:
        ax[i].tick_params(labelleft=False)

    for i in [2,3,4]:
        ax[i].set_ylim(8e-2,80)

    for i in [5,6,7,8]:
        ax[i].set_ylim(2e-2,50)

    for i in [9,10,11,12]:
        ax[i].set_ylim(8e-2,20)

    for i in [13,14,15,16]:
        ax[i].set_ylim(2e-3,2)

    for i in range(16):
        locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12)
        ax[i+1].yaxis.set_major_locator(locmaj)
        locmin = matplotlib.ticker.LogLocator(base=10,subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
        ax[i+1].yaxis.set_minor_locator(locmin)
        ax[i+1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


    #ax[1] .text(0.05, 0.65, r'\boldmath${\rm d}^2 \sigma/{\rm d}z {\rm d}M_{h} ~ [{\rm nb}/{\rm GeV}]$',transform=ax[1].transAxes,size=25)
    ax[1] .text(0.05, 0.75, r'\boldmath$\frac{{\rm d}^2 \sigma}{{\rm d}z {\rm d}M_{h}} ~ [{\rm nb}/{\rm GeV}]$',transform=ax[1].transAxes,size=25)
    ax[1].text(0.05, 0.55, r'$\sqrt{s} = 10.58$'+' '+r'\textrm{GeV}', transform=ax[1].transAxes,size=20)

    thy ,= ax[2].plot([],[],color='black')
    thy_band = ax[2].fill_between([],[],[],color='gold',alpha=1.0)

    ax[1].axis('off')

    handles,labels = [], []
    handles.append(hand[1000])
    handles.append((thy_band,thy))
    labels.append(r'\textbf{\textrm{BELLE}}') 
    labels.append(r'\textbf{\textrm{JAM}}') 
    ax[1].legend(handles,labels,frameon=False,fontsize=22,loc='lower left',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)

    py.tight_layout()
    py.subplots_adjust(hspace=0,wspace=0)


    checkdir('%s/gallery'%wdir)
    filename='%s/gallery/belle-unpolarized-log'%wdir
    if mode==1: filename+='-bands'
    filename+='.png'

    py.savefig(filename)
    print('Saving Belle plot to %s'%filename)


def plot_unpolarized_ratio(wdir):

    print('\ngenerating unpolarized dihadron SIA plot from %s'%(wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if reaction not in predictions['reactions']: return
    if 1000 not in predictions['reactions'][reaction]: return

    filters = conf['datasets']['dihadron']['filters']

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


    nrows,ncols=4,4
    fig = py.figure(figsize=(ncols*5,nrows*3))
    ax = {}
    for i in range(16):
        ax[i+1] = py.subplot(nrows,ncols,i+1)

    #######################
    #--plot absolute values
    #######################

    hand = {}
    #--plot data
    for idx in data:
        if idx==1000: color = 'firebrick'
        else: continue        
    
        Z = data[idx]['z']
        nbins = len(np.unique(Z))+1
        Zmin = data[idx]['zdo']
        Zmax = data[idx]['zup']
        M = data[idx]['M']
        values = data[idx]['value']
        alpha  = data[idx]['alpha']
        BIN = data[idx]['zbin']
        _thy = data[idx]['thy-%d'%ic]
        _std = data[idx]['dthy-%d'%ic]
        for i in range(nbins):
            lprint('generating bin: [%s/%s]'%(i+1,nbins))
            if (i+1) not in BIN: continue
            z,m,val,alp,thy,std = [],[],[],[],[],[]
            for j in range(len(values)):
                if BIN[j] != (i+1): continue
                val.append(values[j])
                alp.append(alpha[j])
                z.append(Z[j])
                m.append(M[j])
                thy.append(_thy[j])
                std.append(_std[j])
                zmin,zmax = Zmin[j],Zmax[j]
            ax[i+1].text(0.05,0.85,r'\boldmath$%s < z < %s$'%(zmin,zmax),transform=ax[i+1].transAxes,size=22)
            thy = np.array(thy)
            std = np.array(std)
            down = thy - std
            up   = thy + std
            thy = ax[i+1].errorbar(m,val/thy,yerr=alp/thy,color=color,fmt='.',ms=10,capsize=3.0)
            ax[i+1].axhline(1,0,1,color='black',alpha=0.5,ls='--')




    for i in range(16):
        ax[i+1].tick_params(axis='both',which='both',top=True,right=True,labelbottom=False,direction='in',labelsize=30)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(0.5)
        ax[i+1].xaxis.set_minor_locator(minorLocator)
        ax[i+1].xaxis.set_major_locator(majorLocator)
        ax[i+1].xaxis.set_tick_params(which='major',length=6)
        ax[i+1].xaxis.set_tick_params(which='minor',length=3)
        ax[i+1].set_xlim(0.30,2.30)
        ax[i+1].set_xticks([0.5,1.0,1.5,2.0])
        ax[i+1].set_ylim(0.50,1.50)
        minorLocator = MultipleLocator(0.05)
        majorLocator = MultipleLocator(0.20)
        ax[i+1].yaxis.set_minor_locator(minorLocator)
        ax[i+1].yaxis.set_major_locator(majorLocator)
        ax[i+1].yaxis.set_tick_params(which='major',length=6)
        ax[i+1].yaxis.set_tick_params(which='minor',length=3)
        #ax[i+1].set_yticks([0.6,0.8,1.0,1.2,1.4])

    for i in [13,14,15,16]:
        ax[i].tick_params(labelbottom=True)
        ax[i].set_xlabel(r'\boldmath$M_{h} ~ [{\rm GeV}]$',size=30)

    for i in [2,3,4,6,7,8,10,11,12,14,15,16]:
        ax[i].tick_params(labelleft=False)


    #ax[4] .text(0.05, 0.65, r'\boldmath${\rm d}^2 \sigma/{\rm d}z {\rm d}m_{\pi\pi} ~ [\mu {\rm b}/{\rm GeV}]$',transform=ax[4].transAxes,size=25)
    ax[16].text(0.05, 0.65, r'$\sqrt{s} = 10.58$'+' '+r'\textrm{GeV}', transform=ax[16].transAxes,size=20)

    thy ,= ax[1].plot([],[],color='black')
    thy_band = ax[1].fill_between([],[],[],color='gold',alpha=1.0)

    #handles,labels = [], []
    #handles.append(hand[1000])
    #handles.append((thy_band,thy))
    #labels.append(r'\textbf{\textrm{BELLE}}') 
    #labels.append(r'\textbf{\textrm{JAM}}') 
    #ax[4].legend(handles,labels,frameon=False,fontsize=22,loc='upper right',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)

    py.tight_layout()
    py.subplots_adjust(hspace=0,wspace=0)


    checkdir('%s/gallery'%wdir)
    filename='%s/gallery/belle-unpolarized-ratio'%wdir
    filename+='.png'

    py.savefig(filename)
    print('Saving Belle plot to %s'%filename)

def plot_unpolarized_difference(wdir):

    print('\ngenerating unpolarized dihadron SIA plot from %s'%(wdir))
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


    nrows,ncols=4,4
    fig = py.figure(figsize=(ncols*5,nrows*3))
    ax = {}
    for i in range(16):
        ax[i+1] = py.subplot(nrows,ncols,i+1)

    #######################
    #--plot absolute values
    #######################

    hand = {}
    #--plot data
    for idx in data:
        if idx==1000: color = 'firebrick'
        else: continue        
    
        Z = data[idx]['z']
        nbins = len(np.unique(Z))+1
        Zmin = data[idx]['zdo']
        Zmax = data[idx]['zup']
        M = data[idx]['M']
        values = data[idx]['value']
        alpha  = data[idx]['alpha']
        BIN = data[idx]['zbin']
        _thy = data[idx]['thy-%d'%ic]
        _std = data[idx]['dthy-%d'%ic]
        for i in range(nbins):
            lprint('generating bin: [%s/%s]'%(i+1,nbins))
            if (i+1) not in BIN: continue
            z,m,val,alp,thy,std = [],[],[],[],[],[]
            for j in range(len(values)):
                if BIN[j] != (i+1): continue
                val.append(values[j])
                alp.append(alpha[j])
                z.append(Z[j])
                m.append(M[j])
                thy.append(_thy[j])
                std.append(_std[j])
                zmin,zmax = Zmin[j],Zmax[j]
            ax[i+1].text(0.05,0.85,r'\boldmath$%s < z < %s$'%(zmin,zmax),transform=ax[i+1].transAxes,size=22)
            thy = np.array(thy)
            std = np.array(std)
            down = thy - std
            up   = thy + std
            thy = ax[i+1].errorbar(m,(np.array(val)-np.array(thy))/np.array(val),yerr=np.array(alp)/np.array(val),color=color,fmt='.',ms=10,capsize=3.0)
            ax[i+1].axhline(0,0,1,color='black',alpha=0.5,ls='--')




    for i in range(16):
        ax[i+1].tick_params(axis='both',which='both',top=True,right=True,labelbottom=False,direction='in',labelsize=30)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(0.5)
        ax[i+1].xaxis.set_minor_locator(minorLocator)
        ax[i+1].xaxis.set_major_locator(majorLocator)
        ax[i+1].xaxis.set_tick_params(which='major',length=6)
        ax[i+1].xaxis.set_tick_params(which='minor',length=3)
        ax[i+1].set_xlim(0.30,2.30)
        ax[i+1].set_xticks([0.5,1.0,1.5,2.0])
        ax[i+1].set_ylim(-0.50,0.50)
        minorLocator = MultipleLocator(0.05)
        majorLocator = MultipleLocator(0.20)
        ax[i+1].yaxis.set_minor_locator(minorLocator)
        ax[i+1].yaxis.set_major_locator(majorLocator)
        ax[i+1].yaxis.set_tick_params(which='major',length=6)
        ax[i+1].yaxis.set_tick_params(which='minor',length=3)
        #ax[i+1].set_yticks([0.6,0.8,1.0,1.2,1.4])

    for i in [13,14,15,16]:
        ax[i].tick_params(labelbottom=True)
        ax[i].set_xlabel(r'\boldmath$M_{h} [{\rm GeV}]$',size=30)

    for i in [2,3,4,6,7,8,10,11,12,14,15,16]:
        ax[i].tick_params(labelleft=False)


    #ax[4] .text(0.05, 0.65, r'\boldmath${\rm d}^2 \sigma/{\rm d}z {\rm d}m_{\pi\pi} ~ [\mu {\rm b}/{\rm GeV}]$',transform=ax[4].transAxes,size=25)
    ax[16].text(0.05, 0.65, r'$\sqrt{s} = 10.58$'+' '+r'\textrm{GeV}', transform=ax[16].transAxes,size=20)

    thy ,= ax[1].plot([],[],color='black')
    thy_band = ax[1].fill_between([],[],[],color='gold',alpha=1.0)

    #handles,labels = [], []
    #handles.append(hand[1000])
    #handles.append((thy_band,thy))
    #labels.append(r'\textbf{\textrm{BELLE}}') 
    #labels.append(r'\textbf{\textrm{JAM}}') 
    #ax[4].legend(handles,labels,frameon=False,fontsize=22,loc='upper right',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)

    py.tight_layout()
    py.subplots_adjust(hspace=0,wspace=0)


    checkdir('%s/gallery'%wdir)
    filename='%s/gallery/belle-unpolarized-difference'%wdir
    filename+='.png'

    py.savefig(filename)
    print('Saving Belle plot to %s'%filename)





def plot_a12R_z1_M1(wdir,mode):

    print('\ngenerating a12R (z1,M1) SIA plot from %s'%(wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if reaction not in predictions['reactions']: return
    if 2000 not in predictions['reactions'][reaction]: return

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


    nrows,ncols=2,4
    fig = py.figure(figsize=(ncols*5,nrows*3))
    ax = {}
    for i in range(8):
        ax[i+1] = py.subplot(nrows,ncols,i+1)

    #######################
    #--plot absolute values
    #######################

    hand = {}
    #--plot data
    for idx in data:
        if idx==2000: color = 'firebrick'
        else: continue        
    
        Z1    = data[idx]['z1']
        M1    = data[idx]['M1']
        M1min = data[idx]['M1min']
        M1max = data[idx]['M1max']
        values = data[idx]['value']
        alpha  = data[idx]['alpha']
        BIN = data[idx]['bin']
        nbins = len(np.unique(BIN))+1
        _thy = data[idx]['thy-%d'%ic]
        _std = data[idx]['dthy-%d'%ic]
        for i in range(nbins):
            lprint('generating bin: [%s/%s]'%(i+1,nbins))
            if (i+1) not in BIN: continue
            if mode==0:
                for k in range(len(predictions_ic)):
                    z1,m1,val,alp,thy,std = [],[],[],[],[],[]
                    for j in range(len(values)):
                        if BIN[j] != (i+1): continue
                        val.append(values[j])
                        alp.append(alpha[j])
                        z1.append(Z1[j])
                        m1.append(M1[j])
                        thy.append(predictions_ic[k][j])
                        std.append(_std[j])
                        _M1min,_M1max = M1min[j],M1max[j]
                        hand[idx] = ax[i+1].errorbar(z1,val,yerr=alp,color=color,fmt='o',ms=2.0,capsize=3.0)
                    if k==0: ax[i+1].text(0.05,0.03,r'\boldmath$%3.2f < M_h < %3.2f ~\rm{GeV}$'%(_M1min,_M1max),transform=ax[i+1].transAxes,size=22)
                    #--plot mean and std of all replicas
                    thy = np.array(thy)
                    thy_plot ,= ax[i+1].plot(z1,thy,color='black',lw=1.0,alpha=0.5)
            if mode==1:
                z1,m1,val,alp,thy,std = [],[],[],[],[],[]
                for j in range(len(values)):
                    if BIN[j] != (i+1): continue
                    val.append(values[j])
                    alp.append(alpha[j])
                    z1.append(Z1[j])
                    m1.append(M1[j])
                    thy.append(_thy[j])
                    std.append(_std[j])
                    _M1min,_M1max = M1min[j],M1max[j]
                    hand[idx] = ax[i+1].errorbar(z1,val,yerr=alp,color=color,fmt='o',ms=2.0,capsize=3.0)
                ax[i+1].text(0.05,0.03,r'\boldmath$%3.2f < M_h < %3.2f ~{\rm GeV}$'%(_M1min,_M1max),transform=ax[i+1].transAxes,size=22)
                #--plot mean and std of all replicas
                thy = np.array(thy)
                std = np.array(std)
                down = thy - std
                up   = thy + std
                thy_plot ,= ax[i+1].plot(z1,thy,color='black')
                thy_band  = ax[i+1].fill_between(z1,down,up,color='gold',alpha=0.4)


    for i in range(8):
        ax[i+1].tick_params(axis='both',which='both',top=True,right=True,labelbottom=False,direction='in',labelsize=30)
        minorLocator = MultipleLocator(0.04)
        majorLocator = MultipleLocator(0.2)
        ax[i+1].xaxis.set_minor_locator(minorLocator)
        ax[i+1].xaxis.set_major_locator(majorLocator)
        ax[i+1].xaxis.set_tick_params(which='major',length=6)
        ax[i+1].xaxis.set_tick_params(which='minor',length=3)
        ax[i+1].set_xlim(0.15,1)
        ax[i+1].set_xticks([0.2,0.4,0.6,0.8])
        ax[i+1].axhline(0,0,1,ls='--',alpha=0.5,color='black')

    for i in [5,6,7,8]:
        ax[i].tick_params(labelbottom=True)
        ax[i].set_xlabel(r'\boldmath$z$',size=30)
        ax[i].xaxis.set_label_coords(0.92,-0.02)

    for i in [2,3,4,6,7,8]:
        ax[i].tick_params(labelleft=False)

    for i in [1,2,3,4]:
        ax[i].set_ylim(-0.040,0.015)
        minorLocator = MultipleLocator(0.005)
        majorLocator = MultipleLocator(0.02)
        ax[i].yaxis.set_minor_locator(minorLocator)
        ax[i].yaxis.set_major_locator(majorLocator)
        ax[i].yaxis.set_tick_params(which='major',length=6)
        ax[i].yaxis.set_tick_params(which='minor',length=3)
        ax[i].set_yticks([-0.02,0])
    for i in [5,6,7,8]:
        ax[i].set_ylim(-0.10,0.02)
        minorLocator = MultipleLocator(0.01)
        majorLocator = MultipleLocator(0.04)
        ax[i].yaxis.set_minor_locator(minorLocator)
        ax[i].yaxis.set_major_locator(majorLocator)
        ax[i].yaxis.set_tick_params(which='major',length=6)
        ax[i].yaxis.set_tick_params(which='minor',length=3)
        ax[i].set_yticks([-0.08,-0.04,0])

    ax[1].text(0.05, 0.20, r'\boldmath$a_{12R}$',transform=ax[1].transAxes,size=40)
    ax[2].text(0.05, 0.20, r'$\sqrt{s} = 10.58$'+' '+r'\textrm{GeV}', transform=ax[2].transAxes,size=25)

    ax[3].text(0.05,0.87,r'$0.2 < \bar{z} < 1.0$'                       ,transform=ax[3].transAxes,size=18)
    ax[3].text(0.05,0.75,r'$2m_{\pi} < \overline{M}_h < 2.00 ~\rm{GeV}$',transform=ax[3].transAxes,size=18)


    thy ,= ax[1].plot([],[],color='black')
    thy_band = ax[1].fill_between([],[],[],color='gold',alpha=1.0)

    handles,labels = [], []
    handles.append(hand[2000])
    handles.append((thy_band,thy))
    labels.append(r'\textbf{\textrm{BELLE}}') 
    labels.append(r'\textbf{\textrm{JAM}}') 
    ax[4].legend(handles,labels,frameon=False,fontsize=22,loc='upper right',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)


    py.tight_layout()
    py.subplots_adjust(hspace=0,wspace=0)


    checkdir('%s/gallery'%wdir)
    filename='%s/gallery/belle-a12R_z1_M1'%wdir
    if mode==1: filename+='-bands'
    filename+='.png'

    py.savefig(filename)
    print('Saving Belle a12R (z1,M1) plot to %s'%filename)

def plot_a12R_M1_M2(wdir,mode):

    print('\ngenerating a12R (M1,M2) SIA plot from %s'%(wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if reaction not in predictions['reactions']: return
    if 2001 not in predictions['reactions'][reaction]: return

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


    nrows,ncols=2,4
    fig = py.figure(figsize=(ncols*5,nrows*3))
    ax = {}
    for i in range(8):
        ax[i+1] = py.subplot(nrows,ncols,i+1)

    #######################
    #--plot absolute values
    #######################

    hand = {}
    #--plot data
    for idx in data:
        if idx==2001: color = 'firebrick'
        else: continue        

        M1    = data[idx]['M1']
        M2    = data[idx]['M2']
        M1min = data[idx]['M1min']
        M1max = data[idx]['M1max']
        values = data[idx]['value']
        alpha  = data[idx]['alpha']
        BIN = data[idx]['bin']
        nbins = len(np.unique(BIN))+1
        _thy = data[idx]['thy-%d'%ic]
        _std = data[idx]['dthy-%d'%ic]
        for i in range(nbins):
            lprint('generating bin: [%s/%s]'%(i+1,nbins))
            if (i+1) not in BIN: continue
            if mode==0:
                for k in range(len(predictions_ic)):
                    m1,m2,val,alp,thy,std = [],[],[],[],[],[]
                    for j in range(len(values)):
                        if BIN[j] != (i+1): continue
                        val.append(values[j])
                        alp.append(alpha[j])
                        m1.append(M1[j])
                        m2.append(M2[j])
                        thy.append(predictions_ic[k][j])
                        std.append(_std[j])
                        _M1min,_M1max = M1min[j],M1max[j]
                        hand[idx] = ax[i+1].errorbar(m2,val,yerr=alp,color=color,fmt='o',ms=2.0,capsize=3.0)
                    if k==0: ax[i+1].text(0.05,0.03,r'\boldmath$%3.2f < M_h < %3.2f ~\rm{GeV}$'%(_M1min,_M1max),transform=ax[i+1].transAxes,size=22)
                    #--plot mean and std of all replicas
                    thy = np.array(thy)
                    thy_plot ,= ax[i+1].plot(m2,thy,color='black',lw=1.0,alpha=0.5)
            if mode==1:
                m1,m2,val,alp,thy,std = [],[],[],[],[],[]
                for j in range(len(values)):
                    if BIN[j] != (i+1): continue
                    val.append(values[j])
                    alp.append(alpha[j])
                    m1.append(M1[j])
                    m2.append(M2[j])
                    thy.append(_thy[j])
                    std.append(_std[j])
                    _M1min,_M1max = M1min[j],M1max[j]
                    hand[idx] = ax[i+1].errorbar(m2,val,yerr=alp,color=color,fmt='o',ms=2.0,capsize=3.0)
                ax[i+1].text(0.05,0.03,r'\boldmath$%3.2f < M_h < %3.2f ~{\rm GeV}$'%(_M1min,_M1max),transform=ax[i+1].transAxes,size=22)
                #--plot mean and std of all replicas
                thy = np.array(thy)
                std = np.array(std)
                down = thy - std
                up   = thy + std
                thy_plot ,= ax[i+1].plot(m2,thy,color='black')
                thy_band  = ax[i+1].fill_between(m2,down,up,color='gold',alpha=0.4)


    for i in range(8):
        ax[i+1].tick_params(axis='both',which='both',top=True,right=True,labelbottom=False,direction='in',labelsize=30)
        minorLocator = MultipleLocator(0.4)
        majorLocator = MultipleLocator(0.1)
        ax[i+1].xaxis.set_minor_locator(minorLocator)
        ax[i+1].xaxis.set_major_locator(majorLocator)
        ax[i+1].xaxis.set_tick_params(which='major',length=6)
        ax[i+1].xaxis.set_tick_params(which='minor',length=3)
        ax[i+1].set_xlim(0.25,2)
        ax[i+1].set_xticks([0.4,0.8,1.2,1.6])
        ax[i+1].axhline(0,0,1,ls='--',alpha=0.5,color='black')

    for i in [5,6,7,8]:
        ax[i].tick_params(labelbottom=True)
        ax[i].set_xlabel(r'\boldmath$\overline{M}_h$',size=30)
        ax[i].xaxis.set_label_coords(0.92,-0.02)

    for i in [2,3,4,6,7,8]:
        ax[i].tick_params(labelleft=False)

    for i in [1,2,3,4]:
        ax[i].set_ylim(-0.080,0.025)
        minorLocator = MultipleLocator(0.01)
        majorLocator = MultipleLocator(0.04)
        ax[i].yaxis.set_minor_locator(minorLocator)
        ax[i].yaxis.set_major_locator(majorLocator)
        ax[i].yaxis.set_tick_params(which='major',length=6)
        ax[i].yaxis.set_tick_params(which='minor',length=3)
        ax[i].set_yticks([-0.06,-0.04,-0.02,0])

    for i in [5,6,7,8]:
        ax[i].set_ylim(-0.20,0.02)
        minorLocator = MultipleLocator(0.01)
        majorLocator = MultipleLocator(0.04)
        ax[i].yaxis.set_minor_locator(minorLocator)
        ax[i].yaxis.set_major_locator(majorLocator)
        ax[i].yaxis.set_tick_params(which='major',length=6)
        ax[i].yaxis.set_tick_params(which='minor',length=3)
        ax[i].set_yticks([-0.16,-0.12,-0.08,-0.04,0])

    ax[1].text(0.05, 0.20, r'\boldmath$a_{12R}$',transform=ax[1].transAxes,size=40)
    ax[2].text(0.05, 0.20, r'$\sqrt{s} = 10.58$'+' '+r'\textrm{GeV}', transform=ax[2].transAxes,size=25)

    ax[3].text(0.05,0.87,r'$0.2 <      z  < 1.0$'                   ,transform=ax[3].transAxes,size=18)
    ax[3].text(0.05,0.75,r'$0.2 < \bar{z} < 1.0$'                   ,transform=ax[3].transAxes,size=18)


    thy ,= ax[1].plot([],[],color='black')
    thy_band = ax[1].fill_between([],[],[],color='gold',alpha=1.0)

    handles,labels = [], []
    handles.append(hand[2001])
    handles.append((thy_band,thy))
    labels.append(r'\textbf{\textrm{BELLE}}') 
    labels.append(r'\textbf{\textrm{JAM}}') 
    ax[4].legend(handles,labels,frameon=False,fontsize=22,loc='upper right',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)


    py.tight_layout()
    py.subplots_adjust(hspace=0,wspace=0)


    checkdir('%s/gallery'%wdir)
    filename='%s/gallery/belle-a12R-M1-M2'%wdir
    if mode==1: filename+='-bands'
    filename+='.png'

    py.savefig(filename)
    print('Saving Belle a12R (M1,M2) plot to %s'%filename)

def plot_a12R_z1_z2(wdir,mode):

    print('\ngenerating a12R (z1,z2) SIA plot from %s'%(wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if reaction not in predictions['reactions']: return
    if 2002 not in predictions['reactions'][reaction]: return

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


    nrows,ncols=3,3
    fig = py.figure(figsize=(ncols*5,nrows*3))
    ax = {}
    for i in range(9):
        ax[i+1] = py.subplot(nrows,ncols,i+1)

    #######################
    #--plot absolute values
    #######################

    hand = {}
    #--plot data
    for idx in data:
        if idx==2002: color = 'firebrick'
        else: continue        

        Z1    = data[idx]['z1']
        Z2    = data[idx]['z2']
        Z1min = data[idx]['z1min']
        Z1max = data[idx]['z1max']
        values = data[idx]['value']
        alpha  = data[idx]['alpha']
        BIN = data[idx]['bin']
        nbins = len(np.unique(BIN))+1
        _thy = data[idx]['thy-%d'%ic]
        _std = data[idx]['dthy-%d'%ic]
        for i in range(nbins):
            lprint('generating bin: [%s/%s]'%(i+1,nbins))
            if (i+1) not in BIN: continue
            if mode==0:
                for k in range(len(predictions_ic)):
                    z1,z2,val,alp,thy,std = [],[],[],[],[],[]
                    for j in range(len(values)):
                        if BIN[j] != (i+1): continue
                        val.append(values[j])
                        alp.append(alpha[j])
                        z1.append(Z1[j])
                        z2.append(Z2[j])
                        thy.append(predictions_ic[k][j])
                        std.append(_std[j])
                        _Z1min,_Z1max = Z1min[j],Z1max[j]
                        hand[idx] = ax[i+1].errorbar(z2,val,yerr=alp,color=color,fmt='o',ms=2.0,capsize=3.0)
                    if k==0: ax[i+1].text(0.05,0.03,r'\boldmath$%3.2f < z < %3.2f$'%(_Z1min,_Z1max),transform=ax[i+1].transAxes,size=22)
                    #--plot mean and std of all replicas
                    thy = np.array(thy)
                    thy_plot ,= ax[i+1].plot(z1,thy,color='black',lw=1.0,alpha=0.5)
            if mode==1:
                z1,z2,val,alp,thy,std = [],[],[],[],[],[]
                for j in range(len(values)):
                    if BIN[j] != (i+1): continue
                    val.append(values[j])
                    alp.append(alpha[j])
                    z1.append(Z1[j])
                    z2.append(Z2[j])
                    thy.append(_thy[j])
                    std.append(_std[j])
                    _Z1min,_Z1max = Z1min[j],Z1max[j]
                    hand[idx] = ax[i+1].errorbar(z2,val,yerr=alp,color=color,fmt='o',ms=2.0,capsize=3.0)
                ax[i+1].text(0.05,0.03,r'\boldmath$%3.2f < z < %3.2f$'%(_Z1min,_Z1max),transform=ax[i+1].transAxes,size=22)
                #--plot mean and std of all replicas
                thy = np.array(thy)
                std = np.array(std)
                down = thy - std
                up   = thy + std
                thy_plot ,= ax[i+1].plot(z2,thy,color='black')
                thy_band  = ax[i+1].fill_between(z2,down,up,color='gold',alpha=0.4)


    for i in range(9):
        ax[i+1].tick_params(axis='both',which='both',top=True,right=True,labelbottom=False,direction='in',labelsize=30)
        minorLocator = MultipleLocator(0.04)
        majorLocator = MultipleLocator(0.2)
        ax[i+1].xaxis.set_minor_locator(minorLocator)
        ax[i+1].xaxis.set_major_locator(majorLocator)
        ax[i+1].xaxis.set_tick_params(which='major',length=6)
        ax[i+1].xaxis.set_tick_params(which='minor',length=3)
        ax[i+1].set_xlim(0.15,1)
        ax[i+1].set_xticks([0.2,0.4,0.6,0.8])
        ax[i+1].axhline(0,0,1,ls='--',alpha=0.5,color='black')

    for i in [7,8,9]:
        ax[i].tick_params(labelbottom=True)
        ax[i].set_xlabel(r'\boldmath$\overline{z}$',size=30)
        ax[i].xaxis.set_label_coords(0.92,-0.02)

    for i in [2,3,5,6,8,9]:
        ax[i].tick_params(labelleft=False)

    for i in [1,2,3]:
        ax[i].set_ylim(-0.07,0.02)
        minorLocator = MultipleLocator(0.01)
        majorLocator = MultipleLocator(0.04)
        ax[i].yaxis.set_minor_locator(minorLocator)
        ax[i].yaxis.set_major_locator(majorLocator)
        ax[i].yaxis.set_tick_params(which='major',length=6)
        ax[i].yaxis.set_tick_params(which='minor',length=3)
        ax[i].set_yticks([-0.06,-0.04,-0.02,0])
    for i in [4,5,6]:
        ax[i].set_ylim(-0.10,0.02)
        minorLocator = MultipleLocator(0.01)
        majorLocator = MultipleLocator(0.04)
        ax[i].yaxis.set_minor_locator(minorLocator)
        ax[i].yaxis.set_major_locator(majorLocator)
        ax[i].yaxis.set_tick_params(which='major',length=6)
        ax[i].yaxis.set_tick_params(which='minor',length=3)
        ax[i].set_yticks([-0.08,-0.04,0])
    for i in [7,8,9]:
        ax[i].set_ylim(-0.15,0.02)
        minorLocator = MultipleLocator(0.01)
        majorLocator = MultipleLocator(0.04)
        ax[i].yaxis.set_minor_locator(minorLocator)
        ax[i].yaxis.set_major_locator(majorLocator)
        ax[i].yaxis.set_tick_params(which='major',length=6)
        ax[i].yaxis.set_tick_params(which='minor',length=3)
        ax[i].set_yticks([-0.12,-0.08,-0.04,0])

    ax[1].text(0.05, 0.20, r'\boldmath$a_{12R}$',transform=ax[1].transAxes,size=40)
    ax[2].text(0.05, 0.20, r'$\sqrt{s} = 10.58$'+' '+r'\textrm{GeV}', transform=ax[2].transAxes,size=25)

    ax[3].text(0.05,0.87,r'$2m_{\pi} <           M_h  < 2.00 ~\rm{GeV}$',transform=ax[3].transAxes,size=18)
    ax[3].text(0.05,0.75,r'$2m_{\pi} < \overline{M}_h < 2.00 ~\rm{GeV}$',transform=ax[3].transAxes,size=18)


    thy ,= ax[1].plot([],[],color='black')
    thy_band = ax[1].fill_between([],[],[],color='gold',alpha=1.0)

    handles,labels = [], []
    handles.append(hand[2002])
    handles.append((thy_band,thy))
    labels.append(r'\textbf{\textrm{BELLE}}') 
    labels.append(r'\textbf{\textrm{JAM}}') 
    ax[4].legend(handles,labels,frameon=False,fontsize=22,loc='upper right',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)


    py.tight_layout()
    py.subplots_adjust(hspace=0,wspace=0)


    checkdir('%s/gallery'%wdir)
    filename='%s/gallery/belle-a12R-z1-z2'%wdir
    if mode==1: filename+='-bands'
    filename+='.png'

    py.savefig(filename)
    print('Saving Belle a12R (z1,z2) plot to %s'%filename)








