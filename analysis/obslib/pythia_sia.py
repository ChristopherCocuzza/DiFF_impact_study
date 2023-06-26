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

    plot_unpolarized(wdir,mode)


def plot_unpolarized(wdir,mode):

    print('\ngenerating unpolarized dihadron SIA plot from %s (from PYTHIA)'%(wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if reaction not in predictions['reactions']: return
    if 103 not in predictions['reactions'][reaction]: return

    filters = conf['datasets'][reaction]['filters']

    conf['aux']=aux.AUX()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    data = predictions['reactions'][reaction]

    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)

    RS = []
    #--get theory by seperating solutions and taking mean
    for idx in data:
        if data[idx]['obs'][0] not in ['sig','sig_rat']: continue
        RS.append(data[idx]['RS'][0])
        predictions = copy.copy(data[idx]['prediction-rep'])
        for ic in range(nc):
            predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
            data[idx]['thy-%d'%ic]  = np.mean(predictions_ic,axis=0)
            data[idx]['dthy-%d'%ic] = np.std(predictions_ic,axis=0)

    RS = np.unique(RS)

    #######################
    #--plot absolute values
    #######################

    hand = {}
    #--plot data
    for l in range(len(RS)):

        nrows,ncols=4,4
        fig = py.figure(figsize=(ncols*5,nrows*3))
        ax = {}
        for i in range(16):
            ax[i+1] = py.subplot(nrows,ncols,i+1)

        ax[16].text(0.05, 0.65, r'$\sqrt{s} = %s$'%RS[l] +' '+r'\textrm{GeV}', transform=ax[16].transAxes,size=20)

        for idx in data:
            if data[idx]['obs'][0] not in ['sig','sig_rat']:continue

            rs = data[idx]['RS'][0]
            if rs != RS[l]: continue

            if 'channel' not in data[idx]: continue
            channel = data[idx]['channel'][0]
   
            if channel=='u': color  = 'purple'
            if channel=='s': color  = 'green'
            if channel=='c': color  = 'orange'
            if channel=='b': color  = 'pink'

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
                lprint('generating bin for %s: [%s/%s]'%(idx,i+1,nbins))
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
                            hand[idx] = ax[i+1].errorbar(m,np.array(val),yerr=np.array(alp),color=color,fmt='o',ms=2.0,capsize=3.0)
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
                        hand[channel] = ax[i+1].errorbar(m,np.array(val),yerr=np.array(alp),color=color,fmt='o',ms=2.0,capsize=3.0)
                    ax[i+1].text(0.05,0.85,r'\boldmath$%s < z < %s$'%(round(zmin,2),round(zmax,2)),transform=ax[i+1].transAxes,size=22)
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


        for i in [1,2,3,4]:
            ax[i].set_ylim(0.0,0.6)
            ax[i].set_yticks([0.1,0.2,0.3,0.4,0.5])

        for i in [5,6,7,8]:
            ax[i].set_ylim(0.0,0.5)
            ax[i].set_yticks([0.1,0.2,0.3,0.4])

        for i in [9,10,11,12]:
            ax[i].set_ylim(0.0,0.4)
            ax[i].set_yticks([0.1,0.2,0.3])

        for i in [13,14,15,16]:
            ax[i].set_ylim(0.0,0.2)
            ax[i].set_yticks([0.05,0.1,0.15])

        for i in [13,14,15,16]:
            ax[i].tick_params(labelbottom=True)
            ax[i].set_xlabel(r'\boldmath$M_h [{\rm GeV}]$',size=30)
            #ax[i].xaxis.set_label_coords(0.95,-0.02)

        for i in [2,3,4,6,7,8,10,11,12,14,15,16]:
            ax[i].tick_params(labelleft=False)

        ax[1] .text(0.05, 0.65, r'\boldmath$\sigma^q/\sigma_{\rm tot}$',transform=ax[1].transAxes,size=25)

        thy ,= ax[1].plot([],[],color='black')
        thy_band = ax[1].fill_between([],[],[],color='gold',alpha=1.0)

        handles,labels = [], []
        if 's' in hand: handles.append(hand['s'])
        if 'c' in hand: handles.append(hand['c'])
        if 'b' in hand: handles.append(hand['b'])
        handles.append((thy_band,thy))
        if 's' in hand: labels.append(r'\textbf{\textrm{PYTHIA \boldmath$s$}}') 
        if 'c' in hand: labels.append(r'\textbf{\textrm{PYTHIA \boldmath$c$}}') 
        if 'b' in hand: labels.append(r'\textbf{\textrm{PYTHIA \boldmath$b$}}') 
        labels.append(r'\textbf{\textrm{JAM}}') 
        ax[4].legend(handles,labels,frameon=False,fontsize=22,loc='upper right',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)

        py.tight_layout()
        py.subplots_adjust(hspace=0,wspace=0)


        checkdir('%s/gallery'%wdir)
        filename='%s/gallery/sia-pythia-RS=%s'%(wdir,RS[l])
        if mode==1: filename+='-bands'
        filename+='.png'

        py.savefig(filename)
        print('Saving Belle plot to %s'%filename)
        py.clf()
















