#!/usr/bin/env python
import sys,os
import numpy as np
import pylab as py
from tools.tools import load,checkdir
from tools.config    import conf,load_config
from analysis.corelib import core

#--matplotlib
import matplotlib
matplotlib.use('Agg')
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rc('text',usetex=True)
import pylab as py
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import ScalarFormatter, NullFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm
import  matplotlib as mpl
from matplotlib import cm
import scipy.stats as stats

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

#--set lhapdf data path
version = int(sys.version[0])
os.environ["LHAPDF_DATA_PATH"] = '/work/JAM/ccocuzza/lhapdf/python%s/sets'%version

def plot_moments(wdir,mode=1,Q2=4):

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()

    if 'tpdf' not in conf['steps'][istep]['active distributions']:
        if 'tpdf' not in conf['steps'][istep]['passive distributions']:
                print('tpdf is not an active or passive distribution')
                return 

    nrows,ncols=2,2
    fig = py.figure(figsize=(ncols*5,nrows*4))
    
    ##############################################
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)
    ax21=py.subplot(nrows,ncols,3)
    ax22=py.subplot(nrows,ncols,4)

    hand = {}
    #--plot our result
    Q2 = 4

    moms = [1,2,3,4,5]
    data = {}
    for mom in moms:
        try:
            data[mom] = load('%s/data/tpdf-moment-%d-Q2=%3.5f.dat'%(wdir,mom,Q2))
        except: 
            tpdf.gen_moments(wdir,Q2,mom=mom)
            data[mom] = load('%s/data/tpdf-moment-%d-Q2=%3.5f.dat'%(wdir,mom,Q2))
        if mom==1:
            m0 = data[mom]['moments']
            minus0 = np.array(m0['uv']) - np.array(m0['dv'])
            plus0  = np.array(m0['uv']) + np.array(m0['dv'])

    for mom in moms:
        if mom==1: continue
        m = data[mom]['moments']
        minus = np.array(m['uv']) - np.array(m['dv'])
        plus  = np.array(m['uv']) + np.array(m['dv'])

        mean = np.mean(minus)
        std  = np.std(minus)
        hand['JAM'] = ax11.errorbar([mean],mom-1,xerr=[std],fmt='ro',markersize=4,elinewidth=1,capsize=2.0)
 
        mean = np.mean(plus)
        std  = np.std(plus)
        hand['JAM'] = ax12.errorbar([mean],mom-1,xerr=[std],fmt='ro',markersize=4,elinewidth=1,capsize=2.0)

        minus = minus/minus0
        plus  = plus/minus0

        mean = np.mean(minus)
        std  = np.std(minus)
        hand['JAM'] = ax21.errorbar([mean],mom-1,xerr=[std],fmt='ro',markersize=4,elinewidth=1,capsize=2.0)
 
        mean = np.mean(plus)
        std  = np.std(plus)
        hand['JAM'] = ax22.errorbar([mean],mom-1,xerr=[std],fmt='ro',markersize=4,elinewidth=1,capsize=2.0)

    #--LQCD results
    #--HadStruc 21 (Egerer et al)
    #--results are divided by minus0
    mom1plus      = 0.2285
    mom1plus_err  = 0.0028
    mom2plus      = 0.0787
    mom2plus_err  = 0.0017
    mom1minus     = 0.2199
    mom1minus_err = 0.0148
    mom2minus     = 0.0714
    mom2minus_err = 0.0030 

    d = 0.1
    hand['HadStruc'] = ax21.errorbar([mom1minus], 1+d, xerr=[mom1minus_err],fmt='gs',markersize=2,elinewidth=1,capsize=2.0)
    hand['HadStruc'] = ax21.errorbar([mom2minus], 2+d, xerr=[mom2minus_err],fmt='gs',markersize=2,elinewidth=1,capsize=2.0)
    hand['HadStruc'] = ax22.errorbar([mom1plus],  1+d, xerr=[mom1plus_err] ,fmt='gs',markersize=2,elinewidth=1,capsize=2.0)
    hand['HadStruc'] = ax22.errorbar([mom2plus],  2+d, xerr=[mom2plus_err] ,fmt='gs',markersize=2,elinewidth=1,capsize=2.0)

    #--result from PNDME 20 (first moment only) (u-d only)
    mom1minus     = 0.208
    mom1minus_err = 0.031
    d = 0.1
    hand['PNDME'] = ax11.errorbar([mom1minus], 1+d, xerr=[mom1minus_err],fmt='ms',markersize=2,elinewidth=1,capsize=2.0)

    #--result from ETMC19 (first moment only) (u-d only)
    mom1minus     = 0.204
    mom1minus_err = 0.023
    d = 0.2
    hand['ETMC'] = ax11.errorbar([mom1minus], 1+d, xerr=[mom1minus_err],fmt='bs',markersize=2,elinewidth=1,capsize=2.0)


 
    ax11.set_xlim(0.00,0.25)
    ax11.set_xticks([0,0.05,0.10,0.15,0.20])

    ax12.set_xlim(0,0.30)
    ax12.set_xticks([0,0.05,0.10,0.15,0.20,0.25])

    ax21.set_xlim(0.00,0.25)
    ax21.set_xticks([0,0.05,0.10,0.15,0.20])

    ax22.set_xlim(0,0.30)
    ax22.set_xticks([0,0.05,0.10,0.15,0.20,0.25])

    
    for ax in [ax11,ax12,ax21,ax22]:
        ax.set_ylim(0.5,4.5)
        ax.set_yticks([1,2,3,4])
        ax.tick_params(axis='both',which='both',direction='in',top=True,right=True,labelsize=15)

    ax12.tick_params(labelleft=False)

    ax11.set_ylabel(r'\boldmath{$n$}',size=20,rotation=0)
    ax11.yaxis.set_label_coords(-0.06, 0.925)
    ax21.set_ylabel(r'\boldmath{$n$}',size=20,rotation=0)
    ax21.yaxis.set_label_coords(-0.06, 0.925)
   
    ax12.text(0.60,0.65,r'$\mu^2 = %d~{\rm GeV}^2$'%Q2,transform=ax12.transAxes, size=16)
    #ax11.text(0.45,0.80,r'\boldmath$\langle x^n \rangle^{u-d} / g_T^{u-d}$',transform=ax11.transAxes, size=25)
    #ax12.text(0.45,0.80,r'\boldmath$\langle x^n \rangle^{u+d} / g_T^{u+d}$',transform=ax12.transAxes, size=25)
    ax11.text(0.55,0.80,r'\boldmath$\langle x^n \rangle^{u-d}$',transform=ax11.transAxes, size=25)
    ax12.text(0.55,0.80,r'\boldmath$\langle x^n \rangle^{u+d}$',transform=ax12.transAxes, size=25)
    ax21.text(0.55,0.80,r'\boldmath$\langle x^n \rangle^{u-d} / g_T$',transform=ax21.transAxes, size=25)
    ax22.text(0.55,0.80,r'\boldmath$\langle x^n \rangle^{u+d} / g_T$',transform=ax22.transAxes, size=25)
 
    #minorLocator = MultipleLocator(0.05)
    #ax.xaxis.set_minor_locator(minorLocator)
    #minorLocator = MultipleLocator(0.05)
    #ax.yaxis.set_minor_locator(minorLocator)
    

    handles, labels = [],[]
    handles.append(hand['JAM'])
    handles.append(hand['HadStruc'])
    handles.append(hand['PNDME'])
    handles.append(hand['ETMC'])
    labels.append(r'\textrm{\textbf{JAMDiFF}}')
    labels.append(r'\textrm{\textbf{HadStruc21}}')
    labels.append(r'\textrm{\textbf{PNDME20}}')
    labels.append(r'\textrm{\textbf{ETMC19}}')
    ax12.legend(handles,labels,loc='lower right',fontsize=16,frameon=0,handletextpad=0.2,ncol=1,columnspacing=1.0)
    
    
    
    
    ###############################################
    py.tight_layout()
    #py.subplots_adjust(left=0.07, bottom=None, right=0.99, top=None, wspace=0, hspace=None)
    py.subplots_adjust(wspace=0.01)

    filename='%s/gallery/moments-Q2=%3.5f'%(wdir,Q2)
    if mode==1: filename+='-bands'
    filename+='.png'
    #filename+='.pdf'
    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    print('Saving figure to %s'%filename)






