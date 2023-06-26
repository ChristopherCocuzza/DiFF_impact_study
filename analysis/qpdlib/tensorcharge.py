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

cwd = 'plots'

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        edgecolor=None,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def get_data(wdir,dist,Q2):
    msg='%s/%s-Q2-%d-xmin-0.000000.dat'
    tab=load(msg%(wdir,dist,Q2))
    return np.array(tab)

def plot(ax,wdir,color,center=False):
    n=50
    x=get_data(wdir,'deltau',4)
    y=get_data(wdir,'deltad',4)

    b=confidence_ellipse(x, y, ax, n_std=1
        ,lw=1,facecolor=color,alpha=0.3)

    if center:
        bb,=ax.plot(np.mean(x),np.mean(y),'%so'%color,markersize=2)

        return (b,bb)

    else:

        return b

def plot_tensorcharge(wdir,mode=1,Q2=4,trunc=False,xmin=0.006,xmax=0.99):

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()

    if 'tpdf' not in conf['steps'][istep]['active distributions']:
        if 'tpdf' not in conf['steps'][istep]['passive distributions']:
                print('tpdf is not an active or passive distribution')
                return 

    nrows,ncols=1,2
    fig = py.figure(figsize=(ncols*5,nrows*4))
    
    ##############################################
    ax=py.subplot(nrows,ncols,1)

    H,L=[],[]
    #--plot our result
    Q2 = 4

    if trunc==False:
        data = load('%s/data/tpdf-moment-1-Q2=%3.5f.dat'%(wdir,Q2))
    else: 
        data = load('%s/data/tpdf-truncmom-xmin=%3.5f-xmax=%3.5f-Q2=%3.5f'%(wdir,xmin,xmax,Q2))
    du   = np.array(data['moments']['uv'])
    dd   = np.array(data['moments']['dv'])
    gT   = du - dd

    if mode==0: 
        bb =ax.scatter(du,dd,color='red',s=10)
        H.append((bb))
    if mode==1: 
        bb,=ax.plot(np.mean(du),np.mean(dd),'ro',markersize=2)
        b  = confidence_ellipse(du,dd,ax,n_std=1,lw=1,facecolor='red',alpha=0.5)
        H.append((b,bb))
    L.append(r'${\rm JAM}$')

    #--plot JAM3D
    if mode==0: 
        du=get_data('%s/data/JAM3D/JAM23/LQCD'%cwd,'deltau',4)
        dd=get_data('%s/data/JAM3D/JAM23/LQCD'%cwd,'deltad',4)
        bb =ax.scatter(du,dd,color='blue',s=8,alpha=0.5)
        H.append((bb))
        
        du=get_data('%s/data/JAM3D/JAM23/noLQCD'%cwd,'deltau',4)
        dd=get_data('%s/data/JAM3D/JAM23/noLQCD'%cwd,'deltad',4)
        bb =ax.scatter(du,dd,color='green',s=8,alpha=0.5)
        H.append((bb))
   
    if mode==1: 
        h=plot(ax,'%s/data/JAM3D/JAM23/LQCD'%cwd,'b',center=True)
        H.append(h)
        
        h=plot(ax,'%s/data/JAM3D/JAM23/noLQCD'%cwd,'g',center=True)
        H.append(h)

    L.append(r'${\rm JAM3D}$')
    L.append(r'${\rm JAM3D (no LQCD)}$')
    
    
    #Goldstein~et~al (4 GeV^2)
    #u = 0.860
    #du=0.248
    #d= -0.119
    #dd=0.060
    #h=ax.errorbar([u],[d],xerr=[du],yerr=[dd],fmt='kD',\
    #            markersize=4,elinewidth=1)
    #H.append(h)
    #L.append(r'$\rm Goldstein~et~al~(2014)$')
   
    #Radici,~Bacchetta~(2018) (4 GeV^2)
    u = 0.39
    du=0.10
    d= -0.11
    dd=0.26
    h=ax.errorbar([u],[d],xerr=[du],yerr=[dd],fmt='ks',\
                markersize=4,elinewidth=1)
    H.append(h)
    L.append(r'\rm Radici,~Bacchetta~(2018)')

    #Gupta~et~al (4 GeV^2)
    u =0.790
    du=0.027
    d= -0.198
    dd=0.010
    h=ax.errorbar([u],[d],xerr=[du],yerr=[dd],fmt='m>',\
                markersize=4,elinewidth=1)
    H.append(h)
    L.append(r'$\rm Gupta~et~al~(2018)$')
 

    #Alexandrou~et~al (4 GeV^2)
    u =0.729
    du=0.022
    d= -0.208
    dd=0.008
    h=ax.errorbar([u],[d],xerr=[du],yerr=[dd],fmt='mx',\
                markersize=4,elinewidth=1)
    H.append(h)
    L.append(r'$\rm Alexandrou~et~al~(2020)$')
    
    
    
    #Pitschmann~et~al (4 GeV^2)
    #u =0.55
    #du=0.08
    #d= -0.11
    #dd=0.02
    #h=ax.errorbar([u],[d],xerr=[du],yerr=[dd],fmt='c^',\
    #            markersize=4,elinewidth=1)
    #H.append(h)
    #L.append(r'$\rm Pitschmann~et~al~(2015)$')
    
    ax.set_xlim(0.25,0.95)
    ax.set_xticks([.4,.6,.8])

    ax.set_ylim(-0.4,0.4)
    ax.set_yticks([-0.3,-0.2,-0.1,0,0.1,0.2,0.3])


    ax.set_xlabel(r'\boldmath{$\delta u$}',size=20)
    ax.set_ylabel(r'\boldmath{$\delta d$}',size=20,rotation=0)
    ax.xaxis.set_label_coords(0.95, -0.02)
    ax.yaxis.set_label_coords(-0.06, 0.925)
    ax.tick_params(axis='both',which='both',direction='in',top=True,right=True,labelsize=15)
   
    ax.text(0.65,0.05,r'$Q^2 = %d~{\rm GeV}^2$'%Q2,transform=ax.transAxes, size=20)
 
    minorLocator = MultipleLocator(0.05)
    ax.xaxis.set_minor_locator(minorLocator)
    minorLocator = MultipleLocator(0.05)
    ax.yaxis.set_minor_locator(minorLocator)
    
    ax.legend(H,L,loc=1,fontsize=12,frameon=0,handletextpad=0.2,ncol=2,columnspacing=1.0)
    
    
    
    
    
    ##############################################
   
    ax=py.subplot(nrows,ncols,2)
    H=[]

    D=[]
    #         gT     -dgT    +dgT    format  distance
    #--our result
    if mode==0:
        ax.scatter(gT,np.zeros(len(gT)),color='red',s=10)
        ax.text(1.3,0+0.8-1,r'$\rm JAM$',size=12)
    if mode==1:
        mean, std = np.mean(gT), np.std(gT)
        D.append([mean ,std   ,std   ,'JAM'     ,'ro',0.08])

    #--JAM22-3D
    #D.append([0.90 ,0.05  ,0.05  ,'JAM3D','bo',0.08])
    #D.append([0.71 ,0.16  ,0.16  ,'JAM3D (no LQCD)','go',0.08])
    du=get_data('%s/data/JAM3D/JAM23/LQCD'%cwd,'deltau',4)
    dd=get_data('%s/data/JAM3D/JAM23/LQCD'%cwd,'deltad',4)
    gT = du - dd
    if mode==0: 
        bb =ax.scatter(gT,np.ones(len(gT)),color='blue',s=8,alpha=0.5)
        ax.text(1.3,1+0.8-1,r'$\rm JAM3D (w/ LQCD)$',size=12)
    if mode==1: 
        mean, std = np.mean(gT), np.std(gT)
        D.append([mean ,std   ,std   ,'JAM3D (w/ LQCD)'     ,'bo',0.08])
    du=get_data('%s/data/JAM3D/JAM23/noLQCD'%cwd,'deltau',4)
    dd=get_data('%s/data/JAM3D/JAM23/noLQCD'%cwd,'deltad',4)
    gT = du - dd
    if mode==0: 
        bb =ax.scatter(gT,2*np.ones(len(gT)),color='green',s=8,alpha=0.5)
        ax.text(1.3,2+0.8-1,r'$\rm JAM3D (no LQCD)$',size=12)
    if mode==1: 
        mean, std = np.mean(gT), np.std(gT)
        D.append([mean ,std   ,std   ,'JAM3D (no LQCD)'     ,'go',0.08])
        ax.errorbar(np.mean(gT),2,xerr=np.std(gT),fmt='go',markersize=4,capsize=0)
    
    
    D.append([0.69 ,0.21  ,0.21  ,"D'Alesio et al (2020)"   ,'ko',0.08])
    D.append([0.57 ,0.21  ,0.21  ,'Benel et al (2019)'      ,'ko',0.08])
    D.append([0.53 ,0.25  ,0.25  ,'Radici, Bacchetta (2018)','ks',0.08])
    D.append([0.61 ,0.25  ,0.15  ,'Kang~et~al~(2015)'       ,'ko',0.08])
    D.append([0.81 ,0.44  ,0.44  ,'Radici~et~al~(2015)'     ,'ko',0.08])
    D.append([0.979,0.25  ,0.25  ,'Goldstein~et~al~(2014)'  ,'kD',0.08])
    D.append([0.64 ,0.32  ,0.28  ,'Anselmino~et~al~(2013)'  ,'ko',0.08])
    
    D.append([0.936,0.025 ,0.025 ,'Alexandrou~et~al~(2020)' ,'mx',0.08])
    D.append([0.989,0.0335,0.0335,'Gupta~et~al~(2018)'      ,'m>',0.08])
    D.append([0.972,0.041 ,0.041 ,'Hasan~et~al~(2018)'      ,'mo',0.08])
    
    D.append([0.66 ,0.10  ,0.10  ,'Pitschmann~et~al~(2015)' ,'c^',0.08])
    
    for i in range(len(D)):
        d=D[i]
        if mode==0: i = i+3
        h=ax.errorbar(d[0],i,xerr=[[d[1]],[d[2]]],fmt='%s'%d[4],markersize=4,capsize=0)
        ax.text(d[0]+d[2]+d[5],i+0.8-1,r'$\rm %s$'%d[3].replace(' ','~'),size=12)
    
    
    
    #ax.axvline(x=0.87,color='k',ls=':',alpha=0.5)
    #ax.axvline(x=0.98,color='k',ls='--',alpha=0.5)
    #ax.axvline(x=0.76,color='k',ls='--',alpha=0.5)
    
    #ax.text(0.275,0.95,r'$\rm This$',size=8)
    #ax.text(0.275,0.2,r'$\rm work$',size=8)
    #ax.annotate(r"$\{$",fontsize=24,
    #    xy=(0.495, 0.2),xycoords='data')
    #ax.text(0.15,0.55,r'$\rm JAM20$',size=8)#color='r')
    #ax.annotate(r"$\{$",fontsize=24,
    #    xy=(0.52, 0.2),xycoords='data')
    
    ax.tick_params(axis='both',which='both',direction='in',top=True,labelsize=15)

    ax.set_xlim(0.1,2.51)
    ax.set_xticks([0.5,1.0,1.5,2.0])

    ax.set_ylim(-1,14)
    ax.set_yticks([])

    ax.set_xlabel(r'\boldmath{$g_{\rm T}$}',size=20)
    ax.xaxis.set_label_coords(0.95, -0.02)
    
    minorLocator = MultipleLocator(0.1)
    ax.xaxis.set_minor_locator(minorLocator)
    
    
    
    
    
    ###############################################
    py.tight_layout()
    py.subplots_adjust(left=0.07
                    , bottom=None
                    , right=0.99
                    , top=None
                    , wspace=0
                    , hspace=None)

    if trunc==False:
        filename='%s/gallery/gT-Q2=%3.5f'%(wdir,Q2)
    else:
        filename='%s/gallery/gT-xmin=%3.5f-xmax=%3.5f-Q2=%3.5f'%(wdir,xmin,xmax,Q2)
    if mode==1: filename+='-bands'
    filename+='.png'
    #filename+='.pdf'
    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    print('Saving figure to %s'%filename)






