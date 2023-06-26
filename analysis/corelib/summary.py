import sys,os
import numpy as np
import scipy.stats
import copy
from subprocess import Popen, PIPE, STDOUT
import pandas as pd

#--matplotlib
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import pylab as py

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config

#--from fitlib
from fitlib.resman import RESMAN

#--from local
from analysis.corelib import core
from analysis.corelib import classifier

def get_z_score(chi2_red,dof):
    x = chi2_red
    cdf = scipy.stats.chi2.cdf(x,dof)
    p_val = 1-cdf
    return np.abs(scipy.stats.norm.ppf(p_val))

def get_norm(wdir,kc): 
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
    tab={}
    for replica in replicas: 
        order=replica['order'][istep]
        params=replica['params'][istep]
        for i in range(len(order)):
            if order[i][0]==2:
                reaction=order[i][1]
                idx=order[i][2]
                if reaction not in tab: tab[reaction]={}
                if idx not in tab[reaction]: tab[reaction][idx]=[] 
                tab[reaction][idx].append(params[i])

        for k in conf['datasets']:
            for kk in conf['datasets'][k]['norm']:
                if conf['datasets'][k]['norm'][kk]['fixed'] == True:  continue
                if conf['datasets'][k]['norm'][kk]['fixed'] == False: continue
                reference_norm = conf['datasets'][k]['norm'][kk]['fixed']
                if k  not in tab: tab[k]={}
                if kk not in tab[k]: tab[k][kk]=[] 
                tab[k][kk].append(tab[k][reference_norm])
                       
    for reaction in tab:
        for idx in tab[reaction]:
            norm=tab[reaction][idx][:]
            tab[reaction][idx]={}
            tab[reaction][idx]['mean']=np.mean(norm)
            tab[reaction][idx]['std']=np.std(norm)
    return tab

def get_chi2(wdir,kc): 

    istep=core.get_istep()
    #replicas=core.get_replicas(wdir)
    #core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    data=predictions['reactions']
    tab={}
    for reaction in data:
        if len(list(data[reaction])) == 0: continue
        if reaction not in tab: 
            tab[reaction]={}
            tab[reaction]['chi2'] = 0
            tab[reaction]['npts'] = 0
        for idx in data[reaction]:
            if idx not in tab[reaction]: tab[reaction][idx]={}
     
 
            value=data[reaction][idx]['value']
            alpha=data[reaction][idx]['alpha']
            if 'rres-rep' in data[reaction][idx].keys():
                rres=np.mean(data[reaction][idx]['rres-rep'],axis=0)
            else:
                rres=0.
            if np.isnan(rres).any(): rres=0.0
            thy=np.mean(data[reaction][idx]['prediction-rep'],axis=0)
            tab['nrep'] = len(data[reaction][idx]['prediction-rep'])
            col=data[reaction][idx]['col'][0]
            if 'target' in            data[reaction][idx]: tar=data[reaction][idx]['target'][0]
            elif 'tar' in             data[reaction][idx]: tar=data[reaction][idx]['tar'][0]
            elif 'particles-in' in    data[reaction][idx]: tar=data[reaction][idx]['particles-in'][0]
            elif 'reaction' in        data[reaction][idx]: tar=data[reaction][idx]['reaction'][0]
            else:                                          tar='-' 
            if tar=='deuteron': tar='d'
            if 'hadron' in            data[reaction][idx]: had=data[reaction][idx]['hadron'][0]
            else:                                          had='-'
            obs=data[reaction][idx]['obs'][0]
            #--rewrite some observables to be more compact
            res=(value-thy)/alpha
            chi2=np.sum(res**2) + np.sum(rres**2)
            npts=res.size
            chi2_npts=chi2/npts

            zscore = get_z_score(chi2,npts)

            tab[reaction][idx]['col']       = col
            tab[reaction][idx]['tar']       = tar
            tab[reaction][idx]['had']       = had
            tab[reaction][idx]['obs']       = obs
            tab[reaction][idx]['chi2']      = chi2
            tab[reaction][idx]['npts']      = npts
            tab[reaction][idx]['chi2_npts'] = chi2_npts
            tab[reaction][idx]['zscore']    = zscore
            tab[reaction]['chi2']          += chi2
            tab[reaction]['npts']          += npts
        tab[reaction]['chi2_npts'] = tab[reaction]['chi2']/tab[reaction]['npts']
        tab[reaction]['zscore']    = get_z_score(tab[reaction]['chi2'],tab[reaction]['npts'])

    return tab

def print_summary(wdir,kc): 
    load_config('%s/input.py'%wdir)
    norm_tab=get_norm(wdir,kc)
    chi2_tab=get_chi2(wdir,kc)

    L = len(os.listdir('%s/msr-inspected'%wdir))

    print('\nsummary of  %s [%d/%d replicas]\n'%(wdir,chi2_tab['nrep'],L))

    #--adjust width of col and obs columns
    lobs,lcol = [],[]
    for reaction in chi2_tab:
        if reaction in ['nrep']: continue
        for idx in chi2_tab[reaction]:
            if idx in ['chi2','npts','chi2_npts','zscore']: continue
            lcol.append(len(chi2_tab[reaction][idx]['col']))
            lobs.append(len(chi2_tab[reaction][idx]['obs']))

    lcol = np.max(lcol)
    lobs = np.max(lobs)

    msg1='%14s '
    msg1+='%10s '
    msg1+=' '*(lcol-2)
    msg1+='%s '
    msg1+='%10s '
    msg1+='%5s '
    msg1+=' '*(lobs-2)
    msg1+='%s '
    msg1+='%5s '
    msg1+='%10s '
    msg1+='%10s '
    msg1+='%10s '
    msg1+='%12s '
    msg1=msg1%('reaction','idx','col','tar','had','obs','npts','chi2','chi2/npts','zscore','norm')
    print(msg1)
    chi2_tot=0
    npts_tot=0
    for reaction in chi2_tab:
        if reaction in ['nrep']: continue
        l = list(chi2_tab[reaction])
        for _ in ['chi2','npts','chi2_npts','zscore']: l.remove(_)
        for idx in sorted(l):
            col=chi2_tab[reaction][idx]['col']
            tar=chi2_tab[reaction][idx]['tar']
            had=chi2_tab[reaction][idx]['had']
            obs=chi2_tab[reaction][idx]['obs']
            npts=chi2_tab[reaction][idx]['npts']
            chi2=chi2_tab[reaction][idx]['chi2']
            chi2_npts=chi2_tab[reaction][idx]['chi2_npts']
            zscore=chi2_tab[reaction][idx]['zscore']
            if reaction in norm_tab:
                if idx in norm_tab[reaction]:
                    mean=norm_tab[reaction][idx]['mean']
                    std =norm_tab[reaction][idx]['std']
                    std =str(int(np.round(std*1000)))
                    norm = '%4.3f'%mean
                    if len(std)==1: norm+=' '
                    norm += '('+std+')'
                else:
                    norm='N/A'
            else: norm = 'N/A'

            chi2_tot+=chi2
            npts_tot+=npts

            msg2 ='%14s '
            msg2+='%10d '
            msg2+=' '*(lcol-len(col)+1)
            msg2+='%s '
            msg2+='%10s '
            msg2+='%5s '
            msg2+=' '*(lobs-len(obs)+1)
            msg2+='%s '
            msg2+='%5d '
            msg2+='%10.2f '
            msg2+='%10.2f '
            msg2+='%10.2f '
            if norm=='N/A': msg2 += '%12s ' 
            else:           msg2 += '%12s ' 
            print(msg2%(reaction,idx,col,tar,had,obs,npts,chi2,chi2_npts,zscore,norm))

    chi2_npts_tot=chi2_tot/npts_tot

    zscore_tot = get_z_score(chi2_tot,npts_tot)

    print("-"*len(msg1))
    #--print chi2 per experiment
    for reaction in chi2_tab:
        if reaction in ['nrep']: continue
        npts      = chi2_tab[reaction]['npts']
        chi2      = chi2_tab[reaction]['chi2']
        chi2_npts = chi2_tab[reaction]['chi2_npts']
        zscore    = chi2_tab[reaction]['zscore']

        msg3 ='%14s '
        msg3+='%10s '
        msg3+=' '*(lcol)
        msg3+='%s '
        msg3+='%10s '
        msg3+='%5s '
        msg3+=' '*(lobs)
        msg3+='%s '
        msg3+='%5d '
        msg3+='%10.2f '
        msg3+='%10.2f '
        msg3+='%10.2f '
        print(msg3%(reaction,' ',' ',' ',' ',' ',npts,chi2,chi2_npts,zscore))

    print("-"*len(msg1))
    msg4 ='%14s '
    msg4+='%10s '
    msg4+=' '*(lcol)
    msg4+='%s '
    msg4+='%10s '
    msg4+='%5s '
    msg4+=' '*(lobs)
    msg4+='%s '
    msg4+='%5d '
    msg4+='%10.2f '
    msg4+='%10.2f '
    msg4+='%10.2f'
    print(msg4%('total',' ',' ',' ',' ',' ',npts_tot,chi2_tot,chi2_npts_tot,zscore_tot))

 



