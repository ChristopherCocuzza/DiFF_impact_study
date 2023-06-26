import os,sys
#--matplotlib
import matplotlib
matplotlib.use('Agg')
import pylab  as py
import pandas as pd
import numpy as np
from scipy.integrate import quad

import numpy
import pandas
import argparse


#--from analysis
from analysis.corelib import core
from analysis.corelib import predict

#--from tools
from tools           import config
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config
from tools.inputmod  import INPUTMOD
from tools.randomstr import id_generator
from tools.config    import load_config,conf

#--from fitlib
from fitlib.resman import RESMAN
from fitlib.parman import PARMAN

#--from qcdlib
from qcdlib.aux import AUX
from qcdlib.alphaS import ALPHAS
from qcdlib.eweak import EWEAK

def simulate(wdir,tar='p',force=True):

    #--generate initial data file
    gen_xlsx(wdir,tar)

    #--modify conf with new data file
    conf = gen_conf(wdir,tar)

    #--get predictions on new data file if not already done
    print('Generating predictions...')
    name = 'simulate-%s'%(tar)
    predict.get_predictions(wdir,force=force,mod_conf=conf,name=name)

    #--update tables
    update_tabs(wdir,tar)



#--generate pseudo-data
def gen_xlsx(wdir,tar):

    checkdir('%s/sim'%wdir)

    #-- the kinem. var.
    data={_:[] for _ in ['col','target','X','Xdo','Xup','Q2','Q2do','Q2up','obs','value','stat_u','syst_u','norm_c','RS','El','Eh','lum']}

    #--get specific points from data file at fitpack/database/pvdis/expdata/1000.xlsx
    fdir = os.environ['FITPACK']
    if tar == 'p': idx = 1000
    if tar == 'd': idx = 1001
    if tar == 'h': idx = 1001
    grid = pd.read_excel(fdir + '/database/EIC/expdata/%s.xlsx'%idx)
    grid = grid.to_dict(orient='list')
    data['X']    = grid['X']
    data['Q2']   = grid['Q2']
    data['Xup']  = grid['Xup']
    data['Xdo']  = grid['Xdo']
    data['Q2up'] = grid['Q2up']
    data['Q2do'] = grid['Q2do']
    data['RS']   = grid['RS']
    data['El']   = grid['El']
    data['Eh']   = grid['Eh']

    obs = 'A_PV_%s'%kind

    for i in range(len(data['X'])):
        data['col']   .append('JAM4EIC')
        data['target'].append(tar)
        data['obs']   .append(obs)
        data['value'] .append(0.0)
        data['stat_u'].append(1e-10)
        data['syst_u'].append(0.0)
        data['norm_c'].append(0.0)

    df=pd.DataFrame(data)
    filename = '%s/sim/simulate-%s.xlsx'%(wdir,tar)
    df.to_excel(filename, index=False)
    print('Generating xlsx file and saving to %s'%filename)

def gen_conf(wdir,tar):

    print('Modifying config with new experimental data file...')

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    exp = 'dihadron_sidis'
    conf['steps'][istep]['datasets'] = {}
    conf['steps'][istep]['datasets'][exp]=[]
    conf['datasets'][exp]['filters']=[]

    #--placeholder index
    idx = 90000
    conf['datasets'][exp]['xlsx'][idx]='./%s/sim/simulate-%s.xlsx'%(wdir,tar)
    conf['steps'][istep]['datasets'][exp].append(idx)

    return conf

def update_tabs(wdir,tar):

    istep=core.get_istep()
    data=load('%s/data/predictions-%d-pvdis-%s-%s.dat'%(wdir,istep,kind,tar))

    blist=[]
    blist.append('thy')
    blist.append('shift')
    blist.append('residuals')
    blist.append('prediction')
    blist.append('N')
    blist.append('Shift')
    blist.append('W2')
    blist.append('alpha')
    blist.append('residuals-rep')
    blist.append('r-residuals')
    blist.append('L')
    blist.append('H')

    #--placeholder index
    idx = 90000
    exp = 'dihadron_sidis'
    tab=data['reactions'][exp][idx]

    #--delete unnecessary data
    for k in blist: 
        try:    del tab[k]
        except: continue

    #--save mean value
    tab['value'] = np.mean(tab['prediction-rep'],axis=0)

    del tab['prediction-rep']
    del tab['shift-rep']

    df=pd.DataFrame(tab)
    filename = '%s/sim/simulate-%s.xlsx'%(wdir,tar)
    df.to_excel(filename, index=False)
    print('Updating xlsx file and saving to %s'%filename)



if __name__=="__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('-w', '--wdir', type=str)
    ap.add_argument('-t', '--tar',  type=str, default='p')

    args = ap.parse_args()

    wdir = args.wdir
    tar  = args.tar

    simulate(wdir,tar)

 
