#!/usr/bin/env python
import os,sys
import subprocess
import numpy as np
import scipy as sp
import pandas as pd
import copy

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config
from tools.randomstr import id_generator

#--from local
from analysis.corelib import core
from analysis.corelib import classifier

def gen_priors(wdir,kc,nsamples):

    print('Generating new priors for %s'%wdir)

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()

    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
    best_cluster=cluster_order[0]
  
    #--get data
    replicas=core.get_replicas(wdir)
    step=conf['steps'][istep]
    samples=[]
    for i in range(len(replicas)):

        if cluster[i]!=best_cluster: continue

        replica=replicas[i]
        order=replica['order'][istep]
        params=replica['params'][istep]
        samples.append(params)

    
    samples=np.array(samples)
    pmin =[np.amin(p) for p in samples.T]
    pmax =[np.amax(p) for p in samples.T]
    mean =[np.mean(p) for p in samples.T]
    std  =[np.std(p)  for p in samples.T]
    cov  = np.cov(samples.T)

    for _ in [pmin,pmax,mean,std]: _ = np.array(_)

    checkdir('%s/msr-opt-priors'%wdir)
    replica=replicas[0] #--template replica
    i = 0
    cnt = 0 #--keep track of failures
    factor = 1.0
    while i < nsamples:
        lprint('%d/%d'%(i+1,nsamples))
        new = np.random.multivariate_normal(mean,cov)
        #--check that new parameters are within limits
        flag = False
        #for j in range(len(samples.T)):
        #    if new[j] < pmin[j]: flag=True
        #    if new[j] > pmax[j]: flag=True
        if flag==False:
            replica['params'][istep]=new
            fname='%s/msr-opt-priors/%s.msr'%(wdir,id_generator(12))
            save(replica,fname)
            i += 1
            cnt = 0
        else: cnt +=1
        if cnt>1000:
            factor+=0.1
            print('Unable to generate a replica after 1000 attempts.  Dividing covariance by %2.1f'%factor)
            cov  = np.cov(samples.T)/factor
            cnt=0

    print 
    print('Saving new priors to %s/msr-opt-priors'%wdir)


