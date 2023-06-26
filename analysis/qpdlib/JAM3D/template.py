#!/usr/bin/env python
import os,sys
import subprocess
import numpy as np
import scipy as sp
import pandas as pd
import copy
import random
import itertools as it


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




class ELLIPSE:

    def __init__(self,samples,N,kappa=1.0):
        self.N=N #--number of samples to be generated
        self.dim=len(samples[0])
        #--generate transformation matrix
        self.y0=np.mean(samples,axis=0)

        print('computing inv cov matrix')
        cov=self.get_cov(samples)
        w,v=np.linalg.eig(cov)
        icov=np.linalg.inv(cov)
        if np.any(np.isnan(icov)): raise ValueError('icov is nan')

        print('computing eigen values')
        v=np.transpose(v)
        for i in range(w.size): v[i]*=w[i]**0.5
        self.T=np.transpose(v)
        if np.any(np.isnan(self.T)): raise ValueError('T is nan')

        print('computing enlargement factor')
        self.F0=np.amax([np.einsum('i,ij,j',y-self.y0,icov,y-self.y0)\
                         for y in samples])**0.5
        self.F=kappa*self.F0
        if np.isnan(self.F): raise ValueError('F is nan')

        print('generating new samples')
        self.gen_new_samples()
        self.get_volume()

    def is_positive_semi_definite(self,cov):
        test=True
        w,v=np.linalg.eig(cov)
        if np.any(w<0): test=False
        if np.any(np.isnan(v)): test=False
        return test

    #--fix cov matrix.

    def fix_cov1(self,samples,cov):
        sigma=np.abs(np.diagonal(cov))**0.5
        cnt=0
        while 1:
            cnt+=1
            #if cnt%100==0:
            #print('\nfixing cov attempts:',cnt)
            fake_samples=[sample+np.random.randn(sigma.size)*sigma for sample in samples]
            cov=np.cov(np.transpose(fake_samples))
            if self.is_positive_semi_definite(cov): break
        return cov

    def vol_prefactor(self,n):
        """Volume constant for an n-dimensional sphere:
        for n even:      (2pi)^(n    /2) / (2 * 4 * ... * n)
        for n odd :  2 * (2pi)^((n-1)/2) / (1 * 3 * ... * n)
        """
        if n % 2 == 0:
            f = 1.
            i = 2
            while i <= n:
                f *= (2. / i * np.pi)
                i += 2
        else:
            f = 2.
            i = 3
            while i <= n:
                f *= (2. / i * np.pi)
                i += 2
        return f

    def make_eigvals_positive(self,cov, targetprod):
        """
        For the symmetric square matrix ``cov``, increase any zero eigenvalues
        to fulfill the given target product of eigenvalues.
        Returns a (possibly) new matrix.
        """
        w, v = np.linalg.eigh(cov)  # Use eigh because we assume a is symmetric.
        mask = w < 1.e-10
        if np.any(mask):
            nzprod = np.product(w[~mask])  # product of nonzero eigenvalues
            nzeros = mask.sum()  # number of zero eigenvalues
            w[mask] = (targetprod / nzprod) ** (1./nzeros)  # adjust zero eigvals
            cov = np.dot(np.dot(v, np.diag(w)), np.linalg.inv(v))  # re-form cov
        return cov

    def fix_cov2(self,samples,cov):
        """
        Ensure that ``cov`` is nonsingular.
        It can be singular when the ellipsoid has zero volume, which happens
        when npoints <= ndim or when enough points are linear combinations
        of other points. (e.g., npoints = ndim+1 but one point is a linear
        combination of others). When this happens, we expand the ellipse
        in the zero dimensions to fulfill the volume expected from
        ``pointvol``.
        """
        print('\nfixing cov')
        npoints=self.N
        expected_vol = np.exp(-1.0 / float(npoints) )
        pointvol= expected_vol / npoints
        targetprod = (npoints * pointvol / self.vol_prefactor(self.dim))**2
        return self.make_eigvals_positive(cov, targetprod)

    def get_cov(self,samples):
        cov=np.cov(np.transpose(samples))
        if self.is_positive_semi_definite(cov):
          return cov
        else:
          #return self.fix_cov1(samples,cov)
          return self.fix_cov2(samples,cov)

    def gen_new_samples(self):

        # generate the unit sphere
        z=np.random.randn(self.N,self.dim)
        r=np.array([np.dot(z[i],z[i])**0.5 for i in range(self.N)])
        X=np.array([z[i]/r[i]*np.random.rand()**(1.0/self.dim) for i in range(self.N)])

        # generate sphere samples
        Y=np.einsum('ij,nj->ni',self.F*self.T,X) + self.y0
        #print(Y[-10:])
        self.Y=[y for y in Y]

    def get_volume(self):
        from scipy.special import gamma
        y0 = np.mean(self.Y,axis=0)
        cov=np.cov(np.transpose(self.Y))
        w,v=np.linalg.eig(cov)
        v=np.transpose(v)
        icov=np.linalg.inv(cov)
        R=np.amax([np.einsum('i,ij,j',y-y0,icov,y-y0) for y in self.Y])**0.5
        for i in range(w.size): w[i]*=R**2
        for i in range(w.size): v[i]*=w[i]**0.5
        self.vol=np.prod(w**0.5)* np.pi**(self.dim/2.)/gamma(self.dim/2.+1)
        self.v=v
        self.w=w
        self.y0=y0

    def status(self):
        if len(self.Y)>0: return True
        else: return False

    def get_sample(self):
        return self.Y.pop()

    def get_samples(self):
        return np.array(self.Y)

class PDF(CORE):

    def __init__(self,task,wdir,last=False):

        self.FLAV=[]
        #self.FLAV.append('g')   # 0
        self.FLAV.append('u')   # 1
        self.FLAV.append('ub')  # 2
        self.FLAV.append('d')   # 3
        self.FLAV.append('db')  # 4
        self.FLAV.append('s')   # 5
        self.FLAV.append('sb')  # 6

        if  task==0:
            self.gen_priors(wdir)
        if  task==1:
            self.plot_priors(wdir)
        if  task==2:
            self.func=self.gen_xf
            self.msg='pdf.get_xf'
            self.loop_over_steps(wdir,'simple',last)
        if  task==3:
            self.func=self.plot_xf
            self.msg='pdf.plot_xf'
            self.loop_over_steps(wdir,None,last)
        if  task==9:
            self.func=self.par_plot
            self.msg='pdf.par_plot'
            self.loop_over_steps(wdir,None,last)

    def gen_priors(self,wdir):

        load_config('%s/input.py'%wdir)
        resman=RESMAN(nworkers=1,parallel=False,datasets=False)
        parman=resman.parman
        params=[parman.gen_flat(setup=False) for _ in range(300)]
        pdf=conf['pdf']

        #--setup kinematics
        X=10**np.linspace(-4,-1,100)
        X=np.append(X,np.linspace(0.1,0.9,100))
        Q2=conf['aux'].Q02


        #--compute XF for all replicas
        XF={}
        cnt=0
        for par in params:
            cnt+=1
            lprint('%d/%d'%(cnt,len(params)))

            parman.set_new_params(par,initial=False)

            for flav in self.FLAV:
                if flav not in XF:  XF[flav]=[]

                if   flav=='u':
                     func=lambda x: pdf.get_C(x,Q2)[1]
                elif flav=='ub':
                     func=lambda x: pdf.get_C(x,Q2)[2]
                elif flav=='d':
                     func=lambda x: pdf.get_C(x,Q2)[3]
                elif flav=='db':
                     func=lambda x: pdf.get_C(x,Q2)[4]
                elif flav=='s':
                     func=lambda x: pdf.get_C(x,Q2)[5]
                elif flav=='sb':
                     func=lambda x:pdf.get_C(x,Q2)[6]

                XF[flav].append([func(x) for x in X])
        print()
        checkdir('%s/data'%wdir)
        save({'X':X,'Q2':Q2,'XF':XF},'%s/data/pdf-prior.dat'%(wdir))

    def plot_priors(self,wdir):

        data=load('%s/data/pdf-prior.dat'%(wdir))
        X=data['X']

        ncols=4
        nrows=len(self.FLAV)/ncols
        if len(self.FLAV)%ncols>1: nrows+=1
        nrows = int(nrows)

        fig = py.figure(figsize=(ncols*3,nrows*2))

        cnt=0
        for flav in self.FLAV:
            cnt+=1
            ax=py.subplot(nrows,ncols,cnt)
            for i in range(len(data['XF'][flav])):
                ax.plot(X,data['XF'][flav][i])
            if flav=='g' : ax.set_ylim(-1,1)
            if flav=='u': ax.set_ylim(-1,1)
            if flav=='ub': ax.set_ylim(-1,1)
            if flav=='d': ax.set_ylim(-1,1)
            if flav=='db': ax.set_ylim(-1,1)
            if flav=='s' : ax.set_ylim(-1,1)
            if flav=='sb': ax.set_ylim(-1,1)
            ax.set_ylabel(flav)

        py.tight_layout()
        checkdir('%s/gallery'%wdir)
        py.savefig('%s/gallery/pdf-prior.pdf'%(wdir))
        py.close()

    def gen_xf(self,wdir,istep):
        if 'pdf' not in conf['steps'][istep]['active distributions']: return
        #if istep!=7: return
        pdf=conf['pdf']

        #--setup kinematics
        X=10**np.linspace(-4,-1,100)
        X=np.append(X,np.linspace(0.1,0.9,100))
        Q2=conf['aux'].Q02
        #Q2=1.0
        #Q2=80.0**2

        #--compute XF for all replicas
        XF={}
        cnt=0
        replicas=self.get_replicas(wdir)
        for replica in replicas:
            cnt+=1
            lprint('%d/%d'%(cnt,len(replicas)))

            #--retrive the parameters for current step and current replica
            #print()
            #print(len(replica['params'][istep]))
            #print(len(self.parman.par))
            #print()
            #for _ in self.parman.order: print(_)

            #--filter
            flag=False
            params=replica['params'][istep]
            order=replica['order'][istep]
            for i in range(len(order)):
                if order[i][0]!=1:continue
                if order[i][1]!='pdf':continue
                #if order[i][2]=='s1 a':
                #   if params[i]<-0.9: flag=True
            if flag: continue


            self.parman.set_new_params(replica['params'][istep],initial=False)
            self.set_passive_params(istep,replica)

            for flav in self.FLAV:
                if flav not in XF:  XF[flav]=[]

                if   flav=='u':
                     func=lambda x: pdf.get_C(x,Q2)[1]
                elif flav=='ub':
                     func=lambda x: pdf.get_C(x,Q2)[2]
                elif flav=='d':
                     func=lambda x: pdf.get_C(x,Q2)[3]
                elif flav=='db':
                     func=lambda x: pdf.get_C(x,Q2)[4]
                elif flav=='s':
                     func=lambda x: pdf.get_C(x,Q2)[5]
                elif flav=='sb':
                     func=lambda x:pdf.get_C(x,Q2)[6]

                XF[flav].append([x*func(x) for x in X])

        print()
        checkdir('%s/data'%wdir)
        if Q2==conf['aux'].Q02:
            save({'X':X,'Q2':Q2,'XF':XF},'%s/data/pdf-%d.dat'%(wdir,istep))
        else:
            save({'X':X,'Q2':Q2,'XF':XF},'%s/data/pdf-%d-%d.dat'%(wdir,istep,int(Q2)))

    def plot_xf(self,wdir,istep):
        if 'pdf' not in conf['steps'][istep]['active distributions']: return

        cluster,colors,nc,order = self.get_clusters(wdir,istep)
        #return
        data=load('%s/data/pdf-%d.dat'%(wdir,istep))
        X=data['X']

        ncols=4
        nrows=len(self.FLAV)/ncols
        if len(self.FLAV)%ncols>1: nrows+=1
        nrows = int(nrows)
        fig = py.figure(figsize=(ncols*3,nrows*2))


        cnt=0
        for flav in self.FLAV:
            cnt+=1
            ax=py.subplot(nrows,ncols,cnt)
            for i in range(len(data['XF'][flav])):
                c=colors[cluster[i]]
                if c=='r':    ax.plot(X,data['XF'][flav][i],'%s-'%c,zorder=10,alpha=0.5)
                else:         ax.plot(X,data['XF'][flav][i],'%s-'%c,zorder= 0,alpha=0.5)

            ax.semilogx()
            if flav=='g' : ax.set_ylim(0,1)
            if flav=='u': ax.set_ylim(0,1)
            if flav=='ub': ax.set_ylim(0,1)
            if flav=='d': ax.set_ylim(0,1)
            if flav=='db': ax.set_ylim(0,1)
            if flav=='s' : ax.set_ylim(0,1)
            if flav=='sb': ax.set_ylim(0,1)
            ax.set_ylabel(flav)

        py.tight_layout()
        checkdir('%s/gallery'%wdir)
        #py.savefig('%s/gallery/pdf-%d.pdf'%(wdir,istep))
        py.savefig('%s/gallery/pdf-%d.pdf'%(wdir,istep))
        py.close()

    def par_plot(self,wdir,istep):

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


        replicas=self.get_replicas(wdir)
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


class PDFPI(CORE):

    def __init__(self,task,wdir,last=False):

        self.FLAV=[]
        #self.FLAV.append('g')   # 0
        self.FLAV.append('u')   # 1
        self.FLAV.append('ub')  # 2
        self.FLAV.append('d')   # 3
        self.FLAV.append('db')  # 4
        self.FLAV.append('s')   # 5
        self.FLAV.append('sb')  # 6

        if  task==0:
            self.gen_priors(wdir)
        if  task==1:
            self.plot_priors(wdir)
        if  task==2:
            self.func=self.gen_xf
            self.msg='pion.pdf.get_xf'
            self.loop_over_steps(wdir,'simple',last)
        if  task==3:
            self.func=self.plot_xf
            self.msg='pion.pdf.plot_xf'
            self.loop_over_steps(wdir,None,last)
        if  task==9:
            self.func=self.par_plot
            self.msg='pion.pdf.par_plot'
            self.loop_over_steps(wdir,None,last)

    def gen_priors(self,wdir):

        load_config('%s/input.py'%wdir)
        resman=RESMAN(nworkers=1,parallel=False,datasets=False)
        parman=resman.parman
        params=[parman.gen_flat(setup=False) for _ in range(300)]
        pdf=conf['pdfpi-']

        #--setup kinematics
        X=10**np.linspace(-4,-1,100)
        X=np.append(X,np.linspace(0.1,0.9,100))
        Q2=conf['aux'].Q02

        #--compute XF for all replicas
        XF={}
        cnt=0
        for par in params:
            cnt+=1
            lprint('%d/%d'%(cnt,len(params)))

            parman.set_new_params(par,initial=False)

            for flav in self.FLAV:
                if flav not in XF:  XF[flav]=[]

                if   flav=='u':
                     func=lambda x: pdf.get_C(x,Q2)[1]
                elif flav=='ub':
                     func=lambda x: pdf.get_C(x,Q2)[2]
                elif flav=='d':
                     func=lambda x: pdf.get_C(x,Q2)[3]
                elif flav=='db':
                     func=lambda x: pdf.get_C(x,Q2)[4]
                elif flav=='s':
                     func=lambda x: pdf.get_C(x,Q2)[5]
                elif flav=='sb':
                     func=lambda x:pdf.get_C(x,Q2)[6]

                XF[flav].append([func(x) for x in X])
        print()
        checkdir('%s/data'%wdir)
        save({'X':X,'Q2':Q2,'XF':XF},'%s/data/pion-pdf-prior.dat'%(wdir))

    def plot_priors(self,wdir):

        data=load('%s/data/pion-pdf-prior.dat'%(wdir))
        X=data['X']

        ncols=4
        nrows=len(self.FLAV)/ncols
        if len(self.FLAV)%ncols>1: nrows+=1
        nrows = int(nrows)

        fig = py.figure(figsize=(ncols*3,nrows*2))

        cnt=0
        for flav in self.FLAV:
            cnt+=1
            ax=py.subplot(nrows,ncols,cnt)
            for i in range(len(data['XF'][flav])):
                ax.plot(X,data['XF'][flav][i])
            if flav=='g' : ax.set_ylim(-1,1)
            if flav=='u': ax.set_ylim(-1,1)
            if flav=='ub': ax.set_ylim(-1,1)
            if flav=='d': ax.set_ylim(-1,1)
            if flav=='db': ax.set_ylim(-1,1)
            if flav=='s' : ax.set_ylim(-1,1)
            if flav=='sb': ax.set_ylim(-1,1)
            ax.set_ylabel(flav)

        py.tight_layout()
        checkdir('%s/gallery'%wdir)
        py.savefig('%s/gallery/pion-pdf-prior.pdf'%(wdir))
        py.close()

    def gen_xf(self,wdir,istep):
        if 'pdfpi-' not in conf['steps'][istep]['active distributions']: return
        #if istep!=7: return
        pdf=conf['pdfpi-']

        #--setup kinematics
        X=10**np.linspace(-4,-1,100)
        X=np.append(X,np.linspace(0.1,0.9,100))
        Q2=conf['aux'].Q02
        #Q2=1.0
        #Q2=80.0**2

        #--compute XF for all replicas
        XF={}
        cnt=0
        replicas=self.get_replicas(wdir)
        for replica in replicas:
            cnt+=1
            lprint('%d/%d'%(cnt,len(replicas)))

            #--retrive the parameters for current step and current replica
            #print()
            #print(len(replica['params'][istep]))
            #print(len(self.parman.par))
            #print()
            #for _ in self.parman.order: print(_)

            #--filter
            flag=False
            params=replica['params'][istep]
            order=replica['order'][istep]
            for i in range(len(order)):
                if order[i][0]!=1:continue
                if order[i][1]!='pdfpi-':continue
                #if order[i][2]=='s1 a':
                #   if params[i]<-0.9: flag=True
            if flag: continue


            self.parman.set_new_params(replica['params'][istep],initial=False)
            self.set_passive_params(istep,replica)

            for flav in self.FLAV:
                if flav not in XF:  XF[flav]=[]

                if   flav=='u':
                     func=lambda x: pdf.get_C(x,Q2)[1]
                elif flav=='ub':
                     func=lambda x: pdf.get_C(x,Q2)[2]
                elif flav=='d':
                     func=lambda x: pdf.get_C(x,Q2)[3]
                elif flav=='db':
                     func=lambda x: pdf.get_C(x,Q2)[4]
                elif flav=='s':
                     func=lambda x: pdf.get_C(x,Q2)[5]
                elif flav=='sb':
                     func=lambda x:pdf.get_C(x,Q2)[6]

                XF[flav].append([x*func(x) for x in X])

        print()
        checkdir('%s/data'%wdir)
        if Q2==conf['aux'].Q02:
            save({'X':X,'Q2':Q2,'XF':XF},'%s/data/pion-pdf-%d.dat'%(wdir,istep))
        else:
            save({'X':X,'Q2':Q2,'XF':XF},'%s/data/pion-pdf-%d-%d.dat'%(wdir,istep,int(Q2)))

    def plot_xf(self,wdir,istep):
        if 'pdfpi-' not in conf['steps'][istep]['active distributions']: return

        cluster,colors,nc,order = self.get_clusters(wdir,istep)
        #return
        data=load('%s/data/pion-pdf-%d.dat'%(wdir,istep))
        X=data['X']

        ncols=4
        nrows=len(self.FLAV)/ncols
        if len(self.FLAV)%ncols>1: nrows+=1
        nrows = int(nrows)
        fig = py.figure(figsize=(ncols*3,nrows*2))


        cnt=0
        for flav in self.FLAV:
            cnt+=1
            ax=py.subplot(nrows,ncols,cnt)
            for i in range(len(data['XF'][flav])):
                c=colors[cluster[i]]
                if c=='r':    ax.plot(X,data['XF'][flav][i],'%s-'%c,zorder=10,alpha=0.5)
                else:         ax.plot(X,data['XF'][flav][i],'%s-'%c,zorder= 0,alpha=0.5)

            ax.semilogx()
            if flav=='g' : ax.set_ylim(0,1)
            if flav=='u': ax.set_ylim(0,1)
            if flav=='ub': ax.set_ylim(0,1)
            if flav=='d': ax.set_ylim(0,1)
            if flav=='db': ax.set_ylim(0,1)
            if flav=='s' : ax.set_ylim(0,1)
            if flav=='sb': ax.set_ylim(0,1)
            ax.set_ylabel(flav)

        py.tight_layout()
        checkdir('%s/gallery'%wdir)
        #py.savefig('%s/gallery/pion-pdf-%d.pdf'%(wdir,istep))
        py.savefig('%s/gallery/pion-pdf-%d.pdf'%(wdir,istep))
        py.close()

    def par_plot(self,wdir,istep):

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

        replicas=self.get_replicas(wdir)
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

class FFKAON(CORE):
    def __init__(self,task,wdir,last=False):
        self.FLAV=[]
        #self.FLAV.append('g')
        self.FLAV.append('u')
        self.FLAV.append('d')
        self.FLAV.append('s')
        self.FLAV.append('sb')
        #self.FLAV.append('c')
        #self.FLAV.append('b')
        #self.FLAV.append('u+ub')
        #self.FLAV.append('d+db')
        #self.FLAV.append('s+sb')
        if  task==0:
            self.gen_priors(wdir)
        if  task==1:
            self.plot_priors(wdir)
        if  task==2:
            self.func=self.gen_xf
            self.msg='ffkaon.get_xf'
            self.loop_over_steps(wdir,'simple',last)
        if  task==3:
            self.func=self.plot_xf
            self.msg='ffkaon.plot_xf'
            self.loop_over_steps(wdir,None,last)
        if  task==4:
            self.func=self.par_plot
            self.msg='ffkaon.par_plot'
            self.loop_over_steps(wdir,None,last)

    def gen_priors(self,wdir):

        load_config('%s/input.py'%wdir)
        resman=RESMAN(nworkers=1,parallel=False,datasets=False)
        parman=resman.parman
        params=[parman.gen_flat(setup=False) for _ in range(300)]
        ffkaon=conf['ffk']

        #--setup kinematics
        X=10**np.linspace(-4,-1,100)
        X=np.append(X,np.linspace(0.1,0.9,100))
        Q2=conf['aux'].Q02

        #--compute XF for all replicas
        XF={}
        cnt=0
        for par in params:
            cnt+=1
            lprint('%d/%d'%(cnt,len(params)))

            parman.set_new_params(par,initial=False)

            for flav in self.FLAV:
                if flav not in XF:  XF[flav]=[]
                if  flav=='u':
                    func=lambda x: ffkaon.get_C(x,Q2)[1]
                elif flav=='ub':
                    func=lambda x: ffkaon.get_C(x,Q2)[2]
                elif flav=='d':
                    func=lambda x: ffkaon.get_C(x,Q2)[3]
                elif flav=='db':
                    func=lambda x: ffkaon.get_C(x,Q2)[4]
                elif flav=='s':
                    func=lambda x: ffkaon.get_C(x,Q2)[5]
                elif flav=='sb':
                    func=lambda x: ffkaon.get_C(x,Q2)[6]
                XF[flav].append([x*func(x) for x in X])

        print()
        checkdir('%s/data'%wdir)
        save({'X':X,'Q2':Q2,'XF':XF},'%s/data/ffkaon-prior.dat'%(wdir))

    def plot_priors(self,wdir):

        data=load('%s/data/ffkaon-prior.dat'%(wdir))
        X=data['X']

        ncols=3
        nrows=len(self.FLAV)/ncols
        if len(self.FLAV)%ncols>1: nrows+=1
        nrows = int(nrows)

        fig = py.figure(figsize=(ncols*3,nrows*2))

        cnt=0
        for flav in self.FLAV:
            cnt+=1
            ax=py.subplot(nrows,ncols,cnt)
            for i in range(len(data['XF'][flav])):
                ax.plot(X,data['XF'][flav][i])
            ax.semilogx()
            #ax.set_ylim(0,1)
            if flav=='u': ax.set_ylim(-1,1)
            if flav=='ub': ax.set_ylim(-1,1)
            if flav=='d': ax.set_ylim(-1,1)
            if flav=='db': ax.set_ylim(-1,1)
            if flav=='s' : ax.set_ylim(-1,1)
            if flav=='sb': ax.set_ylim(-1,1)
            ax.set_ylabel(flav)

        py.tight_layout()
        checkdir('%s/gallery'%wdir)
        py.savefig('%s/gallery/ffkaon-prior.pdf'%(wdir))
        py.close()


    def gen_xf(self,wdir,istep):
        if 'ffk' not in conf['steps'][istep]['active distributions']: return
        ffkaon=conf['ffk']

        #--setup kinematics
        X=np.linspace(0.01,0.99,100)
        Q2=conf['aux'].Q02

        #--compute XF for all replicas
        XF={}
        cnt=0
        replicas=self.get_replicas(wdir)
        for replica in replicas:
            cnt+=1
            lprint('%d/%d'%(cnt,len(replicas)))

            self.parman.set_new_params(replica['params'][istep],initial=False)
            self.set_passive_params(istep,replica)

            for flav in self.FLAV:
                if flav not in XF: XF[flav]=[]
                if  flav=='u':
                    func=lambda x: ffkaon.get_C(x,Q2)[1]
                elif flav=='ub':
                    func=lambda x: ffkaon.get_C(x,Q2)[2]
                elif flav=='d':
                    func=lambda x: ffkaon.get_C(x,Q2)[3]
                elif flav=='db':
                    func=lambda x: ffkaon.get_C(x,Q2)[4]
                elif flav=='s':
                    func=lambda x: ffkaon.get_C(x,Q2)[5]
                elif flav=='sb':
                    func=lambda x: ffkaon.get_C(x,Q2)[6]
                XF[flav].append([x*func(x) for x in X])
        print()
        checkdir('%s/data'%wdir)
        save({'X':X,'Q2':Q2,'XF':XF},'%s/data/ffkaon-%d.dat'%(wdir,istep))

    def plot_xf(self,wdir,istep):
        if 'ffk' not in conf['steps'][istep]['active distributions']: return

        cluster,colors,nc,order = self.get_clusters(wdir,istep)
        data=load('%s/data/ffkaon-%d.dat'%(wdir,istep))
        X=data['X']

        ncols=3
        nrows=len(self.FLAV)/ncols
        if len(self.FLAV)%ncols>0: nrows+=1
        nrows = int(nrows)
        fig = py.figure(figsize=(ncols*3,nrows*2))

        cnt=0
        for flav in self.FLAV:
            cnt+=1
            ax=py.subplot(nrows,ncols,cnt)
            for i in range(len(data['XF'][flav])):
                c=colors[cluster[i]]
                if c=='r':    ax.plot(X,data['XF'][flav][i],'%s-'%c,zorder=10,alpha=0.1)
                else:         ax.plot(X,data['XF'][flav][i],'%s-'%c,zorder= 0,alpha=0.1)

            #ax.semilogx()
            ax.set_ylim(0,0.6)
            if flav=='g': ax.set_ylim(0,0.75)
            if flav=='u': ax.set_ylim(0,0.75)
            if flav=='ub': ax.set_ylim(0,0.75)
            if flav=='d': ax.set_ylim(0,0.75)
            if flav=='db': ax.set_ylim(0,0.75)
            if flav=='s' : ax.set_ylim(0,0.75)
            if flav=='sb': ax.set_ylim(0,0.75)
            ax.set_ylabel(flav)

        py.tight_layout()
        checkdir('%s/gallery'%wdir)
        py.savefig('%s/gallery/ffkaon-%d.pdf'%(wdir,istep))
        py.close()

    def par_plot(self,wdir,istep):
        #self.FLAV.append('g')
        #self.FLAV.append('u')
        #self.FLAV.append('d')
        #self.FLAV.append('s')
        #self.FLAV.append('sb')
        #self.FLAV.append('c')
        #self.FLAV.append('b')

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

        replicas=self.get_replicas(wdir)
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

class FFPION(CORE):

    def __init__(self,task,wdir,last=False):

        self.FLAV=[]
        #self.FLAV.append('g')   # 0
        self.FLAV.append('u')   # 1
        self.FLAV.append('ub')  # 2
        self.FLAV.append('d')   # 3
        self.FLAV.append('db')  # 4
        self.FLAV.append('s')   # 5
        self.FLAV.append('sb')  # 6

        if  task==0:
            self.gen_priors(wdir)
        if  task==1:
            self.plot_priors(wdir)
        if  task==2:
            self.func=self.gen_xf
            self.msg='ffpion.gen_xf'
            self.loop_over_steps(wdir,'simple',last)
        if  task==3:
            self.func=self.plot_xf
            self.msg='ffpion.plot_xf'
            self.loop_over_steps(wdir,None,last)
        if  task==4:
            self.func=self.par_plot
            self.msg='ffpion.par_plot'
            self.loop_over_steps(wdir,None,last)

    def gen_priors(self,wdir):

        load_config('%s/input.py'%wdir)
        resman=RESMAN(nworkers=1,parallel=False,datasets=False)
        parman=resman.parman
        params=[parman.gen_flat(setup=False) for _ in range(300)]
        ffpion=conf['ffpi']

        #--setup kinematics
        X=10**np.linspace(-4,-1,100)
        X=np.append(X,np.linspace(0.1,0.9,100))
        Q2=conf['aux'].Q02

        #--compute XF for all replicas
        XF={}
        cnt=0
        for par in params:
            cnt+=1
            lprint('%d/%d'%(cnt,len(params)))

            parman.set_new_params(par,initial=False)

            for flav in self.FLAV:
                if flav not in XF:  XF[flav]=[]
                if  flav=='u':
                    func=lambda x: ffpion.get_C(x,Q2)[1]
                elif flav=='ub':
                    func=lambda x: ffpion.get_C(x,Q2)[2]
                elif flav=='d':
                    func=lambda x: ffpion.get_C(x,Q2)[3]
                elif flav=='db':
                    func=lambda x: ffpion.get_C(x,Q2)[4]
                elif flav=='s':
                    func=lambda x: ffpion.get_C(x,Q2)[5]
                elif flav=='sb':
                    func=lambda x: ffpion.get_C(x,Q2)[6]
                XF[flav].append([x*func(x) for x in X])

        print()
        checkdir('%s/data'%wdir)
        save({'X':X,'Q2':Q2,'XF':XF},'%s/data/ffpion-prior.dat'%(wdir))

    def plot_priors(self,wdir):

        data=load('%s/data/ffpion-prior.dat'%(wdir))
        X=data['X']

        ncols=3
        nrows=len(self.FLAV)/ncols
        if len(self.FLAV)%ncols>1: nrows+=1
        nrows = int(nrows)

        fig = py.figure(figsize=(ncols*3,nrows*2))

        cnt=0
        for flav in self.FLAV:
            cnt+=1
            ax=py.subplot(nrows,ncols,cnt)
            for i in range(len(data['XF'][flav])):
                ax.plot(X,data['XF'][flav][i])
            ax.semilogx()
            #ax.set_ylim(0,1)
            if flav=='u': ax.set_ylim(-1,1)
            if flav=='ub': ax.set_ylim(-1,1)
            if flav=='d': ax.set_ylim(-1,1)
            if flav=='db': ax.set_ylim(-1,1)
            if flav=='s' : ax.set_ylim(-1,1)
            if flav=='sb': ax.set_ylim(-1,1)
            ax.set_ylabel(flav)

        py.tight_layout()
        checkdir('%s/gallery'%wdir)
        py.savefig('%s/gallery/ffpion-prior.pdf'%(wdir))
        py.close()

    def gen_xf(self,wdir,istep):
        if 'ffpi' not in conf['steps'][istep]['active distributions']: return
        ffpion=conf['ffpi']

        #--setup kinematics
        X=np.linspace(0.01,0.99,100)
        Q2=conf['aux'].Q02

        #--compute XF for all replicas
        XF={}
        cnt=0
        replicas=self.get_replicas(wdir)
        for replica in replicas:
            cnt+=1
            lprint('%d/%d'%(cnt,len(replicas)))

            self.parman.set_new_params(replica['params'][istep],initial=False)
            self.set_passive_params(istep,replica)

            for flav in self.FLAV:
                if flav not in XF: XF[flav]=[]
                if  flav=='u':
                    func=lambda x: ffpion.get_C(x,Q2)[1]
                elif flav=='ub':
                    func=lambda x: ffpion.get_C(x,Q2)[2]
                elif flav=='d':
                    func=lambda x: ffpion.get_C(x,Q2)[3]
                elif flav=='db':
                    func=lambda x: ffpion.get_C(x,Q2)[4]
                elif flav=='s':
                    func=lambda x: ffpion.get_C(x,Q2)[5]
                elif flav=='sb':
                    func=lambda x: ffpion.get_C(x,Q2)[6]
                XF[flav].append([x*func(x) for x in X])
        print()
        checkdir('%s/data'%wdir)
        save({'X':X,'Q2':Q2,'XF':XF},'%s/data/ffpion-%d.dat'%(wdir,istep))

    def plot_xf(self,wdir,istep):
        if 'ffpi' not in conf['steps'][istep]['active distributions']: return

        cluster,colors,nc,order = self.get_clusters(wdir,istep)
        data=load('%s/data/ffpion-%d.dat'%(wdir,istep))
        X=data['X']

        ncols=3
        nrows=len(self.FLAV)/ncols
        if len(self.FLAV)%ncols>0: nrows+=1
        nrows = int(nrows)
        fig = py.figure(figsize=(ncols*3,nrows*2))

        cnt=0
        for flav in self.FLAV:
            cnt+=1
            ax=py.subplot(nrows,ncols,cnt)
            for i in range(len(data['XF'][flav])):
                c=colors[cluster[i]]
                if c=='r':    ax.plot(X,data['XF'][flav][i],'%s-'%c,zorder=10,alpha=0.1)
                else:         ax.plot(X,data['XF'][flav][i],'%s-'%c,zorder= 0,alpha=0.1)

            ax.semilogx()
            #ax.set_ylim(0,1)
            if flav=='g': ax.set_ylim(0,1.5)
            if flav=='u': ax.set_ylim(0,1.5)
            if flav=='ub': ax.set_ylim(0,1.5)
            if flav=='d': ax.set_ylim(0,1.5)
            if flav=='db': ax.set_ylim(0,1.5)
            if flav=='s' : ax.set_ylim(0,1.5)
            if flav=='sb': ax.set_ylim(0,1.5)
            ax.set_ylabel(flav)

        py.tight_layout()
        checkdir('%s/gallery'%wdir)
        py.savefig('%s/gallery/ffpion-%d.pdf'%(wdir,istep))
        py.close()

    def par_plot(self,wdir,istep):
        #self.FLAV.append('g')   # 0
        #self.FLAV.append('u')   # 1
        #self.FLAV.append('ub')  # 2
        #self.FLAV.append('d')   # 3
        #self.FLAV.append('db')  # 4
        #self.FLAV.append('s')   # 5
        #self.FLAV.append('sb')  # 6

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


        replicas=self.get_replicas(wdir)
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

class SIVERS(CORE):

    def __init__(self,task,wdir,last=False):

        self.FLAV=[]
        #self.FLAV.append('g')   # 0
        self.FLAV.append('u')   # 1
        self.FLAV.append('ub')  # 2
        self.FLAV.append('d')   # 3
        self.FLAV.append('db')  # 4
        #self.FLAV.append('s')   # 5
        #self.FLAV.append('sb')  # 6

        if  task==0:
            self.gen_priors(wdir)
        if  task==1:
            self.plot_priors(wdir)
        if  task==2:
            self.func=self.gen_xf
            self.msg='sivers.get_xf'
            self.loop_over_steps(wdir,'simple',last)
        if  task==3:
            self.func=self.plot_xf
            self.msg='sivers.plot_xf'
            self.loop_over_steps(wdir,None,last)
        if  task==4:
            self.func=self.plot_xf_relerr
            self.msg='sivers.plot_xf_relerr'
            self.loop_over_steps(wdir,None,last)
        if  task==5:
            self.func=self.gen_fkT
            self.msg='sivers.gen_fkT'
            self.loop_over_steps(wdir,'simple',last)
        if  task==6:
            self.func=self.plot_fkT
            self.msg='sivers.plot_fkT'
            self.loop_over_steps(wdir,None,last)
        if  task==9:
            self.func=self.par_plot
            self.msg='sivers.par_plot'
            self.loop_over_steps(wdir,None,last)

    def gen_priors(self,wdir):

        load_config('%s/input.py'%wdir)
        resman=RESMAN(nworkers=1,parallel=False,datasets=False)
        parman=resman.parman
        params=[parman.gen_flat(setup=False) for _ in range(300)]
        pdf=conf['sivers']

        #--setup kinematics
        X=10**np.linspace(-4,-1,100)
        X=np.append(X,np.linspace(0.01,0.99,100))
        Q2=conf['aux'].Q02

        #--compute XF for all replicas
        XF={}
        cnt=0
        for par in params:
            cnt+=1
            lprint('%d/%d'%(cnt,len(params)))

            parman.set_new_params(par,initial=False)

            for flav in self.FLAV:
                if flav not in XF:  XF[flav]=[]

                if   flav=='u':
                     func=lambda x: pdf.get_C(x,Q2)[1]
                elif flav=='ub':
                     func=lambda x: pdf.get_C(x,Q2)[2]
                elif flav=='d':
                     func=lambda x: pdf.get_C(x,Q2)[3]
                elif flav=='db':
                     func=lambda x: pdf.get_C(x,Q2)[4]
                elif flav=='s':
                     func=lambda x: pdf.get_C(x,Q2)[5]
                elif flav=='sb':
                     func=lambda x:pdf.get_C(x,Q2)[6]

                XF[flav].append([func(x) for x in X])
        print()
        checkdir('%s/data'%wdir)
        save({'X':X,'Q2':Q2,'XF':XF},'%s/data/sivers-prior.dat'%(wdir))

    def plot_priors(self,wdir):

        data=load('%s/data/sivers-prior.dat'%(wdir))
        X=data['X']

        ncols=4
        nrows=len(self.FLAV)/ncols
        if len(self.FLAV)%ncols>1: nrows+=1
        nrows = int(nrows)

        fig = py.figure(figsize=(ncols*3,nrows*2))

        cnt=0
        for flav in self.FLAV:
            cnt+=1
            ax=py.subplot(nrows,ncols,cnt)
            for i in range(len(data['XF'][flav])):
                ax.plot(X,data['XF'][flav][i])
            if flav=='g' : ax.set_ylim(-1, 1)
            if flav=='u': ax.set_ylim(-1, 1)
            if flav=='ub': ax.set_ylim(-1, 1)
            if flav=='d': ax.set_ylim(-1, 1)
            if flav=='db': ax.set_ylim(-1,1)
            if flav=='s' : ax.set_ylim(-1, 1)
            if flav=='sb': ax.set_ylim(-1, 1)
            ax.set_ylabel(flav)

        py.tight_layout()
        checkdir('%s/gallery'%wdir)
        py.savefig('%s/gallery/sivers-prior.pdf'%(wdir))
        py.close()

    def gen_xf(self,wdir,istep):
        if 'sivers' not in conf['steps'][istep]['active distributions']: return
        #if istep!=7: return
        pdf=conf['sivers']

        #--setup kinematics
        X1=10**np.linspace(-4,-2,100)
        X2=np.linspace(0.011,0.99,100)
        X=np.concatenate([X1,X2])
        #Q2=conf['aux'].Q02
        #Q2=10.0
        #Q2=80.0**2
        Q2array=[2,4,10,100]

        #--compute XF for all replicas
        for Q2 in Q2array:
            XF={}
            cnt=0
            replicas=self.get_replicas(wdir)
            for replica in replicas:
                cnt+=1
                lprint('%d/%d'%(cnt,len(replicas)))

                #--retrive the parameters for current step and current replica
                #print()
                #print(len(replica['params'][istep]))
            	#print(len(self.parman.par))
            	#print()
            	#for _ in self.parman.order: print(_)

            	#--filter
                flag=False
                params=replica['params'][istep]
                order=replica['order'][istep]
                for i in range(len(order)):
                    if order[i][0]!=1:continue
                    if order[i][1]!='pdf':continue
                    #if order[i][2]=='s1 a':
                    #   if params[i]<-0.9: flag=True
                if flag: continue
                    

                self.parman.set_new_params(replica['params'][istep],initial=False)
                self.set_passive_params(istep,replica)
                

                for flav in self.FLAV:
                    if flav not in XF:  XF[flav]=[]

                    if   flav=='u':
                         func=lambda x: pdf.get_C(x,Q2)[1]
                    elif flav=='ub':
                         func=lambda x: pdf.get_C(x,Q2)[2]
                    elif flav=='d':
                         func=lambda x:pdf.get_C(x,Q2)[3]
                    elif flav=='db':
                         func=lambda x: pdf.get_C(x,Q2)[4]
                    elif flav=='s':
                         func=lambda x: pdf.get_C(x,Q2)[5]
                    elif flav=='sb':
                         func=lambda x: pdf.get_C(x,Q2)[6]

                    XF[flav].append([x*func(x) for x in X])
            print()
            checkdir('%s/data'%wdir)
            if Q2==conf['aux'].Q02:
                save({'X':X,'Q2':Q2,'XF':XF},'%s/data/sivers-%d.dat'%(wdir,istep))
            else:
                save({'X':X,'Q2':Q2,'XF':XF},'%s/data/sivers-%d-%d.dat'%(wdir,istep,int(Q2)))

    def plot_xf(self,wdir,istep):
        if 'sivers' not in conf['steps'][istep]['active distributions']: return

        cluster,colors,nc,order = self.get_clusters(wdir,istep)
        #return
        
        replicas=self.get_replicas(wdir)

        Q2array=[2,4,10,100]
        for Q2 in Q2array:
            if Q2==2:
                data1=load('%s/data/sivers-%d.dat'%(wdir,istep))
                save(data1,'%s/npdata/sivers_reps-Q2-%d.dat'%(wdir,int(Q2)))
                #np.save('%s/nparrays/sivers-Q2-%d.npy'%(wdir,int(Q2)),data1)
            elif Q2==10:
                data2=load('%s/data/sivers-%d-%d.dat'%(wdir,istep,int(Q2)))
                save(data2,'%s/npdata/sivers_reps-Q2-%d.dat'%(wdir,int(Q2)))
                #np.save('%s/nparrays/sivers-Q2-%d.npy'%(wdir,int(Q2)),data2)
            elif Q2==100:
                data3=load('%s/data/sivers-%d-%d.dat'%(wdir,istep,int(Q2)))
                save(data3,'%s/npdata/sivers_reps-Q2-%d.dat'%(wdir,int(Q2)))
                #np.save('%s/nparrays/sivers-Q2-%d.npy'%(wdir,int(Q2)),data3)
            elif Q2==4:
                data4=load('%s/data/sivers-%d-%d.dat'%(wdir,istep,int(Q2)))
                save(data4,'%s/npdata/sivers_reps-Q2-%d.dat'%(wdir,int(Q2)))
                #np.save('%s/nparrays/sivers-Q2-%d.npy'%(wdir,int(Q2)),data3)

        X=data1['X']

        ncols=2
        nrows=len(self.FLAV)/ncols
        if len(self.FLAV)%ncols>1: nrows+=1
        nrows = int(nrows)
        fig = py.figure(figsize=(ncols*3,nrows*2))

        cnt=0

           #for flav in self.FLAV:
           #cnt+=1
           #ax=py.subplot(nrows,ncols,cnt)
           #for i in range(len(data['XF'][flav])):
               #c=colors[cluster[i]]
               #if c=='r':    ax.plot(X,data['XF'][flav][i],'%s-'%c,zorder=10,alpha=0.1)
               #else:         ax.plot(X,data['XF'][flav][i],'%s-'%c,zorder= 0,alpha=0.1)
        rand_list=[]
        for j in range(100):
            rand_list.append(random.randint(0, 950))
        for flav in self.FLAV:
            cnt+=1
            ax=py.subplot(nrows,ncols,cnt)
            _data1=data1['XF'][flav]
            _data2=data2['XF'][flav]
            _data3=data3['XF'][flav]
            _data4=data4['XF'][flav]
            for Q2 in Q2array:
                if Q2==4:
                    for i in range(len(replicas)):
                        ax.plot(X,data4['XF'][flav][i],color='y',zorder= 0,alpha=0.3)
                    c='r'
                    meanXF2 = np.mean(_data4,axis=0)
                    stdXF2 = np.std(_data4,axis=0)

                    XFarray=[X,meanXF2,stdXF2]
                    save(XFarray,'%s/npdata/sivers-%s-Q2-%d.dat'%(wdir,flav,int(Q2)))

                    lower2=meanXF2-stdXF2
                    upper2=meanXF2+stdXF2
                    ax.plot(X,meanXF2,'%s-'%c,zorder=10,alpha=0.3)
                    ax.fill_between(X, lower2, upper2,color=c,alpha=0.3)
                elif Q2==10:
                    c='b'
                    meanXF2 = np.mean(_data2,axis=0)
                    #stdXF2 = np.std(_data2,axis=0)

                    #XFarray=[X,meanXF2,stdXF2]
                    #save(XFarray,'%s/npdata/sivers-%s-Q2-%d.dat'%(wdir,flav,int(Q2)))

                    #lower2=meanXF2-stdXF2
                    #upper2=meanXF2+stdXF2
                    ax.plot(X,meanXF2,'%s-'%c,zorder=10,alpha=0.5)
                    #ax.fill_between(X, lower2, upper2,color=c,alpha=0.5)
            
                elif Q2==100:
                    c='g'
                    meanXF3 = np.mean(_data3,axis=0)
                    #stdXF3 = np.std(_data3,axis=0)
                    #lower3=meanXF3-stdXF3
                    #upper3=meanXF3+stdXF3
                    ax.plot(X,meanXF3,'%s-'%c,zorder=10,alpha=0.5)
                    #ax.fill_between(X, lower3, upper3,color=c,alpha=0.5)
                    #for i in rand_list:
                    #    ax.plot(X,data['XF'][flav][i],color='b',zorder= 0,alpha=0.1)

            #ax.semilogx()
            #if flav=='g' : ax.set_ylim(-0.03,0.03)
            if flav=='ub': ax.set_ylim(-0.05,0.05)
            if flav=='db': ax.set_ylim(-0.05,0.05)
            #ax.set_ylim(-0.1,0.1)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylabel('$x(\pi\, F_{FT}^{%s}(x,x))$'%(flav))
            ax.set_xlabel('$x$')

        py.tight_layout()
        checkdir('%s/gallery'%wdir)
        #py.savefig('%s/gallery/pdf-%d.pdf'%(wdir,istep))
        py.savefig('%s/gallery/sivers-%d.pdf'%(wdir,istep))
        py.close()
        
    def gen_fkT(self,wdir,istep):
        if 'sivers' not in conf['steps'][istep]['active distributions']: return
        #if istep!=7: return
        pdf=conf['sivers']
        
        x=0.1

        #--setup kinematics
        KT=np.linspace(0,2,100)
        #Q2=conf['aux'].Q02
        #Q2=10.0
        #Q2=80.0**2
        Q2array=[2,4,10,100]

        #--compute XF for all replicas
        for Q2 in Q2array:
            FkT={}
            cnt=0
            replicas=self.get_replicas(wdir)
            for replica in replicas:
                cnt+=1
                lprint('%d/%d'%(cnt,len(replicas)))

                #--retrive the parameters for current step and current replica
                #print()
                #print(len(replica['params'][istep]))
            	#print(len(self.parman.par))
            	#print()
            	#for _ in self.parman.order: print(_)

            	#--filter
                flag=False
                params=replica['params'][istep]
                order=replica['order'][istep]
                for i in range(len(order)):
                    if order[i][0]!=1:continue
                    if order[i][1]!='pdf':continue
                    #if order[i][2]=='widths1_sea':
                    #    if params[i]<0.1 or params[i]>1.0: flag=True
            	#if flag: continue


                self.parman.set_new_params(replica['params'][istep],initial=False)
                self.set_passive_params(istep,replica)
                
                widths=pdf.get_widths(Q2)
                _pdf = pdf.get_C(x,Q2)
                M = 0.938

                for flav in self.FLAV:
                    if flav not in FkT:  FkT[flav]=[]

                    if   flav=='u':
                         func=lambda kT: 2.0*M**2*_pdf[1]*np.exp(-kT**2/widths[1])/np.pi/widths[1]**2
                    elif flav=='ub':
                         func=lambda kT: 2.0*M**2*_pdf[2]*np.exp(-kT**2/widths[2])/np.pi/widths[2]**2
                    elif flav=='d':
                         func=lambda kT: 2.0*M**2*_pdf[3]*np.exp(-kT**2/widths[3])/np.pi/widths[3]**2
                    elif flav=='db':
                         func=lambda kT: 2.0*M**2*_pdf[4]*np.exp(-kT**2/widths[4])/np.pi/widths[4]**2
                    elif flav=='s':
                         func=lambda kT: 2.0*M**2*_pdf[5]*np.exp(-kT**2/widths[5])/np.pi/widths[5]**2
                    elif flav=='sb':
                         func=lambda kT: 2.0*M**2*_pdf[6]*np.exp(-kT**2/widths[6])/np.pi/widths[6]**2

                    FkT[flav].append([func(kT) for kT in KT])
            print()
            checkdir('%s/data'%wdir)
            if Q2==conf['aux'].Q02:
                save({'KT':KT,'Q2':Q2,'FkT':FkT},'%s/data/siverskT-%d.dat'%(wdir,istep))
            else:
                save({'KT':KT,'Q2':Q2,'FkT':FkT},'%s/data/siverskT-%d-%d.dat'%(wdir,istep,int(Q2)))
                
    def plot_fkT(self,wdir,istep):
        if 'sivers' not in conf['steps'][istep]['active distributions']: return

        cluster,colors,nc,order = self.get_clusters(wdir,istep)
        #return
        
        replicas=self.get_replicas(wdir)

        Q2array=[2,4,10,100]
        for Q2 in Q2array:
            if Q2==2:
                data1=load('%s/data/siverskT-%d.dat'%(wdir,istep))
                save(data1,'%s/npdata/siverskT_reps-Q2-%d.dat'%(wdir,int(Q2)))
                #np.save('%s/nparrays/sivers-Q2-%d.npy'%(wdir,int(Q2)),data1)
            elif Q2==10:
                data2=load('%s/data/siverskT-%d-%d.dat'%(wdir,istep,int(Q2)))
                save(data2,'%s/npdata/siverskT_reps-Q2-%d.dat'%(wdir,int(Q2)))
                #np.save('%s/nparrays/sivers-Q2-%d.npy'%(wdir,int(Q2)),data2)
            elif Q2==100:
                data3=load('%s/data/siverskT-%d-%d.dat'%(wdir,istep,int(Q2)))
                save(data3,'%s/npdata/siverskT_reps-Q2-%d.dat'%(wdir,int(Q2)))
                #np.save('%s/nparrays/sivers-Q2-%d.npy'%(wdir,int(Q2)),data3)
            elif Q2==4:
                data4=load('%s/data/siverskT-%d-%d.dat'%(wdir,istep,int(Q2)))
                save(data4,'%s/npdata/siverskT_reps-Q2-%d.dat'%(wdir,int(Q2)))
                #np.save('%s/nparrays/sivers-Q2-%d.npy'%(wdir,int(Q2)),data3)

        KT=data1['KT']

        ncols=2
        nrows=len(self.FLAV)/ncols
        if len(self.FLAV)%ncols>1: nrows+=1
        nrows = int(nrows)
        fig = py.figure(figsize=(ncols*3,nrows*2))

        cnt=0

           #for flav in self.FLAV:
           #cnt+=1
           #ax=py.subplot(nrows,ncols,cnt)
           #for i in range(len(data['XF'][flav])):
               #c=colors[cluster[i]]
               #if c=='r':    ax.plot(X,data['XF'][flav][i],'%s-'%c,zorder=10,alpha=0.1)
               #else:         ax.plot(X,data['XF'][flav][i],'%s-'%c,zorder= 0,alpha=0.1)
        rand_list=[]
        for j in range(100):
            rand_list.append(random.randint(0, 950))
        for flav in self.FLAV:
            cnt+=1
            ax=py.subplot(nrows,ncols,cnt)
            _data1=data1['FkT'][flav]
            _data2=data2['FkT'][flav]
            _data3=data3['FkT'][flav]
            _data4=data4['FkT'][flav]
            for Q2 in Q2array:
                if Q2==4:
                    for i in range(len(replicas)):
                        ax.plot(KT,np.abs(data4['FkT'][flav][i]),color='y',zorder= 0,alpha=0.3)
                    c='r'
                    meanXF2 = np.mean(_data4,axis=0)
                    stdXF2 = np.std(_data4,axis=0)

                    XFarray=[KT,meanXF2,stdXF2]
                    save(XFarray,'%s/npdata/siverskT-%s-Q2-%d.dat'%(wdir,flav,int(Q2)))

                    lower2=meanXF2-stdXF2
                    upper2=meanXF2+stdXF2
                    ax.plot(KT,meanXF2,'%s-'%c,zorder=10,alpha=0.3)
                    ax.fill_between(KT, np.abs(lower2), np.abs(upper2),color=c,alpha=0.3)
                elif Q2==10:
                    c='b'
                    meanXF2 = np.mean(_data2,axis=0)
                    #stdXF2 = np.std(_data2,axis=0)

                    #XFarray=[X,meanXF2,stdXF2]
                    #save(XFarray,'%s/npdata/sivers-%s-Q2-%d.dat'%(wdir,flav,int(Q2)))

                    #lower2=meanXF2-stdXF2
                    #upper2=meanXF2+stdXF2
                    ax.plot(KT,np.abs(meanXF2),'%s-'%c,zorder=10,alpha=0.5)
                    #ax.fill_between(X, lower2, upper2,color=c,alpha=0.5)
            
                elif Q2==100:
                    c='g'
                    meanXF3 = np.mean(_data3,axis=0)
                    #stdXF3 = np.std(_data3,axis=0)
                    #lower3=meanXF3-stdXF3
                    #upper3=meanXF3+stdXF3
                    ax.plot(KT,np.abs(meanXF3),'%s-'%c,zorder=10,alpha=0.5)
                    #ax.fill_between(X, lower3, upper3,color=c,alpha=0.5)
                    #for i in rand_list:
                    #    ax.plot(X,data['XF'][flav][i],color='b',zorder= 0,alpha=0.1)

            #ax.semilogx()
            #if flav=='g' : ax.set_ylim(-0.03,0.03)
            #ax.set_ylim(-0.1,0.1)
            ax.set_xlim(0.0, 2.0)
            ax.set_ylim(1e-4,10.0)
            if flav=='u': ax.set_ylabel('$-f_{1T}^{\perp %s}(x,k_T)$'%(flav))
            elif flav=='d': ax.set_ylabel('$f_{1T}^{\perp %s}(x,k_T)$'%(flav))
            ax.set_xlabel('$k_T$')
            ax.semilogy()

        py.tight_layout()
        checkdir('%s/gallery'%wdir)
        #py.savefig('%s/gallery/pdf-%d.pdf'%(wdir,istep))
        py.savefig('%s/gallery/siverskT-%d.pdf'%(wdir,istep))
        py.close()

    def plot_xf_relerr(self,wdir,istep):
        if 'sivers' not in conf['steps'][istep]['active distributions']: return

        cluster,colors,nc,order = self.get_clusters(wdir,istep)
        #return

        Q2=10
        data=load('%s/data/sivers-%d-%d.dat'%(wdir,istep,int(Q2)))

        X=data['X']

        ncols=2
        nrows=len(self.FLAV)/ncols
        if len(self.FLAV)%ncols>1: nrows+=1
        nrows = int(nrows)
        fig = py.figure(figsize=(ncols*3,nrows*2))

        cnt=0

        for flav in self.FLAV:
            cnt+=1
            ax=py.subplot(nrows,ncols,cnt)
            _data=data['XF'][flav]
            c='r'
            meanXF = np.mean(_data,axis=0)
            stdXF = np.std(_data,axis=0)

            _interpXF=interpolate.splrep(X,meanXF)
            interpXF=lambda x: interpolate.splev(x,_interpXF)

            _dinterpXF=interpolate.splrep(X,meanXF+stdXF)
            dinterpXF=lambda x: interpolate.splev(x,_dinterpXF)

            momXF=sp.integrate.quad(interpXF,0,1)[0]
            dmomXF=sp.integrate.quad(dinterpXF,0,1)[0]

            mom_err = np.abs((dmomXF-momXF)/momXF)

            print('sivers %s moment rel err=%0.2f'%(flav,mom_err))

            ax.plot(X,np.abs(stdXF/meanXF),'%s-'%c,zorder=10,alpha=0.5)

            if flav=='u': ax.set_ylim(0,0.5)
            #if flav=='ub': ax.set_ylim(-0.05,0.05)
            if flav=='d': ax.set_ylim(0,0.5)
            #if flav=='db': ax.set_ylim(-0.05,0.05)
            #if flav=='s' : ax.set_ylim(-0.05,0.05)
            #if flav=='sb': ax.set_ylim(-0.05,0.05)
            #ax.set_ylim(-0.1,0.1)
            ax.set_xlim(0.0, 0.3)
            ax.set_ylabel(r'$|\Delta F^{%s}_{FT}/F^{%s}_{FT}|$'%(flav,flav))
            ax.set_xlabel('$x$')

        py.tight_layout()
        checkdir('%s/gallery'%wdir)
        #py.savefig('%s/gallery/pdf-%d.pdf'%(wdir,istep))
        py.savefig('%s/gallery/sivers-relerr-%d.pdf'%(wdir,istep))
        py.close()

    def par_plot(self,wdir,istep):

        par=[]
        #par.append(['uv1 N 1', 'uv1 a 1', 'uv1 b 1'])
        #par.append(['u N1 1', 'u a1 1', 'u b1 1'])
        #par.append(['ub N0 1', 'ub a0 1', 'ub b0 1'])
        #par.append(['dv1 N 1', 'dv1 a 1', 'dv1 b 1'])
        #par.append(['d N1 1', 'd a1 1', 'd b1 1'])
        #par.append(['db N0 1','db a0 1','db b0 1'])
        #par.append(['s1 N 1', 's1 a 1', 's1 b 1'])
        #par.append(['s N1 1', 's a1 1', 's b1 1'])
        #par.append(['sb N0 1', 'sb a0 1', 'sb b0 1'])
        #par.append(['widths1_uv', 'widths1_dv'])
        #par.append(['widths1_uv','widths1_sea'])
        
        par.append(['u N0 1', 'u a0 1', 'u b0 1']) #,'u c0 1'])
        par.append(['u N1 1', 'u a1 1', 'u b1 1'])
        par.append(['u N0 2', 'u a0 2', 'u b0 2']) 
        par.append(['u N1 2', 'u a1 2', 'u b1 2'])
        par.append(['u c0 1', 'u d0 1'])
        par.append(['u c1 1', 'u d1 1'])
        #par.append(['ub N0 1', 'ub a0, 1', 'ub b0 1'])
        par.append(['d N0 1', 'd a0 1', 'd b0 1']) #,'d c0 1'])
        par.append(['d N1 1', 'd a1 1', 'd b1 1'])
        par.append(['d N0 2', 'd a0 2', 'd b0 2']) 
        par.append(['d N1 2', 'd a1 2', 'd b1 2'])
        par.append(['d c0 1', 'd d0 1'])
        par.append(['d c1 1', 'd d1 1'])
        #par.append(['db N0 1','db a0 1','db b0 1'])
        par.append(['ub N0 1', 'ub a0 1', 'ub b0 1'])
        par.append(['db N0 1', 'db a0 1', 'db b0 1'])
        #par.append(['sb N0 1', 'sb a0 1', 'sb b0 1'])
        par.append(['widths1_uv', 'widths1_sea'])
        #par.append(['widths1_uv'])

        kind=1
        tag='sivers'

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


        replicas=self.get_replicas(wdir)
        for j in range(len(replicas)):
            #for replica in replicas:
            replica=replicas[j]
            #if cluster[j]!=0: continue
            #print(replica)
            params=replica['params'][istep]

            #if params[5] < 0.25 or params[5] > 1.5: continue #filter on par values
            order=replica['order'][istep]
            for i in range(len(order)):
                if kind != order[i][0]:continue
                if tag  != order[i][1]:continue
                for _ in data[cluster[j]]:
                    if  _ ==order[i][2]:
                        data[cluster[j]][_].append(params[i])

        data=pd.DataFrame(data)

        widths1_uvMean = np.mean(data[0]['widths1_uv'])
        widths1_uvSTD = np.std(data[0]['widths1_uv'])
        meanuv = "{:.6f}".format(widths1_uvMean)
        stduv="{:.6f}".format(widths1_uvSTD)

        widths1_seaMean = np.mean(data[0]['widths1_sea'])
        widths1_seaSTD = np.std(data[0]['widths1_sea'])
        meansea = "{:.6f}".format(widths1_seaMean)
        stdsea="{:.6f}".format(widths1_seaSTD)

        #data=data[data['s1 a'] >-1]#data.query("'s1 a'>-0.9")
        nrows,ncols=len(par),3
        fig = py.figure(figsize=(ncols*3,nrows*1.5))
        cnt=0
        for row in par:
            for _ in row:
                cnt+=1
                if _==None: continue
                #if _== 'u N0 1': R = (-1,0)
                #elif _== 'd N0 1': R = (0,2.5)
                #elif _== 'u a0 1': R = (0,0.2)
                #elif _== 'd a0 1': R = (0,0.05)
                #elif _== 'u b0 1': R = (4.5,6)
                #elif _== 'd b0 1': R = (7.5,20)
                else: R=None
                ax=py.subplot(nrows,ncols,cnt)
                for i in data:
                    if i==0: c='r'
                    if i==1: c='b'
                    if i==2: c='g'
                    if i==3: c='m'

                    ax.hist(data[i][_],bins=50,color=c,range=R,histtype='step')

                if '_' in _: _ = _.replace("_", " ")
                ax.set_xlabel(_)

        fig.text(1.2,0.7,'$Mean Valence: %s$'%meanuv,transform=ax.transAxes)
        fig.text(1.2,0.5,'$Error Valence: %s$'%stduv,transform=ax.transAxes)
        fig.text(1.2,0.3,'$Mean Sea: %s$'%meansea,transform=ax.transAxes)
        fig.text(1.2,0.1,'$Error Sea: %s$'%stdsea,transform=ax.transAxes)

        py.tight_layout()
        checkdir('%s/gallery'%wdir)
        py.savefig('%s/gallery/par-plot-sivers-%d.pdf'%(wdir,istep))
        py.close()

class DSIVERS(CORE):

    def __init__(self,task,wdir,last=False):

        self.FLAV=[]
        #self.FLAV.append('g')   # 0
        self.FLAV.append('u')   # 1
        #self.FLAV.append('ub')  # 2
        self.FLAV.append('d')   # 3
        #self.FLAV.append('db')  # 4
        #self.FLAV.append('s')   # 5
        #self.FLAV.append('sb')  # 6

        if  task==0:
            self.gen_priors(wdir)
        if  task==1:
            self.plot_priors(wdir)
        if  task==2:
            self.func=self.gen_xf
            self.msg='dsivers.gen_xf'
            self.loop_over_steps(wdir,'simple',last)
        if  task==3:
            self.func=self.plot_xf
            self.msg='dsivers.plot_xf'
            self.loop_over_steps(wdir,None,last)
        if  task==4:
            self.func=self.plot_xf_relerr
            self.msg='dsivers.plot_xf_relerr'
            self.loop_over_steps(wdir,None,last)
        if  task==9:
            self.func=self.par_plot
            self.msg='dsivers.par_plot'
            self.loop_over_steps(wdir,None,last)

    def gen_xf(self,wdir,istep):
        if 'sivers' not in conf['steps'][istep]['active distributions']: return
        pdf=conf['dsivers']

        #--setup kinematics
        X=np.linspace(0.01,0.99,100)
        #Q2=conf['aux'].Q02
        Q2array=[2,4,10,100]

        #--compute XF for all replicas
        for Q2 in Q2array:
            XF={}
            cnt=0
            replicas=self.get_replicas(wdir)
            for replica in replicas:
                cnt+=1
                lprint('%d/%d'%(cnt,len(replicas)))

                self.parman.set_new_params(replica['params'][istep],initial=False)
                self.set_passive_params(istep,replica)
                for flav in self.FLAV:
                    if flav not in XF:  XF[flav]=[]
                    if  flav=='u':
                        func=lambda x: pdf.get_C(x,Q2)[1]
                    elif flav=='ub':
                        func=lambda x: pdf.get_C(x,Q2)[2]
                    elif flav=='d':
                        func=lambda x: pdf.get_C(x,Q2)[3]
                    elif flav=='db':
                        func=lambda x: pdf.get_C(x,Q2)[4]
                    elif flav=='s':
                        func=lambda x: pdf.get_C(x,Q2)[5]
                    elif flav=='sb':
                        func=lambda x: pdf.get_C(x,Q2)[6]

                    XF[flav].append([x*func(x) for x in X])
            print()
            checkdir('%s/data'%wdir)
            if Q2==conf['aux'].Q02:
                save({'X':X,'Q2':Q2,'XF':XF},'%s/data/dsivers-%d.dat'%(wdir,istep))
            else:
                save({'X':X,'Q2':Q2,'XF':XF},'%s/data/dsivers-%d-%d.dat'%(wdir,istep,int(Q2)))

    def plot_xf(self,wdir,istep):
        if 'sivers' not in conf['steps'][istep]['active distributions']: return

        cluster,colors,nc,order = self.get_clusters(wdir,istep)
        #data=load('%s/data/collinspi-%d.dat'%(wdir,istep))

        Q2array=[2,4,10,100]
        for Q2 in Q2array:
            if Q2==2:
                data1=load('%s/data/dsivers-%d.dat'%(wdir,istep))
                save(data1,'%s/npdata/dsivers_reps-Q2-%d.dat'%(wdir,int(Q2)))
                #np.save('%s/nparrays/collinspi-Q2-%d.npy'%(wdir,int(Q2)),data1)
            elif Q2==10:
                data2=load('%s/data/dsivers-%d-%d.dat'%(wdir,istep,int(Q2)))
                save(data2,'%s/npdata/dsivers_reps-Q2-%d.dat'%(wdir,int(Q2)))
                #np.save('%s/nparrays/collinspi-Q2-%d.npy'%(wdir,int(Q2)),data2)
            elif Q2==100:
                data3=load('%s/data/dsivers-%d-%d.dat'%(wdir,istep,int(Q2)))
                save(data3,'%s/npdata/dsivers_reps-Q2-%d.dat'%(wdir,int(Q2)))
                #np.save('%s/nparrays/collinspi-Q2-%d.npy'%(wdir,int(Q2)),data3)
            elif Q2==4:
                data4=load('%s/data/dsivers-%d-%d.dat'%(wdir,istep,int(Q2)))
                save(data4,'%s/npdata/dsivers_reps-Q2-%d.dat'%(wdir,int(Q2)))
                #np.save('%s/nparrays/collinspi-Q2-%d.npy'%(wdir,int(Q2)),data3)

        X=data1['X']

        ncols=2
        nrows=len(self.FLAV)/ncols
        if len(self.FLAV)%ncols>0: nrows+=1
        nrows = int(nrows)
        fig = py.figure(figsize=(ncols*3,nrows*2))

        cnt=0
        #for flav in self.FLAV:
            #cnt+=1
            #ax=py.subplot(nrows,ncols,cnt)
            #for i in range(len(data['XF'][flav])):
                #c=colors[cluster[i]]
                #if c=='r':    ax.plot(X,data['XF'][flav][i],'%s-'%c,zorder=10,alpha=0.1)
                #else:         ax.plot(X,data['XF'][flav][i],'%s-'%c,zorder= 0,alpha=0.1)
        rand_list=[]
        for j in range(100):
            rand_list.append(random.randint(0, 950))
        for flav in self.FLAV:
            cnt+=1
            ax=py.subplot(nrows,ncols,cnt)
            _data1=data1['XF'][flav]
            _data2=data2['XF'][flav]
            _data3=data3['XF'][flav]
            for Q2 in Q2array:
                if Q2==2:
                    c='r'
                    meanXF1 = np.mean(_data1,axis=0)
                    #stdXF1 = np.std(_data1,axis=0)
                    #lower1=meanXF1-stdXF1
                    #upper1=meanXF1+stdXF1
                    ax.plot(X,meanXF1,'%s-'%c,zorder=10,alpha=0.5)
                    #ax.fill_between(X, lower1, upper1,color=c,alpha=0.5)
                    #for i in rand_list:
                    #    ax.plot(X,data['XF'][flav][i],color='b',zorder= 0,alpha=0.1)
                elif Q2==10:
                    c='b'
                    meanXF2 = np.mean(_data2,axis=0)
                    stdXF2 = np.std(_data2,axis=0)

                    XFarray=[X,meanXF2,stdXF2]
                    save(XFarray,'%s/npdata/dsivers-%s-Q2-%d.dat'%(wdir,flav,int(Q2)))

                    lower2=meanXF2-stdXF2
                    upper2=meanXF2+stdXF2
                    ax.plot(X,meanXF2,'%s-'%c,zorder=10,alpha=0.5)
                    ax.fill_between(X, lower2, upper2,color=c,alpha=0.5)
                    #for i in rand_list:
                    #    ax.plot(X,data2['XF'][flav][i],color='y',zorder= 0,alpha=0.1)
                elif Q2==100:
                    c='g'
                    meanXF3 = np.mean(_data3,axis=0)
                    #stdXF3 = np.std(_data3,axis=0)
                    #lower3=meanXF3-stdXF3
                    #upper3=meanXF3+stdXF3
                    ax.plot(X,meanXF3,'%s-'%c,zorder=10,alpha=0.5)
                    #ax.fill_between(X, lower3, upper3,color=c,alpha=0.5)

            #ax.semilogx()
            ax.set_xlim(0.1, 1)
            #if flav=='u': ax.set_ylim(-0.1,1)
            #if flav=='d': ax.set_ylim(-1,0.1)
            #if flav=='ub': ax.set_ylim(-0.055,0)
            #if flav=='db': ax.set_ylim(0,0.03)
            #if flav=='s' : ax.set_ylim(-0.055,0.03)
            #if flav=='sb': ax.set_ylim(-0.055,0.03)
            #if flav=='s+sb/db+ub': ax.set_ylim(0,4)
            #if flav=='s+sb': ax.set_ylim(0,2)
            #if flav=='u':ax.set_ylabel('$2M_h\,z^2\,H_1^{\perp(1)\,fav}(z)$')
            #else: ax.set_ylabel('$2M_h\,z^2\,H_1^{\perp(1)\,unf}(z)$')
            if flav=='u':
                ax.set_ylabel(r' $x\frac{d f_{1T}^{\perp(1)u}(x)}{dx}$')
                ax.set_ylim(-0.02, 0.2)
            else:
                ax.set_ylabel(r' $x\frac{d f_{1T}^{\perp(1)d}(x)}{dx}$')
                #ax.set_ylim(-0.3, 0.02)
            ax.set_xlabel('$x$')

        py.tight_layout()
        checkdir('%s/gallery'%wdir)
        py.savefig('%s/gallery/dsivers-%d.pdf'%(wdir,istep))
        py.close()

class COLLINSPI(CORE):

    def __init__(self,task,wdir,last=False):

        self.FLAV=[]
        #self.FLAV.append('g')   # 0
        self.FLAV.append('u')   # 1
        #self.FLAV.append('ub')  # 2
        self.FLAV.append('d')   # 3
        #self.FLAV.append('db')  # 4
        #self.FLAV.append('s')   # 5
        #self.FLAV.append('sb')  # 6

        if  task==0:
            self.gen_priors(wdir)
        if  task==1:
            self.plot_priors(wdir)
        if  task==2:
            self.func=self.gen_xf
            self.msg='collinspi.gen_xf'
            self.loop_over_steps(wdir,'simple',last)
        if  task==3:
            self.func=self.plot_xf
            self.msg='collinspi.plot_xf'
            self.loop_over_steps(wdir,None,last)
        if  task==4:
            self.func=self.plot_xf_relerr
            self.msg='collinspi.plot_xf_relerr'
            self.loop_over_steps(wdir,None,last)
        if  task==9:
            self.func=self.par_plot
            self.msg='collinspi.par_plot'
            self.loop_over_steps(wdir,None,last)

    def gen_priors(self,wdir):

        load_config('%s/input.py'%wdir)
        resman=RESMAN(nworkers=1,parallel=False,datasets=False)
        parman=resman.parman
        params=[parman.gen_flat(setup=False) for _ in range(300)]

        ffpion=conf['collinspi']

        #--setup kinematics
        X1=10**np.linspace(-4,-1,100)
        X2=np.linspace(0.01,0.99,100)
        X=np.concatenate([X1,X2])
        #Q2=conf['aux'].Q02
        Q2=10

        #--compute XF for all replicas
        XF={}
        cnt=0
        for par in params:
            cnt+=1
            lprint('%d/%d'%(cnt,len(params)))

            parman.set_new_params(par,initial=False)

            for flav in self.FLAV:
                if flav not in XF:  XF[flav]=[]
                if  flav=='u':
                    func=lambda x: ffpion.get_C(x,Q2)[1]
                elif flav=='ub':
                    func=lambda x: ffpion.get_C(x,Q2)[2]
                elif flav=='d':
                    func=lambda x: ffpion.get_C(x,Q2)[3]
                elif flav=='db':
                    func=lambda x: ffpion.get_C(x,Q2)[4]
                elif flav=='s':
                    func=lambda x: ffpion.get_C(x,Q2)[5]
                elif flav=='sb':
                    func=lambda x: ffpion.get_C(x,Q2)[6]
                XF[flav].append([func(x) for x in X])

        print()
        checkdir('%s/data'%wdir)
        save({'X':X,'Q2':Q2,'XF':XF},'%s/data/collinspi-prior.dat'%(wdir))

    def plot_priors(self,wdir):

        data=load('%s/data/collinspi-prior.dat'%(wdir))
        X=data['X']

        ncols=3
        nrows=len(self.FLAV)/ncols
        if len(self.FLAV)%ncols>1: nrows+=1
        nrows = int(nrows)

        fig = py.figure(figsize=(ncols*3,nrows*2))

        cnt=0
        for flav in self.FLAV:
            cnt+=1
            ax=py.subplot(nrows,ncols,cnt)
            for i in range(len(data['XF'][flav])):
                ax.plot(X,data['XF'][flav][i])
            #ax.semilogx()
            ax.set_ylim(0,1)
            #if flav=='um': ax.set_ylim(0,1)
            #if flav=='dm': ax.set_ylim(0,0.5)
            #if flav=='ub': ax.set_ylim(0,0.6)
            #if flav=='db': ax.set_ylim(0,0.6)
            #if flav=='s' : ax.set_ylim(0,0.6)
            #if flav=='sb': ax.set_ylim(0,0.6)
            #if flav=='s+sb/db+ub': ax.set_ylim(0,4)
            #if flav=='s+sb': ax.set_ylim(0,2)
            ax.set_ylabel(flav)

        py.tight_layout()
        checkdir('%s/gallery'%wdir)
        py.savefig('%s/gallery/collinspi-prior.pdf'%(wdir))
        py.close()

    def gen_xf(self,wdir,istep):
        if 'collinspi' not in conf['steps'][istep]['active distributions']: return
        ffpion=conf['collinspi']

        #--setup kinematics
        X1=10**np.linspace(-4,-2,100)
        X2=np.linspace(0.011,0.99,100)
        X=np.concatenate([X1,X2])
        #Q2=conf['aux'].Q02
        Q2array=[2,4,10,100]

        #--compute XF for all replicas
        for Q2 in Q2array:
            XF={}
            cnt=0
            replicas=self.get_replicas(wdir)
            for replica in replicas:
                cnt+=1
                lprint('%d/%d'%(cnt,len(replicas)))
                
                self.parman.set_new_params(replica['params'][istep],initial=False)
                self.set_passive_params(istep,replica)
                for flav in self.FLAV:
                    if flav not in XF:  XF[flav]=[]
                    if  flav=='u':
                        func=lambda x: ffpion.get_C(x,Q2)[1]
                    elif flav=='ub':
                        func=lambda x: ffpion.get_C(x,Q2)[2]
                    elif flav=='d':
                        func=lambda x: ffpion.get_C(x,Q2)[3]
                    elif flav=='db':
                        func=lambda x: ffpion.get_C(x,Q2)[4]
                    elif flav=='s':
                        func=lambda x: ffpion.get_C(x,Q2)[5]
                    elif flav=='sb':
                        func=lambda x: ffpion.get_C(x,Q2)[6]

                    #XF[flav].append([2*0.135*x**2*func(x) for x in X])
                    XF[flav].append([x*func(x) for x in X])
            print()
            checkdir('%s/data'%wdir)
            if Q2==conf['aux'].Q02:
                save({'X':X,'Q2':Q2,'XF':XF},'%s/data/collinspi-%d.dat'%(wdir,istep))
            else:
                save({'X':X,'Q2':Q2,'XF':XF},'%s/data/collinspi-%d-%d.dat'%(wdir,istep,int(Q2)))

    def plot_xf(self,wdir,istep):
        if 'collinspi' not in conf['steps'][istep]['active distributions']: return

        cluster,colors,nc,order = self.get_clusters(wdir,istep)
        #data=load('%s/data/collinspi-%d.dat'%(wdir,istep))
        
        replicas=self.get_replicas(wdir)

        Q2array=[2,4,10,100]
        for Q2 in Q2array:
            if Q2==2:
                data1=load('%s/data/collinspi-%d.dat'%(wdir,istep))
                save(data1,'%s/npdata/collinspi_reps-Q2-%d.dat'%(wdir,int(Q2)))
                #np.save('%s/nparrays/collinspi-Q2-%d.npy'%(wdir,int(Q2)),data1)
            elif Q2==10:
                data2=load('%s/data/collinspi-%d-%d.dat'%(wdir,istep,int(Q2)))
                save(data2,'%s/npdata/collinspi_reps-Q2-%d.dat'%(wdir,int(Q2)))
                #np.save('%s/nparrays/collinspi-Q2-%d.npy'%(wdir,int(Q2)),data2)
            elif Q2==100:
                data3=load('%s/data/collinspi-%d-%d.dat'%(wdir,istep,int(Q2)))
                save(data3,'%s/npdata/collinspi_reps-Q2-%d.dat'%(wdir,int(Q2)))
                #np.save('%s/nparrays/collinspi-Q2-%d.npy'%(wdir,int(Q2)),data3)
            elif Q2==4:
                data4=load('%s/data/collinspi-%d-%d.dat'%(wdir,istep,int(Q2)))
                save(data4,'%s/npdata/collinspi_reps-Q2-%d.dat'%(wdir,int(Q2)))
                #np.save('%s/nparrays/collinspi-Q2-%d.npy'%(wdir,int(Q2)),data3)

        X=data1['X']

        ncols=2
        nrows=len(self.FLAV)/ncols
        if len(self.FLAV)%ncols>0: nrows+=1
        nrows = int(nrows)
        fig = py.figure(figsize=(ncols*3,nrows*2))

        cnt=0
        #for flav in self.FLAV:
            #cnt+=1
            #ax=py.subplot(nrows,ncols,cnt)
            #for i in range(len(data['XF'][flav])):
                #c=colors[cluster[i]]
                #if c=='r':    ax.plot(X,data['XF'][flav][i],'%s-'%c,zorder=10,alpha=0.1)
                #else:         ax.plot(X,data['XF'][flav][i],'%s-'%c,zorder= 0,alpha=0.1)
        rand_list=[]
        for j in range(100):
            rand_list.append(random.randint(0, 950))
        for flav in self.FLAV:
            cnt+=1
            ax=py.subplot(nrows,ncols,cnt)
            _data1=data1['XF'][flav]
            _data2=data2['XF'][flav]
            _data3=data3['XF'][flav]
            _data4=data4['XF'][flav]
            for Q2 in Q2array:
                if Q2==4:
                    for i in range(len(replicas)):
                        ax.plot(X,data4['XF'][flav][i],color='y',zorder= 0,alpha=0.3)
                    c='r'
                    meanXF2 = np.mean(_data4,axis=0)
                    stdXF2 = np.std(_data4,axis=0)

                    XFarray=[X,meanXF2,stdXF2]
                    save(XFarray,'%s/npdata/collinspi-%s-Q2-%d.dat'%(wdir,flav,int(Q2)))

                    lower2=meanXF2-stdXF2
                    upper2=meanXF2+stdXF2
                    ax.plot(X,meanXF2,'%s-'%c,zorder=10,alpha=0.3)
                    ax.fill_between(X, lower2, upper2,color=c,alpha=0.3)
                elif Q2==10:
                    c='b'
                    meanXF2 = np.mean(_data2,axis=0)
                    #stdXF2 = np.std(_data2,axis=0)

                    #XFarray=[X,meanXF2,stdXF2]
                    #save(XFarray,'%s/npdata/sivers-%s-Q2-%d.dat'%(wdir,flav,int(Q2)))

                    #lower2=meanXF2-stdXF2
                    #upper2=meanXF2+stdXF2
                    ax.plot(X,meanXF2,'%s-'%c,zorder=10,alpha=0.5)
                    #ax.fill_between(X, lower2, upper2,color=c,alpha=0.5)
            
                elif Q2==100:
                    c='g'
                    meanXF3 = np.mean(_data3,axis=0)
                    #stdXF3 = np.std(_data3,axis=0)
                    #lower3=meanXF3-stdXF3
                    #upper3=meanXF3+stdXF3
                    ax.plot(X,meanXF3,'%s-'%c,zorder=10,alpha=0.5)
                    #ax.fill_between(X, lower3, upper3,color=c,alpha=0.5)
                    #for i in rand_list:
                    #    ax.plot(X,data['XF'][flav][i],color='b',zorder= 0,alpha=0.1)

            #ax.semilogx()
            #ax.set_ylim(-0.04,0.04)
            ax.set_xlim(0.15, 1.0)
            if flav=='u': ax.set_ylim(-0.1,1)
            if flav=='d': ax.set_ylim(-1,0.1)
            if flav=='ub': ax.set_ylim(-0.055,0)
            if flav=='db': ax.set_ylim(0,0.03)
            if flav=='s' : ax.set_ylim(-0.055,0.03)
            if flav=='sb': ax.set_ylim(-0.055,0.03)
            #if flav=='s+sb/db+ub': ax.set_ylim(0,4)
            #if flav=='s+sb': ax.set_ylim(0,2)
            #if flav=='u':ax.set_ylabel('$2M_h\,z^2\,H_1^{\perp(1)\,fav}(z)$')
            #else: ax.set_ylabel('$2M_h\,z^2\,H_1^{\perp(1)\,unf}(z)$')
            if flav=='u':ax.set_ylabel('$z\,H_1^{\perp(1)\,fav}(z)$')
            else: ax.set_ylabel('$z\,H_1^{\perp(1)\,unf}(z)$')
            ax.set_xlabel('$z$')

        py.tight_layout()
        checkdir('%s/gallery'%wdir)
        py.savefig('%s/gallery/collinspi-%d.pdf'%(wdir,istep))
        py.close()

    def plot_xf_relerr(self,wdir,istep):
        if 'collinspi' not in conf['steps'][istep]['active distributions']: return

        cluster,colors,nc,order = self.get_clusters(wdir,istep)
        #return

        Q2=10
        data=load('%s/data/collinspi-%d-%d.dat'%(wdir,istep,int(Q2)))

        X=data['X']

        ncols=2
        nrows=len(self.FLAV)/ncols
        if len(self.FLAV)%ncols>1: nrows+=1
        nrows = int(nrows)
        fig = py.figure(figsize=(ncols*3,nrows*2))

        cnt=0

        for flav in self.FLAV:
            cnt+=1
            ax=py.subplot(nrows,ncols,cnt)
            _data=data['XF'][flav]
            c='r'
            meanXF = np.mean(_data,axis=0)
            stdXF = np.std(_data,axis=0)

            _interpXF=interpolate.splrep(X,meanXF)
            interpXF=lambda x: interpolate.splev(x,_interpXF)

            _dinterpXF=interpolate.splrep(X,meanXF+stdXF)
            dinterpXF=lambda x: interpolate.splev(x,_dinterpXF)

            momXF=sp.integrate.quad(interpXF,0,1)[0]
            dmomXF=sp.integrate.quad(dinterpXF,0,1)[0]

            mom_err = np.abs((dmomXF-momXF)/momXF)

            print('collinspi %s moment rel err=%0.2f'%(flav,mom_err))

            ax.plot(X,np.abs(stdXF/meanXF),'%s-'%c,zorder=10,alpha=0.5)

            if flav=='u': ax.set_ylim(0,0.5)
            #if flav=='ub': ax.set_ylim(-0.05,0.05)
            if flav=='d': ax.set_ylim(0,0.75)
            #if flav=='db': ax.set_ylim(-0.05,0.05)
            #if flav=='s' : ax.set_ylim(-0.05,0.05)
            #if flav=='sb': ax.set_ylim(-0.05,0.05)
            #ax.set_ylim(-0.1,0.1)
            ax.set_xlim(0.2, 0.75)
            if flav=='u':ax.set_ylabel('$|\Delta H_1^{\perp(1)\,fav}/H_1^{\perp(1)\,fav}|$')
            else: ax.set_ylabel('$|\Delta H_1^{\perp(1)\,unf}/H_1^{\perp(1)\,unf}|$')
            ax.set_xlabel('$z$')

        py.tight_layout()
        checkdir('%s/gallery'%wdir)
        #py.savefig('%s/gallery/pdf-%d.pdf'%(wdir,istep))
        py.savefig('%s/gallery/collinspi-relerr-%d.pdf'%(wdir,istep))
        py.close()

    def par_plot(self,wdir,istep):
        #self.FLAV.append('g')
        #self.FLAV.append('u')
        #self.FLAV.append('d')
        #self.FLAV.append('s')
        #self.FLAV.append('sb')
        #self.FLAV.append('c')
        #self.FLAV.append('b')

        par=[]
        #par.append(['u1 N 1', 'u1 a 1', 'u1 b 1'])
        #par.append(['u N1 1', 'u a1 1', 'u b1 1'])
        #par.append(['u1 N 2', 'u1 a 2', 'u1 b 2'])
        #par.append(['u N1 2', 'u a1 2', 'u b1 2'])
        #par.append(['ub N0 1', 'ub a0, 1', 'ub b0 1'])
        #par.append(['d1 N 1', 'd1 a 1', 'd1 b 1'])
        #par.append(['d N1 1', 'd a1 1', 'd b1 1'])
        #par.append(['d1 N 2', 'd1 a 2', 'd1 b 2'])
        #par.append(['d N1 2', 'd a1 2', 'd b1 2'])
        #par.append(['db N0 1','db a0 1','db b0 1'])
        #par.append(['s N0 1', 's a0 1', 's b0 1'])
        #par.append(['sb N0 1', 'sb a0 1', 'sb b0 1'])
        #par.append(['widths1_uv', 'widths1_dv'])
        #par.append(['widths1_fav', 'widths1_ufav'])
        
        par.append(['u N0 1', 'u a0 1', 'u b0 1'])
        par.append(['u N1 1', 'u a1 1', 'u b1 1'])
        par.append(['u N0 2', 'u a0 2', 'u b0 2'])
        par.append(['u N1 2', 'u a1 2', 'u b1 2'])
        par.append(['u c0 1', 'u d0 1'])
        #par.append(['ub N0 1', 'ub a0, 1', 'ub b0 1'])
        par.append(['d N0 1', 'd a0 1', 'd b0 1'])
        par.append(['d N1 1', 'd a1 1', 'd b1 1'])
        par.append(['d N0 2', 'd a0 2', 'd b0 2'])
        par.append(['d N1 2', 'd a1 2', 'd b1 2'])
        par.append(['d c0 1', 'd d0 1'])
        #par.append(['db N0 1','db a0 1','db b0 1'])
        #par.append(['s N0 1', 's a0 1', 's b0 1'])
        #par.append(['sb N0 1', 'sb a0 1', 'sb b0 1'])
        #par.append(['widths1_uv', 'widths1_dv'])
        par.append(['widths1_fav', 'widths1_ufav'])
        par.append(['widths2_fav', 'widths2_ufav'])
        par.append(['widths3_fav', 'widths3_ufav'])
        par.append(['widths4_fav', 'widths4_ufav'])
        par.append(['widths5_fav', 'widths5_ufav'])

        kind=1
        tag='collinspi'

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


        replicas=self.get_replicas(wdir)
        for j in range(len(replicas)):
            #for replica in replicas:
            replica=replicas[j]
            #print(replica)
            #if cluster[j]!=0: continue
            params=replica['params'][istep]
            #if params[8]>11: continue
            #if params[22]>1 or params[24]>5 or params[17]<20 or params[18]>5 or params[7]<0.15 or     params[7]>1.8 or params[19]>0.15: continue #filter of par values
            #if params[5] > 20 or params[11] > 20 or params[13] > 20: continue #filter on par values
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
        meanfav = "{:.6f}".format(widths1_favMean)
        stdfav= "{:.6f}".format(widths1_favSTD)

        widths1_ufavMean = np.mean(data[0]['widths1_ufav'])
        widths1_ufavSTD = np.std(data[0]['widths1_ufav'])
        meanufav = "{:.6f}".format(widths1_ufavMean)
        stdufav="{:.6f}".format(widths1_ufavSTD)

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
                    ax.hist(data[i][_],bins=50,color=c,histtype='step')
                    
                if '_' in _: _ = _.replace("_", " ")
                ax.set_xlabel(_)

        fig.text(1.2,0.7,'$Mean Fav: %s$'%meanfav,transform=ax.transAxes)
        fig.text(1.2,0.5,'$Error Fav: %s$'%stdfav,transform=ax.transAxes)
        fig.text(1.2,0.3,'$Mean Unfav: %s$'%meanufav,transform=ax.transAxes)
        fig.text(1.2,0.1,'$Error Unfav: %s$'%stdufav,transform=ax.transAxes)

        py.tight_layout()
        checkdir('%s/gallery'%wdir)
        py.savefig('%s/gallery/par-plot-collinspi-%d.pdf'%(wdir,istep))
        py.close()

class HTILDEPI(CORE):

    def __init__(self,task,wdir,last=False):

        self.FLAV=[]
        #self.FLAV.append('g')   # 0
        self.FLAV.append('u')   # 1
        #self.FLAV.append('ub')  # 2
        self.FLAV.append('d')   # 3
        #self.FLAV.append('db')  # 4
        #self.FLAV.append('s')   # 5
        #self.FLAV.append('sb')  # 6

        if  task==0:
            self.gen_priors(wdir)
        if  task==1:
            self.plot_priors(wdir)
        if  task==2:
            self.func=self.gen_xf
            self.msg='Htildepi.gen_xf'
            self.loop_over_steps(wdir,'simple',last)
        if  task==3:
            self.func=self.plot_xf
            self.msg='Htildepi.plot_xf'
            self.loop_over_steps(wdir,None,last)
        if  task==4:
            self.func=self.plot_xf_relerr
            self.msg='Htildepi.plot_xf_relerr'
            self.loop_over_steps(wdir,None,last)
        if  task==9:
            self.func=self.par_plot
            self.msg='Htildepi.par_plot'
            self.loop_over_steps(wdir,None,last)

    def gen_priors(self,wdir):

        load_config('%s/input.py'%wdir)
        resman=RESMAN(nworkers=1,parallel=False,datasets=False)
        parman=resman.parman
        params=[parman.gen_flat(setup=False) for _ in range(300)]
        ffpion=conf['Htildepi']

        #--setup kinematics
        X=10**np.linspace(-4,-1,100)
        X=np.append(X,np.linspace(0.1,0.9,100))
        Q2=conf['aux'].Q02

        #--compute XF for all replicas
        XF={}
        cnt=0
        for par in params:
            cnt+=1
            lprint('%d/%d'%(cnt,len(params)))

            parman.set_new_params(par,initial=False)

            for flav in self.FLAV:
                if flav not in XF:  XF[flav]=[]
                if  flav=='u':
                    func=lambda x: ffpion.get_C(x,Q2)[1]
                elif flav=='ub':
                    func=lambda x: ffpion.get_C(x,Q2)[2]
                elif flav=='d':
                    func=lambda x: ffpion.get_C(x,Q2)[3]
                elif flav=='db':
                    func=lambda x: ffpion.get_C(x,Q2)[4]
                elif flav=='s':
                    func=lambda x: ffpion.get_C(x,Q2)[5]
                elif flav=='sb':
                    func=lambda x: ffpion.get_C(x,Q2)[6]
                XF[flav].append([x*func(x) for x in X])

        print()
        checkdir('%s/data'%wdir)
        save({'X':X,'Q2':Q2,'XF':XF},'%s/data/Htildepi-prior.dat'%(wdir))

    def plot_priors(self,wdir):

        data=load('%s/data/Htildepi-prior.dat'%(wdir))
        X=data['X']

        ncols=3
        nrows=len(self.FLAV)/ncols
        if len(self.FLAV)%ncols>1: nrows+=1
        nrows = int(nrows)

        fig = py.figure(figsize=(ncols*3,nrows*2))

        cnt=0
        for flav in self.FLAV:
            cnt+=1
            ax=py.subplot(nrows,ncols,cnt)
            for i in range(len(data['XF'][flav])):
                ax.plot(X,data['XF'][flav][i])
            #ax.semilogx()
            ax.set_ylim(0,1)
            #if flav=='um': ax.set_ylim(0,1)
            #if flav=='dm': ax.set_ylim(0,0.5)
            #if flav=='ub': ax.set_ylim(0,0.6)
            #if flav=='db': ax.set_ylim(0,0.6)
            #if flav=='s' : ax.set_ylim(0,0.6)
            #if flav=='sb': ax.set_ylim(0,0.6)
            #if flav=='s+sb/db+ub': ax.set_ylim(0,4)
            #if flav=='s+sb': ax.set_ylim(0,2)
            ax.set_ylabel(flav)

        py.tight_layout()
        checkdir('%s/gallery'%wdir)
        py.savefig('%s/gallery/Htildepi-prior.pdf'%(wdir))
        py.close()

    def gen_xf(self,wdir,istep):
        if 'Htildepi' not in conf['steps'][istep]['active distributions']: return
        ffpion=conf['Htildepi']

        #--setup kinematics
        X1=10**np.linspace(-4,-2,100)
        X2=np.linspace(0.011,0.99,100)
        X=np.concatenate([X1,X2])
        Q2array=[2,4,10,100]

        #--compute XF for all replicas
        for Q2 in Q2array:
            XF={}
            cnt=0
            replicas=self.get_replicas(wdir)
            for replica in replicas:
                cnt+=1
                lprint('%d/%d'%(cnt,len(replicas)))

                self.parman.set_new_params(replica['params'][istep],initial=False)
                self.set_passive_params(istep,replica)

                for flav in self.FLAV:
                    if flav not in XF:  XF[flav]=[]
                    if  flav=='u':
                        func=lambda x: ffpion.get_C(x,Q2)[1]
                    elif flav=='ub':
                        func=lambda x: ffpion.get_C(x,Q2)[2]
                    elif flav=='d':
                        func=lambda x: ffpion.get_C(x,Q2)[3]
                    elif flav=='db':
                        func=lambda x: ffpion.get_C(x,Q2)[4]
                    elif flav=='s':
                        func=lambda x: ffpion.get_C(x,Q2)[5]
                    elif flav=='sb':
                        func=lambda x: ffpion.get_C(x,Q2)[6]
                    XF[flav].append([x*func(x) for x in X])
            print()
            checkdir('%s/data'%wdir)
            if Q2==conf['aux'].Q02:
                save({'X':X,'Q2':Q2,'XF':XF},'%s/data/Htildepi-%d.dat'%(wdir,istep))
            else:
                save({'X':X,'Q2':Q2,'XF':XF},'%s/data/Htildepi-%d-%d.dat'%(wdir,istep,int(Q2)))

    def plot_xf(self,wdir,istep):
        if 'Htildepi' not in conf['steps'][istep]['active distributions']: return

        cluster,colors,nc,order = self.get_clusters(wdir,istep)
        #data=load('%s/data/Htildepi-%d.dat'%(wdir,istep))
        
        replicas=self.get_replicas(wdir)

        Q2array=[2,4,10,100]
        for Q2 in Q2array:
            if Q2==2:
                data1=load('%s/data/Htildepi-%d.dat'%(wdir,istep))
                save(data1,'%s/npdata/Htildepi_reps-Q2-%d.dat'%(wdir,int(Q2)))
                #np.save('%s/nparrays/Htildepi-Q2-%d.npy'%(wdir,int(Q2)),data1)
            elif Q2==10:
                data2=load('%s/data/Htildepi-%d-%d.dat'%(wdir,istep,int(Q2)))
                save(data2,'%s/npdata/Htildepi_reps-Q2-%d.dat'%(wdir,int(Q2)))
                #np.save('%s/nparrays/Htildepi-Q2-%d.npy'%(wdir,int(Q2)),data2)
            elif Q2==100:
                data3=load('%s/data/Htildepi-%d-%d.dat'%(wdir,istep,int(Q2)))
                save(data3,'%s/npdata/Htildepi_reps-Q2-%d.dat'%(wdir,int(Q2)))
                #np.save('%s/nparrays/Htildepi-Q2-%d.npy'%(wdir,int(Q2)),data3)
            elif Q2==4:
                data4=load('%s/data/Htildepi-%d-%d.dat'%(wdir,istep,int(Q2)))
                save(data4,'%s/npdata/Htildepi_reps-Q2-%d.dat'%(wdir,int(Q2)))
                #np.save('%s/nparrays/Htildepi-Q2-%d.npy'%(wdir,int(Q2)),data3)

        X=data1['X']

        ncols=2
        nrows=len(self.FLAV)/ncols
        if len(self.FLAV)%ncols>0: nrows+=1
        nrows = int(nrows)
        fig = py.figure(figsize=(ncols*3,nrows*2))

        cnt=0
        #for flav in self.FLAV:
            #cnt+=1
            #ax=py.subplot(nrows,ncols,cnt)
            #for i in range(len(data['XF'][flav])):
                #c=colors[cluster[i]]
                #if c=='r':    ax.plot(X,data['XF'][flav][i],'%s-'%c,zorder=10,alpha=0.1)
                #else:         ax.plot(X,data['XF'][flav][i],'%s-'%c,zorder= 0,alpha=0.1)
        rand_list=[]
        for j in range(100):
            rand_list.append(random.randint(0, 950))
        for flav in self.FLAV:
            cnt+=1
            ax=py.subplot(nrows,ncols,cnt)
            _data1=data1['XF'][flav]
            _data2=data2['XF'][flav]
            _data3=data3['XF'][flav]
            _data4=data4['XF'][flav]
            for Q2 in Q2array:
                if Q2==4:
                    for i in range(len(replicas)):
                        ax.plot(X,data4['XF'][flav][i],color='y',zorder= 0,alpha=0.3)
                    c='r'
                    meanXF2 = np.mean(_data4,axis=0)
                    stdXF2 = np.std(_data4,axis=0)

                    XFarray=[X,meanXF2,stdXF2]
                    save(XFarray,'%s/npdata/Htildepi-%s-Q2-%d.dat'%(wdir,flav,int(Q2)))

                    lower2=meanXF2-stdXF2
                    upper2=meanXF2+stdXF2
                    ax.plot(X,meanXF2,'%s-'%c,zorder=10,alpha=0.3)
                    ax.fill_between(X, lower2, upper2,color=c,alpha=0.3)
                elif Q2==10:
                    c='b'
                    meanXF2 = np.mean(_data2,axis=0)
                    #stdXF2 = np.std(_data2,axis=0)

                    #XFarray=[X,meanXF2,stdXF2]
                    #save(XFarray,'%s/npdata/sivers-%s-Q2-%d.dat'%(wdir,flav,int(Q2)))

                    #lower2=meanXF2-stdXF2
                    #upper2=meanXF2+stdXF2
                    ax.plot(X,meanXF2,'%s-'%c,zorder=10,alpha=0.5)
                    #ax.fill_between(X, lower2, upper2,color=c,alpha=0.5)
            
                elif Q2==100:
                    c='g'
                    meanXF3 = np.mean(_data3,axis=0)
                    #stdXF3 = np.std(_data3,axis=0)
                    #lower3=meanXF3-stdXF3
                    #upper3=meanXF3+stdXF3
                    ax.plot(X,meanXF3,'%s-'%c,zorder=10,alpha=0.5)
                    #ax.fill_between(X, lower3, upper3,color=c,alpha=0.5)
                    #for i in rand_list:
                    #    ax.plot(X,data['XF'][flav][i],color='b',zorder= 0,alpha=0.1)

            #ax.semilogx()
            ax.set_xlim(0.01,1.0)
            #ax.set_ylim(-0.25,0.25)
            #if flav=='um': ax.set_ylim(0,1)
            #if flav=='dm': ax.set_ylim(0,0.5)
            #if flav=='ub': ax.set_ylim(0,0.6)
            #if flav=='db': ax.set_ylim(0,0.6)
            #if flav=='s' : ax.set_ylim(0,0.6)
            #if flav=='sb': ax.set_ylim(0,0.6)
            #if flav=='s+sb/db+ub': ax.set_ylim(0,4)
            #if flav=='s+sb': ax.set_ylim(0,2)
            if flav=='u':ax.set_ylabel('$z\, {Htilde}^{fav}(z)$')
            else: ax.set_ylabel('$z\, {Htilde}^{unf}(z)$')

        py.tight_layout()
        checkdir('%s/gallery'%wdir)
        py.savefig('%s/gallery/Htildepi-%d.pdf'%(wdir,istep))
        py.close()

    def plot_xf_relerr(self,wdir,istep):
        if 'Htildepi' not in conf['steps'][istep]['active distributions']: return

        cluster,colors,nc,order = self.get_clusters(wdir,istep)
        #return

        Q2=10
        data=load('%s/data/Htildepi-%d-%d.dat'%(wdir,istep,int(Q2)))

        X=data['X']

        ncols=2
        nrows=len(self.FLAV)/ncols
        if len(self.FLAV)%ncols>1: nrows+=1
        nrows = int(nrows)
        fig = py.figure(figsize=(ncols*3,nrows*2))

        cnt=0

        for flav in self.FLAV:
            cnt+=1
            ax=py.subplot(nrows,ncols,cnt)
            _data=data['XF'][flav]
            c='r'
            meanXF = np.mean(_data,axis=0)
            stdXF = np.std(_data,axis=0)

            _interpXF=interpolate.splrep(X,meanXF)
            interpXF=lambda x: interpolate.splev(x,_interpXF)

            _dinterpXF=interpolate.splrep(X,meanXF+stdXF)
            dinterpXF=lambda x: interpolate.splev(x,_dinterpXF)

            momXF=sp.integrate.quad(interpXF,0,1)[0]
            dmomXF=sp.integrate.quad(dinterpXF,0,1)[0]

            mom_err = np.abs((dmomXF-momXF)/momXF)

            print('Htildepi %s moment rel err=%0.2f'%(flav,mom_err))

            ax.plot(X,np.abs(stdXF/meanXF),'%s-'%c,zorder=10,alpha=0.5)

            if flav=='u': ax.set_ylim(0,3)
            #if flav=='ub': ax.set_ylim(-0.05,0.05)
            if flav=='d': ax.set_ylim(0,3.5)
            #if flav=='db': ax.set_ylim(-0.05,0.05)
            #if flav=='s' : ax.set_ylim(-0.05,0.05)
            #if flav=='sb': ax.set_ylim(-0.05,0.05)
            #ax.set_ylim(-0.1,0.1)
            ax.set_xlim(0.2, 0.75)
            if flav=='u':ax.set_ylabel('$|\Delta {Htilde}^{fav}/{Htilde}^{fav}|$')
            else: ax.set_ylabel('$|\Delta {Htilde}^{unf}/{Htilde}^{unf}|$')
            ax.set_xlabel('$z$')

        py.tight_layout()
        checkdir('%s/gallery'%wdir)
        #py.savefig('%s/gallery/pdf-%d.pdf'%(wdir,istep))
        py.savefig('%s/gallery/Htildepi-relerr-%d.pdf'%(wdir,istep))
        py.close()

    def par_plot(self,wdir,istep):
        #self.FLAV.append('g')
        #self.FLAV.append('u')
        #self.FLAV.append('d')
        #self.FLAV.append('s')
        #self.FLAV.append('sb')
        #self.FLAV.append('c')
        #self.FLAV.append('b')

        par=[]
        #par.append(['u1 N 1', 'u1 a 1', 'u1 b 1'])
        #par.append(['u N1 1', 'u a1 1', 'u b1 1'])
        #par.append(['ub N0 1', 'ub a0, 1', 'ub b0 1'])
        #par.append(['d1 N 1', 'd1 a 1', 'd1 b 1'])
        #par.append(['d N1 1', 'd a1 1', 'd b1 1'])
        #par.append(['db N0 1','db a0 1','db b0 1'])
        #par.append(['s N0 1', 's a0 1', 's b0 1'])
        #par.append(['sb N0 1', 'sb a0 1', 'sb b0 1'])
        #par.append(['widths1_uv', 'widths1_dv'])
        #par.append(['widths1_uv', 'widths1_dv'])
        
        par.append(['u N0 1', 'u a0 1', 'u b0 1']) #,'u c0 1'])
        par.append(['u N0 2', 'u a0 2', 'u b0 2']) # NEW 6/14/2021
        par.append(['u N1 1', 'u a1 1', 'u b1 1'])
        #par.append(['ub N0 1', 'ub a0, 1', 'ub b0 1'])
        par.append(['d N0 1', 'd a0 1', 'd b0 1']) #,'d c0 1'])
        par.append(['d N0 2', 'd a0 2', 'd b0 2']) #NEW 6/14/2021
        par.append(['d N1 1', 'd a1 1', 'd b1 1'])
        #par.append(['db N0 1','db a0 1','db b0 1'])
        par.append(['s N0 1', 's a0 1', 's b0 1'])
        par.append(['s N1 1', 's a1 1', 's b1 1'])
        #par.append(['sb N0 1', 'sb a0 1', 'sb b0 1'])
        par.append(['widths1_uv', 'widths1_sea'])
        #par.append(['widths1_uv'])

        kind=1
        tag='Htildepi'

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

        replicas=self.get_replicas(wdir)
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
                    ax.hist(data[i][_],bins=50,color=c,histtype='step')
                    
                if '_' in _: _ = _.replace("_", " ")
                ax.set_xlabel(_)

        py.tight_layout()
        checkdir('%s/gallery'%wdir)
        py.savefig('%s/gallery/par-plot-Htildepi-%d.pdf'%(wdir,istep))
        py.close()


class SIDIS(CORE):

    def __init__(self,task,wdir,last=False,pdf_flav=None,ff_flav=None):

        if  task==0:
            self.msg='sidis.data_vs_thy'
            self.func=self.data_vs_thy
            self.loop_over_steps(wdir,None,last)

        if  task==1:
            self.msg='sidis.get_predictions'
            self.func=self.get_predictions
            self.pdf_flav=pdf_flav
            self.ff_flav=ff_flav
            hooks={}
            hooks['reaction']='sidis'
            hooks['sidis-pdf-flav']=pdf_flav
            hooks['sidis-ff-flav'] =ff_flav
            self.loop_over_steps(wdir,'full',last,hooks=hooks)

    def _data_vs_thy_3dbinning(self,wdir,istep,col,tar,had, depol):
        cluster,colors,nc,order = self.get_clusters(wdir,istep)
        predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
        if 'sidis' not in predictions['reactions']: return
        data=predictions['reactions']['sidis']
        for _ in data:
            predictions=copy.copy(data[_]['prediction-rep'])
            del data[_]['prediction-rep']
            del data[_]['residuals-rep']
            for ic in range(nc):
                predictions_ic=[predictions[i] for i in range(len(predictions)) if cluster[i]==ic]
                data[_]['thy-%d'%ic]=np.mean(predictions_ic, axis=0)
                data[_]['dthy-%d'%ic]=np.std(predictions_ic, axis=0)

                zbins=[]
                zbins.append([0.2,0.28])
                zbins.append([0.28,0.37])
                zbins.append([0.37,0.49])
                zbins.append([0.49,0.70])
        
                ptbins=[]
                #ptbins.append([0.00,0.23])
                ptbins.append([0.23,0.36])
                ptbins.append([0.36,0.54])
                ptbins.append([0.54,2.0])
                    
                ncols=len(ptbins)
                nrows=len(zbins)
                ddf = pd.DataFrame(data[_])
                if 'xmin' in ddf.columns and 'zmin' in ddf.columns and 'pTmin' in ddf.columns and not ((pd.DataFrame(data[_])).query('hadron=="%s"'%had)).empty and not ((pd.DataFrame(data[_])).query('target=="%s"'%tar)).empty:
                    
                    fig, ax = py.subplots(nrows, ncols, sharey ='row', sharex = 'col')
                    cnt=0
                    for j in range(len(zbins)):
                        for k in range(len(ptbins)):
                            ptmin,ptmax=ptbins[k]
                            zmin,zmax=zbins[j]
                            df = pd.DataFrame(data[_])
                            df=df.query('hadron=="%s"'%had)
                            df=df.query('col=="%s"'%col)
                            df=df.query('target=="%s"'%tar)
                            df=df.query('pT>%f and pT<%f'%(ptmin,ptmax))
                            df=df.query('z>%f and z<%f'%(zmin,zmax))
                            save(df.to_dict(orient='list'),'%s/npdata/%s-Mult-%s-%s_ptmin-%s_ptmax-%s_zmin-%s_zmax-%s_data-%s.dat'%(wdir,col,tar,had,ptmin,ptmax,zmin,zmax,_))
                        
                            for ic in range(nc):
                                c=colors[ic]
                                if c=='r': zorder=10
                                else:      zorder=0
                                thy=df['thy-%d'%ic]
                                dthy=df['dthy-%d'%ic]
                            pT = df.pT
                            value=df.value
                            alpha=df.alpha
                            x=df.x
                            ax[j,k].plot(x, thy, '-')
                            ax[j,k].fill_between(x, thy+dthy, thy-dthy, alpha =0.5)
                            ax[j,k].errorbar(x,value,yerr=alpha,fmt='%s.'%c)
                            if len(zbins)/(j+1) == len(zbins):
                                title='%0.2f < pT < %0.2f'%(ptmin, ptmax)
                                ax[j,k].set_title(title, fontsize=6)
                            if len(ptbins)/(k+1) == 1:
                                ax[j,k].yaxis.set_label_position('right')
                                ax[j,k].set_ylabel('%0.2f < z < %0.2f'%(zmin,zmax), fontsize=6, rotation=270, verticalalignment='bottom')
                            ax[j,k].axhline(0,0,1,color='black', linewidth=0.1)
                    
                    py.text(0.85, 0.85, r'$\pi$' + had[2:], transform=fig.transFigure)
                    ax[len(zbins)-1,len(ptbins)-1].set_xlabel(r'$x$',size=15)
                    obs=df.iloc[0]['obs']
                    if tar == 'p' or tar == 'proton':
                        if had == 'pi+':
                            fig.suptitle(r'%s 2020 %s Proton $\pi$+'%(col, obs))
                        if had == 'pi-':
                            fig.suptitle(r'%s 2020 %s Proton $\pi$-'%(col, obs))
                        if 'sivers' in obs:
                            py.text(0.035,0.8, r'$A^{sin(\phi-\phi_{s})}_{UT}$', rotation=90, transform=fig.transFigure)
                        if 'collins' in obs:
                            py.text(0.035,0.8, r'$A^{sin(\phi+\phi_{s})}_{UT}$', rotation=90, transform=fig.transFigure)
                    py.text(0.035,0.8, r'$A^{sin(\phi-\phi_{s})}_{UT}$', rotation=90, transform=fig.transFigure)
                    py.subplots_adjust(wspace=0, hspace=0)
                    checkdir('%s/gallery'%wdir)
                    py.savefig('%s/gallery/data-vs-thy-3d-%d-%s-%s-%s-%s.pdf'%(wdir,istep,col,tar,had,_))
                    py.close(fig)
                else:
                    continue

    def _data_vs_thy_unpol(self,wdir,istep,col,tar,had):
        cluster,colors,nc,order = self.get_clusters(wdir,istep)
        predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
        if 'sidis' not in predictions['reactions']: return
        data=predictions['reactions']['sidis']

        if col=='HERMES': col='hermes'

        for _ in data:

            predictions=copy.copy(data[_]['prediction-rep'])
            del data[_]['prediction-rep']
            del data[_]['residuals-rep']
            for ic in range(nc):
                predictions_ic=[predictions[i] for i in range(len(predictions))
                if cluster[i]==ic]
                data[_]['thy-%d'%ic]=np.mean(predictions_ic, axis=0)
                data[_]['dthy-%d'%ic]=np.std(predictions_ic, axis=0)

        zbins=[]
        #zbins.append([0.1,0.2])
        zbins.append([0.2,0.25])
        zbins.append([0.25,0.3])
        zbins.append([0.3,0.35])
        zbins.append([0.4,0.45])
        zbins.append([0.5,0.6])
        xbins=[]
        xbins.append([0.09,0.10])
        xbins.append([0.15,0.16])
        xbins.append([0.25,0.26])
        xbins.append([0.40,0.42])

        nbins=len(zbins)*len(xbins)
        ncols=4
        nrows=nbins/ncols
        if nbins%ncols>0:  nrows=+1
        fig = py.figure(figsize=(ncols*3,nrows*1.5))
        cnt=0
        AX={}
        for i in range(len(zbins)):
            for j in range(len(xbins)):
                zmin,zmax=zbins[i]
                xmin,xmax=xbins[j]
                for _ in data:
                    df=pd.DataFrame(data[_])
                    df=df.query('col=="%s"'%col)
                    df=df.query('target=="%s"'%tar)
                    df=df.query('hadron=="%s"'%had)
                    df=df.query('z>%f and z<%f'%(zmin,zmax))
                    df=df.query('x>%f and x<%f'%(xmin,xmax))

                    save(df.to_dict(orient='list'),'%s/npdata/%s-Mult-%s-%s_zmin-%s_zmax-%s_xmin-%s_xmax-%s.dat'%(wdir,col,tar,had,zmin,zmax,xmin,xmax))

                    if df.index.size<2: continue
                    key=(xmin,xmax,zmin,zmax)
                    if key not in AX:
                        cnt+=1
                        AX[key]=py.subplot(nrows,ncols,cnt)


                    Q2mean=np.mean(df['Q2'].values)
                    Xmean=np.mean(df['x'].values)
                    Zmean=np.mean(df['z'].values)


                    title='Q2=%0.2f x=%0.2f z=%0.2f'%(Q2mean,Xmean,Zmean)
                    AX[key].set_title(title)
                    for ic in range(nc):
                        c=colors[ic]
                        if c=='r': zorder=10
                        else:      zorder=0
                        thy=df['thy-%d'%ic]
                        dthy=df['dthy-%d'%ic]
                        pT = df.pT
                        value=df.value
                        alpha=df.alpha
                        #AX[key].errorbar(pT,value/thy,alpha/thy,fmt='%s.'%c) #EDIT
                        AX[key].errorbar(pT,value,yerr=alpha,fmt='%s.'%c)
                        AX[key].plot(pT,thy,'-')
                        AX[key].fill_between(pT,thy+dthy,thy-dthy)
                        if cnt==1: key0=key

        for _ in AX:
           AX[_].set_xticks([0.2,0.4,0.6,0.8])
           #AX[_].axhline(1,color='b',ls=':')
           #AX[_].set_ylim(0.8,1.2)
           AX[_].set_xlim(0,1)
           AX[_].set_xlabel(r'$P_{hT}$',size=15)
           AX[_].xaxis.set_label_coords(0.95, -0.05)

        py.tight_layout()
        checkdir('%s/gallery'%wdir)
        py.savefig('%s/gallery/data-vs-thy-sidis-%d-%s-%s-%s.pdf'%(wdir,istep,col,tar,had))
        py.close()
        

    def _data_vs_thy_sivers(self,wdir,istep,col,tar,had=None, depol=None):
        cluster,colors,nc,order = self.get_clusters(wdir,istep)
        predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
        if 'sidis' not in predictions['reactions']: return
        data=predictions['reactions']['sidis']

        for _ in data:

            predictions=copy.copy(data[_]['prediction-rep'])
            del data[_]['prediction-rep']
            del data[_]['residuals-rep']
            for ic in range(nc):
                predictions_ic=[predictions[i] for i in range(len(predictions))
                               if cluster[i]==ic]
                data[_]['thy-%d'%ic]=np.mean(predictions_ic, axis=0)
                data[_]['dthy-%d'%ic]=np.std(predictions_ic, axis=0)

        #for i in range(2000,2009):
        #    data[i] = pd.DataFrame(data[i])
        #for i in range(2020,2032):
        #    data[i]=pd.DataFrame(data[i])
        #for i in range(2046,2052):
        #    data[i] = pd.DataFrame(data[i])
        #for i in range(2026,2032):
        #    data[i]=pd.DataFrame(data[i])
        #for i in range(2512,2528):
        #    data[i]=pd.DataFrame(data[i])
        for i in it.chain(range(2000, 2009), range(2020,2032), range(2046,2052), range(2026,2032), range(2512,2528)):
            try:
                data[i]=pd.DataFrame(data[i])
            except:
                data[i]=pd.DataFrame(columns=['target', 'hadron', 'dependence', 'col', 'obs', 'dep', 'Dependence', 'x', 'pT', 'z'])
                print('Could not load file "%f"'%i)
        ncols=3
        nrows=1
        if col=='HERMES':
            col = 'hermes'
            if tar=='proton' or tar=='p':
                if had=='pi0':
                    if depol=='False':
                        py.figure(figsize=(ncols*4,nrows*4))
                        tab=pd.concat([data[2521].query('target=="proton"'),data[2522].query('target=="proton"'),data[2523].query('target=="proton"')])

                        for ic in range(nc):
                            cnt=0
                            for dep in ['x','z','pT']:
                                cnt+=1
                                if dep=='pT': _dep='PT'
                                else: _dep=dep
                                ax=py.subplot(nrows,ncols,cnt)
                                _tab=tab.query('dependence=="%s"'%dep)
                                pi0=_tab.query('hadron=="pi0"')
                                save(pi0.to_dict(orient='list'),'%s/npdata/HERMES-2020-Sivers-proton-pi0-%s.dat'%(wdir,dep))
                                #np.save('%s/nparrays/HERMES-Sivers-proton-pi0.npy'%wdir,pi0)
                                if dep=='pT': label='P_{hT}'
                                if dep=='x': label='x'
                                if dep=='z': label='z'
                                ax.errorbar(pi0[dep],pi0['value'],yerr=pi0['alpha'],fmt='g.')
                                ax.plot(pi0[dep],pi0['thy-%d'%ic],color='g',alpha=0.5)
                                ax.fill_between(pi0[dep],(pi0['thy-%d'%ic]-pi0['dthy-%d'%ic]),(pi0['thy-%d'%ic]+pi0['dthy-%d'%ic]),color='g',alpha=0.5)
                                ax.set_xlabel(r'$%s$'%label,size=20); ax.set_ylabel(r'$A_{UT}^{\sin(\phi_h-\phi_S)}$',size=20)
                                ax.set_title('HERMES p')

                        py.tight_layout()
                        checkdir('%s/gallery'%wdir)
                        py.savefig('%s/gallery/data-vs-thy-sidis-sivers-%d-%s-2020-%s-%s.pdf'%(wdir,istep,col,tar,had))
                        py.close()
                        
                elif depol==None:
                    py.figure(figsize=(ncols*4,nrows*4))
                    tab=pd.concat([data[2006].query('target=="proton"'),data[2007].query('target=="proton"'),data[2008].query('target=="proton"')])

                    for ic in range(nc):
                        cnt=0
                        for dep in ['x','z','pT']:
                            cnt+=1
                            if dep=='pT': _dep='PT'
                            else: _dep=dep
                            ax=py.subplot(nrows,ncols,cnt)
                            _tab=tab.query('Dependence=="%s"'%_dep)
                            pi0=_tab.query('hadron=="pi0"')
                            save(pi0.to_dict(orient='list'),'%s/npdata/HERMES-Sivers-proton-pi0-%s.dat'%(wdir,dep))
                            #np.save('%s/nparrays/HERMES-Sivers-proton-pi0.npy'%wdir,pi0)
                            if dep=='pT': label='P_{hT}'
                            if dep=='x': label='x'
                            if dep=='z': label='z'
                            ax.errorbar(pi0[dep],pi0['value'],yerr=pi0['alpha'],fmt='g.')
                            ax.plot(pi0[dep],pi0['thy-%d'%ic],color='g',alpha=0.5)
                            ax.fill_between(pi0[dep],(pi0['thy-%d'%ic]-pi0['dthy-%d'%ic]),(pi0['thy-%d'%ic]+pi0['dthy-%d'%ic]),color='g',alpha=0.5)
                            ax.set_xlabel(r'$%s$'%label,size=20); ax.set_ylabel(r'$A_{UT}^{\sin(\phi_h-\phi_S)}$',size=20)
                            ax.set_title('HERMES p')

                    py.tight_layout()
                    checkdir('%s/gallery'%wdir)
                    py.savefig('%s/gallery/data-vs-thy-sidis-sivers-%d-%s-%s-%s.pdf'%(wdir,istep,col,tar,had))
                    py.close()
                else:
                    py.figure(figsize=(ncols*4,nrows*4))
                    tab=pd.concat([data[2000].query('target=="proton"'),data[2001].query('target=="proton"'),data[2002].query('target=="proton"'), \
                    data[2003].query('target=="proton"'),data[2004].query('target=="proton"'),data[2005].query('target=="proton"')])

                    for ic in range(nc):
                        cnt=0
                        for dep in ['x','z','pT']:
                            cnt+=1
                            if dep=='pT': _dep='PT'
                            else: _dep=dep
                            ax=py.subplot(nrows,ncols,cnt)
                            _tab=tab.query('Dependence=="%s"'%_dep)
                            pip=_tab.query('hadron=="pi+"')
                            pim=_tab.query('hadron=="pi-"')
                            save(pip.to_dict(orient='list'),'%s/npdata/HERMES-Sivers-proton-pip-%s.dat'%(wdir,dep))
                            save(pim.to_dict(orient='list'),'%s/npdata/HERMES-Sivers-proton-pim-%s.dat'%(wdir,dep))
                            #np.save('%s/nparrays/HERMES-Sivers-proton-pip.npy'%wdir,pip)
                            #np.save('%s/nparrays/HERMES-Sivers-proton-pim.npy'%wdir,pim)
                            if dep=='pT': label='P_{hT}'
                            if dep=='x': label='x'
                            if dep=='z': label='z'
                            ax.errorbar(pip[dep],pip['value'],yerr=pip['alpha'],fmt='b.')
                            ax.errorbar(pim[dep],pim['value'],yerr=pim['alpha'],fmt='r.')
                            ax.plot(pip[dep],pip['thy-%d'%ic],color='b',alpha=0.5)
                            ax.plot(pim[dep],pim['thy-%d'%ic],color='r',alpha=0.5)
                            ax.fill_between(pip[dep],(pip['thy-%d'%ic]-pip['dthy-%d'%ic]),(pip['thy-%d'%ic]+pip['dthy-%d'%ic]),color='b',alpha=0.5)
                            ax.fill_between(pim[dep],(pim['thy-%d'%ic]-pim['dthy-%d'%ic]),(pim['thy-%d'%ic]+pim['dthy-%d'%ic]),color='r',alpha=0.5)
                            ax.set_xlabel(r'$%s$'%label,size=20); ax.set_ylabel(r'$A_{UT}^{\sin(\phi_h-\phi_S)}$',size=20)
                            ax.set_title('HERMES p')

                    py.tight_layout()
                    checkdir('%s/gallery'%wdir)
                    py.savefig('%s/gallery/data-vs-thy-sidis-sivers-%d-%s-%s-%s.pdf'%(wdir,istep,col,tar,had))
                    py.close()
                    
                                
                                
                    
                    
        else:
            col = 'compass'
            if tar=='proton' and had=='h+-':
                py.figure(figsize=(ncols*4,nrows*4))
                tab=pd.concat([data[2020].query('target=="proton"'),data[2021].query('target=="proton"'),data[2022].query('target=="proton"'), \
                data[2023].query('target=="proton"'),data[2024].query('target=="proton"'),data[2025].query('target=="proton"')])

                for ic in range(nc):
                    cnt=0
                    for dep in ['x','z','pT']:
                        cnt+=1
                        if dep=='pT': _dep='PT'
                        else: _dep=dep
                        if dep=='pT': label='P_{hT}'
                        if dep=='x': label='x'
                        if dep=='z': label='z'
                        ax=py.subplot(nrows,ncols,cnt)
                        _tab=tab.query('Dependence=="%s"'%_dep)
                        pip=_tab.query('hadron=="h+"')
                        pim=_tab.query('hadron=="h-"')
                        ax.errorbar(pip[dep],pip['value'],yerr=pip['alpha'],fmt='b.')
                        ax.errorbar(pim[dep],pim['value'],yerr=pim['alpha'],fmt='r.')
                        ax.plot(pip[dep],pip['thy-%d'%ic],color='b',alpha=0.5)
                        ax.plot(pim[dep],pim['thy-%d'%ic],color='r',alpha=0.5)
                        ax.fill_between(pip[dep],(pip['thy-%d'%ic]-pip['dthy-%d'%ic]),(pip['thy-%d'%ic]+pip['dthy-%d'%ic]),color='b',alpha=0.5)
                        ax.fill_between(pim[dep],(pim['thy-%d'%ic]-pim['dthy-%d'%ic]),(pim['thy-%d'%ic]+pim['dthy-%d'%ic]),color='r',alpha=0.5)
                        ax.set_xlabel(r'$%s$'%label,size=20); ax.set_ylabel(r'$A_{UT}^{\sin(\phi_h-\phi_S)}$',size=20)
                        ax.set_title('COMPASS p')

                py.tight_layout()
                checkdir('%s/gallery'%wdir)
                py.savefig('%s/gallery/data-vs-thy-sidis-sivers-%d-%s-%s-%s.pdf'%(wdir,istep,col,tar,had))
                py.close()

                #This code needs q2bins
                q2bins=[]
                q2bins.append([1.0,4.0])
                q2bins.append([4.0,6.25])
                q2bins.append([6.25,16.0])
                q2bins.append([16.0,81.0])

                nbins=len(q2bins)*3
                ncols= 3
                nrows=nbins/ncols
                if nbins%ncols>0: nrows +=1
                py.figure(figsize=(ncols*4,nrows*4))

                tab=pd.concat([data[2046].query('target=="proton"'),data[2047].query('target=="proton"'),data[2048].query('target=="proton"'), \
                data[2049].query('target=="proton"'),data[2050].query('target=="proton"'),data[2051].query('target=="proton"')])

                for ic in range(nc):
                    cnt=0
                    for i in range(len(q2bins)):
                        for dep in ['x','z','pT']:
                                cnt+=1
                                if dep=='pT': _dep='PT'
                                else: _dep=dep
                                if dep=='pT': label='P_{hT}'
                                if dep=='x': label='x'
                                if dep=='z': label='z'
                                #if dep=='Q2': label='Q^2'
                                #add for loop for over q2bins and the have plot stuff go under this for loop
                                #for i in range(len(q2bins)):
                                q2min,q2max=q2bins[i]
                                ax=py.subplot(nrows,ncols,cnt)
                                _tab=tab.query('Dependence=="%s"'%_dep)
                                _tab=_tab.query('Q2>%f and Q2<%f'%(q2min,q2max))
                                pip=_tab.query('hadron=="h+"')
                                pim=_tab.query('hadron=="h-"')
                                ax.errorbar(pip[dep],pip['value'],yerr=pip['alpha'],fmt='b.')
                                ax.errorbar(pim[dep],pim['value'],yerr=pim['alpha'],fmt='r.')
                                ax.plot(pip[dep],pip['thy-%d'%ic],color='b',alpha=0.5)
                                ax.plot(pim[dep],pim['thy-%d'%ic],color='r',alpha=0.5)
                                ax.fill_between(pip[dep],(pip['thy-%d'%ic]-pip['dthy-%d'%ic]),(pip['thy-%d'%ic]+pip['dthy-%d'%ic]),color='b',alpha=0.5)
                                ax.fill_between(pim[dep],(pim['thy-%d'%ic]-pim['dthy-%d'%ic]),(pim['thy-%d'%ic]+pim['dthy-%d'%ic]),color='r',alpha=0.5)
                                ax.set_xlabel(r'$%s$'%label,size=20); ax.set_ylabel(r'$A_{UT}^{\sin(\phi_h-\phi_S)}$',size=20)
                                ax.set_title('COMPASS p %0.2f$<Q^2<$%0.2f'%(q2min,q2max))

                py.tight_layout()
                checkdir('%s/gallery'%wdir)
                py.savefig('%s/gallery/data-vs-thy-sidis-sivers-%d-%s-2-%s-%s.pdf'%(wdir,istep,col,tar,had))
                py.close()

            elif tar=='proton' and had=='pi+-':
                py.figure(figsize=(ncols*4,nrows*4))
                tab=pd.concat([data[2512].query('target=="proton"'),data[2513].query('target=="proton"'),data[2514].query('target=="proton"'), \
                data[2515].query('target=="proton"'),data[2516].query('target=="proton"'),data[2517].query('target=="proton"')])

                for ic in range(nc):
                    cnt=0
                    for dep in ['x','z','pT']:
                        cnt+=1
                        if dep=='pT': _dep='pt'
                        else: _dep=dep
                        if dep=='pT': label='P_{hT}'
                        if dep=='x': label='x'
                        if dep=='z': label='z'
                        ax=py.subplot(nrows,ncols,cnt)
                        _tab=tab.query('dependence=="%s"'%_dep)
                        pip=_tab.query('hadron=="pi+"')
                        pim=_tab.query('hadron=="pi-"')
                        save(pip.to_dict(orient='list'),'%s/npdata/COMPASS-Sivers-proton-pip-%s.dat'%(wdir,dep))
                        save(pim.to_dict(orient='list'),'%s/npdata/COMPASS-Sivers-proton-pim-%s.dat'%(wdir,dep))
                        #np.save('%s/nparrays/COMPASS-Sivers-proton-pip.npy'%wdir,pip)
                        #np.save('%s/nparrays/COMPASS-Sivers-proton-pim.npy'%wdir,pim)
                        ax.errorbar(pip[dep],pip['value'],yerr=pip['alpha'],fmt='b.')
                        ax.errorbar(pim[dep],pim['value'],yerr=pim['alpha'],fmt='r.')
                        ax.plot(pip[dep],pip['thy-%d'%ic],color='b',alpha=0.5)
                        ax.plot(pim[dep],pim['thy-%d'%ic],color='r',alpha=0.5)
                        ax.fill_between(pip[dep],(pip['thy-%d'%ic]-pip['dthy-%d'%ic]),(pip['thy-%d'%ic]+pip['dthy-%d'%ic]),color='b',alpha=0.5)
                        ax.fill_between(pim[dep],(pim['thy-%d'%ic]-pim['dthy-%d'%ic]),(pim['thy-%d'%ic]+pim['dthy-%d'%ic]),color='r',alpha=0.5)
                        ax.set_xlabel(r'$%s$'%label,size=20); ax.set_ylabel(r'$A_{UT}^{\sin(\phi_h-\phi_S)}$',size=20)
                        ax.set_title('COMPASS p')
                py.tight_layout()
                checkdir('%s/gallery'%wdir)
                py.savefig('%s/gallery/data-vs-thy-sidis-sivers-%d-%s-%s-%s.pdf'%(wdir,istep,col,tar,had))
                py.close()
            else:
                py.figure(figsize=(ncols*4,nrows*4))
                tab=pd.concat([data[2026].query('target=="deuteron"'),data[2027].query('target=="deuteron"'),data[2028].query('target=="deuteron"'), \
                data[2029].query('target=="deuteron"'),data[2030].query('target=="deuteron"'),data[2031].query('target=="deuteron"')])

                for ic in range(nc):
                    cnt=0
                    for dep in ['x','z','pT']:
                        cnt+=1
                        if dep=='pT': _dep='PT'
                        else: _dep=dep
                        if dep=='pT': label='P_{hT}'
                        if dep=='x': label='x'
                        if dep=='z': label='z'
                        ax=py.subplot(nrows,ncols,cnt)
                        _tab=tab.query('Dependence=="%s"'%_dep)
                        pip=_tab.query('hadron=="pi+"')
                        pim=_tab.query('hadron=="pi-"')
                        save(pip.to_dict(orient='list'),'%s/npdata/COMPASS-Sivers-deuteron-pip-%s.dat'%(wdir,dep))
                        save(pim.to_dict(orient='list'),'%s/npdata/COMPASS-Sivers-deuteron-pim-%s.dat'%(wdir,dep))
                        #np.save('%s/nparrays/COMPASS-Sivers-deuteron-pip.npy'%wdir,pip)
                        #np.save('%s/nparrays/COMPASS-Sivers-deuteron-pim.npy'%wdir,pim)
                        ax.errorbar(pip[dep],pip['value'],yerr=pip['alpha'],fmt='b.')
                        ax.errorbar(pim[dep],pim['value'],yerr=pim['alpha'],fmt='r.')
                        ax.plot(pip[dep],pip['thy-%d'%ic],color='b',alpha=0.5)
                        ax.plot(pim[dep],pim['thy-%d'%ic],color='r',alpha=0.5)
                        ax.fill_between(pip[dep],(pip['thy-%d'%ic]-pip['dthy-%d'%ic]),(pip['thy-%d'%ic]+pip['dthy-%d'%ic]),color='b',alpha=0.5)
                        ax.fill_between(pim[dep],(pim['thy-%d'%ic]-pim['dthy-%d'%ic]),(pim['thy-%d'%ic]+pim['dthy-%d'%ic]),color='r',alpha=0.5)
                        ax.set_xlabel(r'$%s$'%label,size=20); ax.set_ylabel(r'$A_{UT}^{\sin(\phi_h-\phi_S)}$',size=20)
                        ax.set_title('COMPASS d')
                py.tight_layout()
                checkdir('%s/gallery'%wdir)
                py.savefig('%s/gallery/data-vs-thy-sidis-sivers-%d-%s-%s-%s.pdf'%(wdir,istep,col,tar,had))
                py.close()

    def _data_vs_thy_collins(self,wdir,istep,col,tar,had=None, depol=None):
        cluster,colors,nc,order = self.get_clusters(wdir,istep)
        predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
        if 'sidis' not in predictions['reactions']: return
        data=predictions['reactions']['sidis']

        for _ in data:

            predictions=copy.copy(data[_]['prediction-rep'])
            del data[_]['prediction-rep']
            del data[_]['residuals-rep']
            for ic in range(nc):
                predictions_ic=[predictions[i] for i in range(len(predictions))
                               if cluster[i]==ic]
                data[_]['thy-%d'%ic]=np.mean(predictions_ic, axis=0)
                data[_]['dthy-%d'%ic]=np.std(predictions_ic, axis=0)

        #for i in range(4000,4006):
        #    data[i] = pd.DataFrame(data[i])
        for i in [3006, 3014, 3015, 3026, 3000, 3003, 3016, 3004,3018, 3027, 3025, 3010, 3012, 3005, 3013, 3703, 3704, 3705, 4000, 4001, 4002, 4003, 4004, 4005]:
            try:
                data[i]=pd.DataFrame(data[i])
            except:
                data[i]=pd.DataFrame(columns=['target', 'hadron', 'dependence', 'col', 'obs', 'dep', 'Dependence'])
                print('Could not load file "%f"'%i)
        ncols=3
        nrows=1
        if col=='HERMES':
            if had=='pi0':
                if depol=='False':
                    py.figure(figsize=(ncols*4,nrows*4))
                    tab=pd.concat([data[3705].query('target=="p"'),data[3703].query('target=="p"'),data[3704].query('target=="p"')])

                    for ic in range(nc):
                        cnt=0
                        for dep in ['x','z','pT']:
                            cnt+=1
                            if dep=='pT': _dep='pt'
                            else: _dep=dep
                            ax=py.subplot(nrows,ncols,cnt)
                            _tab=tab.query('dependence=="%s"'%dep)
                            pi0=_tab.query('hadron=="pi0"')
                            if dep=='pT': label='P_{hT}'
                            if dep=='x': label='x'
                            if dep=='z': label='z'
                            save(pi0.to_dict(orient='list'),'%s/npdata/HERMES-Collins-2020-proton-pi0-%s.dat'%(wdir,dep))
                            #np.save('%s/nparrays/HERMES-Collins-proton-pi0.npy'%wdir,pi0)
                            ax.errorbar(pi0[dep],pi0['value'],yerr=pi0['alpha'],fmt='g.')
                            ax.plot(pi0[dep],pi0['thy-%d'%ic],color='g',alpha=0.5)
                            ax.fill_between(pi0[dep],(pi0['thy-%d'%ic]-pi0['dthy-%d'%ic]),(pi0['thy-%d'%ic]+pi0['dthy-%d'%ic]),color='g',alpha=0.5)
                            ax.set_xlabel(r'$%s$'%label,size=20); ax.set_ylabel(r'$A_{UT}^{\sin(\phi_h+\phi_S)}$',size=20)
                            ax.set_title('HERMES p')

                    py.tight_layout()
                    checkdir('%s/gallery'%wdir)
                    py.savefig('%s/gallery/data-vs-thy-sidis-collins-%d-%s-2020-%s-%s.pdf'%(wdir,istep,col,tar,had))
                    py.close()
            elif depol =='None':
                py.figure(figsize=(ncols*4,nrows*4))
                tab=pd.concat([data[3006].query('target=="proton"'),data[3014].query('target=="proton"'),data[3015].query('target=="proton"')])

                for ic in range(nc):
                    cnt=0
                    for dep in ['x','z','pT']:
                        cnt+=1
                        if dep=='pT': _dep='pt'
                        else: _dep=dep
                        ax=py.subplot(nrows,ncols,cnt)
                        _tab=tab.query('dependence=="%s"'%_dep)
                        pi0=_tab.query('hadron=="pi0"')
                        if dep=='pT': label='P_{hT}'
                        if dep=='x': label='x'
                        if dep=='z': label='z'
                        save(pi0.to_dict(orient='list'),'%s/npdata/HERMES-Collins-proton-pi0-%s.dat'%(wdir,dep))
                        #np.save('%s/nparrays/HERMES-Collins-proton-pi0.npy'%wdir,pi0)
                        ax.errorbar(pi0[dep],pi0['value'],yerr=pi0['alpha'],fmt='g.')
                        ax.plot(pi0[dep],pi0['thy-%d'%ic],color='g',alpha=0.5)
                        ax.fill_between(pi0[dep],(pi0['thy-%d'%ic]-pi0['dthy-%d'%ic]),(pi0['thy-%d'%ic]+pi0['dthy-%d'%ic]),color='g',alpha=0.5)
                        ax.set_xlabel(r'$%s$'%label,size=20); ax.set_ylabel(r'$A_{UT}^{\sin(\phi_h+\phi_S)}$',size=20)
                        ax.set_title('HERMES p')

                py.tight_layout()
                checkdir('%s/gallery'%wdir)
                py.savefig('%s/gallery/data-vs-thy-sidis-collins-%d-%s-%s-%s.pdf'%(wdir,istep,col,tar,had))
                py.close()
            else:
                py.figure(figsize=(ncols*4,nrows*4))
                tab=pd.concat([data[3026].query('target=="proton"'),data[3000].query('target=="proton"'),data[3003].query('target=="proton"'), \
                    data[3016].query('target=="proton"'),data[3004].query('target=="proton"'),data[3018].query('target=="proton"')])
                for ic in range(nc):
                    cnt=0
                    for dep in ['x','z','pT']:
                        cnt+=1
                        if dep=='pT': _dep='pt'
                        else: _dep=dep
                        ax=py.subplot(nrows,ncols,cnt)
                        _tab=tab.query('dependence=="%s"'%_dep)
                        pip=_tab.query('hadron=="pi+"')
                        pim=_tab.query('hadron=="pi-"')
                        if dep=='pT': label='P_{hT}'
                        if dep=='x': label='x'
                        if dep=='z': label='z'
                        save(pip.to_dict(orient='list'),'%s/npdata/HERMES-Collins-proton-pip-%s.dat'%(wdir,dep))
                        save(pim.to_dict(orient='list'),'%s/npdata/HERMES-Collins-proton-pim-%s.dat'%(wdir,dep))
                        #np.save('%s/nparrays/HERMES-Collins-proton-pip.npy'%wdir,pip)
                        #np.save('%s/nparrays/HERMES-Collins-proton-pim.npy'%wdir,pim)
                        ax.errorbar(pip[dep],pip['value'],yerr=pip['alpha'],fmt='b.')
                        ax.errorbar(pim[dep],pim['value'],yerr=pim['alpha'],fmt='r.')
                        ax.plot(pip[dep],pip['thy-%d'%ic],color='b',alpha=0.5)
                        ax.plot(pim[dep],pim['thy-%d'%ic],color='r',alpha=0.5)
                        ax.fill_between(pip[dep],(pip['thy-%d'%ic]-pip['dthy-%d'%ic]),(pip['thy-%d'%ic]+pip['dthy-%d'%ic]),color='b',alpha=0.5)
                        ax.fill_between(pim[dep],(pim['thy-%d'%ic]-pim['dthy-%d'%ic]),(pim['thy-%d'%ic]+pim['dthy-%d'%ic]),color='r',alpha=0.5)
                        ax.set_xlabel(r'$%s$'%label,size=20); ax.set_ylabel(r'$A_{UT}^{\sin(\phi_h+\phi_S)}$',size=20)
                        ax.set_title('HERMES p')
                py.tight_layout()
                checkdir('%s/gallery'%wdir)
                py.savefig('%s/gallery/data-vs-thy-sidis-collins-%d-%s-%s-pi+-.pdf'%(wdir,istep,col,tar))
                py.close()

        else:
            col='compass'
            if tar=='proton':
                py.figure(figsize=(ncols*4,nrows*4))
                tab=pd.concat([data[3027].query('target=="proton"'),data[3025].query('target=="proton"'),data[3010].query('target=="proton"'), \
                    data[3012].query('target=="proton"'),data[3005].query('target=="proton"'),data[3013].query('target=="proton"')])

                for ic in range(nc):
                    cnt=0
                    for dep in ['x','z','pT']:
                        cnt+=1
                        if dep=='pT': _dep='pt'
                        else: _dep=dep
                        ax=py.subplot(nrows,ncols,cnt)
                        _tab=tab.query('dependence=="%s"'%_dep)
                        pip=_tab.query('hadron=="pi+"')
                        pim=_tab.query('hadron=="pi-"')
                        if dep=='pT': label='P_{hT}'
                        if dep=='x': label='x'
                        if dep=='z': label='z'
                        save(pip.to_dict(orient='list'),'%s/npdata/COMPASS-Collins-proton-pip-%s.dat'%(wdir,dep))
                        save(pim.to_dict(orient='list'),'%s/npdata/COMPASS-Collins-proton-pim-%s.dat'%(wdir,dep))
                        #np.save('%s/nparrays/COMPASS-Collins-proton-pip.npy'%wdir,pip)
                        #np.save('%s/nparrays/COMPASS-Collins-proton-pim.npy'%wdir,pim)
                        ax.errorbar(pip[dep],-pip['value'],yerr=pip['alpha'],fmt='b.')
                        ax.errorbar(pim[dep],-pim['value'],yerr=pim['alpha'],fmt='r.')
                        ax.plot(pip[dep],-pip['thy-%d'%ic],color='b',alpha=0.5)
                        ax.plot(pim[dep],-pim['thy-%d'%ic],color='r',alpha=0.5)
                        ax.fill_between(pip[dep],-(pip['thy-%d'%ic]-pip['dthy-%d'%ic]),-(pip['thy-%d'%ic]+pip['dthy-%d'%ic]),color='b',alpha=0.5)
                        ax.fill_between(pim[dep],-(pim['thy-%d'%ic]-pim['dthy-%d'%ic]),-(pim['thy-%d'%ic]+pim['dthy-%d'%ic]),color='r',alpha=0.5)
                        ax.set_xlabel(r'$%s$'%label,size=20); ax.set_ylabel(r'$A_{UT}^{\sin(\phi_h+\phi_S)}$',size=20)
                        ax.set_title('COMPASS p')
                py.tight_layout()
                checkdir('%s/gallery'%wdir)
                py.savefig('%s/gallery/data-vs-thy-sidis-collins-%d-%s-%s-pi+-.pdf'%(wdir,istep,col,tar))
                py.close()
            else:
                py.figure(figsize=(ncols*4,nrows*4))
                tab=pd.concat([data[4000].query('target=="deuteron"'),data[4001].query('target=="deuteron"'),data[4002].query('target=="deuteron"'), \
                    data[4003].query('target=="deuteron"'),data[4004].query('target=="deuteron"'),data[4005].query('target=="deuteron"')])

                for ic in range(nc):
                    cnt=0
                    for dep in ['x','z','pT']:
                        cnt+=1
                        if dep=='pT': _dep='pT'
                        else: _dep=dep
                        if dep=='pT': label='P_{hT}'
                        if dep=='x': label='x'
                        if dep=='z': label='z'
                        ax=py.subplot(nrows,ncols,cnt)
                        _tab=tab.query('dependence=="%s"'%_dep)
                        pip=_tab.query('hadron=="pi+"')
                        pim=_tab.query('hadron=="pi-"')
                        save(pip.to_dict(orient='list'),'%s/npdata/COMPASS-Collins-deuteron-pip-%s.dat'%(wdir,dep))
                        save(pim.to_dict(orient='list'),'%s/npdata/COMPASS-Collins-deuteron-pim-%s.dat'%(wdir,dep))
                        #np.save('%s/nparrays/COMPASS-Collins-deuteron-pip.npy'%wdir,pip)
                        #np.save('%s/nparrays/COMPASS-Collins-deuteron-pim.npy'%wdir,pim)
                        ax.errorbar(pip[dep],-pip['value'],yerr=pip['alpha'],fmt='b.')
                        ax.errorbar(pim[dep],-pim['value'],yerr=pim['alpha'],fmt='r.')
                        ax.plot(pip[dep],-pip['thy-%d'%ic],color='b',alpha=0.5)
                        ax.plot(pim[dep],-pim['thy-%d'%ic],color='r',alpha=0.5)
                        ax.fill_between(pip[dep],-(pip['thy-%d'%ic]-pip['dthy-%d'%ic]),-(pip['thy-%d'%ic]+pip['dthy-%d'%ic]),color='b',alpha=0.5)
                        ax.fill_between(pim[dep],-(pim['thy-%d'%ic]-pim['dthy-%d'%ic]),-(pim['thy-%d'%ic]+pim['dthy-%d'%ic]),color='r',alpha=0.5)
                        ax.set_xlabel(r'$%s$'%label,size=20); ax.set_ylabel(r'$A_{UT}^{\sin(\phi_h+\phi_S)}$',size=20)
                        ax.set_title('COMPASS d')
                py.tight_layout()
                checkdir('%s/gallery'%wdir)
                py.savefig('%s/gallery/data-vs-thy-sidis-collins-%d-%s-%s-pi+-.pdf'%(wdir,istep,col,tar))
                py.close()

    def _data_vs_thy_sinphis(self,wdir,istep,col,tar,had=None):
        cluster,colors,nc,order = self.get_clusters(wdir,istep)
        predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
        if 'sidis' not in predictions['reactions']: return
        data=predictions['reactions']['sidis']

        for _ in data:

            predictions=copy.copy(data[_]['prediction-rep'])
            del data[_]['prediction-rep']
            del data[_]['residuals-rep']
            for ic in range(nc):
                predictions_ic=[predictions[i] for i in range(len(predictions))
                               if cluster[i]==ic]
                data[_]['thy-%d'%ic]=np.mean(predictions_ic, axis=0)
                data[_]['dthy-%d'%ic]=np.std(predictions_ic, axis=0)

        for i in it.chain(range(9000,9110,11),range(10010,10076,11)):
            try:
                data[i]=pd.DataFrame(data[i])
            except:
                data[i]=pd.DataFrame(['x', 'z', 'pT', 'target', 'hadron', 'dependence', 'value', 'alpha'])
                print('Could not load file "%f"'%i)
        #for i in it.chain(range(9011,9110,11),range(10010,10076,11)):
        #    try:
        #        tab+=pd.concat([data[i].query('target=="proton"')])
        #    except: continue
        if col =='COMPASS':
            tab=pd.concat([data[9011].query('target=="proton"'),data[9022].query('target=="proton"'),data[9033].query('target=="proton"'), data[9044].query('target=="proton"')])
            hadlist = ['h+', 'h-']
        elif col =='HERMES' or col =='hermes':
            tab=pd.concat([data[9055].query('target=="proton"'),data[9066].query('target=="proton"'),data[9088].query('target=="proton"'),data[9099].query('target=="proton"'),data[10021].query('target=="proton"'),data[10032].query('target=="proton"')]) #data[9077].query('target=="proton"'), data[10010].query('target=="proton"'),,data[10043].query('target=="proton"')])
            hadlist = ['pi+', 'pi-', 'pi0']
            
        i = 0
        colorlist = ['b', 'r', 'g']
        for had in hadlist:
            i += 1
            ncols=2
            nrows=1
            py.figure(figsize=(ncols*4,nrows*4))
            for ic in range(nc):
                cnt=0
                for dep in ['x', 'z']:
                    cnt+=1
                    ax=py.subplot(nrows,ncols,cnt)
                    _dep=dep
                    _tab=tab.query('dependence=="%s"'%_dep)
                    pip=_tab.query('hadron=="%s"'%had)
                    if dep=='x': label='x'
                    if dep=='z': label='z'
                    save(pip.to_dict(orient='list'),'%s/npdata/COMPASS-sinphiS-proton-pip-%s.dat'%(wdir,dep))
                    ax.errorbar(pip[dep],-pip['value'],yerr=pip['alpha'],fmt='%s.'%colorlist[i-1])
                    ax.plot(pip[dep],-pip['thy-%d'%ic],color='%s'%colorlist[i-1],alpha=0.5)
                    ax.fill_between(pip[dep],-(pip['thy-%d'%ic]-pip['dthy-%d'%ic]),-(pip['thy-%d'%ic]+pip['dthy-%d'%ic]),color='%s'%colorlist[i-1],alpha=0.5)
                    ax.set_xlabel(r'$%s$'%label,size=20); ax.set_ylabel(r'$A_{UT}^{\sin(\phi_{S})}$',size=20)

            py.tight_layout()
            checkdir('%s/gallery'%wdir)
            py.savefig('%s/gallery/data-vs-thy-sidis-sinphis-%d-%s-%s-%s.pdf'%(wdir,istep,col,tar,had))
            py.close()

    def data_vs_thy(self,wdir,istep):
        #--Calling unpolarized data
        #self._data_vs_thy_unpol(wdir,istep,'HERMES','proton','pi+')
        #self._data_vs_thy_unpol(wdir,istep,'HERMES','proton','pi-')
        #self._data_vs_thy_unpol(wdir,istep,'HERMES','deuteron','pi+')
        #self._data_vs_thy_unpol(wdir,istep,'HERMES','deuteron','pi-')
        #self._data_vs_thy_unpol(wdir,istep,'HERMES','proton','k+')
        #self._data_vs_thy_unpol(wdir,istep,'HERMES','proton','k-')
        #self._data_vs_thy_unpol(wdir,istep,'HERMES','deuteron','k+')
        #self._data_vs_thy_unpol(wdir,istep,'HERMES','deuteron','k-')


        #--Calling Sivers data
        self._data_vs_thy_sivers(wdir, istep, 'HERMES', 'proton', 'pi0')
        #self._data_vs_thy_sivers(wdir, istep, 'HERMES', 'proton', 'pi+-')
        #self._data_vs_thy_sivers(wdir, istep, 'COMPASS', 'proton','h+-')
        self._data_vs_thy_sivers(wdir, istep, 'COMPASS', 'deuteron','pi+-')
        self._data_vs_thy_sivers(wdir, istep, 'COMPASS', 'proton','pi+-')

        #--Calling Collins Data
        #self._data_vs_thy_collins(wdir,istep,'HERMES', 'proton', 'pi+-')
        self._data_vs_thy_collins(wdir,istep,'HERMES', 'proton', 'pi0')
        self._data_vs_thy_collins(wdir,istep,'COMPASS','proton')
        self._data_vs_thy_collins(wdir,istep,'COMPASS','deuteron')
        
        self._data_vs_thy_3dbinning(wdir,istep,'HERMES','p','pi+', 'FALSE')
        self._data_vs_thy_3dbinning(wdir,istep,'HERMES','p','pi-', 'FALSE')
        self._data_vs_thy_3dbinning(wdir,istep,'HERMES','proton','pi+', 'FALSE')
        self._data_vs_thy_3dbinning(wdir,istep,'HERMES','proton','pi-', 'FALSE')

        #--Calling SinPhiS data
        #self._data_vs_thy_sinphis(wdir,istep,'COMPASS','proton','pi+-')
        self._data_vs_thy_sinphis(wdir,istep,'HERMES','proton','pi+-')

    def get_predictions(self,wdir,istep):
        #if istep!=6: return
        obsres={}
        if 'sidis'  in conf['datasets'] : obsres['sidis']  =self.resman.sidisres


        #--setup big table to store all we want
        data={}
        data['order']=self.order
        data['params']=[]
        data['reactions']={}
        data['res']=[]
        data['rres']=[]
        data['nres']=[]

        for _ in obsres:
            tabs=copy.copy(obsres[_].tabs)
            #--create a space to store all the predictions from replicas
            for idx in tabs:
                tabs[idx]['prediction-rep']=[]
                tabs[idx]['residuals-rep']=[]
            data['reactions'][_]=tabs

        replicas=self.get_replicas(wdir)

        cnt=0
        for replica in replicas:
            cnt+=1
            lprint('%d/%d'%(cnt,len(replicas)))

            #--retrive the parameters for current step and current replica
            step=conf['steps'][istep]
            if 'fix parameters' in step:
                for dist in step['fix parameters']:
                    for par in step['fix parameters'][dist]:
                        #--set prior parameters values for passive distributions
                        for iistep in step['dep']:
                            prior_order=replica['order'][iistep]
                            prior_params=replica['params'][iistep]
                            for i in range(len(prior_order)):
                                _,_dist,_par = prior_order[i]
                                if  dist==_dist and par==_par:
                                    conf['params'][dist][par]['value']=prior_params[i]


            self.parman.par=copy.copy(replica['params'][istep])
            data['params']=np.append(data['params'],self.parman.par)

            #--compute residuals (==theory)
            res,rres,nres=self.resman.get_residuals(self.parman.par)
            data['res'].append(res)
            data['rres'].append(rres)
            data['nres'].append(nres)

            #--save predictions of the current step and current replica at data
            for _ in obsres:
                for idx in data['reactions'][_]:
                    prediction=copy.copy(obsres[_].tabs[idx]['prediction'])
                    residuals=copy.copy(obsres[_].tabs[idx]['residuals'])
                    data['reactions'][_][idx]['prediction-rep'].append(prediction)
                    data['reactions'][_][idx]['residuals-rep'].append(residuals)

        #--convert tables to numpy array before saving
        for _ in ['res','rres','nres']:
            data[_]=np.array(data[_])

        checkdir('%s/data'%wdir)

        if   self.pdf_flav==None: pdf='all'
        else:                     pdf=self.pdf_flav
        if   self.ff_flav==None: ff='all'
        else:                    ff=self.ff_flav
        fname='%s/data/predictions-%d-sidis-pdf-%s-ff-%s.dat'%(wdir,istep,pdf,ff)
        save(data,fname)


class SIA(CORE):

    def __init__(self,task,wdir,last=False):

        if  task==0:
            self.msg='sia.data_vs_thy'
            self.func=self.data_vs_thy
            self.loop_over_steps(wdir,None,last)

    def _data_vs_thy_sia(self,wdir,istep,had=None):

        cluster,colors,nc,order = self.get_clusters(wdir,istep)
        predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
        if 'sia' not in predictions['reactions']: return
        data=predictions['reactions']['sia']

        for _ in data:

            predictions=copy.copy(data[_]['prediction-rep'])
            del data[_]['prediction-rep']
            del data[_]['residuals-rep']
            for ic in range(nc):
                predictions_ic=[predictions[i] for i in range(len(predictions))
                               if cluster[i]==ic]
                data[_]['thy-%d'%ic]=np.mean(predictions_ic, axis=0)
                data[_]['dthy-%d'%ic]=np.std(predictions_ic, axis=0)

        for i in [1000,1001,1002,1003,1004,1005,2008,2009,3000,3001,3002,3003]:
            data[i] = pd.DataFrame(data[i])

        tab1=pd.concat([data[1000],data[1001]])
        tab2=pd.concat([data[1002],data[1003]])
        tab3=pd.concat([data[1004],data[1005]])
        tab4=pd.concat([data[2008],data[2009]])
        tab5=pd.concat([data[3000],data[3001]])
        tab6=pd.concat([data[3002],data[3003]])

        zbins=[]
        zbins.append([0.15,0.2])
        zbins.append([0.2,0.3])
        zbins.append([0.3,0.4])
        zbins.append([0.4,0.5])
        zbins.append([0.5,0.7])
        zbins.append([0.7,0.9])

        for tab in ['tab1','tab2','tab3','tab4','tab5','tab6']:
            if tab=='tab1':
                ncols,nrows=1,1
                py.figure(figsize=(ncols*4,nrows*4))
                _tab=tab1
                for ic in range(nc):
                    auc=_tab.query('obs=="AUC-0-PT"')
                    aul=_tab.query('obs=="AUL-0-PT"')
                    save(auc.to_dict(orient='list'),'%s/npdata/BaBar-Collins-AUC-PT.dat'%wdir)
                    save(aul.to_dict(orient='list'),'%s/npdata/BaBar-Collins-AUL-PT.dat'%wdir)
                    #np.save('%s/nparrays/BaBar-Collins-AUC-PT.npy'%wdir,auc)
                    #np.save('%s/nparrays/BaBar-Collins-AUL-PT.npy'%wdir,aul)
                    ax=py.subplot(1,1,1)
                    ax.errorbar(auc['pT'],auc['value'],yerr=auc['alpha'],fmt='b.',label='$A_{UC}\, (\%)$')
                    ax.errorbar(aul['pT'],aul['value'],yerr=aul['alpha'],fmt='r.',label='$A_{UL}\, (\%)$')
                    ax.legend(loc='upper left')
                    ax.plot(auc['pT'],auc['thy-%d'%ic],color='b',alpha=0.5)
                    ax.plot(aul['pT'],aul['thy-%d'%ic],color='r',alpha=0.5)
                    ax.fill_between(auc['pT'],(auc['thy-%d'%ic]-auc['dthy-%d'%ic]),(auc['thy-%d'%ic]+auc['dthy-%d'%ic]),color='b',alpha=0.5)
                    ax.fill_between(aul['pT'],(aul['thy-%d'%ic]-aul['dthy-%d'%ic]),(aul['thy-%d'%ic]+aul['dthy-%d'%ic]),color='r',alpha=0.5)
                    ax.set_xlabel('$P_{hT}$',size=20)
                    ax.set_title('BaBar')
                py.tight_layout()
                checkdir('%s/gallery'%wdir)
                py.savefig('%s/gallery/data-vs-thy-sia-%d-BaBar-pT.pdf'%(wdir,istep))
                py.close()
            elif tab=='tab6':
                ncols,nrows=1,1
                py.figure(figsize=(ncols*4,nrows*4))
                _tab=tab6
                for ic in range(nc):
                    auc=_tab.query('obs=="AUC-0-PT"')
                    aul=_tab.query('obs=="AUL-0-PT"')
                    save(auc.to_dict(orient='list'),'%s/npdata/BES3-Collins-AUC-PT.dat'%wdir)
                    save(aul.to_dict(orient='list'),'%s/npdata/BES3-Collins-AUL-PT.dat'%wdir)
                    #np.save('%s/nparrays/BES3-Collins-AUC-PT.npy'%wdir,auc)
                    #np.save('%s/nparrays/BES3-Collins-AUL-PT.npy'%wdir,aul)
                    ax=py.subplot(1,1,1)
                    ax.errorbar(auc['pT'],auc['value'],yerr=auc['alpha'],fmt='b.',label='$A_{UC}\, (\%)$')
                    ax.errorbar(aul['pT'],aul['value'],yerr=aul['alpha'],fmt='r.',label='$A_{UL}\, (\%)$')
                    ax.legend(loc='upper left')
                    ax.plot(auc['pT'],auc['thy-%d'%ic],color='b',alpha=0.5)
                    ax.plot(aul['pT'],aul['thy-%d'%ic],color='r',alpha=0.5)
                    ax.fill_between(auc['pT'],(auc['thy-%d'%ic]-auc['dthy-%d'%ic]),(auc['thy-%d'%ic]+auc['dthy-%d'%ic]),color='b',alpha=0.5)
                    ax.fill_between(aul['pT'],(aul['thy-%d'%ic]-aul['dthy-%d'%ic]),(aul['thy-%d'%ic]+aul['dthy-%d'%ic]),color='r',alpha=0.5)
                    ax.set_xlabel('$P_{hT}$',size=20)
                    ax.set_title('BESIII')
                py.tight_layout()
                checkdir('%s/gallery'%wdir)
                py.savefig('%s/gallery/data-vs-thy-sia-%d-BES3-pT.pdf'%(wdir,istep))
                py.close()
            else:
                if tab=='tab2':
                    ncols,nrows=3,2
                    ntab=tab2
                    filename='%s/gallery/data-vs-thy-sia-%d-BaBar1-z2.pdf'%(wdir,istep)
                    arrnameAUC='%s/npdata/BaBar-Collins-AUC-z2'%wdir
                    arrnameAUL='%s/npdata/BaBar-Collins-AUL-z2'%wdir
                    #arrnameAUC='%s/nparrays/BaBar-Collins-AUC-z2.npy'%wdir
                    #arrnameAUL='%s/nparrays/BaBar-Collins-AUL-z2.npy'%wdir
                if tab=='tab3':
                    ncols,nrows=2,2
                    ntab=tab3
                    filename='%s/gallery/data-vs-thy-sia-%d-Belle-z2.pdf'%(wdir,istep)
                    arrnameAUC='%s/npdata/Belle-Collins-AUC-z2'%wdir
                    arrnameAUL='%s/npdata/Belle-Collins-AUL-z2'%wdir
                    #arrnameAUC='%s/nparrays/Belle-Collins-AUC-z2.npy'%wdir
                    #arrnameAUL='%s/nparrays/Belle-Collins-AUL-z2.npy'%wdir
                if tab=='tab4':
                    ncols,nrows=2,2
                    ntab=tab4
                    filename='%s/gallery/data-vs-thy-sia-%d-BaBar2-z2.pdf'%(wdir,istep)
                    arrnameAUC='%s/npdata/BaBar2-Collins-AUC-z2'%wdir
                    arrnameAUL='%s/npdata/BaBar2-Collins-AUL-z2'%wdir
                    #arrnameAUC='%s/nparrays/BaBar2-Collins-AUC-z2.npy'%wdir
                    #arrnameAUL='%s/nparrays/BaBar2-Collins-AUL-z2.npy'%wdir
                if tab=='tab5':
                    #zbins=[[0.2,0.3],[0.3,0.5],[0.5,0.9]]
                    zbins=[[0.2,0.9]]
                    ncols,nrows=1,1
                    ntab=tab5
                    filename='%s/gallery/data-vs-thy-sia-%d-BES3-z2.pdf'%(wdir,istep)
                    arrnameAUC='%s/npdata/BES3-Collins-AUC-z2'%wdir
                    arrnameAUL='%s/npdata/BES3-Collins-AUL-z2'%wdir
                py.figure(figsize=(ncols*4,nrows*4))
                for ic in range(nc):
                    cnt=0
                    for j in range(len(zbins)):
                        cnt+=1
                        zmin,zmax=zbins[j]
                        _tab=ntab.query('z1>%f and z1<%f'%(zmin,zmax))
                        if _tab.empty == True:
                            cnt-=1
                            continue
                        ax=py.subplot(nrows,ncols,cnt)
                        auc=_tab.query('obs=="AUC-0-PT-INT"')
                        aul=_tab.query('obs=="AUL-0-PT-INT"')
                        save(auc.to_dict(orient='list'),arrnameAUC+'-z1bin%s.dat'%j)
                        save(aul.to_dict(orient='list'),arrnameAUL+'-z1bin%s.dat'%j)
                        #np.save(arrnameAUC,auc)
                        #np.save(arrnameAUL,aul)
                        ax.errorbar(auc['z2'],auc['value'],yerr=auc['alpha'],fmt='b.',label='$A_{UC}\, (\%)$')
                        ax.errorbar(aul['z2'],aul['value'],yerr=aul['alpha'],fmt='r.',label='$A_{UL}\, (\%)$')
                        if cnt==1: ax.legend(loc='upper left')
                        ax.plot(auc['z2'],auc['thy-%d'%ic],color='b',alpha=0.5)
                        ax.plot(aul['z2'],aul['thy-%d'%ic],color='r',alpha=0.5)
                        ax.fill_between(auc['z2'],(auc['thy-%d'%ic]-auc['dthy-%d'%ic]),(auc['thy-%d'%ic]+auc['dthy-%d'%ic]),color='b',alpha=0.5)
                        ax.fill_between(aul['z2'],(aul['thy-%d'%ic]-aul['dthy-%d'%ic]),(aul['thy-%d'%ic]+aul['dthy-%d'%ic]),color='r',alpha=0.5)
                        ax.set_xlabel('$Z_2$',size=20)
                        ax.set_title('%0.2f$\leq Z_1\leq$%0.2f'%(zmin,zmax))

                py.tight_layout()
                checkdir('%s/gallery'%wdir)
                py.savefig(filename)
                py.close()

    def data_vs_thy(self,wdir,istep):
        self._data_vs_thy_sia(wdir,istep,'pion')


class AN(CORE):

    def __init__(self,task,wdir,last=False):

        if  task==0:
            self.msg='an.data_vs_thy'
            self.func=self.data_vs_thy_AN
            self.loop_over_steps(wdir,None,last)

    def _data_vs_thy_AN(self,wdir,istep,col,tar,had=None, noniso=False):

        cluster,colors,nc,order = self.get_clusters(wdir,istep)
        predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
        if 'AN' not in predictions['reactions']: return
        data=predictions['reactions']['AN']
        for _ in data:

            predictions=copy.copy(data[_]['prediction-rep'])
            del data[_]['prediction-rep']
            del data[_]['residuals-rep']
            for ic in range(nc):
                predictions_ic=[predictions[i] for i in range(len(predictions))
                               if cluster[i]==ic]
                data[_]['thy-%d'%ic]=np.mean(predictions_ic, axis=0)
                data[_]['dthy-%d'%ic]=np.std(predictions_ic, axis=0)

        #for i in range(1000,1004):
        #    data[i]=pd.DataFrame(data[i])
        #for i in range(2000,2004):
        #    data[i]=pd.DataFrame(data[i])
        #for i in range(3000,3004):
        #    data[i]=pd.DataFrame(data[i])
        #for i in [4000,4001]:
        #    data[i]=pd.DataFrame(data[i])
        for i in it.chain(range(1000, 1004), range(2000,2005), range(3000,3006), range(4000,4005)):
            try:
                data[i]=pd.DataFrame(data[i])
            except:
                data[i]=pd.DataFrame(columns=['target', 'hadron', 'dependence', 'col', 'obs', 'dep', 'Dependence', 'x', 'z', 'pT', 'xF', 'value', 'alpha'])
                print('Could not load file "%f"'%i)

        if col=='BRAHMS':
            ncols=2
            nrows=1
            tab1=pd.concat([data[1000],data[1002]])
            tab2=pd.concat([data[1001],data[1003]])
            py.figure(figsize=(ncols*4,nrows*4))
            cnt=0
            for ntab in ['tab1','tab2']:
                cnt+=1
                ax=py.subplot(nrows,ncols,cnt)
                if ntab=='tab1':
                    tab=tab1
                    title ='BRAHMS' r' $\theta=2.3^{\circ}$'
                    arrnamepip='%s/npdata/BRAHMS-AN-theta2.3-pip.dat'%wdir
                    arrnamepim='%s/npdata/BRAHMS-AN-theta2.3-pim.dat'%wdir
                    #arrnamepip='%s/nparrays/BRAHMS-AN-theta2.3-pip.npy'%wdir
                    #arrnamepim='%s/nparrays/BRAHMS-AN-theta2.3-pim.npy'%wdir
                else:
                    tab=tab2
                    title='BRAHMS' r' $\theta=4^{\circ}$'
                    arrnamepip='%s/npdata/BRAHMS-AN-theta4-pip.dat'%wdir
                    arrnamepim='%s/npdata/BRAHMS-AN-theta4-pim.dat'%wdir
                    #arrnamepip='%s/nparrays/BRAHMS-AN-theta4-pip.npy'%wdir
                    #arrnamepim='%s/nparrays/BRAHMS-AN-theta4-pim.npy'%wdir
                for ic in range(nc):
                    dep='xF'
                    label='x_F'
                    _tab=tab.query('obs=="AN"')
                    pip=_tab.query('hadron=="pi+"')
                    pim=_tab.query('hadron=="pi-"')
                    save(pip.to_dict(),arrnamepip)
                    save(pim.to_dict(),arrnamepim)
                    #np.save(arrnamepip,pip)
                    #np.save(arrnamepim,pim)
                    ax.errorbar(pip[dep],pip['value'],yerr=pip['alpha'],fmt='b.')
                    ax.errorbar(pim[dep],pim['value'],yerr=pim['alpha'],fmt='r.')
                    ax.plot(pip[dep],pip['thy-%d'%ic],color='b',alpha=0.5)
                    ax.plot(pim[dep],pim['thy-%d'%ic],color='r',alpha=0.5)
                    ax.fill_between(pip[dep],(pip['thy-%d'%ic]-pip['dthy-%d'%ic]),(pip['thy-%d'%ic]+pip['dthy-%d'%ic]),color='b',alpha=0.5)
                    ax.fill_between(pim[dep],(pim['thy-%d'%ic]-pim['dthy-%d'%ic]),(pim['thy-%d'%ic]+pim['dthy-%d'%ic]),color='r',alpha=0.5)
                    ax.set_xlabel(r'$%s$'%label,size=20); ax.set_ylabel(r'$A^{\pi^\pm}_{N}$',size=20)
                    ax.set_title(title)
            py.tight_layout()
            checkdir('%s/gallery'%wdir)
            py.savefig('%s/gallery/data-vs-thy-AN--%d-%s-%s-%s.pdf'%(wdir,istep,col,tar,had))
            py.close()


        elif col=='STAR' and had=='pi0':
            if noniso == 'False':
                ncols=2
                nrows=2
                tab1=data[2000]
                tab2=data[2001]
                tab3=data[2002]
                tab4=data[2003]
                cnt=0
                py.figure(figsize=(ncols*4,nrows*4))
                for ntab in ['tab1','tab2','tab3','tab4']:
                    cnt+=1
                    ax=py.subplot(nrows,ncols,cnt)
                    if ntab=='tab1':
                        tab=tab1
                        arrname='%s/npdata/STAR-AN-pi0-eta3.3-4.1.dat'%wdir
                        #arrname='%s/nparrays/STAR-AN-pi0-eta3.3-4.1.npy'%wdir
                    elif ntab=='tab2':
                        tab=tab2.query('xF > 0')
                        arrname='%s/npdata/STAR-AN-pi0-eta3.3.dat'%wdir
                        #arrname='%s/nparrays/STAR-AN-pi0-eta3.3.npy'%wdir
                    elif ntab=='tab3':
                        tab=tab3
                        arrname='%s/npdata/STAR-AN-pi0-eta3.68.dat'%wdir
                        #arrname='%s/nparrays/STAR-AN-pi0-eta3.68.npy'%wdir
                    elif ntab=='tab4':
                        tab=tab4.query('xF > 0')
                        arrname='%s/npdata/STAR-AN-pi0-eta3.7.dat'%wdir
                        #arrname='%s/nparrays/STAR-AN-pi0-eta3.7.npy'%wdir
                    for ic in range(nc):
                        dep='xF'
                        _tab=tab.query('obs=="AN"')
                        pi0=_tab.query('hadron=="pi0"')
                        save(pi0.to_dict(),arrname)
                        #np.save(arrname,pi0)
                        label='x_F'
                        ax.errorbar(pi0[dep],pi0['value'],yerr=pi0['alpha'],fmt='g.')
                        ax.plot(pi0[dep],pi0['thy-%d'%ic],color='g',alpha=0.5)
                        ax.fill_between(pi0[dep],(pi0['thy-%d'%ic]-pi0['dthy-%d'%ic]),(pi0['thy-%d'%ic]+pi0['dthy-%d'%ic]),color='g',alpha=0.5)
                        ax.set_xlabel(r'$%s$'%label,size=20); ax.set_ylabel(r'$A_{N}^{\pi^0}$',size=20)
                        #ax.set_ylim(0,0.12)
                        ax.set_title('STAR')
                py.tight_layout()
                checkdir('%s/gallery'%wdir)
                py.savefig('%s/gallery/data-vs-thy-AN--%d-%s-%s-%s.pdf'%(wdir,istep,col,tar,had))
                py.close()

                tab5=data[3000]
                tab6=data[3001]
                tab7=data[3002]
                tab8=data[3003]
                cnt=0
                nrows,ncols=4,2
                py.figure(figsize=(ncols*8,nrows*6))
                for ntab in ['tab5','tab6','tab7','tab8']:
                    if ntab=='tab5':
                        cnt+=1
                        ax=py.subplot(nrows,ncols,cnt)
                        tab=tab5
                        arrname='%s/npdata/STAR20-AN-xF-200.dat'%wdir
                        #if os.path.exists("%s/STAR2020/pi0_AN_vs_xF-200.dat"%wdir): os.remove("%s/STAR2020/pi0_AN_vs_xF-200.dat"%wdir)
                        #f=open("%s/STAR2020/pi0_AN_vs_xF-200.dat"%wdir, "a+")
                    elif ntab=='tab6':
                        cnt+=1
                        ax=py.subplot(nrows,ncols,cnt)
                        tab=tab6
                        arrname='%s/npdata/STAR20-AN-xF-500.dat'%wdir
                        #if os.path.exists("%s/STAR2020/pi0_AN_vs_xF-500.dat"%wdir): os.remove("%s/STAR2020/pi0_AN_vs_xF-500.dat"%wdir)
                    elif ntab=='tab7':
                        tab=tab7
                        arrname='%s/npdata/STAR20-AN-pT-200.dat'%wdir
                    elif ntab=='tab8':
                        tab=tab8
                        arrname='%s/npdata/STAR20-AN-pT-500.dat'%wdir

                    if ntab in ['tab5','tab6']: dep='xF'
                    else: dep='pT'

                    if dep=='xF':
                        for ic in range(nc):
                            _tab=tab.query('obs=="AN"')
                            pi0=_tab.query('hadron=="pi0"')
                            if ntab=='tab5': rs='200'
                            else: rs='500'
                            #print(pi0)
                            save(pi0.to_dict(),arrname)
                            #np.save(arrname,pi0)
                            label='x_F'
                            ax.errorbar(pi0[dep],pi0['value'],yerr=pi0['alpha'],fmt='g.')
                            ax.plot(pi0[dep],pi0['thy-%d'%ic],color='g',alpha=0.5)
                            ax.fill_between(pi0[dep],(pi0['thy-%d'%ic]-pi0['dthy-%d'%ic]),(pi0['thy-%d'%ic]+pi0['dthy-%d'%ic]),color='g',alpha=0.5)
                            ax.set_xlabel(r'$%s$'%label,size=30); ax.set_ylabel(r'$A_{N}^{\pi^0}$',size=30)
                            ax.set_ylim(0,0.12)
                            ax.set_title('STAR-2020 %s GeV'%rs)
                            #bin=0
                            #f.write('bin   asy   (+/-)err\r\n')
                            #for i in range(len(pi0[dep])):
                            #    f.write('%d   %.4f   %.4f\r\n'%(bin,pi0['thy-%d'%ic][i],pi0['dthy-%d'%ic][i]))
                            #    bin+=1
                    else:
                        xF200=[0.21,0.26,0.32]
                        xF500=[0.22,0.27,0.33]
                        for _xF in [['1',0.18,0.24],['2',0.24,0.3],['3',0.3,0.36]]:
                            b=_xF[0]
                            if ntab=='tab7':
                                rs='200'
                                xF=xF200[int(b)-1]
                            else:
                                rs='500'
                                xF=xF500[int(b)-1]
                            #if os.path.exists("%s/STAR2020/pi0_AN_vs_PT-%s-xFbin%s.dat"%(wdir,rs,b)): os.remove("%s/STAR2020/pi0_AN_vs_PT-%s-xFbin%s.dat"%(wdir,rs,b))
                            #f=open("%s/STAR2020/pi0_AN_vs_PT-%s-xFbin%s.dat"%(wdir,rs,b), "a+")
                            cnt+=1
                            ax=py.subplot(nrows,ncols,cnt)
                            xFmin,xFmax=_xF[1],_xF[2]
                            for ic in range(nc):
                                _tab=tab.query('obs=="AN"').query('xF>%.2f'%xFmin).query('xF<%.2f'%xFmax)
                                pi0=_tab.query('hadron=="pi0"').reset_index()
                                save(pi0.to_dict(),arrname)
                                #np.save(arrname,pi0)
                                label='P_T'
                                ax.errorbar(pi0[dep],pi0['value'],yerr=pi0['alpha'],fmt='g.')
                                ax.plot(pi0[dep],pi0['thy-%d'%ic],color='g',alpha=0.5)
                                ax.fill_between(pi0[dep],(pi0['thy-%d'%ic]-pi0['dthy-%d'%ic]),(pi0['thy-%d'%ic]+pi0['dthy-%d'%ic]),color='g',alpha=0.5)
                                ax.set_xlabel(r'$%s$'%label,size=20); ax.set_ylabel(r'$A_{N}^{\pi^0}$',size=20)
                                ax.set_ylim(0,0.12)
                                ax.set_title(r'STAR-2020 %s GeV $x_F=%.2f$'%(rs,xF))
                                #if rs=='200': bin=0
                                #elif rs=='500' and b != '3': bin=1
                                #else: bin=3
                                #f.write('bin   asy   (+/-)err\r\n')
                                #for i in range(len(pi0[dep])):
                                #    f.write('%d   %.4f   %.4f\r\n'%(bin,pi0['thy-%d'%ic][i],pi0['dthy-%d'%ic][i]))
                                #    bin+=1
                py.tight_layout()
                checkdir('%s/gallery'%wdir)
                py.savefig('%s/gallery/data-vs-thy-AN--%d-%s20-%s-%s.pdf'%(wdir,istep,col,tar,had))
                py.close()
                
                tab11=data[2004]
                cnt=0
                xFbins=[]
                xFbins.append(0.28)
                xFbins.append(0.32)
                xFbins.append(0.37)
                xFbins.append(0.43)
                xFbins.append(0.5)
                xFbins.append(0.6)

                nrows,ncols=2,3
                py.figure(figsize=(ncols*8,nrows*6))
                ntab=tab11
                filename='%s/gallery/data-vs-thy-AN-%d-STAR08-pT.pdf'%(wdir,istep)
                arrname='%s/npdata/STAR08-AN-pT'%wdir
                for ic in range(nc):
                    cnt=0
                    for j in range(len(xFbins)):

                        cnt+=1
                        xF=xFbins[j]
                        _tab=ntab.query('xF==%f'%(xF))
                        if _tab.empty == True:
                                cnt-=1
                                continue
                        ax=py.subplot(nrows,ncols,cnt)
                        AN=_tab.query('obs=="AN"')
                        save(AN.to_dict(orient='list'),arrname+'-xFbin%s.dat'%j)
                        ax.errorbar(AN['pT'],AN['value'],yerr=AN['alpha'],fmt='b.',label='$A_{N}$')
                        if cnt==1: ax.legend(loc='upper left')
                        ax.plot(AN['pT'],AN['thy-%d'%ic],color='b',alpha=0.5)
                        ax.fill_between(AN['pT'],(AN['thy-%d'%ic]-AN['dthy-%d'%ic]),(AN['thy-%d'%ic]+AN['dthy-%d'%ic]),color='b',alpha=0.5)
                        ax.set_xlabel('$p_T$',size=20)
                        ax.set_title(r'$x_F=%.2f$'%(xF))

                py.tight_layout()
                checkdir('%s/gallery'%wdir)
                py.savefig(filename)
                py.close()
                
            if noniso == 'True':
                tab9=data[3004]
                tab10=data[3005]

                cnt=0
                nrows,ncols=2,2
                py.figure(figsize=(ncols*8,nrows*6))
                for ntab in ['tab9', 'tab10']:
                    if ntab=='tab9':
                        cnt+=1
                        ax=py.subplot(nrows,ncols,cnt)
                        tab=tab9
                        arrname='%s/npdata/STAR20-Nonisolated-AN-xF-200.dat'%wdir
                        #if os.path.exists("%s/STAR2020/pi0_AN_vs_xF-200.dat"%wdir): os.remove("%s/STAR2020/pi0_AN_vs_xF-200.dat"%wdir)
                        #f=open("%s/STAR2020/pi0_AN_vs_xF-200.dat"%wdir, "a+")
                    elif ntab=='tab10':
                        cnt+=1
                        ax=py.subplot(nrows,ncols,cnt)
                        tab=tab10
                        arrname='%s/npdata/STAR20-Nonisolated-AN-xF-500.dat'%wdir
                        #if os.path.exists("%s/STAR2020/pi0_AN_vs_xF-500.dat"%wdir): os.remove("%s/STAR2020/pi0_AN_vs_xF-500.dat"%wdir)
                        #f=open("%s/STAR2020/pi0_AN_vs_xF-500.dat"%wdir, "a+")
                    if ntab in ['tab9','tab10']: dep='xF'
                    else: dep='pT'
                    if dep=='xF':
                        for ic in range(nc):
                            _tab=tab.query('obs=="AN"')
                            pi0=_tab.query('hadron=="pi0"')
                            if ntab=='tab9': rs='Non-Isolated 200'
                            elif ntab=='tab10': rs='Non-Isolated 500'
                            #print(pi0)
                            save(pi0.to_dict(),arrname)
                            #np.save(arrname,pi0)
                            label='x_F'
                            ax.errorbar(pi0[dep],pi0['value'],yerr=pi0['alpha'],fmt='g.')
                            ax.plot(pi0[dep],pi0['thy-%d'%ic],color='g',alpha=0.5)
                            ax.fill_between(pi0[dep],(pi0['thy-%d'%ic]-pi0['dthy-%d'%ic]),(pi0['thy-%d'%ic]+pi0['dthy-%d'%ic]),color='g',alpha=0.5)
                            ax.set_xlabel(r'$%s$'%label,size=20); ax.set_ylabel(r'$A_{N}^{\pi^0}$',size=20)
                            #ax.set_ylim(0,0.12)
                            ax.set_title('STAR-2020 %s GeV'%rs)
                            #bin=0
                            #f.write('bin   asy   (+/-)err\r\n')
                            #for i in range(len(pi0[dep])):
                            #    f.write('%d   %.4f   %.4f\r\n'%(bin,pi0['thy-%d'%ic][i],pi0['dthy-%d'%ic][i]))
                            #    bin+=1
                py.tight_layout()
                checkdir('%s/gallery'%wdir)
                py.savefig('%s/gallery/data-vs-thy-AN-Nonisolated-%d-%s20-%s-%s.pdf'%(wdir,istep,col,tar,had))
                py.close()
            
        elif col=='STAR' and had=='jet':
            tab9=data[4002]
            tab10=data[4003]
            dep='xF'
            cnt=0
            nrows,ncols=1,2
            py.figure(figsize=(ncols*4,nrows*4))
            for ntab in ['tab9','tab10']:
                if ntab=='tab9':
                    cnt+=1
                    ax=py.subplot(nrows,ncols,cnt)
                    tab=tab9
                    arrname='%s/npdata/STAR20-AN-xF-jet-200.dat'%wdir
                    #if os.path.exists("%s/STAR2020/pi0_AN_vs_xF-jet-200.dat"%wdir): os.remove("%s/STAR2020/pi0_AN_vs_xF-jet-200.dat"%wdir)
                    #f=open("%s/STAR2020/pi0_AN_vs_xF-jet-200.dat"%wdir, "a+")
                elif ntab=='tab10':
                    cnt+=1
                    ax=py.subplot(nrows,ncols,cnt)
                    tab=tab10
                    arrname='%s/npdata/STAR20-AN-xF-jet-500.dat'%wdir
                    #if os.path.exists("%s/STAR2020/pi0_AN_vs_xF-jet-500.dat"%wdir): os.remove("%s/STAR2020/pi0_AN_vs_xF-jet-500.dat"%wdir)
                    #f=open("%s/STAR2020/pi0_AN_vs_xF-jet-500.dat"%wdir, "a+")

                for ic in range(nc):
                    _tab=tab.query('obs=="AN"')
                    jet=_tab.query('hadron=="jet"')
                    if ntab=='tab9': rs='200'
                    else: rs='500'
                    #print(jet)
                    save(jet.to_dict(),arrname)
                    #np.save(arrname,jet)
                    label='x_F'
                    ax.errorbar(jet[dep],jet['value'],yerr=jet['alpha'],fmt='g.')
                    ax.plot(jet[dep],jet['thy-%d'%ic],color='g',alpha=0.5)
                    ax.fill_between(jet[dep],(jet['thy-%d'%ic]-jet['dthy-%d'%ic]),(jet['thy-%d'%ic]+jet['dthy-%d'%ic]),color='g',alpha=0.5)
                    ax.set_xlabel(r'$%s$'%label,size=20); ax.set_ylabel(r'$A_{N}^{jet}$',size=20)
                    #ax.set_ylim(-0.01,0.01)
                    ax.set_title('STAR-2020 %s GeV'%rs)
                    #if rs=='200': bin=0
                    #else: bin=2
                    #f.write('bin   asy   (+/-)err\r\n')
                    #for i in range(len(jet[dep])):
                    #    f.write('%d   %.6f   %.6f\r\n'%(bin,jet['thy-%d'%ic][i],jet['dthy-%d'%ic][i]))
                    #    bin+=1
            py.tight_layout()
            checkdir('%s/gallery'%wdir)
            py.savefig('%s/gallery/data-vs-thy-AN--%d-%s20-%s-%s.pdf'%(wdir,istep,col,tar,had))
            py.close()
            
        elif col=='ANDY' and had=='jet':
            tab11=data[4004]
            dep='xF'
            cnt=0
            nrows,ncols=1,2
            py.figure(figsize=(ncols*4,nrows*4))
            for ntab in ['tab11']:
                if ntab=='tab11':
                    cnt+=1
                    ax=py.subplot(nrows,ncols,cnt)
                    tab=tab11
                    arrname='%s/npdata/ANDY-AN-xF-jet-500.dat'%wdir
                    #if os.path.exists("%s/STAR2020/pi0_AN_vs_xF-jet-200.dat"%wdir): os.remove("%s/STAR2020/pi0_AN_vs_xF-jet-200.dat"%wdir)
                    #f=open("%s/STAR2020/pi0_AN_vs_xF-jet-200.dat"%wdir, "a+")
                for ic in range(nc):
                    _tab=tab.query('obs=="AN"')
                    jet_=_tab.query('hadron=="jet"')
                    jet=_tab.query('xF>=0')
                    rs='500'
                    #print(jet)
                    save(jet.to_dict(),arrname)
                    #np.save(arrname,jet)
                    label='x_F'
                    ax.errorbar(jet[dep],jet['value'],yerr=jet['alpha'],fmt='g.')
                    ax.plot(jet[dep],jet['thy-%d'%ic],color='g',alpha=0.5)
                    ax.fill_between(jet[dep],(jet['thy-%d'%ic]-jet['dthy-%d'%ic]),(jet['thy-%d'%ic]+jet['dthy-%d'%ic]),color='g',alpha=0.5)
                    ax.set_xlabel(r'$%s$'%label,size=20); ax.set_ylabel(r'$A_{N}^{jet}$',size=20)
                    #ax.set_ylim(-0.01,0.01)
                    ax.set_title('ANDY %s GeV'%rs)
                    #if rs=='200': bin=0
                    #else: bin=2
                    #f.write('bin   asy   (+/-)err\r\n')
                    #for i in range(len(jet[dep])):
                    #    f.write('%d   %.6f   %.6f\r\n'%(bin,jet['thy-%d'%ic][i],jet['dthy-%d'%ic][i]))
                    #    bin+=1
            py.tight_layout()
            checkdir('%s/gallery'%wdir)
            if col=='ANDY': py.savefig('%s/gallery/data-vs-thy-AN--%d-%s-%s-%s.pdf'%(wdir,istep,col,tar,had))
            else: py.savefig('%s/gallery/data-vs-thy-AN--%d-%s20-%s-%s.pdf'%(wdir,istep,col,tar,had))
            py.close()
            
            
            
    def _data_vs_thy_HIJ(self,wdir,istep,col,tar,had=None):
        cluster,colors,nc,order = self.get_clusters(wdir,istep)
        predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
        if 'AN' not in predictions['reactions']: return
        data=predictions['reactions']['AN']
        for _ in data:

            predictions=copy.copy(data[_]['prediction-rep'])
            del data[_]['prediction-rep']
            del data[_]['residuals-rep']
            for ic in range(nc):
                predictions_ic=[predictions[i] for i in range(len(predictions))
                               if cluster[i]==ic]
                data[_]['thy-%d'%ic]=np.mean(predictions_ic, axis=0)
                data[_]['dthy-%d'%ic]=np.std(predictions_ic, axis=0)
                
        def plotter(hadron,dependence,GeV,DF):
            #col = 0
            #py.style.use('fivethirtyeight')
            if had == 'pi+' or had == 'pi-':
                if dependence == 'z':
                    bins = pTlist
                    con = 'pT'
                if dependence == 'pT':
                    bins = zlist
                    con = 'z_avg'
                if dependence == 'jperp':
                    bins = jperpzlist
                    con = 'z_avg'
                if dependence == 'jperp_pTbin':
                    bins = pTlist
                    con = 'pT'
                _dep = dependence
                
                if col=='STAR18':
                    fig = py.subplots(len(bins),len(etalist),figsize=(12,10),sharex=True,sharey=True)
                    for i in range(len(bins)*len(etalist)):
                        ax = py.subplot(len(bins),len(etalist),i+1)
                        title='%0.2f < eta < %0.2f'%(etalist[i%2][0], etalist[i%2][1])
                        if i==0 or i == 1:
                            ax.set_title(title,fontsize=20)
                        if i==4 or i==5:
                            ax.set_xlabel(r'$%s$'%str(_dep),size=20)
                        if i==0 or i==2 or i==4:
                            ax.set_ylabel(r'$%s = %0.2f$'%(con,bins[int(i/2)]),fontsize=20)
                        ax.axhline(0,0,1,color='black',linewidth=0.1,alpha=0.5)
                        _pip=DF
                       
                        if i==0 or i==1: k=0
                        if i==2 or i==3: k=1
                        if i==4 or i==5: k=2
                        pip1=_pip.query('%s==%s'%(con,bins[k]))
                        pip2=pip1.query('etamin==%s'%(etalist[i%2][0]))
                        pip3=pip2.query('etamax==%s'%(etalist[i%2][1]))
                        save(pip3.to_dict(orient='list'),'%s/npdata/%s-HIJ-%s-%s.dat'%(wdir,col,hadron,dep))

                        ax.errorbar(pip3[_dep],pip3['value'],yerr=pip3['alpha'],fmt='g.')
                        ax.plot(pip3[_dep],pip3['thy-%d'%ic],color='r',alpha=0.5)
                        ax.fill_between(pip3[_dep],(pip3['thy-%d'%ic]-pip3['dthy-%d'%ic]),(pip3['thy-%d'%ic]+pip3['dthy-%d'%ic]),alpha=0.5)#,color=colorlist[i-1],alpha=0.5)
                    #py.tight_layout()
                    checkdir('%s/gallery'%wdir)
                    py.savefig('%s/gallery/data-vs-thy-AN-HIJ-%s-%s-%s-%s-GeV.pdf'%(wdir,col,had,_dep,GeV))
                    py.close()
                
                if col=='STAR22':
                    if dependence=='jperp': fig = py.subplots(len(bins),len(etalist),figsize=(12,10),sharex=True,sharey=True)
                    elif dependence=='pT': fig = py.subplots(len(bins),len(etalist),figsize=(8,5),sharex=True,sharey=True)
                    elif dependence=='z': fig = py.subplots(len(bins),len(etalist),figsize=(12,16),sharex=True,sharey=True)
                    elif dependence=='jperp_pTbin': fig = py.subplots(len(bins),len(etalist),figsize=(12,10),sharex=True,sharey=True)
                    for i in range(len(bins)*len(etalist)):
                        ax = py.subplot(len(bins),len(etalist),i+1)
                        title='%0.2f < eta < %0.2f'%(etalist[i%2][0], etalist[i%2][1])
                        if i==0 or i == 1:
                            ax.set_title(title,fontsize=20)
                        if i==len(bins)*len(etalist)-2 or i==len(bins)*len(etalist)-1:
                            ax.set_xlabel(r'$%s$'%str(_dep),size=20)
                        if i==0 or i==2 or i==4 or i==6 or i==8 or i==10:
                            ax.set_ylabel(r'$%s = %0.2f$'%(con,bins[int(i/2)]),fontsize=15)
                        ax.axhline(0,0,1,color='black',linewidth=0.1,alpha=0.5)
                        _pip=DF
                       
                        if i==0 or i==1: k=0
                        if i==2 or i==3: k=1
                        if i==4 or i==5: k=2
                        if i==6 or i==7: k=3
                        if i==8 or i==9: k=4
                        if i==10 or i==11: k=5
                    
                        pip1=_pip.query('%s==%s'%(con,bins[k]))
                        pip2=pip1.query('etamin==%s'%(etalist[i%2][0]))
                        pip3=pip2.query('etamax==%s'%(etalist[i%2][1]))
                        save(pip3.to_dict(orient='list'),'%s/npdata/%s-HIJ-%s-%s.dat'%(wdir,col,hadron,dep))

                        if dependence=='jperp_pTbin': _dep='jperp'
                        ax.errorbar(pip3[_dep],pip3['value'],yerr=pip3['alpha'],fmt='g.')
                        ax.plot(pip3[_dep],pip3['thy-%d'%ic],color='r',alpha=0.5)
                        ax.fill_between(pip3[_dep],(pip3['thy-%d'%ic]-pip3['dthy-%d'%ic]),(pip3['thy-%d'%ic]+pip3['dthy-%d'%ic]),alpha=0.5)#,color=colorlist[i-1],alpha=0.5)
                    #py.tight_layout()
                    checkdir('%s/gallery'%wdir)
                    if dependence=='jperp_pTbin': _dep='jperp_pTbin'
                    py.savefig('%s/gallery/data-vs-thy-AN-HIJ-%s-%s-%s-%s-GeV.pdf'%(wdir,col,had,_dep,GeV))
                    py.close()
                
            elif had == 'pi0':
                bins = pi0pTlist
                con = 'pT'
                _dep = dependence
                    
                fig = py.subplots(len(bins),len(etalist),figsize=(12,18))
                for k in range(5000,5006):
                    try:
                         pip=pd.DataFrame(data[k])
                    except:
                        pip=pd.DataFrame(['x', 'z', 'pT', 'target', 'hadron', 'dependence', 'value', 'alpha', 'Dependence','jperp','etamin','etamax'])
                        print('Could not load file "%f"'%k)
                    i = k - 5000
                    ax = py.subplot(3,2,i+1)
                    if i==4 or i==5:
                        ax.set_xlabel(r'$jperp$',size=20)
                    ax.axhline(0,0,1,color='black',linewidth=0.1,alpha=0.5)
                    save(pip.to_dict(orient='list'),'%s/npdata/STAR-HIJ-proton-pip-%s.dat'%(wdir,dep))

                    ax.errorbar(pip['z'],pip['value'],yerr=pip['alpha'],fmt='g.')
                    ax.plot(pip['z'],pip['thy-%d'%ic],alpha=0.5)
                    ax.fill_between(pip['z'],(pip['thy-%d'%ic]-pip['dthy-%d'%ic]),(pip['thy-%d'%ic]+pip['dthy-%d'%ic]),alpha=0.5)#,color=colorlist[i-1],alpha=0.5)
                    ax.text(0.3, 0.85, '%s, jperp = %s, GeV = %s'%(k, pi0jperplist[i], pip.iloc[0]['rs']), transform=ax.transAxes)
                    #ax[len(zbins)-1,len(ptbins)-1].set_xlabel(r'$x_{F}$',size=15)
                    #obs=df.iloc[0]['obs']
                #py.tight_layout()
                checkdir('%s/gallery'%wdir)
                py.savefig('%s/gallery/data-vs-thy-AN-HIJ-%s-%s-%s.pdf'%(wdir,col,had,'jperp'))
                py.close()

                
        tab = pd.DataFrame([])
        #for i in range(5006,5076):
        if col=='STAR18': #5006, 5007, 5030, 5031 do not survive the cut on z
            for i in [5010,5011,5014,5015,5018,5019,5022,5023,5026,5027,5034,5035,5038,5039]:
                try:
                    data[i]=pd.DataFrame(data[i])
                except:
                    data[i]=pd.DataFrame(['x', 'z_avg', 'pT', 'target', 'hadron', 'dependence', 'value', 'alpha', 'Dependence','jperp','etamin','etamax'])
                    print('Could not load file "%f"'%i)
                tab = pd.concat([tab, data[i].query('target=="p"')])
        
        elif col=='STAR22':
            #for i in list(range(5042,5054))+list(range(5074,5076)):
            for i in list(range(5042,5066))+list(range(5068,5076)): #5066, 5067 do not survive the cut on z
                try:
                    data[i]=pd.DataFrame(data[i])
                except:
                    data[i]=pd.DataFrame(['x', 'z_avg', 'pT', 'target', 'hadron', 'dependence', 'value', 'alpha', 'Dependence','jperp','etamin','etamax'])
                    print('Could not load file "%f"'%i)
                tab = pd.concat([tab, data[i].query('target=="p"')])
                
        hadlist = ['pi+', 'pi-']
        colorlist = sns.color_palette('Set1')
        if col=='STAR18': zlist = [0.14,0.24,0.38]
        elif col=='STAR22': zlist = [0.22]
            
        if col=='STAR18': jperpzlist = [0.13,0.23,0.37]
        elif col=='STAR22': jperpzlist = [0.14,0.24,0.35,0.50]
        
        if col=='STAR18': pTlist = [10.6,20.6,31]
        elif col=='STAR22': pTlist=[6.7,9.0,10.7,13.6,17.5,22.0]
            
        pi0pTlist = [3]
        pi0etalist = [2.9,3.8]
        pi0jperplist = [0,0,0.13,0.245,0.340,0.460]
        if col=='STAR18': dep = ['pT','z','jperp']
        elif col=='STAR22': dep = ['pT','z','jperp','jperp_pTbin']
        GeVlist = [200,500]
        if had == 'pi+-' and col=='STAR18':
            etalist=[[-1,0],[0,1]]
            for had in hadlist: 
                if had == 'pi+' or had == 'pi-':
                    tab1 = tab
                    df1 = tab1.query('hadron=="%s"'%had)
                    for _dep in dep:
                        df2 = df1.query('Dependence=="%s"'%_dep)
                        df = df2
                        plotter(had,_dep,500,df)
        if had == 'pi+-' and col=='STAR22':
            etalist=[[-0.9,0],[0,0.9]]
            for had in hadlist: 
                if had == 'pi+' or had == 'pi-':
                    tab1 = tab
                    df1 = tab1.query('hadron=="%s"'%had)
                    for _dep in dep:
                        df2 = df1.query('Dependence=="%s"'%_dep)
                        df = df2
                        plotter(had,_dep,200,df)
        if had == 'pi0':
            tabpi0 = pd.DataFrame([])
            plotter(had,'z',200,tabpi0)
            

    def data_vs_thy_AN(self,wdir,istep):
        self._data_vs_thy_AN(wdir,istep,'BRAHMS','proton','pi+-')
        self._data_vs_thy_AN(wdir,istep,'STAR','proton','pi0','True')
        #self._data_vs_thy_AN(wdir,istep,'STAR','proton','pi0','False')
        self._data_vs_thy_AN(wdir,istep,'STAR','proton','jet')
        self._data_vs_thy_AN(wdir,istep,'ANDY','proton','jet')
        #--Calling HIJ data
        self._data_vs_thy_HIJ(wdir,istep,'STAR18','p','pi+-')
        self._data_vs_thy_HIJ(wdir,istep,'STAR22','p','pi+-')
        
class DY(CORE):

    def __init__(self,task,wdir,last=False):

        if  task==0:
            self.msg='dy.data_vs_thy'
            self.func=self.data_vs_thy_dy
            self.loop_over_steps(wdir,None,last)

    def _data_vs_thy_dy(self,wdir,istep,col,tar,beam):

        cluster,colors,nc,order = self.get_clusters(wdir,istep)
        predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
        if 'dy' not in predictions['reactions']: return
        data=predictions['reactions']['dy']
        for _ in data:

            predictions=copy.copy(data[_]['prediction-rep'])
            del data[_]['prediction-rep']
            del data[_]['residuals-rep']
            for ic in range(nc):
                predictions_ic=[predictions[i] for i in range(len(predictions))
                               if cluster[i]==ic]
                data[_]['thy-%d'%ic]=np.mean(predictions_ic, axis=0)
                data[_]['dthy-%d'%ic]=np.std(predictions_ic, axis=0)

        for i in range(1000,1004):
            data[i]=pd.DataFrame(data[i])

        if col=='COMPASS':
            ncols=2
            nrows=2
            tab1=data[1000]
            tab2=data[1001]
            tab3=data[1002]
            tab4=data[1003]
            py.figure(figsize=(ncols*4,nrows*4))
            cnt=0
            for ntab in ['tab1','tab2','tab3','tab4']:
                cnt+=1
                ax=py.subplot(nrows,ncols,cnt)
                if ntab=='tab1':
                    _tab=tab1
                    dep='xbeam'
                    label='x_b'
                    title ='COMPASS Sivers'
                    arrname='%s/npdata/COMPASS-dysivers-xbeam.dat'%wdir
                elif ntab=='tab2':
                    _tab=tab2
                    dep='xtarget'
                    label='x_t'
                    title='COMPASS Sivers'
                    arrname='%s/npdata/COMPASS-dysivers-xtarget.dat'%wdir
                elif ntab=='tab3':
                    _tab=tab3
                    dep='xF'
                    label='x_F'
                    title='COMPASS Sivers'
                    arrname='%s/npdata/COMPASS-dysivers-xF.dat'%wdir
                else:
                    _tab=tab4
                    dep='qT'
                    label='q_T'
                    title='COMPASS Sivers'
                    arrname='%s/npdata/COMPASS-dysivers-qT.dat'%wdir
                for ic in range(nc):
                    tab=_tab
                    save(tab.to_dict(),arrname)
                    ax.errorbar(tab[dep],tab['value'],yerr=tab['alpha'],fmt='r.')
                    ax.plot(tab[dep],tab['thy-%d'%ic],color='r',alpha=0.5)
                    ax.fill_between(tab[dep],(tab['thy-%d'%ic]-tab['dthy-%d'%ic]),(tab['thy-%d'%ic]+tab['dthy-%d'%ic]),color='r',alpha=0.5)
                    ax.set_xlabel(r'$%s$'%label,size=20); ax.set_ylabel(r'$A^{\sin\phi}_{UT}$',size=20)
                    ax.set_title(title)
            py.tight_layout()
            checkdir('%s/gallery'%wdir)
            py.savefig('%s/gallery/data-vs-thy-dysivers-%d-%s-%s-%s.pdf'%(wdir,istep,col,tar,beam))
            py.close()

    def data_vs_thy_dy(self,wdir,istep):
        self._data_vs_thy_dy(wdir,istep,'COMPASS','proton','pi-')

class WZ(CORE):

    def __init__(self,task,wdir,last=False):

        if  task==0:
            self.msg='wz.data_vs_thy'
            self.func=self.data_vs_thy_wz
            self.loop_over_steps(wdir,None,last)

    def _data_vs_thy_wz(self,wdir,istep,col):

        cluster,colors,nc,order = self.get_clusters(wdir,istep)
        predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
        if 'wz' not in predictions['reactions']: return
        data=predictions['reactions']['wz']
        for _ in data:

            predictions=copy.copy(data[_]['prediction-rep'])
            del data[_]['prediction-rep']
            del data[_]['residuals-rep']
            for ic in range(nc):
                predictions_ic=[predictions[i] for i in range(len(predictions))
                               if cluster[i]==ic]
                data[_]['thy-%d'%ic]=np.mean(predictions_ic, axis=0)
                data[_]['dthy-%d'%ic]=np.std(predictions_ic, axis=0)

        for i in range(2000,2003):
            data[i]=pd.DataFrame(data[i])

        if col=='STAR':
            tab1a=data[2000].query('obs=="ANW+"')
            tab1b=data[2000].query('obs=="ANW-"')
            tab2a=data[2001].query('obs=="ANW+"')
            tab2b=data[2001].query('obs=="ANW-"')
            tab3=data[2002]
            nrows,ncols=2,2
            py.figure(figsize=(ncols*4,nrows*4))
            cnt=0
            for ntab in ['tab1a','tab1b','tab2a','tab2b']:
                cnt+=1
                ax=py.subplot(nrows,ncols,cnt)
                if ntab=='tab1a':
                    tab=tab1a
                    title ='STAR' r' $W^+$'
                    dep='pT'
                    xlabel='P_T^{W^+}'
                    ylabel='A_N^{W^+}'
                    c='b'
                    arrname='%s/npdata/STAR-Wp-%s.dat'%(wdir,dep)
                elif ntab=='tab1b':
                    tab=tab1b
                    title ='STAR' r' $W^-$'
                    dep='pT'
                    xlabel='P_T^{W^-}'
                    ylabel='A_N^{W^-}'
                    c='r'
                    arrname='%s/npdata/STAR-Wm-%s.dat'%(wdir,dep)
                elif ntab=='tab2a':
                    tab=tab2a
                    title='STAR' r' $W^+$'
                    dep='y'
                    xlabel='y_{W^+}'
                    ylabel='A_N^{W^+}'
                    c='b'
                    arrname='%s/npdata/STAR-Wp-%s.dat'%(wdir,dep)
                elif ntab=='tab2b':
                    tab=tab2b
                    title='STAR' r' $W^-$'
                    dep='y'
                    xlabel='y_{W^-}'
                    ylabel='A_N^{W^-}'
                    c='r'
                    arrname='%s/npdata/STAR-Wm-%s.dat'%(wdir,dep)
                for ic in range(nc):
                    save(tab.to_dict(),arrname)
                    ax.errorbar(tab[dep],tab['value'],yerr=tab['alpha'],fmt='%s.'%c)
                    ax.plot(tab[dep],tab['thy-%d'%ic],color='%s'%c,alpha=0.5)
                    ax.fill_between(tab[dep],(tab['thy-%d'%ic]-tab['dthy-%d'%ic]),(tab['thy-%d'%ic]+tab['dthy-%d'%ic]),color='%s'%c,alpha=0.5)
                    ax.set_xlabel(r'$%s$'%xlabel,size=20); ax.set_ylabel(r'$%s$'%ylabel,size=20)
                    ax.set_title(title)
            py.tight_layout()
            checkdir('%s/gallery'%wdir)
            py.savefig('%s/gallery/data-vs-thy-Wsivers-%d-%s.pdf'%(wdir,istep,col))
            py.close()

            tab=tab3
            py.figure(figsize=(4,4))
            ax=py.subplot(1,1,1)
            title='STAR' r' $Z$'
            dep='y'
            label='y_Z'
            arrname='%s/npdata/STAR-Z-y.dat'%wdir
            for ic in range(nc):
                Z=tab.query('obs=="ANZ"')
                save(Z.to_dict(),arrname)
                ax.errorbar(Z[dep],Z['value'],yerr=Z['alpha'],fmt='g.')
                ax.errorbar(Z[dep],Z['thy-%d'%ic],yerr=Z['dthy-%d'%ic],fmt='m.')
                ax.set_xlabel(r'$%s$'%label,size=20); ax.set_ylabel(r'$A_N^Z$',size=20)
                ax.set_title(title)
            py.tight_layout()
            checkdir('%s/gallery'%wdir)
            py.savefig('%s/gallery/data-vs-thy-Zsivers-%d-%s.pdf'%(wdir,istep,col))
            py.close()


    def data_vs_thy_wz(self,wdir,istep):
        self._data_vs_thy_wz(wdir,istep,'STAR')








