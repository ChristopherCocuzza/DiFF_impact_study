#!/usr/bin/env python
import sys,os
import numpy as np
import copy
from subprocess import Popen, PIPE, STDOUT
import lhapdf

#--from tools
from tools.tools    import load,save,checkdir,lprint
from tools.config   import conf,load_config

from analysis.corelib import core

from fitlib.resman import RESMAN

cwd = os.getcwd()

#--index conventions:
#--1: down
#--2: up
#--3: strange
#--4: charm
#--5: bottom
#--6: top
#--21: gluon
#--negative values for antiquarks

iflavs = [-5,-4,-3,-2,-1,1,2,3,4,5,21]

X1=10**np.linspace(-4,-1)
X2=np.linspace(0.101,0.99)
X=np.append(X1,X2)

#--LO
order = 0

def write_lhapdf_conf(wdir):
    script = """
    Verbosity: 1
    Interpolator: logcubic
    Extrapolator: continuation
    ForcePositive: 0
    AlphaS_Type: analytic
    MZ: 91.1876
    MUp: 0.002
    MDown: 0.005
    MStrange: 0.10
    MCharm: 1.29
    MBottom: 4.19
    MTop: 172.9
    Pythia6LambdaV5Compat: true"""
    F=open('%s/data/lhapdf.conf'%wdir,'w')
    F.writelines(script)
    F.close()
    
def rename_tables(wdir,dirname,newname):

    print('\nrenaming LHAPDF tables for %s to %s'%(dirname,newname))
    old = '%s/data/%s'%(wdir,dirname)
    new = '%s/data/%s'%(wdir,newname)
    
    checkdir(new)
    F=os.listdir(old)

    cnt=0
    for f in F:
        cnt+=1
        lprint('progress %d/%d'%(cnt,len(F)))
        cmd=['cp','%s/%s'%(old,f),'%s/%s'%(new,f.replace(dirname,newname))]
        p=Popen(cmd,  stdout=PIPE, stderr=STDOUT)
        output = p.stdout.read()
    print



class QCF:

    def __init__(self):
        pass

    def gen_cj_grid(self):
    
        Q2=[1.30000E+00,1.50159E+00,1.75516E+00,2.07810E+00\
                ,2.49495E+00,3.04086E+00,3.76715E+00,4.50000E+00\
                ,4.75000E+00,6.23113E+00,8.37423E+00,1.15549E+01\
                ,1.64076E+01,2.40380E+01,3.64361E+01,5.73145E+01\
                ,9.38707E+01,1.60654E+02,2.88438E+02,5.45587E+02\
                ,1.09231E+03,2.32646E+03,5.30043E+03,1.29956E+04\
                ,3.45140E+04,1.00000E+05]
    
        X=[1.00000E-06,1.28121E-06,1.64152E-06,2.10317E-06\
               ,2.69463E-06,3.45242E-06,4.42329E-06,5.66715E-06\
               ,7.26076E-06,9.30241E-06,1.19180E-05,1.52689E-05\
               ,1.95617E-05,2.50609E-05,3.21053E-05,4.11287E-05\
               ,5.26863E-05,6.74889E-05,8.64459E-05,1.10720E-04\
               ,1.41800E-04,1.81585E-04,2.32503E-04,2.97652E-04\
               ,3.80981E-04,4.87518E-04,6.26039E-04,8.00452E-04\
               ,1.02297E-03,1.30657E-03,1.66759E-03,2.12729E-03\
               ,2.71054E-03,3.44865E-03,4.37927E-03,5.54908E-03\
               ,7.01192E-03,8.83064E-03,1.10763E-02,1.38266E-02\
               ,1.71641E-02,2.11717E-02,2.59364E-02,3.15062E-02\
               ,3.79623E-02,4.53425E-02,5.36750E-02,6.29705E-02\
               ,7.32221E-02,8.44039E-02,9.64793E-02,1.09332E-01\
               ,1.23067E-01,1.37507E-01,1.52639E-01,1.68416E-01\
               ,1.84794E-01,2.01731E-01,2.19016E-01,2.36948E-01\
               ,2.55242E-01,2.73927E-01,2.92954E-01,3.12340E-01\
               ,3.32036E-01,3.52019E-01,3.72282E-01,3.92772E-01\
               ,4.13533E-01,4.34326E-01,4.55495E-01,4.76836E-01\
               ,4.98342E-01,5.20006E-01,5.41818E-01,5.63773E-01\
               ,5.85861E-01,6.08077E-01,6.30459E-01,6.52800E-01\
               ,6.75387E-01,6.98063E-01,7.20830E-01,7.43683E-01\
               ,7.66623E-01,7.89636E-01,8.12791E-01,8.35940E-01\
               ,8.59175E-01,8.82485E-01,9.05866E-01,9.29311E-01\
               ,9.52817E-01,9.76387E-01,1.00000E+00]
    
        return X,Q2
    
    def gen_grid(self,dist):
        if    dist=='tpdf' : return self.gen_cj_grid()
    
    def _gen_table(self,dist):
    
        X,Q2=self.gen_grid(dist)
        qpd=conf[dist]
    
        nx=len(X)
        nQ2=len(Q2)
    
        #--fill table
        table={iflav:[]  for iflav in iflavs}  
        npts=nQ2*nx
        #--note that one has the corresponding indices
        #--u:   2 -> 1
        #--ub: -2 -> 2
        #--d:   1 -> 3
        #--db: -1 -> 4
        #--s:   2 -> 5
        #--sb: -2 -> 6
        for iQ2 in range(nQ2):
            for ix in range(nx):
                table[21].append(qpd.get_xF([X[ix]],[Q2[iQ2]],'g'))
                table[ 1].append(qpd.get_xF([X[ix]],[Q2[iQ2]],'d'))
                table[ 2].append(qpd.get_xF([X[ix]],[Q2[iQ2]],'u'))
                table[ 3].append(qpd.get_xF([X[ix]],[Q2[iQ2]],'s'))
                table[-1].append(qpd.get_xF([X[ix]],[Q2[iQ2]],'db'))
                table[-2].append(qpd.get_xF([X[ix]],[Q2[iQ2]],'ub'))
                table[-3].append(qpd.get_xF([X[ix]],[Q2[iQ2]],'sb'))
                if Q2[iQ2] < conf['aux'].mc2: q2 = conf['aux'].mc2+1 
                else:                         q2 = Q2[iQ2]
                table[ 4].append(qpd.get_xF([X[ix]],[q2],'c'))
                table[-4].append(qpd.get_xF([X[ix]],[q2],'cb'))
                if Q2[iQ2] < conf['aux'].mb2: q2 = conf['aux'].mb2+1 
                else:                         q2 = Q2[iQ2]
                table[ 5].append(qpd.get_xF([X[ix]],[q2],'b'))
                table[-5].append(qpd.get_xF([X[ix]],[q2],'bb'))

        #--remap tables to match with lhapdf format
        for iflav in iflavs: 
            new_list=[]
            for ix in range(nx):
                for inQ2 in range(nQ2):
                    idx=ix+inQ2*nx
                    new_list.append(table[iflav][idx])
            table[iflav]=new_list

        return X,Q2,table
    
    def gen_lhapdf_dat_file(self,X,Q2,table, wdir, file_name,setlabel):
        lines=[]  
        lines.append('PdfType: replica')
        lines.append('Format: lhagrid1')
        lines.append('---')
        line=''
        for _ in X: line+=('%10.5e '%_).upper()
        lines.append(line)
        line=''
        for _ in Q2: line+=('%10.5e '%_**0.5).upper()
        lines.append(line)
        line = ''
        for i in iflavs: line += str(i) + ' '
        lines.append(line)
 
        nx=len(X)
        nQ2=len(Q2)
    
        for i in range(nx*nQ2):
            line=''
            for iflav in iflavs:
                line+=('%10.5e '%table[iflav][i]).upper()
            lines.append(line)
        lines.append('---')
        lines=[l+'\n' for l in lines]
        idx=str(setlabel).zfill(4)
        tab=open('%s/data/%s/%s_%s.dat'%(wdir, file_name, file_name,idx),'w')
        tab.writelines(lines)
        tab.close()
    
    def gen_lhapdf_info_file(self,X,Q2,nrep,wdir,dist,file_name,info):
    
        aS=[conf['alphaS'].get_alphaS(_,order) for _ in Q2]
        mZ=conf['aux'].mZ
        mb=conf['aux'].mb
        mc=conf['aux'].mc
        alphaSMZ=conf['aux'].alphaSMZ
        xmin=X[0]
        xmax=X[-1]
        Qmin=Q2[0]**0.5
        Qmax=Q2[-1]**0.5
    
    
        lines=[]
        lines.append('SetDesc:         "<description>"')
        lines.append('SetIndex:        <index>')
        lines.append('Authors:         <authors>')
        lines.append('Reference:       reference')
        lines.append('Format:          lhagrid1')
        lines.append('DataVersion:     1')
        lines.append('NumMembers:      %d'%nrep)
        lines.append('Particle:        particle')
        #lines.append('Flavors:         [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 21]')
        lines.append('Flavors:         [1, 2, 3, -1, -2, -3]')
        lines.append('OrderQCD:        1')
        lines.append('FlavorScheme:    variable')
        lines.append('NumFlavors:      5')
        lines.append('ErrorType:       replicas')
        lines.append('XMin:            %0.2e'%xmin)
        lines.append('XMax:            %0.2e'%xmax)
        lines.append('QMin:            %0.2e'%Qmin)
        lines.append('QMax:            %0.2e'%Qmax)
        lines.append('MZ:              %f'%mZ)
        lines.append('MUp:             0.0')
        lines.append('MDown:           0.0')
        lines.append('MStrange:        0.0')
        lines.append('MCharm:          %f'%mc)
        lines.append('MBottom:         %f'%mb)
        lines.append('MTop:            180.0')
        lines.append('AlphaS_MZ:       %f'%alphaSMZ)
        lines.append('AlphaS_OrderQCD: 1')
        lines.append('AlphaS_Type:     ipol')
        line='AlphaS_Qs: ['
        for _ in Q2: line+=('%10.5e, '%_**0.5).upper()
        line=line.rstrip(',')+']'
        lines.append(line)
        line='AlphaS_Vals: ['
        for _ in aS: line+=('%10.5e, '%_).upper()
        line=line.rstrip(',')+']'
        lines.append(line)
        lines.append('AlphaS_Lambda4: 0')
        lines.append('AlphaS_Lambda5: 0')
        if dist=='transversity':
            line='widths_uv: ['
            for _ in self.widths_uv: line+=('%10.5e, '%_**0.5).upper()
            line=line.rstrip(',')+']'
            lines.append(line)
        if dist=='collinspi':
            line='widths_fav: ['
            for _ in self.widths_fav: line+=('%10.5e, '%_).upper()
            line=line.rstrip(',')+']'
            lines.append(line)
            line='widths_unf: ['
            for _ in self.widths_unf: line+=('%10.5e, '%_).upper()
            line=line.rstrip(',')+']'
            lines.append(line)

 
        for i in range(len(lines)):
            for _ in info:
                lines[i]=lines[i].replace(_,info[_])
    
    
        lines=[l+'\n' for l in lines]
        tab=open('%s/data/%s/%s.info'%(wdir, file_name, file_name),'w')
        tab.writelines(lines)
        tab.close()
    
    def gen_tables(self,wdir,dist, file_name,info,info_only=False):
   
        print('\ngenerating QCF LHAPDF tables for %s using %s'%(dist,wdir))
    
        write_lhapdf_conf(wdir)
        load_config('%s/input.py'%wdir)
        istep=core.get_istep()
        core.mod_conf(istep) #--set conf as specified in istep   
    
        resman=RESMAN(nworkers=1,parallel=False,datasets=False)
        parman=resman.parman
    
        jar=load('%s/data/jar-%d.dat'%(wdir,istep))
        replicas=jar['replicas']
    
        #--check order consistency
        order=jar['order']
        parman.order = order

        #--create output dir
        checkdir(wdir + '/data/')
        checkdir(wdir + '/data/%s/' % file_name)

        #--gen lhapdf_data_files
        TABLE = {}
        for iflav in iflavs:
            TABLE[iflav] = []
        if info_only==False:
            cnt=0
            for par in replicas:
                lprint('progress: %d/%d'%(cnt+1,len(replicas)))
                parman.set_new_params(par,initial=False)
                X,Q2,table = self._gen_table(dist)
                for iflav in iflavs:
                    TABLE[iflav].append(table[iflav])
                self.gen_lhapdf_dat_file(X,Q2,table, wdir, file_name,cnt+1)
                cnt+=1
            print()
      
        #--save mean value with index 0000 
        for iflav in iflavs:
            TABLE[iflav] = np.mean(TABLE[iflav],axis=0)
        self.gen_lhapdf_dat_file(X,Q2,TABLE, wdir, file_name,0)

        #--gen_lhapdf_info_file
        X,Q2=self.gen_grid(dist)
        nrep=len(replicas)
        self.gen_lhapdf_info_file(X,Q2,nrep,wdir,dist,file_name,info)

        print('Saving LHAPDF table to %s/data/%s'%(wdir,file_name))

      










