#!/usr/bin/env python
import os,sys
#--set lhapdf data path
version = int(sys.version[0])
os.environ["LHAPDF_DATA_PATH"] = '/work/JAM/ccocuzza/lhapdf/python%s/sets'%version
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import kmeanconf as kc
import time

#--from corelib
from analysis.corelib import core, inspect, predict, classifier, optpriors, jar, mlsamples, summary

#--from qpdlib
from analysis.qpdlib import diff, tpdf, tensorcharge, moments

#--from obslib
from analysis.obslib import sia, pythia_sia, sidis, star

#--from parlib
from analysis.parlib  import params

#--primary working directory
try: wdir=sys.argv[1]
except: wdir = None

######################
##--Initial Processing
######################

inspect.get_msr_inspected(wdir,limit=100)
predict.get_predictions(wdir,force=False)
classifier.gen_labels(wdir,kc)
jar.gen_jar_file(wdir,kc)
summary.print_summary(wdir,kc)


###################################################
##--Plot observables
###################################################

sia.plot_obs(wdir,mode=1)
pythia_sia.plot_obs(wdir,mode=1)
sidis.plot_obs(wdir,mode=1)
star.plot_obs(wdir,mode=1)


###################################################
##--Plot QCFs
###################################################

tpdf.gen_xf(wdir,Q2=4)
tpdf.plot_xf(wdir,Q2=4,mode=0)
tpdf.plot_xf(wdir,Q2=4,mode=1)

tpdf.gen_moments(wdir,Q2=4)
tensorcharge.plot_tensorcharge(wdir,mode=1)
tensorcharge.plot_tensorcharge(wdir,mode=0)

diff.gen_D (wdir,Q2=100)
diff.plot_D(wdir,Q2=100,mode=0)
diff.plot_D(wdir,Q2=100,mode=1)

diff.gen_H (wdir,Q2=100)
diff.plot_H(wdir,Q2=100,mode=0)
diff.plot_H(wdir,Q2=100,mode=1)

###################################################
##--Plot parameters
###################################################

params.plot_norms (wdir,'dihadron')
params.plot_params(wdir,'diffpippim',hist=False)
params.plot_params(wdir,'diffpippim',hist=True)
params.plot_params(wdir,'tdiffpippim',hist=False)
params.plot_params(wdir,'tdiffpippim',hist=True)
params.plot_params(wdir,'tpdf',hist=False)
params.plot_params(wdir,'tpdf',hist=True)




