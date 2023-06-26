import os 
import numpy as np
conf={} 
 
 
#--setup posterior sampling 
 
conf['bootstrap']=True
conf['flat par']=True
conf['ftol']=1e-3 
 
#--setup qcd evolution 
 
conf['dglap mode']='truncated' 
conf['alphaSmode']='backward' 
conf['order'] = 'LO' 
conf['Q20']   = 1
 
#--datasets

conf['datasets']={}

#--lepton-lepton reactions

##--SIA dihadron (pi+,pi-)
conf['datasets']['dihadron_sia']={}
conf['datasets']['dihadron_sia']['filters']=[]
#--filters for single pion pair
#--filter out points near kinematic limit
conf['datasets']['dihadron_sia']['filters'].append("gamma<0.7")
conf['datasets']['dihadron_sia']['filters'].append("gamma_belle<0.7")
#--filter out Kaon peak
conf['datasets']['dihadron_sia']['filters'].append("M>0.495 or M<0.485")
#--filter out D0 peak
conf['datasets']['dihadron_sia']['filters'].append("M>1.875 or M<1.865")
conf['datasets']['dihadron_sia']['filters'].append("M<2.00")
#--filter low z
conf['datasets']['dihadron_sia']['filters'].append("z>0.25")
#--filters for double pion pair
conf['datasets']['dihadron_sia']['filters'].append("gamma1<0.7")
conf['datasets']['dihadron_sia']['filters'].append("gamma2<0.7")
conf['datasets']['dihadron_sia']['filters'].append("M1>0.495 or M1<0.485")
conf['datasets']['dihadron_sia']['filters'].append("M2>0.495 or M2<0.485")
conf['datasets']['dihadron_sia']['filters'].append("M1>1.875 or M1<1.865")
conf['datasets']['dihadron_sia']['filters'].append("M2>1.875 or M2<1.865")
conf['datasets']['dihadron_sia']['filters'].append("z1>0.25")
conf['datasets']['dihadron_sia']['filters'].append("z2>0.25")
conf['datasets']['dihadron_sia']['filters'].append("M1<2.00")
conf['datasets']['dihadron_sia']['filters'].append("M2<2.00")
conf['datasets']['dihadron_sia']['xlsx']={}
conf['datasets']['dihadron_sia']['xlsx'][1000]='dihadron_sia/expdata/1000.xlsx'  # BELLE SIA unpolarized
conf['datasets']['dihadron_sia']['xlsx'][103] ='dihadron_sia/pythia/103.xlsx'    # PYTHIA s/tot, RS = 10.58
conf['datasets']['dihadron_sia']['xlsx'][104] ='dihadron_sia/pythia/104.xlsx'    # PYTHIA c/tot, RS = 10.58
conf['datasets']['dihadron_sia']['xlsx'][203] ='dihadron_sia/pythia/203.xlsx'    # PYTHIA s/tot, RS = 30.73 
conf['datasets']['dihadron_sia']['xlsx'][204] ='dihadron_sia/pythia/204.xlsx'    # PYTHIA c/tot, RS = 30.73 
conf['datasets']['dihadron_sia']['xlsx'][205] ='dihadron_sia/pythia/205.xlsx'    # PYTHIA b/tot, RS = 30.73 
conf['datasets']['dihadron_sia']['xlsx'][303] ='dihadron_sia/pythia/303.xlsx'    # PYTHIA s/tot, RS = 50.88 
conf['datasets']['dihadron_sia']['xlsx'][304] ='dihadron_sia/pythia/304.xlsx'    # PYTHIA c/tot, RS = 50.88 
conf['datasets']['dihadron_sia']['xlsx'][305] ='dihadron_sia/pythia/305.xlsx'    # PYTHIA b/tot, RS = 50.88 
conf['datasets']['dihadron_sia']['xlsx'][403] ='dihadron_sia/pythia/403.xlsx'    # PYTHIA s/tot, RS = 71.04 
conf['datasets']['dihadron_sia']['xlsx'][404] ='dihadron_sia/pythia/404.xlsx'    # PYTHIA c/tot, RS = 71.04 
conf['datasets']['dihadron_sia']['xlsx'][405] ='dihadron_sia/pythia/405.xlsx'    # PYTHIA b/tot, RS = 71.04 
conf['datasets']['dihadron_sia']['xlsx'][503] ='dihadron_sia/pythia/503.xlsx'    # PYTHIA s/tot, RS = 91.19 
conf['datasets']['dihadron_sia']['xlsx'][504] ='dihadron_sia/pythia/504.xlsx'    # PYTHIA c/tot, RS = 91.19 
conf['datasets']['dihadron_sia']['xlsx'][505] ='dihadron_sia/pythia/505.xlsx'    # PYTHIA b/tot, RS = 91.19 
#-----------------------------------------------------------------------------------------------------------------------
conf['datasets']['dihadron_sia']['xlsx'][2000]='dihadron_sia/expdata/2000.xlsx'  # BELLE SIA transverse, binned in z1, M1
conf['datasets']['dihadron_sia']['xlsx'][2001]='dihadron_sia/expdata/2001.xlsx'  # BELLE SIA transverse, binned in M1, M2
conf['datasets']['dihadron_sia']['xlsx'][2002]='dihadron_sia/expdata/2002.xlsx'  # BELLE SIA transverse, binned in z1, z2
#-----------------------------------------------------------------------------------------------------------------------
conf['datasets']['dihadron_sia']['norm']={}
conf['datasets']['dihadron_sia']['norm'][1000]={'value':    1.00000e+00,'fixed':False,'min':0.95,'max':1.05}


##--SIDIS dihadron (pi+,pi-)
conf['datasets']['dihadron_sidis']={}
conf['datasets']['dihadron_sidis']['filters']=[]
#--filter out Kaon peak
conf['datasets']['dihadron_sidis']['filters'].append("M>0.495 or M<0.485")
#--filter out D0 peak
conf['datasets']['dihadron_sidis']['filters'].append("M>1.875 or M<1.865")
conf['datasets']['dihadron_sidis']['filters'].append("M<2.00")
#--very difficult to fit lowest z bin
conf['datasets']['dihadron_sidis']['filters'].append("z>0.25")
#-----------------------------------------------------------------------------------------------------------------------
conf['datasets']['dihadron_sidis']['xlsx']={}
conf['datasets']['dihadron_sidis']['xlsx'][2000]='dihadron_sidis/expdata/2000.xlsx'  # HERMES SIDIS p , binned in x
conf['datasets']['dihadron_sidis']['xlsx'][2001]='dihadron_sidis/expdata/2001.xlsx'  # HERMES SIDIS p , binned in M
conf['datasets']['dihadron_sidis']['xlsx'][2002]='dihadron_sidis/expdata/2002.xlsx'  # HERMES SIDIS p , binned in z
conf['datasets']['dihadron_sidis']['xlsx'][2100]='dihadron_sidis/expdata/2100.xlsx'  # COMPASS SIDIS p, binned in x
conf['datasets']['dihadron_sidis']['xlsx'][2101]='dihadron_sidis/expdata/2101.xlsx'  # COMPASS SIDIS p, binned in M
conf['datasets']['dihadron_sidis']['xlsx'][2102]='dihadron_sidis/expdata/2102.xlsx'  # COMPASS SIDIS p, binned in z
conf['datasets']['dihadron_sidis']['xlsx'][2110]='dihadron_sidis/expdata/2110.xlsx'  # COMPASS SIDIS D, binned in x
conf['datasets']['dihadron_sidis']['xlsx'][2111]='dihadron_sidis/expdata/2111.xlsx'  # COMPASS SIDIS D, binned in M
conf['datasets']['dihadron_sidis']['xlsx'][2112]='dihadron_sidis/expdata/2112.xlsx'  # COMPASS SIDIS D, binned in z
#-----------------------------------------------------------------------------------------------------------------------
conf['datasets']['dihadron_sidis']['norm']={}
conf['datasets']['dihadron_sidis']['norm'][2000]={'value':    1.00000e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['dihadron_sidis']['norm'][2001]={'value':    1.00000e+00,'fixed':2000,'min':0.5,'max':1.5}
conf['datasets']['dihadron_sidis']['norm'][2002]={'value':    1.00000e+00,'fixed':2000,'min':0.5,'max':1.5}
conf['datasets']['dihadron_sidis']['norm'][2100]={'value':    1.00000e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['dihadron_sidis']['norm'][2101]={'value':    1.00000e+00,'fixed':2100,'min':0.5,'max':1.5}
conf['datasets']['dihadron_sidis']['norm'][2102]={'value':    1.00000e+00,'fixed':2100,'min':0.5,'max':1.5}
conf['datasets']['dihadron_sidis']['norm'][2110]={'value':    1.00000e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['dihadron_sidis']['norm'][2111]={'value':    1.00000e+00,'fixed':2110,'min':0.5,'max':1.5}
conf['datasets']['dihadron_sidis']['norm'][2112]={'value':    1.00000e+00,'fixed':2110,'min':0.5,'max':1.5}


##--pp dihadron (pi+,pi-)
conf['datasets']['dihadron_pp']={}
conf['datasets']['dihadron_pp']['filters']=[]
#--filter out Kaon peak
conf['datasets']['dihadron_pp']['filters'].append("M>0.495 or M<0.485")
#--filter out D0 peak
conf['datasets']['dihadron_pp']['filters'].append("M>1.875 or M<1.865")
conf['datasets']['dihadron_pp']['filters'].append("M<2.00")
#--very difficult to fit lowest z bin
conf['datasets']['dihadron_pp']['filters'].append("z>0.25")
#-----------------------------------------------------------------------------------------------------------------------
conf['datasets']['dihadron_pp']['xlsx']={}
conf['datasets']['dihadron_pp']['xlsx'][1000]='dihadron_pp/expdata/1000.xlsx'  # STAR, RS=200, R<0.7, binned in M, cross section
#-----------------------------------------------------------------------------------------------------------------------
conf['datasets']['dihadron_pp']['xlsx'][2100]='dihadron_pp/expdata/2100.xlsx'  # STAR, RS=200, R<0.3, binned in PhT, eta < 0
conf['datasets']['dihadron_pp']['xlsx'][2101]='dihadron_pp/expdata/2101.xlsx'  # STAR, RS=200, R<0.3, binned in PhT, eta > 0
conf['datasets']['dihadron_pp']['xlsx'][2102]='dihadron_pp/expdata/2102.xlsx'  # STAR, RS=200, R<0.3, binned in M,   eta < 0
conf['datasets']['dihadron_pp']['xlsx'][2103]='dihadron_pp/expdata/2103.xlsx'  # STAR, RS=200, R<0.3, binned in M,   eta > 0
conf['datasets']['dihadron_pp']['xlsx'][2104]='dihadron_pp/expdata/2104.xlsx'  # STAR, RS=200, R<0.3, binned in eta
#-----------------------------------------------------------------------------------------------------------------------
conf['datasets']['dihadron_pp']['xlsx'][2300]='dihadron_pp/expdata/2300.xlsx'  # STAR, RS=200, R<0.3, binned in PhT, eta < 0
conf['datasets']['dihadron_pp']['xlsx'][2301]='dihadron_pp/expdata/2301.xlsx'  # STAR, RS=200, R<0.3, binned in PhT, eta > 0
conf['datasets']['dihadron_pp']['xlsx'][2302]='dihadron_pp/expdata/2302.xlsx'  # STAR, RS=200, R<0.3, binned in M,   eta < 0
conf['datasets']['dihadron_pp']['xlsx'][2303]='dihadron_pp/expdata/2303.xlsx'  # STAR, RS=200, R<0.3, binned in M,   eta > 0
conf['datasets']['dihadron_pp']['xlsx'][2304]='dihadron_pp/expdata/2304.xlsx'  # STAR, RS=200, R<0.3, binned in eta
#-----------------------------------------------------------------------------------------------------------------------
conf['datasets']['dihadron_pp']['xlsx'][3001]='dihadron_pp/expdata/3001.xlsx'  # STAR, RS=500, binned in PhT,eta > 0
conf['datasets']['dihadron_pp']['xlsx'][3002]='dihadron_pp/expdata/3002.xlsx'  # STAR, RS=500, binned in M,  eta < 0
conf['datasets']['dihadron_pp']['xlsx'][3003]='dihadron_pp/expdata/3003.xlsx'  # STAR, RS=500, binned in M,  eta > 0
conf['datasets']['dihadron_pp']['xlsx'][3004]='dihadron_pp/expdata/3004.xlsx'  # STAR, RS=500, binned in eta
#-----------------------------------------------------------------------------------------------------------------------
conf['datasets']['dihadron_pp']['norm']={}
#-----------------------------------------------------------------------------------------------------------------------
conf['datasets']['dihadron_pp']['norm'][1000]={'value':    1.00000e+00,'fixed':False,'min':0.5,'max':1.5}
#-----------------------------------------------------------------------------------------------------------------------
conf['datasets']['dihadron_pp']['norm'][2100]={'value':    1.00000e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['dihadron_pp']['norm'][2101]={'value':    1.00000e+00,'fixed':2100 ,'min':0.5,'max':1.5}
conf['datasets']['dihadron_pp']['norm'][2102]={'value':    1.00000e+00,'fixed':2100 ,'min':0.5,'max':1.5}
conf['datasets']['dihadron_pp']['norm'][2103]={'value':    1.00000e+00,'fixed':2100 ,'min':0.5,'max':1.5}
conf['datasets']['dihadron_pp']['norm'][2104]={'value':    1.00000e+00,'fixed':2100 ,'min':0.5,'max':1.5}
#-----------------------------------------------------------------------------------------------------------------------
conf['datasets']['dihadron_pp']['norm'][2300]={'value':    1.00000e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['dihadron_pp']['norm'][2301]={'value':    1.00000e+00,'fixed':2300 ,'min':0.5,'max':1.5}
conf['datasets']['dihadron_pp']['norm'][2302]={'value':    1.00000e+00,'fixed':2300 ,'min':0.5,'max':1.5}
conf['datasets']['dihadron_pp']['norm'][2303]={'value':    1.00000e+00,'fixed':2300 ,'min':0.5,'max':1.5}
conf['datasets']['dihadron_pp']['norm'][2304]={'value':    1.00000e+00,'fixed':2300 ,'min':0.5,'max':1.5}
#-----------------------------------------------------------------------------------------------------------------------
conf['datasets']['dihadron_pp']['norm'][3001]={'value':    1.00000e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['dihadron_pp']['norm'][3002]={'value':    1.00000e+00,'fixed':3001 ,'min':0.5,'max':1.5}
conf['datasets']['dihadron_pp']['norm'][3003]={'value':    1.00000e+00,'fixed':3001 ,'min':0.5,'max':1.5}
conf['datasets']['dihadron_pp']['norm'][3004]={'value':    1.00000e+00,'fixed':3001 ,'min':0.5,'max':1.5}
#-----------------------------------------------------------------------------------------------------------------------




##--moments
conf['datasets']['moments']={}
conf['datasets']['moments']['filters']=[]
conf['datasets']['moments']['xlsx']={}
conf['datasets']['moments']['xlsx'][2000]='moments/expdata/2000.xlsx'  # ETMC   delta u
conf['datasets']['moments']['xlsx'][2001]='moments/expdata/2001.xlsx'  # PNDME, delta u 
conf['datasets']['moments']['xlsx'][3000]='moments/expdata/3000.xlsx'  # ETMC,  delta d 
conf['datasets']['moments']['xlsx'][3001]='moments/expdata/3001.xlsx'  # PNDME, delta d 
conf['datasets']['moments']['norm']={}


#--parameters
conf['params']={}


#--pi+ pi- dihadron fragmentation functions
conf['params']['diffpippim']={}

#--can use different M0 grids for different flavors
#--s, c, and g must be a subset of u
conf['D1 M0'] = {}
conf['D1 M0']['u'] = [0.28, 0.40, 0.50, 0.70, 0.75, 0.80, 0.90, 1.00, 1.20, 1.30, 1.40, 1.60, 1.80, 2.00]
conf['D1 M0']['s'] = [0.28, 0.50, 0.75, 1.00, 1.20, 1.60, 2.00]
conf['D1 M0']['c'] = [0.28, 0.50, 0.75, 1.00, 1.20, 1.60, 2.00]
conf['D1 M0']['b'] = [0.28, 0.50, 0.75, 1.00, 1.20, 1.60, 2.00]
conf['D1 M0']['g'] = [0.28, 0.70, 1.00, 1.40, 2.00]

conf['params']['diffpippim']['u 0.28 N1']   ={'value':    0.10000e+00, 'min': 0.01, 'max':    5,'fixed':False}
conf['params']['diffpippim']['u 0.28 a1']   ={'value':    0.00000e+00, 'min':   -2, 'max':   10,'fixed':False}
conf['params']['diffpippim']['u 0.28 b1']   ={'value':    3.00000e+00, 'min': 0.01, 'max':    5,'fixed':False}

conf['params']['diffpippim']['u 0.4 N1']    ={'value':    0.10000e+00, 'min': 0.01, 'max':   15,'fixed':False}
conf['params']['diffpippim']['u 0.4 a1']    ={'value':    0.00000e+00, 'min':   -2, 'max':   10,'fixed':False}
conf['params']['diffpippim']['u 0.4 b1']    ={'value':    3.00000e+00, 'min': 0.01, 'max':    5,'fixed':False}

conf['params']['diffpippim']['u 0.5 N1']    ={'value':    0.10000e+00, 'min': 0.01, 'max':   15,'fixed':False}
conf['params']['diffpippim']['u 0.5 a1']    ={'value':    0.00000e+00, 'min':   -2, 'max':   10,'fixed':False}
conf['params']['diffpippim']['u 0.5 b1']    ={'value':    3.00000e+00, 'min': 0.01, 'max':    5,'fixed':False}

conf['params']['diffpippim']['u 0.7 N1']    ={'value':    0.10000e+00, 'min': 0.01, 'max':   15,'fixed':False}
conf['params']['diffpippim']['u 0.7 a1']    ={'value':    0.00000e+00, 'min':   -2, 'max':   10,'fixed':False}
conf['params']['diffpippim']['u 0.7 b1']    ={'value':    3.00000e+00, 'min': 0.01, 'max':    5,'fixed':False}

conf['params']['diffpippim']['u 0.75 N1']   ={'value':    0.10000e+00, 'min': 0.01, 'max':   10,'fixed':False}
conf['params']['diffpippim']['u 0.75 a1']   ={'value':    0.00000e+00, 'min':   -2, 'max':   10,'fixed':False}
conf['params']['diffpippim']['u 0.75 b1']   ={'value':    3.00000e+00, 'min': 0.01, 'max':    5,'fixed':False}

conf['params']['diffpippim']['u 0.8 N1']    ={'value':    0.10000e+00, 'min': 0.01, 'max':   10,'fixed':False}
conf['params']['diffpippim']['u 0.8 a1']    ={'value':    0.00000e+00, 'min':   -2, 'max':   10,'fixed':False}
conf['params']['diffpippim']['u 0.8 b1']    ={'value':    3.00000e+00, 'min': 0.01, 'max':    5,'fixed':False}

conf['params']['diffpippim']['u 0.9 N1']    ={'value':    0.10000e+00, 'min': 0.01, 'max':    4,'fixed':False}
conf['params']['diffpippim']['u 0.9 a1']    ={'value':    0.00000e+00, 'min':   -2, 'max':   10,'fixed':False}
conf['params']['diffpippim']['u 0.9 b1']    ={'value':    3.00000e+00, 'min': 0.01, 'max':    5,'fixed':False}

conf['params']['diffpippim']['u 1.0 N1']    ={'value':    0.10000e+00, 'min': 0.01, 'max':    2,'fixed':False}
conf['params']['diffpippim']['u 1.0 a1']    ={'value':    0.00000e+00, 'min':   -2, 'max':   10,'fixed':False}
conf['params']['diffpippim']['u 1.0 b1']    ={'value':    3.00000e+00, 'min': 0.01, 'max':    5,'fixed':False}

conf['params']['diffpippim']['u 1.2 N1']    ={'value':    0.10000e+00, 'min': 0.01, 'max':    2,'fixed':False}
conf['params']['diffpippim']['u 1.2 a1']    ={'value':    0.00000e+00, 'min':   -2, 'max':   10,'fixed':False}
conf['params']['diffpippim']['u 1.2 b1']    ={'value':    3.00000e+00, 'min': 0.01, 'max':    5,'fixed':False}

conf['params']['diffpippim']['u 1.3 N1']    ={'value':    0.10000e+00, 'min': 0.01, 'max':    2,'fixed':False}
conf['params']['diffpippim']['u 1.3 a1']    ={'value':    0.00000e+00, 'min':   -2, 'max':   10,'fixed':False}
conf['params']['diffpippim']['u 1.3 b1']    ={'value':    3.00000e+00, 'min': 0.01, 'max':    5,'fixed':False}

conf['params']['diffpippim']['u 1.4 N1']    ={'value':    0.10000e+00, 'min': 0.01, 'max':    2,'fixed':False}
conf['params']['diffpippim']['u 1.4 a1']    ={'value':    0.00000e+00, 'min':   -2, 'max':   10,'fixed':False}
conf['params']['diffpippim']['u 1.4 b1']    ={'value':    3.00000e+00, 'min': 0.01, 'max':    5,'fixed':False}

conf['params']['diffpippim']['u 1.6 N1']    ={'value':    0.10000e+00, 'min': 0.01, 'max':    1,'fixed':False}
conf['params']['diffpippim']['u 1.6 a1']    ={'value':    0.00000e+00, 'min':   -2, 'max':   10,'fixed':False}
conf['params']['diffpippim']['u 1.6 b1']    ={'value':    3.00000e+00, 'min': 0.01, 'max':    5,'fixed':False}

conf['params']['diffpippim']['u 1.8 N1']    ={'value':    0.10000e+00, 'min': 0.01, 'max':    1,'fixed':False}
conf['params']['diffpippim']['u 1.8 a1']    ={'value':    0.00000e+00, 'min':   -2, 'max':   10,'fixed':False}
conf['params']['diffpippim']['u 1.8 b1']    ={'value':    3.00000e+00, 'min': 0.01, 'max':    5,'fixed':False}

conf['params']['diffpippim']['u 2.0 N1']    ={'value':    0.10000e+00, 'min': 0.01, 'max':    1,'fixed':False}
conf['params']['diffpippim']['u 2.0 a1']    ={'value':    0.00000e+00, 'min':   -2, 'max':   10,'fixed':False}
conf['params']['diffpippim']['u 2.0 b1']    ={'value':    3.00000e+00, 'min': 0.01, 'max':    5,'fixed':False}

for M in conf['D1 M0']['u']: conf['params']['diffpippim']['u %s N2'%M]   ={'value':    0.00000e+00, 'min':   -5, 'max':    5,'fixed':False}
for M in conf['D1 M0']['u']: conf['params']['diffpippim']['u %s a2'%M]   ={'value':    0.00000e+00, 'min':   -2, 'max':   10,'fixed':False}
for M in conf['D1 M0']['u']: conf['params']['diffpippim']['u %s b2'%M]   ={'value':    3.00000e+00, 'min': 0.01, 'max':    5,'fixed':False}

for M in conf['D1 M0']['u']: conf['params']['diffpippim']['u %s N3'%M]   ={'value':    0.00000e+00, 'min':   -2, 'max':    2,'fixed':False}
for M in conf['D1 M0']['u']: conf['params']['diffpippim']['u %s a3'%M]   ={'value':    0.00000e+00, 'min':   -2, 'max':   10,'fixed':False}
for M in conf['D1 M0']['u']: conf['params']['diffpippim']['u %s b3'%M]   ={'value':    3.00000e+00, 'min': 0.01, 'max':    5,'fixed':False}

for M in conf['D1 M0']['s']: conf['params']['diffpippim']['s %s N1'%M]   ={'value':    0.00000e+00, 'min':    0, 'max':    3,'fixed':False}
for M in conf['D1 M0']['s']: conf['params']['diffpippim']['s %s a1'%M]   ={'value':    0.00000e+00, 'min':    0, 'max':    6,'fixed':False}
for M in conf['D1 M0']['s']: conf['params']['diffpippim']['s %s b1'%M]   ={'value':    3.00000e+00, 'min':    1, 'max':   10,'fixed':False}

for M in conf['D1 M0']['c']: conf['params']['diffpippim']['c %s N1'%M]   ={'value':    0.00000e+00, 'min':    0, 'max':    3,'fixed':False}
for M in conf['D1 M0']['c']: conf['params']['diffpippim']['c %s a1'%M]   ={'value':    0.00000e+00, 'min':   -2, 'max':    6,'fixed':False}
for M in conf['D1 M0']['c']: conf['params']['diffpippim']['c %s b1'%M]   ={'value':    3.00000e+00, 'min':    2, 'max':   10,'fixed':False}

for M in conf['D1 M0']['b']: conf['params']['diffpippim']['b %s N1'%M]   ={'value':    0.00000e+00, 'min':    0, 'max':    3,'fixed':False}
for M in conf['D1 M0']['b']: conf['params']['diffpippim']['b %s a1'%M]   ={'value':    0.00000e+00, 'min':   -2, 'max':    6,'fixed':False}
for M in conf['D1 M0']['b']: conf['params']['diffpippim']['b %s b1'%M]   ={'value':    3.00000e+00, 'min':    2, 'max':   10,'fixed':False}

for M in conf['D1 M0']['g']: conf['params']['diffpippim']['g %s N1'%M]   ={'value':    0.00000e+00, 'min':    0, 'max':    3,'fixed':False}
for M in conf['D1 M0']['g']: conf['params']['diffpippim']['g %s a1'%M]   ={'value':    0.00000e+00, 'min':   -2, 'max':   10,'fixed':False}
for M in conf['D1 M0']['g']: conf['params']['diffpippim']['g %s b1'%M]   ={'value':    3.00000e+00, 'min': 0.01, 'max':   10,'fixed':False}


#--pi+ pi- interference dihadron fragmentation functions
conf['params']['tdiffpippim']={}

conf['H1 M0'] = {}
conf['H1 M0']['u'] = [0.28, 0.50, 0.70, 0.85, 1.00, 1.20, 1.60, 2.00]

for M in conf['H1 M0']['u']: conf['params']['tdiffpippim']['u %s N1'%M]   ={'value':    0.00000e+00, 'min':   -1, 'max':    0,'fixed':False}
for M in conf['H1 M0']['u']: conf['params']['tdiffpippim']['u %s a1'%M]   ={'value':    0.00000e+00, 'min':   -1, 'max':    5,'fixed':False}
for M in conf['H1 M0']['u']: conf['params']['tdiffpippim']['u %s b1'%M]   ={'value':    3.00000e+00, 'min': 0.01, 'max':   10,'fixed':False}

for M in conf['H1 M0']['u']: conf['params']['tdiffpippim']['u %s N2'%M]   ={'value':    0.00000e+00, 'min':   -1, 'max':    0,'fixed':False}
for M in conf['H1 M0']['u']: conf['params']['tdiffpippim']['u %s a2'%M]   ={'value':    0.00000e+00, 'min':   -1, 'max':    5,'fixed':False}
for M in conf['H1 M0']['u']: conf['params']['tdiffpippim']['u %s b2'%M]   ={'value':    3.00000e+00, 'min': 0.01, 'max':   10,'fixed':False}


#--fit D1u, D1s, D1c, D1b, D1g to Belle data and PYTHIA data all energies
#--transversity pdf parameters
conf['params']['tpdf']={}
conf['db factor'] = -1
conf['tpdf_choice'] = 'valence'

conf['params']['tpdf']['uv N']    ={'value':   0.0, 'min':   0.0, 'max':   1.0, 'fixed': False}
conf['params']['tpdf']['uv a']    ={'value':   0.2, 'min':  0.09, 'max':  0.26, 'fixed': False}
conf['params']['tpdf']['uv b']    ={'value':   4.0, 'min':     0, 'max':    20, 'fixed': False}
conf['params']['tpdf']['uv c']    ={'value':   0.0, 'min':   -20, 'max':    20, 'fixed': False}
conf['params']['tpdf']['uv d']    ={'value':   0.0, 'min':   -20, 'max':    20, 'fixed': False}

conf['params']['tpdf']['dv N']   ={'value':    0.0, 'min':  -1.0, 'max':   1.0, 'fixed': False}
conf['params']['tpdf']['dv a']   ={'value':    0.2, 'min':  0.09, 'max':  0.26, 'fixed': False}
conf['params']['tpdf']['dv b']   ={'value':    4.0, 'min':     0, 'max':    20, 'fixed': False}
conf['params']['tpdf']['dv c']   ={'value':    0.0, 'min':   -20, 'max':    20, 'fixed': False}
conf['params']['tpdf']['dv d']   ={'value':    0.0, 'min':   -20, 'max':    20, 'fixed': False}

conf['params']['tpdf']['ub N']    ={'value':   0.0, 'min':  -1.0, 'max':   1.0, 'fixed': False}
conf['params']['tpdf']['ub a']    ={'value':   0.2, 'min':  0.09, 'max':  0.26, 'fixed': False}
conf['params']['tpdf']['ub b']    ={'value':   4.0, 'min':     0, 'max':    20, 'fixed': False}
conf['params']['tpdf']['ub c']    ={'value':   0.0, 'min':   -20, 'max':    20, 'fixed': False}
conf['params']['tpdf']['ub d']    ={'value':   0.0, 'min':   -20, 'max':    20, 'fixed': False}

#conf['params']['tpdf']['db N']   ={'value':    0.0, 'min':  -1.0, 'max':   1.0, 'fixed': 'ub N'}
#conf['params']['tpdf']['db a']   ={'value':    0.2, 'min':  0.09, 'max':  0.26, 'fixed': 'ub a'}
#conf['params']['tpdf']['db b']   ={'value':    4.0, 'min':     0, 'max':    20, 'fixed': 'ub b'}
#conf['params']['tpdf']['db c']   ={'value':    0.0, 'min':   -20, 'max':    20, 'fixed': 'ub c'}
#conf['params']['tpdf']['db d']   ={'value':    0.0, 'min':   -20, 'max':    20, 'fixed': 'ub d'}



#--steps
conf['steps']={}

istep=7
#--full simultaneous fit
conf['D1 positivity'] = True
conf['H1 positivity'] = True
conf['SofferBound'] = True
conf['steps'][istep]={}
conf['steps'][istep]['dep']=[4,5,6]
conf['steps'][istep]['active distributions'] =['tpdf','tdiffpippim','diffpippim']
conf['steps'][istep]['passive distributions']=[]
conf['steps'][istep]['datasets']={}
conf['steps'][istep]['datasets']['dihadron_sia']=[]
conf['steps'][istep]['datasets']['dihadron_sia'].append(1000)  # BELLE SIA unpolarized
conf['steps'][istep]['datasets']['dihadron_sia'].append(103)  # PYTHIA s/tot, RS = 10.58
conf['steps'][istep]['datasets']['dihadron_sia'].append(104)  # PYTHIA c/tot, RS = 10.58
conf['steps'][istep]['datasets']['dihadron_sia'].append(203)  # PYTHIA s/tot, RS = 30.73 
conf['steps'][istep]['datasets']['dihadron_sia'].append(204)  # PYTHIA c/tot, RS = 30.73 
conf['steps'][istep]['datasets']['dihadron_sia'].append(205)  # PYTHIA b/tot, RS = 30.73 
conf['steps'][istep]['datasets']['dihadron_sia'].append(303)  # PYTHIA s/tot, RS = 50.88 
conf['steps'][istep]['datasets']['dihadron_sia'].append(304)  # PYTHIA c/tot, RS = 50.88 
conf['steps'][istep]['datasets']['dihadron_sia'].append(305)  # PYTHIA b/tot, RS = 50.88 
conf['steps'][istep]['datasets']['dihadron_sia'].append(403)  # PYTHIA s/tot, RS = 71.04 
conf['steps'][istep]['datasets']['dihadron_sia'].append(404)  # PYTHIA c/tot, RS = 71.04 
conf['steps'][istep]['datasets']['dihadron_sia'].append(405)  # PYTHIA b/tot, RS = 71.04 
conf['steps'][istep]['datasets']['dihadron_sia'].append(503)  # PYTHIA s/tot, RS = 91.19 
conf['steps'][istep]['datasets']['dihadron_sia'].append(504)  # PYTHIA c/tot, RS = 91.19 
conf['steps'][istep]['datasets']['dihadron_sia'].append(505)  # PYTHIA b/tot, RS = 91.19 
conf['steps'][istep]['datasets']['dihadron_sia'].append(2000)  # BELLE SIA transverse, binned in z1, M1
conf['steps'][istep]['datasets']['dihadron_sia'].append(2001)  # BELLE SIA transverse, binned in M1, M2
conf['steps'][istep]['datasets']['dihadron_sia'].append(2002)  # BELLE SIA transverse, binned in z1, z2
conf['steps'][istep]['datasets']['dihadron_sidis']=[]
conf['steps'][istep]['datasets']['dihadron_sidis'].append(2000)  # HERMES SIDIS p , binned in x 
conf['steps'][istep]['datasets']['dihadron_sidis'].append(2001)  # HERMES SIDIS p , binned in M 
conf['steps'][istep]['datasets']['dihadron_sidis'].append(2002)  # HERMES SIDIS p , binned in z 
conf['steps'][istep]['datasets']['dihadron_sidis'].append(2100)  # COMPASS SIDIS p, binned in x 
conf['steps'][istep]['datasets']['dihadron_sidis'].append(2101)  # COMPASS SIDIS p, binned in M 
conf['steps'][istep]['datasets']['dihadron_sidis'].append(2102)  # COMPASS SIDIS p, binned in z 
conf['steps'][istep]['datasets']['dihadron_sidis'].append(2110)  # COMPASS SIDIS D, binned in x 
conf['steps'][istep]['datasets']['dihadron_sidis'].append(2111)  # COMPASS SIDIS D, binned in M 
conf['steps'][istep]['datasets']['dihadron_sidis'].append(2112)  # COMPASS SIDIS D, binned in z 
conf['steps'][istep]['datasets']['dihadron_pp']=[]
conf['steps'][istep]['datasets']['dihadron_pp'].append(2100)  # STAR, RS=200, R<0.3, binned in PhT, eta < 0
conf['steps'][istep]['datasets']['dihadron_pp'].append(2101)  # STAR, RS=200, R<0.3, binned in PhT, eta > 0
conf['steps'][istep]['datasets']['dihadron_pp'].append(2102)  # STAR, RS=200, R<0.3, binned in M,   eta < 0
conf['steps'][istep]['datasets']['dihadron_pp'].append(2103)  # STAR, RS=200, R<0.3, binned in M,   eta > 0
conf['steps'][istep]['datasets']['dihadron_pp'].append(2104)  # STAR, RS=200, R<0.3, binned in eta
#conf['steps'][istep]['datasets']['dihadron_pp'].append(2300)  # STAR, RS=200, R<0.3, binned in PhT, eta < 0
#conf['steps'][istep]['datasets']['dihadron_pp'].append(2301)  # STAR, RS=200, R<0.3, binned in PhT, eta > 0
#conf['steps'][istep]['datasets']['dihadron_pp'].append(2302)  # STAR, RS=200, R<0.3, binned in M,   eta < 0
#conf['steps'][istep]['datasets']['dihadron_pp'].append(2303)  # STAR, RS=200, R<0.3, binned in M,   eta > 0
#conf['steps'][istep]['datasets']['dihadron_pp'].append(2304)  # STAR, RS=200, R<0.3, binned in eta
conf['steps'][istep]['datasets']['dihadron_pp'].append(3001)  # STAR, RS=500, binned in PhT,eta > 0
conf['steps'][istep]['datasets']['dihadron_pp'].append(3002)  # STAR, RS=500, binned in M,  eta < 0
conf['steps'][istep]['datasets']['dihadron_pp'].append(3003)  # STAR, RS=500, binned in M,  eta > 0
conf['steps'][istep]['datasets']['dihadron_pp'].append(3004)  # STAR, RS=500, binned in eta
#conf['steps'][istep]['datasets']['moments']=[]
#conf['steps'][istep]['datasets']['moments'].append(2000)   # ETMC,  delta u
#conf['steps'][istep]['datasets']['moments'].append(2001)   # PNDME, delta u
#conf['steps'][istep]['datasets']['moments'].append(3000)   # ETMC,  delta d
#conf['steps'][istep]['datasets']['moments'].append(3001)   # PNDME, delta d 

conf['FILT'] = {_: [] for _ in ['exp','par','value']}

conf['FILT']['exp'].append(('dihadron_sia',1000,10))
conf['FILT']['exp'].append(('dihadron_sia',103,10))
conf['FILT']['exp'].append(('dihadron_sia',104,10))
conf['FILT']['exp'].append(('dihadron_sia',203,10))
conf['FILT']['exp'].append(('dihadron_sia',204,10))
conf['FILT']['exp'].append(('dihadron_sia',205,10))
conf['FILT']['exp'].append(('dihadron_sia',303,10))
conf['FILT']['exp'].append(('dihadron_sia',304,10))
conf['FILT']['exp'].append(('dihadron_sia',305,10))
conf['FILT']['exp'].append(('dihadron_sia',403,10))
conf['FILT']['exp'].append(('dihadron_sia',404,10))
conf['FILT']['exp'].append(('dihadron_sia',405,10))
conf['FILT']['exp'].append(('dihadron_sia',503,10))
conf['FILT']['exp'].append(('dihadron_sia',504,10))
conf['FILT']['exp'].append(('dihadron_sia',505,10))
conf['FILT']['exp'].append(('dihadron_sia',2000,10))
conf['FILT']['exp'].append(('dihadron_sia',2001,10))
conf['FILT']['exp'].append(('dihadron_sia',2002,10))
conf['FILT']['exp'].append(('dihadron_sidis',2000,10))
conf['FILT']['exp'].append(('dihadron_sidis',2001,10))
conf['FILT']['exp'].append(('dihadron_sidis',2002,10))
conf['FILT']['exp'].append(('dihadron_sidis',2100,10))
conf['FILT']['exp'].append(('dihadron_sidis',2101,10))
conf['FILT']['exp'].append(('dihadron_sidis',2102,10))
conf['FILT']['exp'].append(('dihadron_sidis',2110,10))
conf['FILT']['exp'].append(('dihadron_sidis',2111,10))
conf['FILT']['exp'].append(('dihadron_sidis',2112,10))
conf['FILT']['exp'].append(('dihadron_pp',1000,10))
conf['FILT']['exp'].append(('dihadron_pp',2100,10))
conf['FILT']['exp'].append(('dihadron_pp',2101,10))
conf['FILT']['exp'].append(('dihadron_pp',2102,10))
conf['FILT']['exp'].append(('dihadron_pp',2103,10))
conf['FILT']['exp'].append(('dihadron_pp',2104,10))
conf['FILT']['exp'].append(('dihadron_pp',2300,10))
conf['FILT']['exp'].append(('dihadron_pp',2301,10))
conf['FILT']['exp'].append(('dihadron_pp',2302,10))
conf['FILT']['exp'].append(('dihadron_pp',2303,10))
conf['FILT']['exp'].append(('dihadron_pp',2304,10))
conf['FILT']['exp'].append(('dihadron_pp',3001,10))
conf['FILT']['exp'].append(('dihadron_pp',3002,10))
conf['FILT']['exp'].append(('dihadron_pp',3003,10))
conf['FILT']['exp'].append(('dihadron_pp',3004,10))

#conf['FILT']['exp'].append(('moments',2000,20))
#conf['FILT']['exp'].append(('moments',2001,20))
#conf['FILT']['exp'].append(('moments',3000,20))
#conf['FILT']['exp'].append(('moments',3001,20))















