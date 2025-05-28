# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:02:10 2021

@author: Maria
"""
import numpy as np
from gating import *
from createdbs import *
from dbssyn import *
from numba import jit
import matplotlib.pyplot as plt

#check maybe to also change gegi for pd sstate
@jit(nopython=True, cache=True)
def FOGnetwork(pd,stim,freq,Istim, timespike, tmax, dt, v1, v2, v3, v4, v7, n, sw):

#Usage: FOGnetwork(pd,stim,freq)
#
#Variables:
#pd - Variable to determine whether network is under the healthy or 
#Parkinsonian condition. For healthy, pd = 0, for Parkinson's, pd = 1.
#stim - Variable to determine whether deep brain stimulation is on.
#If DBS is off, stim = 0. If DBS is on, stim = 1.
#freq - Determines the frequency of stimulation, in Hz.
#sw - determines of theta switch for dbs is on or off. 
#  
#Author: Mariia Popova, UKE; based on Rosa So, Duke University 
#Updated 2/02/2021
     
    ##Membrane parameters
    Cm=1 
    #In order of Th,STN,GP,PPN,Str,PRF,LC or Th,STN,GPe,GPi,PPN,SNr,Str,PRF,LC
    gl=np.array([0.05, 2.25, 0.1, 0.1, 0.4, 0.3]); El=np.array([-70, -60, -65, -67, -70, -17])
    gna=np.array([3, 37, 120, 30, 100, 120]); Ena=np.array([50, 55, 55, 45, 50, 45]) 
    gk=np.array([5, 45, 30, 3.2, 80, 10, 20]); Ek=np.array([-75, -80, -80, -95, -100, -95, -72])
    gt=np.array([5, 0.5, 0.5]); Et=0
    gca=np.array([0, 2, 0.15]); Eca=np.array([0, 140, 120])
    gahp=np.array([0, 20, 10]) #eahp and ek are the same excluding th
    gcort=0.15; Ecort=0      #cortex par for ppn
    Bcort=1 #ms^-1
    gm=1; Em=-100 #for striatum muscarinic current
    #stn coupling 
    gc1=0.1; gc2=0.1
    
    k1=np.array([0, 15, 10])    #dissociation const of ahp current 
    kca=np.array([0, 22.5, 15]) #calcium pump rate constant

    #synapse params alike in rubin SNr same to Gpi, Str same to Gpi
    A=np.array([0, 3, 2, 2, 3, 2, 2, 3, 3, 3, 2, 3]) 
    B=np.array([0, 0.1, 0.04, 0.08, 0.1, 0.08, 0.08, 0.1, 0.1, 0.1, 0.08, 0.1]) #maybe change 0.08 to so
    the=np.array([0, 30, 20, 20, 30, 20, 20, 30, 30, 30, 20, 30]) #maybe change alike so for prf cnf lc???
    
    ##Synapse parameters
    #In order of Igesn,Isnge,Igege,Isngi,Igegi,Igith 
    gsyn = np.array([1, 0.3, 1, 0.3, 1, .08])   #alike in Rubin gsyn and in So Esyn
    Esyn = np.array([-85, 0, -85, 0, -85, -85]) #alike in Rubin gsyn and in So Esyn

    tau=5; gpeak1=0.3; gpeak=0.43 #parameters for second-order alpha synapse

    gsynstr=np.array([0.8, 1.65, 17, 1.65, 0.05]); ggaba=0.1; gcorstr=0.07
    Esynstr=np.array([-85, 0, -85, -85, -85, -85, -85]); tau_i=13 #parameters for striatum synapses in order gaba-rec crtx strge gestr strgi strsnr snrstr

    # #gsynppn = np.array([0.061, 0.22, 0.061, 0.061, 0.22, 0.061, 0.061, 0.061, 0.061]) 
    # #gsynppn = np.array([0.061, 0.22, 0, 0, 0.22, 0, 0.061, 0.061, 0])
    # gsynppn = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    # Esynppn = np.array([0, -85, 0, 0, -85, 0, 0, 0, 0]) #in order snppn gippn ppnsn ppngi snrppn ppnprf, prfppn, cnfppn, ppnge
    # #gsynppn = np.array([0, 0.17, 0.2, 0.18, 0.17, 0.18]); Esynppn = np.array([0, -85, 0, 0, -85, 0])
    # tau=5; gpeak1=0.3; gpeak=0.43 #parameters for second-order alpha synapse
    # #gsynsnr=np.array([0.15, 0.15, 0.15, 0.15, 0.15]) 
    # gsynsnr=np.array([0.061, 0.061, 0.061, 0.061, 0.061])
    # Esynsnr=np.array([0, 0, 0, 0, -85]) #for "to" snr synapses in order stn, prf, cnf, ppn, gpe
    # gsynstr=np.array([0.8, 1.65, 17, 1, 1]); ggaba=0.1; gcorstr=0.07
    # Esynstr=np.array([-85, 0, -85, -85, -85, -85, -85]); tau_i=13 #parameters for striatum synapses in order gaba-rec crtx strge gestr strgi strsnr snrstr
    # #gsyncnf=np.array([0.22, 0.22, 0.22]); Esyncnf=np.array([0,-85,0]) #cnf in order prf, snr, ppn
    # gsyncnf=np.array([0.061, 0.061, 0.061]); Esyncnf=np.array([0,-85,0]) #cnf in order prf, snr, ppn
    # gsynlc=np.array([0.061, 0.061, 0.061]); Esynlc=np.array([0, 0, 0]) #why these values?!!! - alike PPN? maybe change???
    # gsynprf = np.array([0.061, 0.061, 0.061]); Esynprf=np.array([0, -85, 0]) #why these values?!!!; prf in order cnf
    # gsynsnc = np.array([0.061, 0.061, 0.061, 0.061, 0.061, 0.061, 0.061, 0.061, 0.061, 0.061, 0.061, 0.061]) 
    # Esynsnc=np.array([0, 0, -85, 0, -85, -85, -85, -85, -85, -85, -85, -85]) #snc in order lcsnc, ppnsnc, snrsnc, stnsnc, sncsnr, sncstn, sncgi, sncge, sncstr #why these values?!!!
    # gsynctx=np.array([0.061, 0.061, 0.061, 0.061]) #chosen how???? alike LC?
    # Esynctx=np.array([0,0,0,-85]) #ctx in order lcctx, prfctx, ppnctx, gectx

    Esyn_e = 0
    Esyn_i = -85

    gsyn_e = 0.15 #0.15
    gsyn_i = 0.8

    #time step
    t=np.arange(0,tmax,dt)

    ##Setting initial matrices  
    #n - number of neurons in each population
    vth=np.zeros(shape=(n,len(t)))   #thalamic membrane voltage
    vsn=np.zeros(shape=(n,len(t)))   #STN membrane voltage
    vge=np.zeros(shape=(n,len(t)))   #GPe membrane voltage
    vgi=np.zeros(shape=(n,len(t)))   #GPi membrane voltage
    vstr=np.zeros(shape=(n,len(t)))  #striatum membrane voltage
    vef=np.zeros(shape=(n,len(t)))   #STN effs membrane voltage

    Z4=np.zeros(n)  #for 2 order alpha-synapse gpi-th current
    S4=np.zeros(n)  #for alpha-synapse gpi-th current
    S3=np.zeros(n)  #for alpha-synapse ge-ge/sn/str/snr/snc current
    S3_1=np.zeros(n) #for dummy gesn current
    S2=np.zeros(n)  #for alpha-synapse snge/gi/snr current
    Z2=np.zeros(n)  #for 2 order alpha-synapse sn current
    S2_1=np.zeros(n) #for dummy snge current
    S3_2=np.zeros(n) #for dummy gege current
    S6=np.zeros(n)  #for alpha-synapse gpi-ppn/snc current
    Sc=np.zeros(shape=(n,len(t))) #for cortex-ppn synapse
    S1c=np.zeros(n) #for striatum gaba-rec
    #for synapses striatum-gaba-rec
    gall=np.random.choice(np.arange(n),n,replace=False)
    gbll=np.random.choice(np.arange(n),n,replace=False)
    gcll=np.random.choice(np.arange(n),n,replace=False)
    gdll=np.random.choice(np.arange(n),n,replace=False)
    S10=np.zeros(n) #for alpha-synapse str-ge/gi/snr/snc current

    ##with or without DBS
    if stim==0: 
        Idbs=np.zeros(len(t))
        Idbs1=np.zeros(len(t))
    else:
        #Idbs=createdbs(freq,tmax,dt) #creating DBS train with frequency freq
        Idbs = dbssyn(freq,tmax,dt,sw) #creating DBS train with frequency
        Idbs1 = dbssyn(60,tmax,dt,sw) #creating DBS train with frequency freq, low-freq for ssnr

    ##initial conditions 
    vth[:,0]=v1
    vsn[:,0]=v2
    vge[:,0]=v3
    vgi[:,0]=v4    
    vstr[:,0]=v7  #for striatum D2
    vef[:,0]=v2   #for STN ef

    #helper variables for gating and synapse params - starting parameters
    R2=stn_rinf(vsn[:,0])        #r for stn
    H1=th_hinf(vth[:,0])         #h for th
    R1=th_rinf(vth[:,0])         #r for th
    N2=stn_ninf(vsn[:,0])        #n for stn
    H2=stn_hinf(vsn[:,0])        #h for stn
    C2=stn_cinf(vsn[:,0])        #c for stn
    CA2=np.array([0.1])          #intracellular concentration of Ca2+ in muM for stn
    CA3=CA2                      #for gpe
    CA4=CA2                      #for gpi
    N3=gpe_ninf(vge[:,0])        #n for gpe
    H3=gpe_hinf(vge[:,0])        #h for gpe
    R3=gpe_rinf(vge[:,0])        #r for gpe
    N4=gpe_ninf(vgi[:,0])        #n for gpi
    H4=gpe_hinf(vgi[:,0])        #h for gpi
    R4=gpe_rinf(vgi[:,0])        #r for gpi
    #for STN efferents
    N2e=stn_ninf(vef[:,0])
    H2e=stn_hinf(vef[:,0])
    R2e=stn_rinf(vef[:,0])
    CA2e=CA2
    C2e=stn_cinf(vef[:,0])

    #striatum gating
    m7=str_alpham(vstr[:,0])/(str_alpham(vstr[:,0])+str_betam(vstr[:,0]))
    h7=str_alphah(vstr[:,0])/(str_alphah(vstr[:,0])+str_betah(vstr[:,0]))
    n7=str_alphan(vstr[:,0])/(str_alphan(vstr[:,0])+str_betan(vstr[:,0]))
    p7=str_alphap(vstr[:,0])/(str_alphap(vstr[:,0])+str_betap(vstr[:,0]))

    timespikeint=np.ones_like(timespike)
    timespikeint = (np.round(timespike,0,timespikeint)).astype(np.int64) #index when 1 for ppn
    looptimespikeint=(timespikeint/dt).astype(np.int64)

    gpetim = np.zeros(n) #time counter for STDP
    gpetim_1=np.zeros(n) #for dummy time counter
    gpetim_2=np.zeros(n) #for dummy time counter
    w_1 = np.zeros(n) #new weight for 1 stdp syn
    w_2 = np.zeros(n) #new weight for 1 stdp syn
    w_1fin = np.zeros(n) #weight for 1 stdp syn
    w_2fin = np.zeros(n) #weight for 1 stdp syn
    sntim = np.zeros(n) #time counter for STDP
    sntim_1=np.zeros(n) #for dummy time counter
    w_1sn = np.zeros(n) #new weight for 1 stdp syn
    w_2sn = np.zeros(n) #new weight for 1 stdp syn
    w_1finsn = np.zeros(n) #weight for 1 stdp syn
    w_2finsn = np.zeros(n) #weight for 1 stdp syn
    gpitim = np.zeros(n) #time counter for STDP
    w_1gi = np.zeros(n) #new weight for 1 stdp syn
    w_2gi = np.zeros(n) #new weight for 1 stdp syn
    w_1fingi = np.zeros(n) #weight for 1 stdp syn
    w_2fingi = np.zeros(n) #weight for 1 stdp syn
    
    ##Time loop
    for i in range(1, len(t)):
    
        #condition for cortex current for ppn and striatum
        if np.sum(looptimespikeint==i)==1:
            Sc[:,i-1] = 1
    
        #previous values
        V1=vth[:,i-1];    V2=vsn[:,i-1];     V3=vge[:,i-1];    V4=vgi[:,i-1];   
        V7=vstr[:,i-1]; 
        Ve=vef[:,i-1];   
        
        #Synapse parameters 
        S2_1[1:n]=S2[0:n-1];S2_1[0]=S2[n-1]    #dummy synapse for snge current as there is 1 stn to 2 ge
        S3_1[0:n-1]=S3[1:n];S3_1[-1]=S3[0]     #dummy synapse for gesn current as there is 1 ge to 2 stn
        S3_2[2:n]=S3[0:n-2];S3_2[:2]=S3[n-2:n] #dummy synapse for gege current as there is 1 ge to 2 ge
        S11cr=S1c[gall];S12cr=S1c[gbll];S13cr=S1c[gcll];S14cr=S1c[gdll] #dummy striatum crtx current
        
        gpetim_1[0:n-1]=gpetim[1:n];gpetim_1[-1]=gpetim[0]     #dummy counter for stdp
        gpetim_2[2:n]=gpetim[0:n-2];gpetim_2[:2]=gpetim[n-2:n] #dummy counter for stdp
        sntim_1[1:n]=sntim[0:n-1];sntim_1[0]=sntim[n-1]    #dummy counter for stdp

        #membrane parameters - gating variables
        m1=th_minf(V1);  m2=stn_minf(V2); m3=gpe_minf(V3); m4=gpe_minf(V4) #gpe and gpi are modeled similarily

        n2=stn_ninf(V2); n3=gpe_ninf(V3); n4=gpe_ninf(V4);

        h1=th_hinf(V1);  h2=stn_hinf(V2); h3=gpe_hinf(V3); h4=gpe_hinf(V4) 

        p1=th_pinf(V1); 

        a2=stn_ainf(V2); a3=gpe_ainf(V3); a4=gpe_ainf(V4) #for low-treshold ca

        b2=stn_binf(R2)

        s3=gpe_sinf(V3); s4=gpe_sinf(V4)

        r1=th_rinf(V1);  r2=stn_rinf(V2); r3=gpe_rinf(V3); r4=gpe_rinf(V4)

        c2=stn_cinf(V2)

        #for effs
        m2e=stn_minf(Ve); n2e=stn_ninf(Ve); h2e=stn_hinf(Ve); a2e=stn_ainf(Ve); b2e=stn_binf(R2e)
        r2e=stn_rinf(Ve); c2e=stn_cinf(Ve)
    
        #membrane parameters - time constants
        tn2=stn_taun(V2); tn3=gpe_taun(V3); tn4=gpe_taun(V4);

        th1=th_tauh(V1); th2=stn_tauh(V2); th3=gpe_tauh(V3); th4=gpe_tauh(V4)

        tr1=th_taur(V1); tr2=stn_taur(V2); tr3=30; tr4=30; 

        tc2=stn_tauc(V2); 

        #for effs
        tn2e=stn_taun(Ve); th2e=stn_tauh(Ve); tr2e=stn_taur(Ve); tc2e=stn_tauc(Ve)
    
        #thalamic cell currents
        Il1=gl[0]*(V1-El[0])
        Ina1=gna[0]*(m1**3)*H1*(V1-Ena[0])
        Ik1=gk[0]*((0.75*(1-H1))**4)*(V1-Ek[0]) #misspelled in So paper
        It1=gt[0]*(p1**2)*R1*(V1-Et)
        Igith=1.4*gsyn[5]*(V1-Esyn[5])*S4 #for alpha-synapse second order kinetics
    
        #STN cell currents
        Il2=gl[1]*(V2-El[1])
        Ik2=gk[1]*(N2**4)*(V2-Ek[1])
        Ina2=gna[1]*(m2**3)*H2*(V2-Ena[1])
        It2=gt[1]*(a2**3)*(b2**2)*(V2-Eca[1]) #misspelled in So paper
        Ica2=gca[1]*(C2**2)*(V2-Eca[1])
        Iahp2=gahp[1]*(V2-Ek[1])*(CA2/(CA2+k1[1])) #cause ek and eahp are the same
        Igesn=0.5*(gsyn[0]*(V2-Esyn[0])*(S3+S3_1))  #first-order kinetics 1ge to 2sn
        #Iappstn=36
        Iappstn=35 #38.5
        Icorsn=0.5*gcort*(Sc[:,i-1])*(V2-Ecort) #alike ppn
        Ic1=gc1*(V2-Ve)
        #efferents
        Il2e=gl[1]*(Ve-El[1])
        Ik2e=gk[1]*(N2e**4)*(Ve-Ek[1])
        Ina2e=gna[1]*(m2e**3)*H2e*(Ve-Ena[1])
        It2e=gt[1]*(a2e**3)*(b2e**2)*(Ve-Eca[1])
        Ica2e=gca[1]*(C2e**2)*(Ve-Eca[1])
        Iahp2e=gahp[1]*(Ve-Ek[1])*(CA2e/(CA2e+k1[1]))
        Ic2=gc2*(Ve-V2)
        Igesne=0.5*(gsyn[0]*(Ve-Esyn[0])*(S3+S3_1)) #first-order kinetics 1ge to 2sn
        Icorsne=0.5*gcort*(Sc[:,i-1])*(Ve-Ecort) #alike ppn
    
        #GPe cell currents
        Il3=gl[2]*(V3-El[2])
        Ik3=gk[2]*(N3**4)*(V3-Ek[2])  
        Ina3=gna[2]*(m3**3)*H3*(V3-Ena[2])
        It3=gt[2]*(a3**3)*R3*(V3-Eca[2]) #Eca as in Rubin and Terman
        Ica3=gca[2]*(s3**2)*(V3-Eca[2])  #misspelled in So paper
        Iahp3=gahp[2]*(V3-Ek[2])*(CA3/(CA3+k1[2])) #as Ek is the same with Eahp
        #Isnge=0.5*(gsyn[1]*(V3-Esyn[1])*(S2+S2_1)) #second-order kinetics 1sn to 2ge
        #Igege=0.5*((gsyn[2]+0.8*pd)*(V3-Esyn[2])*0.5*(S3_1+S3_2))
        tij_1 = gpetim - gpetim_1
        tij_2 = gpetim - gpetim_2
        tij_1pos = np.array([i for i in tij_1 if i >= 0])
        tij_1neg = np.array([i for i in tij_1 if i < 0])
        tij_2pos = np.array([i for i in tij_2 if i >= 0])
        tij_2neg = np.array([i for i in tij_2 if i < 0])
        w_1[tij_1>=0] = -1*np.exp(dt*(-4*tij_1pos)/8)
        w_2[tij_2>=0] = -1*np.exp(dt*(-4*tij_2pos)/8)
        w_1[tij_1<0] = ((dt*5*np.abs(tij_1neg))/8)*np.exp(dt*(-2*np.abs(tij_1neg))/8)
        w_2[tij_2<0] = ((dt*5*np.abs(tij_2neg))/8)*np.exp(dt*(-2*np.abs(tij_2neg))/8)
        w_1fin += 0.004*w_1 
        w_2fin += 0.004*w_2 
        w_1fin[w_1fin<0.0001]=0.0001
        w_2fin[w_2fin<0.0001]=0.0001
        w_1fin[w_1fin>0.6]=0.6
        w_2fin[w_2fin>0.6]=0.6
        Igege=0.5*((gsyn[2]+0.8*pd)*(V3-Esyn[2])*(w_1fin*S3_1+w_2fin*S3_2))
        #Iappgpe=16.5 
        #Iappgpe=9.5 #gege plasticity enabled
        Iappgpe=9.5 #plasticity enabled
        tij_1sn = sntim - gpetim
        tij_2sn = sntim_1 - gpetim
        tij_1possn = np.array([i for i in tij_1sn if i >= 0])
        tij_1negsn = np.array([i for i in tij_1sn if i < 0])
        tij_2possn = np.array([i for i in tij_2sn if i >= 0])
        tij_2negsn = np.array([i for i in tij_2sn if i < 0])
        w_1sn[tij_1sn>=0] = 0.002*np.exp(-1*dt*np.abs(tij_1possn)/12)
        w_2sn[tij_2sn>=0] = 0.002*np.exp(-1*dt*np.abs(tij_2possn)/12)
        w_1sn[tij_1sn<0] = -0.002*1.1*np.exp(-1*dt*np.abs(tij_1negsn)/27.5)
        w_2sn[tij_2sn<0] = -0.002*1.1*np.exp(-1*dt*np.abs(tij_2negsn)/27.5)
        w_1finsn += w_1sn 
        w_2finsn += w_2sn 
        w_1finsn[w_1finsn<0]=0
        w_2finsn[w_2finsn<0]=0
        w_1finsn[w_1finsn>1]=1
        w_2finsn[w_2finsn>1]=1
        #print(w_1finsn)
        Isnge=0.5*(gsyn[1]*(V3-Esyn[1])*(w_1finsn*S2+w_2finsn*S2_1)) #second-order kinetics 1sn to 2ge
        #str-gpe synapse 
        Istrge=0.1*gsyn_i*(V3-Esynstr[2])*S10 #1str to 1ge
        #cortical connection
        Icorge=0.5*gcort*(Sc[:,i-1])*(V3-Ecort) #alike ppn
        
        #GPi cell currents
        Il4=gl[2]*(V4-El[2])
        Ik4=gk[2]*(N4**4)*(V4-Ek[2])
        Ina4=gna[2]*(m4**3)*H4*(V4-Ena[2]) #Eca as in Rubin and Terman
        It4=gt[2]*(a4**3)*R4*(V4-Eca[2])   #misspelled in So paper
        Ica4=gca[2]*(s4**2)*(V4-Eca[2]) 
        Iahp4=gahp[2]*(V4-Ek[2])*(CA4/(CA4+k1[2])) #as Ek is the same with Eahp
        #Isngi=0.5*(gsyn[3]*(V4-Esyn[3])*(S2+S2_1)) #second-order kinetics 1sn to 2gi
        tij_1gi = sntim - gpitim
        tij_2gi = sntim_1 - gpitim
        tij_1posgi = np.array([i for i in tij_1gi if i >= 0])
        tij_1neggi = np.array([i for i in tij_1gi if i < 0])
        tij_2posgi = np.array([i for i in tij_2gi if i >= 0])
        tij_2neggi = np.array([i for i in tij_2gi if i < 0])
        w_1gi[tij_1gi>=0] = 0.002*np.exp(-1*dt*np.abs(tij_1posgi)/12)
        w_2gi[tij_2gi>=0] = 0.002*np.exp(-1*dt*np.abs(tij_2posgi)/12)
        w_1gi[tij_1gi<0] = -0.002*1.1*np.exp(-1*dt*np.abs(tij_1neggi)/27.5)
        w_2gi[tij_2gi<0] = -0.002*1.1*np.exp(-1*dt*np.abs(tij_2neggi)/27.5)
        w_1fingi += w_1gi 
        w_2fingi += w_2gi 
        w_1fingi[w_1fingi<0]=0
        w_2fingi[w_2fingi<0]=0
        w_1fingi[w_1fingi>1]=1
        w_2fingi[w_2fingi>1]=1
        Isngi=0.5*(gsyn[3]*(V4-Esyn[3])*(w_1fingi*S2+w_2fingi*S2_1)) #second-order kinetics 1sn to 2gi
        Igegi=0.5*(gsyn[4]*(V4-Esyn[4])*(S3_1+S3_2)) #first-order kinetics 1ge to 2gi
        Iappgpi=16.5 #17
        #str-gpi synapse
        Istrgi=gsyn_i*(V4-Esynstr[4])*S10 #1str to 1gi
  
        #Striatum D2 cell currents
        Ina7=gna[4]*(m7**3)*h7*(V7-Ena[4])
        Ik7=gk[4]*(n7**4)*(V7-Ek[4])
        Il7=gl[3]*(V7-El[3])
        #Im7=(2.6-2.5*pd)*gm*p7*(V7-Em) 
        Im7=(2.6-0.2*pd)*gm*p7*(V7-Em) 
        #Im7=(2.6-bopt*pd)*gm*p7*(V7-Em)
        Igaba7=(ggaba/4)*(V7-Esynstr[0])*(S11cr+S12cr+S13cr+S14cr) #maybe change to 3.5 for direct and indirect #recieves input from 40% remaining
        #Icorstr=(6*gcorstr-0.3*pd)*(V7-Esynstr[1])*Sc[:,i-1] #optimized
        Icorstr=0.5*(5*gcorstr-0.3*pd)*(V7-Esynstr[1])*(Sc[:,i-1]) #optimized 
        #ge-str synapse
        Igestr=gsyn_i*(V7-Esynstr[3])*S3 #1ge to 1str
        Iappstr=4
    
        #Differential Equations for cells using forward Euler method
        
        #thalamic
        vth[:,i]=V1+dt*(1/Cm*(-Il1-Ik1-Ina1-It1-Igith+Istim[i]))
        H1=H1+dt*((h1-H1)/th1)
        R1=R1+dt*((r1-R1)/tr1)

        #for cortex
        Sc[:,i]=Sc[:,i-1]+dt*(-Bcort*Sc[:,i-1])
    
        #STN
        #vsn[:,i]=V2+dt*(1/Cm*(-Il2-Ik2-Ina2-It2-Ica2-Iahp2-Igesn-Isncsn+Iappstn+Idbs[i]-Ippnsn-Icorsn-Ic1)) #currently STN-DBS
        ch = V2+dt*(1/Cm*(-Il2-Ik2-Ina2-It2-Ica2-Iahp2-Igesn+Iappstn-Icorsn-Ic1))
        if len(ch[np.logical_and(ch>-10, Idbs[i]>10)])!=0:
            ch[np.logical_and(ch>-10, Idbs[i]>10)]=ch[np.logical_and(ch>-10, Idbs[i]>10)]
        ch[np.logical_not(np.logical_and(ch>-10, Idbs[i]>10))]=\
            ch[np.logical_not(np.logical_and(ch>-10, Idbs[i]>10))]+dt*(1/Cm*(0.1*Idbs[i])) #why 0.01???
        vsn[:,i]=ch
        N2=N2+dt*(0.75*(n2-N2)/tn2)
        H2=H2+dt*(0.75*(h2-H2)/th2)
        R2=R2+dt*(0.2*(r2-R2)/tr2)
        CA2=CA2+dt*(3.75*1e-5*(-Ica2-It2-kca[1]*CA2))
        C2=C2+dt*(0.08*(c2-C2)/tc2)
    
        #STN effs
        vef[:,i]=Ve+dt*(1/Cm*(-Il2e-Ik2e-Ina2e-It2e-Ica2e-Iahp2e+Idbs[i]-Ic2-Igesne+Iappstn-Icorsne)) #currently STN-DBS 
        N2e=N2e+dt*(0.75*(n2e-N2e)/tn2e)
        H2e=H2e+dt*(0.75*(h2e-H2e)/th2e)
        R2e=R2e+dt*(0.2*(r2e-R2e)/tr2e)   
        CA2e=CA2e+dt*(3.75*1e-5*(-Ica2e-It2e-kca[1]*CA2e))
        C2e=C2e+dt*(0.08*(c2e-C2e)/tc2e)
        #for second-order alpha-synapse
        a=np.where(np.logical_and(vef[:,i-1]<-10, vef[:,i]>-10))[0]
        u=np.zeros(n) 
        u[a]=gpeak/(tau*np.exp(-1))/dt 
        S2=S2+dt*Z2
        zdot=u-2/tau*Z2-1/(tau**2)*S2
        Z2=Z2+dt*zdot  
        #spiking times
        timsn=np.where(np.logical_and(vef[:,i-1]<-10, vef[:,i]>-10))[0]
        sntim[timsn] = i
        #print(sntim)
    
        #GPe
        vge[:,i]=V3+dt*(1/Cm*(-Il3-Ik3-Ina3-It3-Ica3-Iahp3+Iappgpe-Isnge-Istrge-Igege-Icorge))
        N3=N3+dt*(0.1*(n3-N3)/tn3) #misspelled in So paper
        H3=H3+dt*(0.05*(h3-H3)/th3) #misspelled in So paper
        R3=R3+dt*(1*(r3-R3)/tr3) #misspelled in So paper
        CA3=CA3+dt*(1*1e-4*(-Ica3-It3-kca[2]*CA3))
        #ge-sn/str/snr/snc synapse
        S3=S3+dt*(A[2]*(1-S3)*Hinf(V3-the[2])-B[2]*S3)
        #spiking times
        tim=np.where(np.logical_and(vge[:,i-1]<-10, vge[:,i]>-10))[0]
        gpetim[tim] = i
        #print(gpetim)
    
        #GPi
        vgi[:,i]=V4+dt*(1/Cm*(-Il4-Ik4-Ina4-It4-Ica4-Iahp4+Iappgpi-Isngi-Igegi-Istrgi))
        N4=N4+dt*(0.1*(n4-N4)/tn4) #misspelled in So paper
        H4=H4+dt*(0.05*(h4-H4)/th4) #misspelled in So paper
        R4=R4+dt*(1*(r4-R4)/tr4) #misspelled in So paper
        CA4=CA4+dt*(1*1e-4*(-Ica4-It4-kca[2]*CA4))
        #for second-order alpha-synapse
        a=np.where(np.logical_and(vgi[:,i-1]<-10, vgi[:,i]>-10))[0]
        u=np.zeros(n) 
        u[a]=gpeak1/(tau*np.exp(-1))/dt 
        S4=S4+dt*Z4 
        zdot=u-2/tau*Z4-1/(tau**2)*S4
        Z4=Z4+dt*zdot
        #for gpi-ppn/snc synapse
        S6=S6+dt*(A[3]*(1-S6)*Hinf(V4-the[3])-B[3]*S6)
        #spiking times
        timgi=np.where(np.logical_and(vgi[:,i-1]<-10, vgi[:,i]>-10))[0]
        gpitim[timgi] = i
        #print(sntim)
    
        #Striatum D2
        vstr[:,i]=V7+(dt/Cm)*(-Ina7-Ik7-Il7-Im7-Igaba7-Icorstr+Iappstr-Igestr)
        m7=m7+dt*(str_alpham(V7)*(1-m7)-str_betam(V7)*m7)
        h7=h7+dt*(str_alphah(V7)*(1-h7)-str_betah(V7)*h7)
        n7=n7+dt*(str_alphan(V7)*(1-n7)-str_betan(V7)*n7)
        p7=p7+dt*(str_alphap(V7)*(1-p7)-str_betap(V7)*p7)
        S1c=S1c+dt*((str_Ggaba(V7)*(1-S1c))-(S1c/tau_i))
        #for str-gpe/gpi/str/snc synapse
        S10=S10+dt*(A[6]*(1-S10)*Hinf(V7-the[6])-B[6]*S10) 
    

    # plt.figure()
    # plt.plot(Sc[0,:])

    return vsn, vge, vgi, vth, vstr, Idbs

    
    
