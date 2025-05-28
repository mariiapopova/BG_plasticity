# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:12:45 2021

@author: Maria
"""
import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def dbssyn(f,tmax,dt,sw):
    #transmission + synaptic delay: td 
    td=0 #2 ms for trasmission and .5 ms for synaptic delay prev-2
    ti=0 
    tf=tmax+td #in miliseconds 
    t=np.arange(ti,tf,dt)

    #DBS input
    fdbs=f
    #in miliseconds
    dbsi=(dt)/dt
    dbsf=tmax/dt

    #I Kernel time constant
    taus=1.7  #For excitatory synapse (Markram)
    #taus=8.3 #For inhibitory synapse (Markram)

    #transmission + synaptic delay: td 
    td=int(td/dt) #convert to simulation step scale!!!!! integer?

    #input spike train
    sp=np.zeros(len(t))

    #Synapse parameters # Each column 1,2,3 means F,D,P respectively and each row means
    #Excitatory and inhibitory synapse (1: excitatory, 2: inhibitory)
    #In this study we just used the first row, excitatory synapses. (Markram)
    tauf=np.array([[670,17,326], [376,21,62]])
    taud=np.array([[138,671,329], [45,706,144]])
    U=np.array([[.09,.5,.29], [.016,.25,.32]])
    A=np.array([[.00025,.00025,.00025], [.00025,.00025,.00025]]) #(250pA Tsodyks)
    #A=np.array([[1,1,1], [1,1,1]])
    n=100 
    A=n*A  #change the strength of A (order of magnitude of totall number of synapses) 10 to 10
    ie=np.ones(2)

    fid=2.5 #synaptic fidelity 
    we=fid*200
    wi=0 
    #Percentage of excitatory and inhibitory synapses:
    exper=np.array([45,38,17])
    #ne=np.zeros(3)
    #for 1 synapse n1=1 and so forth (approximately giving 2 pA exc. current)
    #ne=10   #for 10 synapses (approximately giving 20 pA exc. current)
    #ne=100  #for 100 synapses (approximately giving 200 pA exc. current)
    #ne=1000 #for 1000 synapses (approximately giving 2 nA exc. current)
    #ni=wi*np.array([13,10,6]) % for 1 synapse (approximately giving 10 pA inhibitory current)
    inper=np.array([8,76,16])
    per=np.vstack((exper,inper))
    weg=np.array([we,wi])
    wegst=np.vstack((weg,weg,weg)).T
    perfin=wegst*per
    #ne=ni
    #ni=np.zeros(3)
    #ni=10   #for 10 synapses (approximately giving 100 pA inh. current)
    #ni=100  #for 100 synapses (approximately giving 1 nA inh. current)
    #ni=1000 #for 1000 synapses (approximately giving 10 nA inh. current)
    A=A*perfin #!!!!check!!!!

    #Compute EPSC
    u=np.zeros(len(t))
    x=np.ones(len(t))
    I=np.zeros(len(t))
    PSC=np.zeros(shape=(len(ie),np.shape(A)[1],len(t)))

    #Compute neuron firing pattern with and without synaptic input:
    for q in range(len(ie)):
        if q==0: 
            w=1
        else:
            w=-1
            
        for p in range(np.shape(A)[1]): #check!!
            T=round((tmax/fdbs)/dt)
            dbs=np.arange(dbsi,dbsf,T)
            ts=dbs.astype(np.int64)    #uncomment for DBS only
            fir=np.ones(int((tmax/dt)/10))
            zer=np.zeros(int((tmax/dt)/10))
            if sw==1:
                turner = np.hstack((fir,zer,fir,zer,fir,zer,fir,zer,fir,zer))
            else:
                turner = np.ones(int(tmax/dt))
            sp[ts]=1/dt
            for i in range(int(td),len(t)-1):
                # u[i+1] = u[i] + dt*(-(u[i]/tauf[q,p])+U[q,p]*(1-u[i])*sp[i-td])
                # x[i+1] = x[i] + dt*((1/taud[q,p])*(1-x[i]) - u[i+1]*x[i]*sp[i-td])
                # I[i+1] = I[i] + dt*((-1/taus)*I[i] + A[q,p]*u[i+1]*x[i]*sp[i-td]);
                u[i+1] = u[i] + dt*(-(u[i]/tauf[q,p])+U[q,p]*(1-u[i])*sp[i-td]*turner[i-td])
                x[i+1] = x[i] + dt*((1/taud[q,p])*(1-x[i]) - u[i+1]*x[i]*sp[i-td]*turner[i-td])
                I[i+1] = I[i] + dt*((-1/taus)*I[i] + A[q,p]*u[i+1]*x[i]*sp[i-td]*turner[i-td]); 
                #replace I with Iwo for no depletion
            PSC[q,p,:] = w*I

    PSC_exc=np.sum(PSC[0,:,:],0)
    PSC_inh=np.sum(PSC[1,:,:],0)
    PSC_all=PSC_exc+PSC_inh
    
    return PSC_all


