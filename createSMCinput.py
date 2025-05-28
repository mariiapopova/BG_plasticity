# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:20:46 2021

@author: Maria
"""
import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def createSMCinput(tmax,dt,freq,cv):
    #creates Sensorimotor Cortex (SMC) input to thalamic cells

    #Variables:
    #tmax - length of input train (msec)
    #dt - time step (msec)
    #freq - frequency of input train
    #cv - coefficient of variation of input train (gamma distribution)

    #Output
    #Istim - Input train from SMC
    #timespike - Timing of each input pulse

    t=np.arange(0, tmax, dt)   #create time vector
    amp=3.5                    #pulse amplitude in muA/cm^2
    Istim=np.zeros(len(t)) #empty stim array
    dur=5                      #pulse duration in ms
    p=int(round(dur/dt))
    pulse=amp*np.ones(p)  #create a pulse
    timespike=np.empty(0)

    #set counters
    i=0
    j=0 
    A = 1/cv**2 #shape parameter 
    B = freq/A #scale parameter 
    
    if cv==0:
        instfreq=freq #take freq
    else:
        instfreq=np.random.gamma(A,B) #take freq from gamma distribution

    ipi=tmax/instfreq #calculate inter-spike interval!!! was 1000 before
    ip=int(round(ipi/dt)) 
    i+=ip  #inter-spike interval
    
    #loop to create SMC input
    while i<len(t) and i+p<(len(t)): 
        timespike=np.hstack((timespike,np.array([t[i]])))       #calculate time when spike happens (begins)
        Istim[i:i+p]=pulse #create stimulus spike 
        #!!! check indices !!!
        #A = 1/cv**2;
        #B = freq / A;
        #to create variability in ipi
        if cv==0:
            instfreq=freq
        else:
            instfreq=np.random.gamma(A,B)
            
        ipi=tmax/instfreq
        i+=ip #move for one ipi
        j+=1             #increase j

    return Istim, timespike



