# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:06:44 2021

@author: Maria
"""
import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def createdbs(f,tmax,dt):
    #Creates DBS train of frequency f, of length tmax (msec) with time step dt (msec)

    t=np.arange(0,tmax,dt)    #create a time vector
    DBS=np.zeros(len(t))  #create zeros for DBS
    amp=300                   #muAmp/cm^2
    pulse=amp*np.ones(int(0.3/dt)) #create a pulse with an amp and a width 0.3ms

    i=0 #set a counter
    
    #loop to create DBS train
    while i<len(t):
        DBS[i:i+int(0.3/dt)]=pulse #create a pulse
        instfreq=f            #set frequency
        isi=tmax/instfreq     #inter-spike interval in ms
        i+=int(round(isi/dt))      #move i

    return DBS