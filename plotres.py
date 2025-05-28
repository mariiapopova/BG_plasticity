# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 18:45:58 2021

@author: Maria
"""
from findfreq import *
from calculateEI import *
import matplotlib.pyplot as plt

def plotres(vsn, vge, vgi, vth, vstr, t,timespike,tmax,Istim,dt):

    ##calculate freqeuncy for plotting
    fr1=findfreq(vsn[0,:])
    fr2=findfreq(vge[0,:])
    fr3=findfreq(vgi[0,:])  
    fr5=findfreq(vth[0,:])
    fr6=findfreq(vstr[0,:])
    
    ##Calculation of error index
    GN=calculateEI(t,vth,timespike,tmax) #for thalamus
    #GN=calculateEI(t,vppn,timespike,tmax) #for PPN
    
    titleval=GN #variable for plotting title
    
    ##Plots membrane potential for one cell in each nucleus  
    plt.figure() 
    plt.subplot(2,3,1) 
    plt.plot(t,vth[0,:])
    plt.plot(t,Istim[0:int(tmax/dt)],'r'); #plot for 1st neuron both Istim and Vth
    plt.xlim(0, tmax)
    plt.ylim(-100, 20)
    plt.title('Thalamus, FR: %s Hz' %(int(round(fr5))))
    plt.ylabel('Vm (mV)')
    plt.xlabel('Time (msec)')
    
    plt.subplot(2,3,2) #for 1st STN neuron
    plt.plot(t,vsn[0,:])
    plt.xlim(0, tmax)
    plt.ylim(-100, 80) 
    plt.title('STN, FR: %s Hz' %(int(round(fr1))))
    plt.ylabel('Vm (mV)') 
    plt.xlabel('Time (msec)')
    
    plt.subplot(2,3,3) #for 1st GPe neuron
    plt.plot(t,vge[0,:])
    plt.xlim(0, tmax)
    plt.ylim(-100, 80) 
    plt.title('GPe, FR: %s Hz' %(int(round(fr2))))
    plt.ylabel('Vm (mV)') 
    plt.xlabel('Time (msec)')
    
    plt.subplot(2,3,4) #for 1st GPi neuron
    plt.plot(t,vgi[0,:])
    plt.xlim(0, tmax)
    plt.ylim(-100, 80) 
    plt.title('GPi, FR: %s Hz' %(int(round(fr3))))
    plt.ylabel('Vm (mV)')
    plt.xlabel('Time (msec)')
    
    plt.subplot(2,3,5) #for 1st striatum neuron
    plt.plot(t,vstr[0,:])
    plt.xlim(0, tmax)
    plt.ylim(-100, 80) 
    plt.title('Striatum, FR: %s Hz' %(int(round(fr6))))
    plt.ylabel('Vm (mV)')
    plt.xlabel('Time (msec)')

    plt.suptitle('Firing patterns in freezing of gait network \n Thalamic relay EI: %s' %(titleval))
    
    plt.show()
    
    return GN