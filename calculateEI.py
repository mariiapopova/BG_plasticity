# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 18:56:23 2021

@author: Maria
"""
import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def calculateEI(t,vth,timespike,tmax):
    #Calculates the Error Index (EI)
    #Input:
    #t - time vector (msec)
    #vth - Array with membrane potentials of each thalamic cell
    #timespike - Time of each SMC input pulse
    #tmax - maximum time taken into consideration for calculation
    #Output:
    #er - Error index
    m=len(vth)

    e=np.zeros(m)
    b1=np.where(timespike>=200)[0][0] #ignore first 200msec
    b2=np.where(timespike<=tmax-25)[0][-1] #ignore last 25 msec
    
    a=np.empty(0).astype(np.int64)
    b=np.empty(0).astype(np.int64)
    
    
    for i in range(m):
        compare=np.empty(0)
        for j in range(1,len(vth[i,:])): 
            if vth[i,j-1]<-40 and vth[i,j]>-40:
                compare=np.hstack((compare,np.array([t[j]])))
        for p in range(b1,b2):
            if p!=b2:
                a=np.where(np.logical_and(compare>=timespike[p], compare<timespike[p]+25).astype(np.int64))[0]
                b=np.where(np.logical_and(compare>=timespike[p]+25, compare<timespike[p+1]).astype(np.int64))[0]
            elif b2==len(timespike):
                a=np.where(np.logical_and(compare>=timespike[p], compare<tmax).astype(np.int64))[0]
                b=np.empty(0).astype(np.int64)
            else:
                a=np.where(np.logical_and(compare>=timespike[p], compare<timespike[p+1]).astype(np.int64))[0]
                b=np.where(np.logical_and(compare>=timespike[p]+25, compare<timespike[p+1]).astype(np.int64))[0]
            
            if len(a)==0:
                e[i]+=1
            elif len(a)>1:
                e[i]+=1
            
            if len(b)!=0:
                e[i]+=len(b)
    
    res=np.mean(e/(b2-b1+1))

    return res
