# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:52:01 2021

@author: Maria
"""
import scipy.signal
import numpy as np

def findfreq(sig): #in Hz
    #delta = 1e-5 #remember that only with dt 0.01 and t 1000 ms
    val = sig
    h=scipy.signal.detrend(val)
    peaks=scipy.signal.find_peaks(h,height=23)[0]
    #fr=1/(np.mean(np.diff(peaks))*delta)
    fr=np.size(peaks)
    if np.size(peaks)<1:
        fr=0
           
    return fr