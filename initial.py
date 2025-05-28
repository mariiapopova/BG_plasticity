# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 18:03:16 2021

@author: Maria
"""
import numpy as np
from createSMCinput import *
from FOGnetwork import *
from plotres import *
import time
#%matplotlib auto


def Initial():
    
    start_time = time.time()
    #%% Set initial conditions
    
    #time variables
    tmax=1000              #maximum time (ms)
    dt=0.01                #timestep (ms)
    t=np.arange(0,tmax,dt) #time vector
    n=12                   #number of neurons in each nucleus (TH, STN, GPe, GPi)
    f=130 #for continuous
    f1 = 200 #for theta high
    f2 = 50 #for theta low
    sw1 = 0 #theta is off
    sw2 = 1 #theta is on
    
    #initial membrane voltages for all cells - random is a little different from matlab
    v1=-62+np.random.randn(1,n)*5
    v2=-62+np.random.randn(1,n)*5
    v3=-62+np.random.randn(1,n)*5
    v4=-62+np.random.randn(1,n)*5
    v7=-62+np.random.randn(1,n)*5 #for striatum #previous 63.8!
    
    #Sensorimotor cortex input to talamic cells
    Istim, timespike = createSMCinput(tmax,dt,14,0.2)
    #%% Running FOGnetwork
    
    #healthy
    vsn, vge, vgi, vth, vstr, bla = FOGnetwork(0,0,0,Istim,timespike,tmax, dt, v1, v2, v3, v4, v7, n, sw1) #healthy
    #h = plotres(vsn, vge, vgi, vth, vstr, t, timespike, tmax, Istim, dt)
    #pd
    vsn1, vge1, vgi1, vth1, vstr1, bla1 = FOGnetwork(1,0,0,Istim,timespike,tmax, dt, v1, v2, v3, v4, v7, n, sw1) #PD
    #pd = plotres(vsn1, vge1, vgi1, vth1, vstr1, t,timespike,tmax,Istim,dt)
    #dbs cont
    vsn2, vge2, vgi2, vth2, vstr2, bla2 = FOGnetwork(1,1,f,Istim,timespike,tmax, dt, v1, v2, v3, v4, v7, n, sw1) #PD with DBS
    #dbs = plotres(vsn2, vge2, vgi2, vth2, vstr2, t,timespike,tmax,Istim,dt)
    #dbs theta high
    vsn3, vge3, vgi3, vth3, vstr3, bla3 = FOGnetwork(1,1,f1,Istim,timespike,tmax, dt, v1, v2, v3, v4, v7, n, sw2) #PD with DBS theta high
    #dbs2 = plotres(vsn3, vge3, vgi3, vth3, vstr3, t,timespike,tmax,Istim,dt)
    #dbs theta low
    vsn4, vge4, vgi4, vth4, vstr4, bla4 = FOGnetwork(1,1,f2,Istim,timespike,tmax, dt, v1, v2, v3, v4, v7, n, sw2) #PD with DBS theta low
    #dbs3 = plotres(vsn4, vge4, vgi4, vth4, vstr4, t,timespike,tmax,Istim,dt)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return (vsn, vge, vgi, vth, vstr, bla, \
           vsn1, vge1, vgi1, vth1, vstr1, bla1, \
           vsn2, vge2, vgi2, vth2, vstr2, bla2,\
           vsn3, vge3, vgi3, vth3, vstr3, bla3,\
           vsn4, vge4, vgi4, vth4, vstr4, bla4,\
           t, timespike, tmax)

if __name__ == "__main__":
    # execute only if run as a script
    fin_res=Initial()

# #%% Plotting routines
# #GPe spectras
# sig = fin_res[1][0]
# fft_res = np.abs(np.fft.rfft(np.fft.ifftshift(sig-np.mean(sig))))
# ff_freqs = np.fft.rfftfreq(n= sig.shape[0],d = 1e-5)

# sig1 = fin_res[7][0]
# fft_res1 = np.abs(np.fft.rfft(np.fft.ifftshift(sig1-np.mean(sig1))))
# ff_freqs1 = np.fft.rfftfreq(n= sig1.shape[0],d = 1e-5)

# sig2 = fin_res[13][0]
# fft_res2 = np.abs(np.fft.rfft(np.fft.ifftshift(sig2-np.mean(sig2))))
# ff_freqs2 = np.fft.rfftfreq(n= sig2.shape[0],d = 1e-5)

# sig3 = fin_res[19][0]
# fft_res3 = np.abs(np.fft.rfft(np.fft.ifftshift(sig3-np.mean(sig3))))
# ff_freqs3 = np.fft.rfftfreq(n= sig3.shape[0],d = 1e-5)

# sig4 = fin_res[25][0]
# fft_res4 = np.abs(np.fft.rfft(np.fft.ifftshift(sig4-np.mean(sig4))))
# ff_freqs4 = np.fft.rfftfreq(n= sig4.shape[0],d = 1e-5)

# plt.figure(figsize=(15,8))
# plt.plot(ff_freqs[:50],fft_res[:50],label='Healthy')
# plt.plot(ff_freqs1[:50],fft_res1[:50],label='PD')
# plt.plot(ff_freqs2[:50],fft_res2[:50],label='cDBS, 130 Hz')
# plt.plot(ff_freqs3[:50],fft_res3[:50],label='tDBS, 200 Hz')
# plt.plot(ff_freqs4[:50],fft_res4[:50],label='tDBS, 50 Hz')
# plt.ylabel('Amplitude [a.u.]')
# plt.xlabel('Frequency [Hz]')
# plt.legend()
# plt.title('GPe psd')
# plt.show()

# # f, t, Sxx = signal.spectrogram((fin_res[14][0][1000:]), 100e3, nperseg=20000)
# # plt.figure(figsize=(15,8))
# # plt.pcolormesh(t, f, Sxx, shading='gouraud')
# # plt.ylabel('Frequency [Hz]')
# # plt.xlabel('Time [sec]')
# # plt.ylim([0, 50])
# # plt.show()

# #to calculate crosscorelations for gpe:
# hcc = []
# pdcc = []
# dbscc = []
# dbs1cc = []
# dbs2cc = []
# for i in range(1,12):
#     hcc= np.append(hcc,np.max(np.correlate(fin_res[1][0], fin_res[1][i], 'full')))
#     pdcc= np.append(pdcc,np.max(np.correlate(fin_res[7][0], fin_res[7][i], 'full')))
#     dbscc = np.append(dbscc,np.max(np.correlate(fin_res[13][0], fin_res[13][i], 'full')))
#     dbs1cc = np.append(dbs1cc,np.max(np.correlate(fin_res[19][0], fin_res[19][i], 'full')))
#     dbs2cc = np.append(dbs2cc,np.max(np.correlate(fin_res[25][0], fin_res[25][i], 'full')))

# data = [hcc,pdcc,dbscc,dbs1cc,dbs2cc]
# fig=plt.figure()
# # Creating axes instance
# ax = fig.add_axes([0, 0, 1, 1])

# # Creating plot
# bp = ax.boxplot(data)
# plt.show()

# from scipy import stats
# a,b = stats.ttest_ind(hcc, pdcc)
# c,d = stats.ttest_ind(pdcc,dbscc)
# e,f = stats.ttest_ind(pdcc,dbs1cc)
# g,h = stats.ttest_ind(pdcc,dbs2cc)
# print("Healthy-PD:",a,b)
# print("PD-DBS:",c,d)
# print("PD-DBS1:",e,f)
# print("PD-DBS2:",g,h)