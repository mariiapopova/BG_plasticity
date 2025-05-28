# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:09:29 2021

@author: Maria
"""
from initial import Initial
import matplotlib.pyplot as plt
from findfreq import *
from calculateEI import *
import datetime
import pickle
import numpy as np

n=300 #number of runs
 
name = "Data/data_%s.pckl" % datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")
file = open(name, 'wb')

for i in range(n):
    
    vsn, vge, vgi, vth, vstr, bla,\
    vsn1, vge1, vgi1, vth1, vstr1, bla1,\
    vsn2, vge2, vgi2, vth2, vstr2, bla2,\
    vsn3, vge3, vgi3, vth3, vstr3, bla3,\
    vsn4, vge4, vgi4, vth4, vstr4, bla4,\
    t, timespike, tmax=Initial()
    
    #for healthy
    ##calculate freqeuncy for plotting
    fr1=findfreq(vsn[0,:])
    fr2=findfreq(vge[0,:])
    fr3=findfreq(vgi[0,:])  
    fr5=findfreq(vth[0,:])
    fr6=findfreq(vstr[0,:])

    ##Calculation of error index
    GN=calculateEI(t,vth,timespike,tmax) #for thalamus
    #GN=calculateEI(t,vppn,timespike,tmax) #for PPN
    
    
    #for pd
    ##calculate freqeuncy for plotting
    fr1_1=findfreq(vsn1[0,:])
    fr2_1=findfreq(vge1[0,:])
    fr3_1=findfreq(vgi1[0,:])  
    fr5_1=findfreq(vth1[0,:])
    fr6_1=findfreq(vstr1[0,:])
    
    ##Calculation of error index
    GN1=calculateEI(t,vth1,timespike,tmax) #for thalamus
    #GN1=calculateEI(t,vppn1,timespike,tmax) #for PPN
    
    
    #for dbs
    ##calculate freqeuncy for plotting
    fr1_2=findfreq(vsn2[0,:])
    fr2_2=findfreq(vge2[0,:])
    fr3_2=findfreq(vgi2[0,:])  
    fr5_2=findfreq(vth2[0,:])
    fr6_2=findfreq(vstr2[0,:])
    
    ##Calculation of error index
    GN2=calculateEI(t,vth2,timespike,tmax) #for thalamus
    #GN2=calculateEI(t,vppn,timespike,tmax) #for PPN

    #for dbs1
    ##calculate freqeuncy for plotting
    fr1_3=findfreq(vsn3[0,:])
    fr2_3=findfreq(vge3[0,:])
    fr3_3=findfreq(vgi3[0,:])  
    fr5_3=findfreq(vth3[0,:])
    fr6_3=findfreq(vstr3[0,:])
    
    ##Calculation of error index
    GN3=calculateEI(t,vth3,timespike,tmax) #for thalamus
    #GN2=calculateEI(t,vppn,timespike,tmax) #for PPN

    #for dbs2
    ##calculate freqeuncy for plotting
    fr1_4=findfreq(vsn4[0,:])
    fr2_4=findfreq(vge4[0,:])
    fr3_4=findfreq(vgi4[0,:])  
    fr5_4=findfreq(vth4[0,:])
    fr6_4=findfreq(vstr4[0,:])
    
    ##Calculation of error index
    GN4=calculateEI(t,vth4,timespike,tmax) #for thalamus
    #GN2=calculateEI(t,vppn,timespike,tmax) #for PPN

    #GPe spectras
    sig = vge[0]
    fft_res = np.abs(np.fft.rfft(np.fft.ifftshift(sig-np.mean(sig))))
    ff_freqs = np.fft.rfftfreq(n= sig.shape[0],d = 1e-5)

    sig1 = vge1[0]
    fft_res1 = np.abs(np.fft.rfft(np.fft.ifftshift(sig1-np.mean(sig1))))
    ff_freqs1 = np.fft.rfftfreq(n= sig1.shape[0],d = 1e-5)

    sig2 = vge2[0]
    fft_res2 = np.abs(np.fft.rfft(np.fft.ifftshift(sig2-np.mean(sig2))))
    ff_freqs2 = np.fft.rfftfreq(n= sig2.shape[0],d = 1e-5)

    sig3 = vge3[0]
    fft_res3 = np.abs(np.fft.rfft(np.fft.ifftshift(sig3-np.mean(sig3))))
    ff_freqs3 = np.fft.rfftfreq(n= sig3.shape[0],d = 1e-5)

    sig4 = vge4[0]
    fft_res4 = np.abs(np.fft.rfft(np.fft.ifftshift(sig4-np.mean(sig4))))
    ff_freqs4 = np.fft.rfftfreq(n= sig4.shape[0],d = 1e-5)
        
    ##create dictionaries
    h_data = {"sn": fr1, "ge": fr2, 'gi': fr3, 'th': fr5, 'str': fr6, 'ei': GN, 'four_freq': ff_freqs, 'four_res': fft_res}    

    pd_data = {"sn": fr1_1, "ge": fr2_1, 'gi': fr3_1, 'th': fr5_1, 'str': fr6_1, 'ei': GN1, 'four_freq': ff_freqs1, 'four_res': fft_res1}  

    dbs_data = {"sn": fr1_2, "ge": fr2_2, 'gi': fr3_2, 'th': fr5_2, 'str': fr6_2, 'ei': GN2, 'four_freq': ff_freqs2, 'four_res': fft_res2}  

    dbs_ht_data = {'four_freq': ff_freqs3, 'four_res': fft_res3}

    dbs_lt_data = {'four_freq': ff_freqs4, 'four_res': fft_res4}

    dbs_data1 = {"sn": fr1_3, "ge": fr2_3, 'gi': fr3_3, 'th': fr5_3, 'str': fr6_3, 'ei': GN3} 

    dbs_data2 = {"sn": fr1_4, "ge": fr2_4, 'gi': fr3_4, 'th': fr5_4, 'str': fr6_4, 'ei': GN4} 

    datalist = [h_data, pd_data, dbs_data, dbs_ht_data, dbs_lt_data, dbs_data1, dbs_data2]          
    
    ##write into a file
    pickle.dump(datalist, file)

    print(i+1)        
    
    
file.close()      

#%% Postprocessing
        
#Create an object generator function to load all the pickles
def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

#load pickles
items = loadall(name)   
c = list(items)
c_ar=np.array(c)

#initializing
sn_dbs=np.zeros(np.shape(c_ar)[0])
ge_dbs=np.zeros(np.shape(c_ar)[0])
gi_dbs=np.zeros(np.shape(c_ar)[0])
th_dbs=np.zeros(np.shape(c_ar)[0])
stri_dbs=np.zeros(np.shape(c_ar)[0])
ei_dbs=np.zeros(np.shape(c_ar)[0])

sn_pd=np.zeros(np.shape(c_ar)[0])
ge_pd=np.zeros(np.shape(c_ar)[0])
gi_pd=np.zeros(np.shape(c_ar)[0])
th_pd=np.zeros(np.shape(c_ar)[0])
stri_pd=np.zeros(np.shape(c_ar)[0])
ei_pd=np.zeros(np.shape(c_ar)[0])

sn=np.zeros(np.shape(c_ar)[0])
ge=np.zeros(np.shape(c_ar)[0])
gi=np.zeros(np.shape(c_ar)[0])
th=np.zeros(np.shape(c_ar)[0])
stri=np.zeros(np.shape(c_ar)[0])
ei=np.zeros(np.shape(c_ar)[0])

sn_dbs1=np.zeros(np.shape(c_ar)[0])
ge_dbs1=np.zeros(np.shape(c_ar)[0])
gi_dbs1=np.zeros(np.shape(c_ar)[0])
th_dbs1=np.zeros(np.shape(c_ar)[0])
stri_dbs1=np.zeros(np.shape(c_ar)[0])
ei_dbs1=np.zeros(np.shape(c_ar)[0])

sn_dbs2=np.zeros(np.shape(c_ar)[0])
ge_dbs2=np.zeros(np.shape(c_ar)[0])
gi_dbs2=np.zeros(np.shape(c_ar)[0])
th_dbs2=np.zeros(np.shape(c_ar)[0])
stri_dbs2=np.zeros(np.shape(c_ar)[0])
ei_dbs2=np.zeros(np.shape(c_ar)[0])

f_freq_h = np.zeros((len(c_ar[:,2][0]['four_freq']), np.shape(c_ar)[0]))
f_res_h = np.zeros((len(c_ar[:,2][0]['four_freq']), np.shape(c_ar)[0]))

f_freq_pd = np.zeros((len(c_ar[:,2][0]['four_freq']), np.shape(c_ar)[0]))
f_res_pd = np.zeros((len(c_ar[:,2][0]['four_freq']), np.shape(c_ar)[0]))

f_freq_dbs = np.zeros((len(c_ar[:,2][0]['four_freq']), np.shape(c_ar)[0]))
f_res_dbs = np.zeros((len(c_ar[:,2][0]['four_freq']), np.shape(c_ar)[0]))

f_freq_dbsht = np.zeros((len(c_ar[:,2][0]['four_freq']), np.shape(c_ar)[0]))
f_res_dbsht = np.zeros((len(c_ar[:,2][0]['four_freq']), np.shape(c_ar)[0]))

f_freq_dbslt = np.zeros((len(c_ar[:,2][0]['four_freq']), np.shape(c_ar)[0]))
f_res_dbslt = np.zeros((len(c_ar[:,2][0]['four_freq']), np.shape(c_ar)[0]))

#%%
#convenient format
for i in range(np.shape(c_ar)[0]):
    sn_dbs[i]=c_ar[:,2][i]['sn']
    ge_dbs[i]=c_ar[:,2][i]['ge']
    gi_dbs[i]=c_ar[:,2][i]['gi']
    th_dbs[i]=c_ar[:,2][i]['th']
    stri_dbs[i]=c_ar[:,2][i]['str']
    ei_dbs[i]=c_ar[:,2][i]['ei'] 

    sn_pd[i]=c_ar[:,1][i]['sn']
    ge_pd[i]=c_ar[:,1][i]['ge']
    gi_pd[i]=c_ar[:,1][i]['gi']
    th_pd[i]=c_ar[:,1][i]['th']
    stri_pd[i]=c_ar[:,1][i]['str']
    ei_pd[i]=c_ar[:,1][i]['ei']

    sn[i]=c_ar[:,0][i]['sn']
    ge[i]=c_ar[:,0][i]['ge']
    gi[i]=c_ar[:,0][i]['gi']
    th[i]=c_ar[:,0][i]['th']
    stri[i]=c_ar[:,0][i]['str']
    ei[i]=c_ar[:,0][i]['ei']

    f_freq_h[:,i]=c_ar[:,0][i]['four_freq']
    f_res_h[:,i]=c_ar[:,0][i]['four_res']

    f_freq_pd[:,i]=c_ar[:,1][i]['four_freq']
    f_res_pd[:,i]=c_ar[:,1][i]['four_res']

    f_freq_dbs[:,i]=c_ar[:,2][i]['four_freq']
    f_res_dbs[:,i]=c_ar[:,2][i]['four_res']

    f_freq_dbsht[:,i]=c_ar[:,3][i]['four_freq']
    f_res_dbsht[:,i]=c_ar[:,3][i]['four_res']

    f_freq_dbslt[:,i]=c_ar[:,4][i]['four_freq']
    f_res_dbslt[:,i]=c_ar[:,4][i]['four_res']

    sn_dbs1[i]=c_ar[:,5][i]['sn']
    ge_dbs1[i]=c_ar[:,5][i]['ge']
    gi_dbs1[i]=c_ar[:,5][i]['gi']
    th_dbs1[i]=c_ar[:,5][i]['th']
    stri_dbs1[i]=c_ar[:,5][i]['str']
    ei_dbs1[i]=c_ar[:,5][i]['ei'] 

    sn_dbs2[i]=c_ar[:,6][i]['sn']
    ge_dbs2[i]=c_ar[:,6][i]['ge']
    gi_dbs2[i]=c_ar[:,6][i]['gi']
    th_dbs2[i]=c_ar[:,6][i]['th']
    stri_dbs2[i]=c_ar[:,6][i]['str']
    ei_dbs2[i]=c_ar[:,6][i]['ei'] 

#means and stds
ei_mean=np.mean(ei)
sn_mean=np.mean(sn)
ge_mean=np.mean(ge)
gi_mean=np.mean(gi)
th_mean=np.mean(th)
stri_mean=np.mean(stri)

ei_std=np.std(ei)
sn_std=np.std(sn)
ge_std=np.std(ge)
gi_std=np.std(gi)
th_std=np.std(th)
stri_std=np.std(stri)

ei_mean_pd=np.mean(ei_pd)
sn_mean_pd=np.mean(sn_pd)
ge_mean_pd=np.mean(ge_pd)
gi_mean_pd=np.mean(gi_pd)
th_mean_pd=np.mean(th_pd)
stri_mean_pd=np.mean(stri_pd)

ei_std_pd=np.std(ei_pd)
sn_std_pd=np.std(sn_pd)
ge_std_pd=np.std(ge_pd)
gi_std_pd=np.std(gi_pd)
th_std_pd=np.std(th_pd)
stri_std_pd=np.std(stri_pd)

ei_mean_dbs=np.mean(ei_dbs)
sn_mean_dbs=np.mean(sn_dbs)
ge_mean_dbs=np.mean(ge_dbs)
gi_mean_dbs=np.mean(gi_dbs)
th_mean_dbs=np.mean(th_dbs)
stri_mean_dbs=np.mean(stri_dbs)

ei_std_dbs=np.std(ei_dbs)
sn_std_dbs=np.std(sn_dbs)
ge_std_dbs=np.std(ge_dbs)
gi_std_dbs=np.std(gi_dbs)
th_std_dbs=np.std(th_dbs)
stri_std_dbs=np.std(stri_dbs)

ei_mean_dbs1=np.mean(ei_dbs1)
sn_mean_dbs1=np.mean(sn_dbs1)
ge_mean_dbs1=np.mean(ge_dbs1)
gi_mean_dbs1=np.mean(gi_dbs1)
th_mean_dbs1=np.mean(th_dbs1)
stri_mean_dbs1=np.mean(stri_dbs1)

ei_std_dbs1=np.std(ei_dbs1)
sn_std_dbs1=np.std(sn_dbs1)
ge_std_dbs1=np.std(ge_dbs1)
gi_std_dbs1=np.std(gi_dbs1)
th_std_dbs1=np.std(th_dbs1)
stri_std_dbs1=np.std(stri_dbs1)

ei_mean_dbs2=np.mean(ei_dbs2)
sn_mean_dbs2=np.mean(sn_dbs2)
ge_mean_dbs2=np.mean(ge_dbs2)
gi_mean_dbs2=np.mean(gi_dbs2)
th_mean_dbs2=np.mean(th_dbs2)
stri_mean_dbs2=np.mean(stri_dbs2)

ei_std_dbs2=np.std(ei_dbs2)
sn_std_dbs2=np.std(sn_dbs2)
ge_std_dbs2=np.std(ge_dbs2)
gi_std_dbs2=np.std(gi_dbs2)
th_std_dbs2=np.std(th_dbs2)
stri_std_dbs2=np.std(stri_dbs2)

fr_mean_h = np.mean(f_freq_h,axis=1)
res_mean_h = np.mean(f_res_h,axis=1)
res_mean_pd = np.mean(f_res_pd,axis=1)
res_mean_dbs = np.mean(f_res_dbs,axis=1)
res_mean_dbsht = np.mean(f_res_dbsht,axis=1)
res_mean_dbslt = np.mean(f_res_dbslt,axis=1)
res_std_h = np.std(f_res_h,axis=1)
res_std_pd = np.std(f_res_pd,axis=1)
res_std_dbs = np.std(f_res_dbs,axis=1)
res_std_dbsht = np.std(f_res_dbsht,axis=1)
res_std_dbslt = np.std(f_res_dbslt,axis=1)

#%% plotting
state = ['Healthy', 'PD', 'DBS','DBS High Theta', 'DBS Low Theta']
x_pos = np.arange(len(state))

ei=np.array([ei_mean, ei_mean_pd, ei_mean_dbs,ei_mean_dbs1,ei_mean_dbs2])
ei_std_full=np.array([ei_std, ei_std_pd, ei_std_dbs,ei_std_dbs1,ei_std_dbs2])

sn=np.array([sn_mean, sn_mean_pd, sn_mean_dbs,sn_mean_dbs1,sn_mean_dbs2])
sn_std_full=np.array([sn_std, sn_std_pd, sn_std_dbs,sn_std_dbs1,sn_std_dbs2])

ge=np.array([ge_mean, ge_mean_pd, ge_mean_dbs,ge_mean_dbs1,ge_mean_dbs2])
ge_std_full=np.array([ge_std, ge_std_pd, ge_std_dbs,ge_std_dbs1,ge_std_dbs2])

gi=np.array([gi_mean, gi_mean_pd, gi_mean_dbs,gi_mean_dbs1,gi_mean_dbs2])
gi_std_full=np.array([gi_std, gi_std_pd, gi_std_dbs,gi_std_dbs1,gi_std_dbs2])

th=np.array([th_mean, th_mean_pd, th_mean_dbs,th_mean_dbs1,th_mean_dbs2])
th_std_full=np.array([th_std, th_std_pd, th_std_dbs,th_std_dbs1,th_std_dbs2])

stri=np.array([stri_mean, stri_mean_pd, stri_mean_dbs,stri_mean_dbs1,stri_mean_dbs2])
stri_std_full=np.array([stri_std, stri_std_pd, stri_std_dbs,stri_std_dbs1,stri_std_dbs2])

def plot(mean, std, title, lable):
    fig, ax = plt.subplots()
    ax.bar(x_pos, mean, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel(lable)
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(state)
    ax.yaxis.grid(True)
    #plt.show()
    return fig
    
plot(ei, ei_std_full, "Error Index", "Error Index")
plot(sn, sn_std_full, "STN", "Firing rate, Hz")
plot(ge, ge_std_full, "GPe", "Firing rate, Hz")
plot(gi, gi_std_full, "GPi", "Firing rate, Hz")
plot(th, th_std_full, "Thalamus", "Firing rate, Hz")
plot(stri, stri_std_full, "Striatum", "Firing rate, Hz")

plt.figure(figsize=(15,8))
plt.plot(fr_mean_h[:50],res_mean_h[:50],label='Healthy')
plt.fill_between(fr_mean_h[:50],res_mean_h[:50]-res_std_h[:50],res_mean_h[:50]+res_std_h[:50],alpha=0.1,color='blue')
plt.plot(fr_mean_h[:50],res_mean_pd[:50],label='PD')
plt.fill_between(fr_mean_h[:50],res_mean_pd[:50]-res_std_pd[:50],res_mean_pd[:50]+res_std_pd[:50],alpha=0.1,color='orange')
plt.plot(fr_mean_h[:50],res_mean_dbs[:50],label='cDBS, 130 Hz')
plt.fill_between(fr_mean_h[:50],res_mean_dbs[:50]-res_std_dbs[:50],res_mean_dbs[:50]+res_std_dbs[:50],alpha=0.1,color='green')
plt.plot(fr_mean_h[:50],res_mean_dbsht[:50],label='tDBS, 200 Hz')
plt.fill_between(fr_mean_h[:50],res_mean_dbsht[:50]-res_std_dbsht[:50],res_mean_dbsht[:50]+res_std_dbsht[:50],alpha=0.1,color='red')
plt.plot(fr_mean_h[:50],res_mean_dbslt[:50],label='tDBS, 50 Hz')
plt.fill_between(fr_mean_h[:50],res_mean_dbslt[:50]-res_std_dbslt[:50],res_mean_dbslt[:50]+res_std_dbslt[:50],alpha=0.1,color='purple')
plt.ylabel('Amplitude [a.u.]')
plt.xlabel('Frequency [Hz]')
plt.legend()
plt.title('Mean GPe psd')
plt.show()
