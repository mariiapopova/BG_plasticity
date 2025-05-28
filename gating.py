import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def Hinf(V):
    return 1/(1+np.exp(-(V+57)/2))

#GPe, GPi, SNr gating parameters
    
@jit(nopython=True, cache=True)
def gpe_ainf(V):
    return 1/(1+np.exp(-(V+57)/2))

@jit(nopython=True, cache=True)
def gpe_hinf(V):
    return 1/(1+np.exp((V+58)/12))

@jit(nopython=True, cache=True)
def gpe_minf(V):
    return 1/(1+np.exp(-(V+37)/10))

@jit(nopython=True, cache=True)
def gpe_ninf(V):
    return 1/(1+np.exp(-(V+50)/14))

@jit(nopython=True, cache=True)
def gpe_rinf(V):
    return 1/(1+np.exp((V+70)/2))

@jit(nopython=True, cache=True)
def gpe_sinf(V):
    return 1/(1+np.exp(-(V+35)/2))

@jit(nopython=True, cache=True)
def gpe_tauh(V):
    return 0.05+0.27/(1+np.exp(-(V+40)/-12))

@jit(nopython=True, cache=True)
def gpe_taun(V):
    return 0.05+0.27/(1+np.exp(-(V+40)/-12))

#LC gating parameters
    
@jit(nopython=True, cache=True)    
def lc_binf(V):
    return (1/(1+np.exp(0.069*(V+53.3))))**4

@jit(nopython=True, cache=True)
def lc_minf(V):
    a=0.1*(V+29.7)/(1-np.exp(-(V+29.7)/10))
    b=4*np.exp(-(V+54.7)/18)
    return a/(a+b)

@jit(nopython=True, cache=True)
def lc_qinf(V):
    a=0.01*(V+45.7)/(1-np.exp(-(V+45.7)/10))
    b=0.125*np.exp(-(V+55.7)/80)
    ninf=a/(a+b)
    binf=(1/(1+np.exp(0.069*(V+53.3))))**4
    return ninf**4+(0.21*(47.7/20))*binf

@jit(nopython=True, cache=True)
def lc_tauq(V):
    tb=1.24+2.678/(1+np.exp((V+50)/16.027))
    a=0.01*(V+45.7)/(1-np.exp(-(V+45.7)/10))
    b=0.125*np.exp(-(V+55.7)/80)
    tn=0.52/(a+b)
    return (tb+tn)/2

#PPN gating parameters
 
@jit(nopython=True, cache=True)
def ppn_hinf(V):
    alpha=0.12*np.exp(-(V+51)/18)
    beta=4/(1+np.exp(-(V+28)/5))
    return alpha/(alpha+beta)

@jit(nopython=True, cache=True)
def ppn_taumt(V):
    return (0.612+1/(np.exp(-(V+134)/16.7)+np.exp((V+18.8)/18.2)))/6.9

@jit(nopython=True, cache=True)
def ppn_taumk(V):
    alpha=0.032*(V+63.8)/(1-np.exp(-(V+63.8)/5))
    beta=0.5*(np.exp(-(V+68.8)/40))
    return 1/(alpha+beta)

@jit(nopython=True, cache=True)
def ppn_taumh(V):
    return 1/(np.exp(-15.45-0.086*V)+np.exp(-1.17+0.0701*V))

@jit(nopython=True, cache=True)
def ppn_taum(V):
    alpha=0.32*(V+55)/(1-np.exp(-(V+55)/4))
    beta=-0.28*(V+28)/(1-np.exp((V+28)/5))
    return 1/(alpha+beta)

@jit(nopython=True, cache=True)
def ppn_tauht(V):
    if ((V+2)<-81).any():
        tau=np.exp((V+469)/66.6)/3.74
    else:
        tau=(28+np.exp(-(V+24)/10.5))/3.74
    return tau

@jit(nopython=True, cache=True)
def ppn_tauhnap(V):
    return 6000/np.cosh((V+48)/6)

@jit(nopython=True, cache=True)
def ppn_tauh(V):
    alpha=0.12*np.exp(-(V+51)/18)
    beta=4/(1+np.exp(-(V+28)/5))
    return 1/(alpha+beta)

@jit(nopython=True, cache=True)
def ppn_mtinf(V):
    return 1/(1+np.exp(-(V+59)/6.2))

@jit(nopython=True, cache=True)
def ppn_mnapinf(V):
    return 1/(1+np.exp((V+40)/(-6)))

@jit(nopython=True, cache=True)
def ppn_mkinf(V):
    alpha=0.032*(V+63.8)/(1-np.exp(-(V+63.8)/5))
    beta=0.5*(np.exp(-(V+68.8)/40))
    return alpha/(alpha+beta)

@jit(nopython=True, cache=True)
def ppn_minf(V):
    alpha=0.32*(V+55)/(1-np.exp(-(V+55)/4))
    beta=-0.28*(V+28)/(1-np.exp((V+28)/5))
    return alpha/(alpha+beta)

@jit(nopython=True, cache=True)
def ppn_mhinf(V):
    return 1/(1+np.exp((V+85)/5.5))

@jit(nopython=True, cache=True)
def ppn_htinf(V):
    return 1/(1+np.exp((V+82)/4))

@jit(nopython=True, cache=True)
def ppn_hnapinf(V):
    return 1/(1+np.exp((V+48)/6))

#STN gating parameters
@jit(nopython=True, cache=True)
def stn_taur(V):
    return 7.1+17.5/(1+np.exp(-(V-68)/-2.2))

@jit(nopython=True, cache=True)
def stn_taun(V):
    return 1+100/(1+np.exp(-(V+80)/-26))

@jit(nopython=True, cache=True)
def stn_tauh(V):
    return 1+500/(1+np.exp(-(V+57)/-3))

@jit(nopython=True, cache=True)
def stn_tauc(V):
    return 1+10/(1+np.exp((V+80)/26))

@jit(nopython=True, cache=True)
def stn_rinf(V):
    return 1/(1+np.exp((V+67)/2))

@jit(nopython=True, cache=True)
def stn_ninf(V):
    return 1/(1+np.exp(-(V+32)/8.0))

@jit(nopython=True, cache=True)
def stn_minf(V):
    return 1/(1+np.exp(-(V+30)/15))

@jit(nopython=True, cache=True)
def stn_hinf(V):
    return 1/(1+np.exp((V+39)/3.1))

@jit(nopython=True, cache=True)
def stn_cinf(V):
    return 1/(1+np.exp(-(V+20)/8))

@jit(nopython=True, cache=True)
def stn_binf(R):
    return 1/(1+np.exp(-(R-0.4)/0.1))-1/(1+np.exp(0.4/0.1))

@jit(nopython=True, cache=True)
def stn_ainf(V):
    return 1/(1+np.exp(-(V+63)/7.8))

#Striatum gating parameters
    
@jit(nopython=True, cache=True)
def str_Ggaba(V):
    return 2*(1+np.tanh(V/4))

@jit(nopython=True, cache=True)
def str_betap(V):
    return (-3.209*10**-4*(30+V))/(1-np.exp((30+V)/9))

@jit(nopython=True, cache=True)
def str_betan(V):
    return 0.5*np.exp((-57-V)/40)

@jit(nopython=True, cache=True)
def str_betam(V):
    return 0.28*(V+27)/((np.exp((27+V)/5))-1)

@jit(nopython=True, cache=True)
def str_betah(V):
    return 4/(1+np.exp((-27-V)/5))

@jit(nopython=True, cache=True)
def str_alphap(V):
    return (3.209*10**-4*(30+V))/(1-np.exp((-30-V)/9))

@jit(nopython=True, cache=True)
def str_alphan(V):
    return (0.032*(52+V))/(1-np.exp((-52-V)/5))

@jit(nopython=True, cache=True)
def str_alpham(V):
    return (0.32*(54+V))/(1-np.exp((-54-V)/4))

@jit(nopython=True, cache=True)
def str_alphah(V):
    return 0.128*np.exp((-50-V)/18)

#Thalamus gating parameters
 
@jit(nopython=True, cache=True)
def th_taur(V):
    return 0.15*(28+np.exp(-(V+25)/10.5))

@jit(nopython=True, cache=True)
def th_tauh(V):
    a=0.128*np.exp(-(V+46)/18)
    b=4/(1+np.exp(-(V+23)/5))
    return 1/(a+b)

@jit(nopython=True, cache=True)
def th_rinf(V):
    return 1/(1+np.exp((V+84)/4))

@jit(nopython=True, cache=True)
def th_pinf(V):
    return 1/(1+np.exp(-(V+60)/6.2))

@jit(nopython=True, cache=True)
def th_minf(V):
    return 1/(1+np.exp(-(V+37)/7))

@jit(nopython=True, cache=True)
def th_hinf(V):
    return 1/(1+np.exp((V+41)/4))

